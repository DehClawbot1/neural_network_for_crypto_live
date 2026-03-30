"""
Hardware detection and optimization for AMD Ryzen 7 5700X + Radeon RX 9060 XT.

Provides a single HW object that every training module can import to get:
  - n_jobs for sklearn parallelism
  - LightGBM device params
  - PyTorch device selection (DirectML on Windows for AMD GPUs)
  - Thread/worker counts calibrated to 8C/16T
"""

import os
import multiprocessing
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── CPU ──────────────────────────────────────────────────────────────────────
PHYSICAL_CORES = 8
LOGICAL_CORES = 16
# Leave 2 logical cores free for OS / dashboard / websocket
SKLEARN_JOBS = min(LOGICAL_CORES - 2, multiprocessing.cpu_count() - 2)
SKLEARN_JOBS = max(1, SKLEARN_JOBS)

# ── GPU detection ────────────────────────────────────────────────────────────
_gpu_backend = None  # "directml", "rocm", "cuda", or None
_torch_device = "cpu"

def _detect_gpu():
    global _gpu_backend, _torch_device

    # 1. Try DirectML (best path for AMD GPUs on Windows)
    try:
        import torch_directml  # noqa: F401
        _gpu_backend = "directml"
        _torch_device = "privateuseone:0"
        logging.info("[HW] AMD GPU detected via DirectML (torch-directml)")
        return
    except ImportError:
        pass

    # 2. Try ROCm (Linux AMD path)
    try:
        import torch
        if torch.cuda.is_available() and "AMD" in (torch.cuda.get_device_name(0) or ""):
            _gpu_backend = "rocm"
            _torch_device = "cuda:0"
            logging.info("[HW] AMD GPU detected via ROCm/HIP")
            return
    except Exception:
        pass

    # 3. Try CUDA (NVIDIA fallback)
    try:
        import torch
        if torch.cuda.is_available():
            _gpu_backend = "cuda"
            _torch_device = "cuda:0"
            logging.info("[HW] NVIDIA GPU detected via CUDA")
            return
    except Exception:
        pass

    logging.info("[HW] No GPU acceleration available; using CPU with %d threads", SKLEARN_JOBS)

_detect_gpu()


# ── Public API ───────────────────────────────────────────────────────────────

def get_torch_device():
    """Return the best available torch device string."""
    return _torch_device

def get_gpu_backend():
    """Return 'directml', 'rocm', 'cuda', or None."""
    return _gpu_backend

def get_sklearn_jobs():
    """Return n_jobs for sklearn parallel estimators."""
    return SKLEARN_JOBS

def get_lightgbm_params():
    """
    Return extra params dict for LightGBM GPU acceleration.
    AMD GPUs support OpenCL-based GPU training if LightGBM was built with GPU support.
    """
    # LightGBM GPU requires OpenCL; AMD GPUs support this on Windows
    # but the pip wheel may not include GPU support.  Try it and fall back.
    try:
        import lightgbm as lgb
        # Quick probe: if gpu_use_dp param is accepted, GPU build is present
        booster = lgb.Booster(params={"device": "gpu", "gpu_use_dp": False, "num_leaves": 2})
        del booster
        logging.info("[HW] LightGBM GPU mode available")
        return {"device": "gpu", "gpu_use_dp": False}
    except Exception:
        logging.info("[HW] LightGBM GPU not available; using CPU with %d threads", SKLEARN_JOBS)
        return {"n_jobs": SKLEARN_JOBS}

def get_parallel_env_vars():
    """Set environment variables for maximum CPU utilisation."""
    env = {
        "OMP_NUM_THREADS": str(PHYSICAL_CORES),
        "MKL_NUM_THREADS": str(PHYSICAL_CORES),
        "OPENBLAS_NUM_THREADS": str(PHYSICAL_CORES),
        "NUMEXPR_MAX_THREADS": str(PHYSICAL_CORES),
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env

# Apply CPU threading env vars on import
get_parallel_env_vars()
