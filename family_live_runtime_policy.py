from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FAMILY_CONFIGS = {
    "btc": {
        "supported_models": (
            "entry_edge",
            "fill_probability",
            "slippage_liquidity",
            "exit_quality",
            "regime_calibration",
        ),
        "required_live_entry_models": (
            "entry_edge",
            "fill_probability",
            "slippage_liquidity",
            "regime_calibration",
        ),
        "required_live_exit_models": ("exit_quality",),
        "live_scope_policy": "btc_full_stack_required_for_live_entries",
    },
    "weather_temperature": {
        "supported_models": ("entry_edge", "exit_quality", "regime_calibration"),
        "required_live_entry_models": ("entry_edge", "regime_calibration"),
        "required_live_exit_models": ("exit_quality",),
        "live_scope_policy": "weather_forecast_and_regime_only_until_execution_models_exist",
    },
}


@dataclass(frozen=True)
class FamilyLiveRuntimeState:
    family: str
    signal_present: bool
    supported_models: tuple[str, ...]
    required_live_entry_models: tuple[str, ...]
    required_live_exit_models: tuple[str, ...]
    live_scope_policy: str
    promoted_models: tuple[str, ...]
    entry_ready: bool
    exit_ready: bool
    missing_entry_models: tuple[str, ...]
    missing_exit_models: tuple[str, ...]

    @property
    def live_ready(self) -> bool:
        return self.entry_ready and self.exit_ready

    @property
    def reason(self) -> str:
        parts: list[str] = []
        if not self.signal_present:
            parts.append("no_offline_signal")
        if self.missing_entry_models:
            parts.append("missing_entry_models=" + ",".join(self.missing_entry_models))
        if self.missing_exit_models:
            parts.append("missing_exit_models=" + ",".join(self.missing_exit_models))
        return "; ".join(parts) if parts else "approved"


def _normalize_family_configs(raw_configs: dict | None) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for family, defaults in DEFAULT_FAMILY_CONFIGS.items():
        raw = raw_configs.get(family, {}) if isinstance(raw_configs, dict) else {}
        configs[family] = {
            "supported_models": tuple(raw.get("supported_models") or defaults["supported_models"]),
            "required_live_entry_models": tuple(raw.get("required_live_entry_models") or defaults["required_live_entry_models"]),
            "required_live_exit_models": tuple(raw.get("required_live_exit_models") or defaults["required_live_exit_models"]),
            "live_scope_policy": str(raw.get("live_scope_policy") or defaults["live_scope_policy"]),
        }
    return configs


def build_family_live_runtime_policy(payload: dict | None) -> dict[str, FamilyLiveRuntimeState]:
    payload = payload if isinstance(payload, dict) else {}
    configs = _normalize_family_configs(payload.get("family_configs"))
    promoted_by_family = {family: set() for family in configs}
    for row in payload.get("results", []) if isinstance(payload.get("results"), list) else []:
        family = str(row.get("family") or "").strip().lower()
        model_name = str(row.get("model_name") or "").strip()
        if family in promoted_by_family and model_name:
            promoted = row.get("promoted") is True or str(row.get("promoted", "")).strip().lower() in {"1", "true", "yes", "on"}
            if promoted:
                promoted_by_family[family].add(model_name)

    signal_present = bool(payload)
    states: dict[str, FamilyLiveRuntimeState] = {}
    for family, cfg in configs.items():
        promoted_models = tuple(sorted(promoted_by_family.get(family, set())))
        promoted_set = set(promoted_models)
        missing_entry = tuple(m for m in cfg["required_live_entry_models"] if m not in promoted_set)
        missing_exit = tuple(m for m in cfg["required_live_exit_models"] if m not in promoted_set)
        states[family] = FamilyLiveRuntimeState(
            family=family,
            signal_present=signal_present,
            supported_models=tuple(cfg["supported_models"]),
            required_live_entry_models=tuple(cfg["required_live_entry_models"]),
            required_live_exit_models=tuple(cfg["required_live_exit_models"]),
            live_scope_policy=cfg["live_scope_policy"],
            promoted_models=promoted_models,
            entry_ready=(signal_present and not missing_entry),
            exit_ready=(signal_present and not missing_exit),
            missing_entry_models=missing_entry,
            missing_exit_models=missing_exit,
        )
    return states


def load_family_live_runtime_policy(signal_file: str | Path) -> dict[str, FamilyLiveRuntimeState]:
    path = Path(signal_file)
    if not path.exists():
        return build_family_live_runtime_policy({})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return build_family_live_runtime_policy({})
    return build_family_live_runtime_policy(payload)
