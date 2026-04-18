"""Champion/Challenger evaluation system.

Keeps two model slots at all times:
  champion  — the current production model (serves live decisions)
  challenger — trained on fresher data, runs in shadow mode only

Shadow mode means the challenger scores every candidate signal without
affecting any trade decision.  Its predictions are logged alongside the
champion's live decisions.  After enough shadow cycles and enough trades
where we can compare both predictions against actual outcomes, the
challenger is promoted only if it is measurably better.

PROMOTION CRITERIA (all must pass):
  1. min_shadow_cycles   — challenger must complete N shadow cycles (default 3)
  2. min_shadow_trades   — must have observed at least M trades where we can
                           compare both models (default 50)
  3. sharpe_delta        — challenger Sortino on shadow trades > champion + margin
  4. precision_delta     — challenger precision on shadow trades > champion + margin
  5. no_regression_gate  — challenger must not significantly underperform champion
                           on any sub-window (prevents lucky aggregate hiding a regime)

DEMOTION:
  If the champion degrades on live trades (rolling Sortino drops below a floor),
  the last known challenger is immediately re-evaluated.  If it passes the
  single-window comparison it is promoted directly (emergency fallback path).

STORAGE:
  weights/{family}/champion/          — production model (always present)
  weights/{family}/challenger/        — challenger model (may not exist)
  logs/{family}/shadow_eval.csv       — per-signal shadow scores (rolling window)
  logs/{family}/champion_challenger.json — state (cycle count, trade count, etc.)

INTEGRATION:
  1. Offline training loop calls ChallengerSlot.register_challenger() after training
  2. Supervisor calls ChallengerSlot.score_shadow(signal_row) on every candidate
     signal — returns {challenger_prob, challenger_action} for logging only
  3. After each trade closes, call ChallengerSlot.record_outcome(trade_result)
  4. ChallengerSlot.try_promote() returns True if promotion conditions met,
     and atomically swaps champion ← challenger
  5. Retrainer calls ChallengerSlot.try_promote() after each shadow cycle
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Shadow evaluation record
# ---------------------------------------------------------------------------

@dataclass
class ShadowRecord:
    """One row of shadow evaluation: both models scored the same signal."""
    shadow_at:          str
    family:             str
    signal_id:          str          # token_id or unique key
    signal_label:       str
    champion_prob:      float | None
    champion_action:    str          # "buy" | "skip" | "unavailable"
    challenger_prob:    float | None
    challenger_action:  str
    outcome:            float | None = None   # realized return, filled after close
    outcome_at:         str | None   = None
    outcome_binary:     int | None   = None   # 1 = positive return, 0 = loss

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class ChampionChallengerState:
    family:              str
    champion_version:    str = ""
    challenger_version:  str = ""
    shadow_cycles:       int = 0    # number of times challenger ran shadow pass
    shadow_trades_seen:  int = 0    # trades where both models were scored + outcome known
    last_shadow_at:      str = ""
    last_promote_at:     str = ""
    last_promote_reason: str = ""
    challenger_sharpe:   float = float("nan")
    champion_sharpe:     float = float("nan")
    challenger_precision: float = float("nan")
    champion_precision:   float = float("nan")
    promoted:            bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChampionChallengerState":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ---------------------------------------------------------------------------
# Helper: Sortino ratio
# ---------------------------------------------------------------------------

def _sortino(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    if len(returns) < 5:
        return float("nan")
    mean_r = float(np.mean(returns))
    down   = returns[returns < 0]
    if len(down) == 0:
        return 10.0
    dd_std = float(np.std(down, ddof=1))
    if dd_std == 0:
        return float("nan")
    return float(mean_r / dd_std * math.sqrt(periods_per_year))


def _precision(actions: pd.Series, outcomes: pd.Series, threshold: str = "buy") -> float:
    """Precision of 'buy' actions: fraction of buy decisions that resulted in positive return."""
    bought = actions == threshold
    if bought.sum() == 0:
        return float("nan")
    return float(outcomes[bought].mean())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ChallengerSlot:
    """
    Manages champion/challenger lifecycle for one market family.

    Designed to be instantiated once per process and reused across retrain cycles.

    Thread safety: single-process assumption (no locking).
    """

    # Promotion thresholds — all overridable via env vars
    _MIN_SHADOW_CYCLES   = _env_int("CC_MIN_SHADOW_CYCLES", 3)
    _MIN_SHADOW_TRADES   = _env_int("CC_MIN_SHADOW_TRADES", 50)
    _SHARPE_MARGIN       = _env_float("CC_SHARPE_MARGIN", 0.0)       # challenger must beat champ by this
    _PRECISION_MARGIN    = _env_float("CC_PRECISION_MARGIN", 0.01)   # +1pp precision minimum
    _REGRESSION_WINDOW   = _env_int("CC_REGRESSION_WINDOW", 20)      # sub-window size for regression check
    _REGRESSION_FLOOR    = _env_float("CC_REGRESSION_FLOOR", -0.10)  # challenger sortino delta floor in sub-windows
    _EMERGENCY_FLOOR     = _env_float("CC_CHAMPION_EMERGENCY_FLOOR", -0.30)  # champion sortino floor before demotion

    _SHADOW_LOG_MAX_ROWS = 5000

    def __init__(
        self,
        family: str,
        weights_dir: str = "weights",
        logs_dir:    str = "logs",
    ) -> None:
        self.family      = family.lower().strip()
        self.weights_dir = Path(weights_dir)
        self.logs_dir    = Path(logs_dir)

        self._champion_dir   = self.weights_dir / self.family / "champion"
        self._challenger_dir = self.weights_dir / self.family / "challenger"
        self._shadow_log     = self.logs_dir / self.family / "shadow_eval.csv"
        self._state_path     = self.logs_dir / self.family / "champion_challenger.json"

        self._champion_dir.mkdir(parents=True, exist_ok=True)
        self._challenger_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / self.family).mkdir(parents=True, exist_ok=True)

        self._state = self._load_state()

        # Cached model objects (loaded lazily)
        self._champion_model: dict | None   = None
        self._challenger_model: dict | None = None

    # -- State persistence ---------------------------------------------------

    def _load_state(self) -> ChampionChallengerState:
        if self._state_path.exists():
            try:
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                return ChampionChallengerState.from_dict(raw)
            except Exception:
                pass
        return ChampionChallengerState(family=self.family)

    def _save_state(self) -> None:
        try:
            self._state_path.write_text(
                json.dumps(self._state.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("[CC/%s] state save failed: %s", self.family, exc)

    # -- Model loading -------------------------------------------------------

    def _load_model(self, directory: Path) -> dict | None:
        """Load the task_entry_edge artifact from a slot directory."""
        import joblib
        for fname in [
            "task_entry_edge.joblib",
            "tp_classifier.joblib",
            "stage1_tp_classifier.joblib",
            "meta_model_bundle.pkl",
        ]:
            p = directory / fname
            if p.exists():
                try:
                    bundle = joblib.load(p)
                    if isinstance(bundle, dict) and "model" in bundle:
                        log.debug("[CC/%s] loaded model from %s", self.family, p)
                        return bundle
                except Exception as exc:
                    log.warning("[CC/%s] failed to load %s: %s", self.family, p, exc)
        return None

    @property
    def champion(self) -> dict | None:
        if self._champion_model is None:
            self._champion_model = self._load_model(self._champion_dir)
        return self._champion_model

    @property
    def challenger(self) -> dict | None:
        if self._challenger_model is None:
            self._challenger_model = self._load_model(self._challenger_dir)
        return self._challenger_model

    def _invalidate_cache(self) -> None:
        self._champion_model   = None
        self._challenger_model = None

    # -- Slot management -----------------------------------------------------

    def register_challenger(
        self,
        source_dir: str | Path,
        version_label: str | None = None,
    ) -> bool:
        """
        Copy freshly trained artifacts into the challenger slot.
        Does NOT touch the champion slot.

        source_dir: the candidate weights dir produced by train_brain_models()
                    or TaskModelSuite.train_all()
        """
        src = Path(source_dir)
        if not src.exists():
            log.warning("[CC/%s] register_challenger: source dir not found: %s", self.family, src)
            return False

        # Wipe old challenger
        shutil.rmtree(self._challenger_dir, ignore_errors=True)
        self._challenger_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for f in src.iterdir():
            if f.suffix in (".joblib", ".pkl", ".zip", ".json"):
                try:
                    shutil.copy2(f, self._challenger_dir / f.name)
                    copied += 1
                except Exception as exc:
                    log.warning("[CC/%s] copy %s failed: %s", self.family, f.name, exc)

        if copied == 0:
            log.warning("[CC/%s] no artifacts copied from %s", self.family, src)
            return False

        version = version_label or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        self._state.challenger_version  = version
        self._state.shadow_cycles       = 0
        self._state.shadow_trades_seen  = 0
        self._state.promoted            = False
        self._save_state()
        self._invalidate_cache()

        log.info("[CC/%s] challenger registered: version=%s  files=%d", self.family, version, copied)
        return True

    def _install_champion(self, source_dir: Path, version: str) -> None:
        """Atomically replace champion slot from source_dir."""
        shutil.rmtree(self._champion_dir, ignore_errors=True)
        self._champion_dir.mkdir(parents=True, exist_ok=True)
        for f in source_dir.iterdir():
            if f.suffix in (".joblib", ".pkl", ".zip", ".json"):
                shutil.copy2(f, self._champion_dir / f.name)
        self._state.champion_version = version
        self._invalidate_cache()
        log.info("[CC/%s] champion installed: version=%s", self.family, version)

    def bootstrap_champion_from_weights(self) -> bool:
        """
        If the champion slot is empty, copy the current active weights
        (weights/{family}/*.joblib) into it as the initial champion.
        Called once at startup.
        """
        if list(self._champion_dir.glob("*.joblib")) or list(self._champion_dir.glob("*.pkl")):
            return False   # already seeded

        src = self.weights_dir / self.family
        copied = 0
        for f in src.glob("task_*.joblib"):
            try:
                shutil.copy2(f, self._champion_dir / f.name)
                copied += 1
            except Exception:
                pass
        if copied:
            version = "bootstrap_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            self._state.champion_version = version
            self._save_state()
            self._invalidate_cache()
            log.info("[CC/%s] champion bootstrapped from %s (%d files)", self.family, src, copied)
        return copied > 0

    # -- Shadow scoring ------------------------------------------------------

    def score_shadow(self, signal_row: dict) -> dict[str, Any]:
        """
        Score a signal row with both champion and challenger.
        Returns a dict of shadow metadata (does NOT modify signal_row).
        This is called on every candidate — the result is logged but never
        used to make a trade decision.
        """
        result: dict[str, Any] = {
            "champion_prob":     None,
            "champion_action":   "unavailable",
            "challenger_prob":   None,
            "challenger_action": "unavailable",
        }

        for slot_name, bundle, prefix in [
            ("champion",   self.champion,   "champion"),
            ("challenger", self.challenger, "challenger"),
        ]:
            if bundle is None:
                continue
            try:
                model    = bundle["model"]
                features = bundle.get("features", [])
                if not features or model is None:
                    continue
                present = [f for f in features if f in signal_row]
                if not present:
                    continue
                X = pd.DataFrame([{f: signal_row.get(f, float("nan")) for f in features}])
                # Encode object columns
                from sklearn.preprocessing import LabelEncoder
                for c in X.columns:
                    if X[c].dtype == object:
                        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
                from sklearn.impute import SimpleImputer
                X_imp = SimpleImputer(strategy="median").fit_transform(X)
                pm = model.predict_proba(X_imp)
                prob = float(pm[0, 1] if pm.shape[1] > 1 else pm[0, 0])
                result[f"{prefix}_prob"]   = round(prob, 6)
                result[f"{prefix}_action"] = "buy" if prob >= 0.55 else "skip"
            except Exception as exc:
                log.debug("[CC/%s] %s score_shadow failed: %s", self.family, slot_name, exc)

        return result

    def log_shadow_signal(
        self,
        signal_row: dict,
        shadow_result: dict,
        signal_id: str | None = None,
    ) -> None:
        """Append a shadow record to the rolling shadow_eval.csv log."""
        rec = ShadowRecord(
            shadow_at       = datetime.now(timezone.utc).isoformat(),
            family          = self.family,
            signal_id       = signal_id or str(signal_row.get("token_id", "")),
            signal_label    = str(signal_row.get("signal_label", "")),
            champion_prob   = shadow_result.get("champion_prob"),
            champion_action = shadow_result.get("champion_action", "unavailable"),
            challenger_prob = shadow_result.get("challenger_prob"),
            challenger_action = shadow_result.get("challenger_action", "unavailable"),
        )
        self._append_shadow(rec)

    def _append_shadow(self, rec: ShadowRecord) -> None:
        df_new = pd.DataFrame([rec.to_dict()])
        try:
            if self._shadow_log.exists():
                existing = pd.read_csv(self._shadow_log, low_memory=False)
                df_new = pd.concat([existing, df_new], ignore_index=True).tail(self._SHADOW_LOG_MAX_ROWS)
            df_new.to_csv(self._shadow_log, index=False)
        except Exception as exc:
            log.warning("[CC/%s] shadow log write failed: %s", self.family, exc)

    # -- Outcome recording ---------------------------------------------------

    def record_outcome(self, signal_id: str, realized_return: float) -> None:
        """
        After a trade closes, match its signal_id in the shadow log and
        record the realized return.  Increments shadow_trades_seen when matched.
        """
        if not self._shadow_log.exists():
            return
        try:
            df = pd.read_csv(self._shadow_log, low_memory=False)
            mask = df["signal_id"].astype(str) == str(signal_id)
            if not mask.any():
                return
            df.loc[mask, "outcome"]        = realized_return
            df.loc[mask, "outcome_binary"] = int(realized_return > 0)
            df.loc[mask, "outcome_at"]     = datetime.now(timezone.utc).isoformat()
            df.to_csv(self._shadow_log, index=False)
            # Count matched + resolved rows for this cycle
            resolved = df["outcome"].notna().sum()
            self._state.shadow_trades_seen = int(resolved)
            self._save_state()
        except Exception as exc:
            log.warning("[CC/%s] record_outcome failed: %s", self.family, exc)

    # -- Evaluation ----------------------------------------------------------

    def _load_shadow_resolved(self) -> pd.DataFrame:
        """Load shadow log rows that have actual outcomes recorded."""
        if not self._shadow_log.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self._shadow_log, low_memory=False)
            df = df[df["outcome"].notna()].copy()
            for col in ("champion_prob", "challenger_prob", "outcome"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()

    def evaluate_shadow(self) -> dict[str, Any]:
        """
        Compare champion vs challenger on resolved shadow trades.
        Returns a full metrics dict with promotion verdict.
        """
        df = self._load_shadow_resolved()
        n  = len(df)

        result: dict[str, Any] = {
            "family":             self.family,
            "n_resolved":         n,
            "shadow_cycles":      self._state.shadow_cycles,
            "min_shadow_cycles":  self._MIN_SHADOW_CYCLES,
            "min_shadow_trades":  self._MIN_SHADOW_TRADES,
            "promote":            False,
            "block_reasons":      [],
            "champion_sortino":   float("nan"),
            "challenger_sortino": float("nan"),
            "champion_precision": float("nan"),
            "challenger_precision": float("nan"),
        }

        blocks: list[str] = []

        # ── Gate 1: minimum shadow cycles ────────────────────────────────
        if self._state.shadow_cycles < self._MIN_SHADOW_CYCLES:
            blocks.append(
                f"shadow_cycles={self._state.shadow_cycles} < required={self._MIN_SHADOW_CYCLES}"
            )

        # ── Gate 2: minimum shadow trades ────────────────────────────────
        if n < self._MIN_SHADOW_TRADES:
            blocks.append(
                f"shadow_trades={n} < required={self._MIN_SHADOW_TRADES}"
            )

        if blocks:
            result["block_reasons"] = blocks
            return result

        # ── Compute per-model metrics ─────────────────────────────────────
        returns = df["outcome"].to_numpy()

        # Champion actions (based on champion_prob threshold)
        champ_buy  = df["champion_action"].astype(str) == "buy"
        chall_buy  = df["challenger_action"].astype(str) == "buy"

        # Sortino on trades each model would have taken
        champ_returns = returns[champ_buy.to_numpy()] if champ_buy.any() else np.array([])
        chall_returns = returns[chall_buy.to_numpy()] if chall_buy.any() else np.array([])

        champ_sortino = _sortino(champ_returns)
        chall_sortino = _sortino(chall_returns)

        champ_prec = _precision(df["champion_action"],   df["outcome_binary"].fillna(0).astype(int))
        chall_prec = _precision(df["challenger_action"], df["outcome_binary"].fillna(0).astype(int))

        result.update({
            "champion_sortino":    champ_sortino,
            "challenger_sortino":  chall_sortino,
            "champion_precision":  champ_prec,
            "challenger_precision": chall_prec,
            "champion_n_buys":     int(champ_buy.sum()),
            "challenger_n_buys":   int(chall_buy.sum()),
        })

        # Update state
        self._state.champion_sharpe    = champ_sortino
        self._state.challenger_sharpe  = chall_sortino
        self._state.champion_precision = champ_prec if not math.isnan(champ_prec) else 0.0
        self._state.challenger_precision = chall_prec if not math.isnan(chall_prec) else 0.0

        # ── Gate 3: Sortino delta ─────────────────────────────────────────
        if not math.isnan(chall_sortino) and not math.isnan(champ_sortino):
            sortino_delta = chall_sortino - champ_sortino
            result["sortino_delta"] = sortino_delta
            if sortino_delta < self._SHARPE_MARGIN:
                blocks.append(
                    f"sortino_delta={sortino_delta:.4f} < required={self._SHARPE_MARGIN:.4f}"
                )
        elif math.isnan(chall_sortino):
            blocks.append("challenger_sortino=nan (insufficient buy decisions)")

        # ── Gate 4: Precision delta ────────────────────────────────────────
        if not math.isnan(chall_prec) and not math.isnan(champ_prec):
            prec_delta = chall_prec - champ_prec
            result["precision_delta"] = prec_delta
            if prec_delta < self._PRECISION_MARGIN:
                blocks.append(
                    f"precision_delta={prec_delta:.4f} < required={self._PRECISION_MARGIN:.4f}"
                )
        elif math.isnan(chall_prec):
            blocks.append("challenger_precision=nan")

        # ── Gate 5: No-regression check (sub-window) ──────────────────────
        # Split resolved trades into sub-windows; challenger must not badly
        # underperform champion in any window.
        w = self._REGRESSION_WINDOW
        if n >= w * 2:
            n_windows  = n // w
            reg_blocks = []
            for i in range(n_windows):
                sub = df.iloc[i * w: (i + 1) * w]
                sub_ret   = sub["outcome"].to_numpy()
                sub_champ = sub["champion_action"].astype(str) == "buy"
                sub_chall = sub["challenger_action"].astype(str) == "buy"
                s_champ   = _sortino(sub_ret[sub_champ.to_numpy()]) if sub_champ.any() else float("nan")
                s_chall   = _sortino(sub_ret[sub_chall.to_numpy()]) if sub_chall.any() else float("nan")
                if not math.isnan(s_champ) and not math.isnan(s_chall):
                    delta = s_chall - s_champ
                    if delta < self._REGRESSION_FLOOR:
                        reg_blocks.append(f"window_{i}_delta={delta:.3f}")
            if reg_blocks:
                blocks.append(
                    f"regression_in_sub_windows: {', '.join(reg_blocks)}"
                )
        result["regression_check_windows"] = n // w if n >= w * 2 else 0

        result["block_reasons"] = blocks
        result["promote"] = len(blocks) == 0
        return result

    # -- Promotion -----------------------------------------------------------

    def try_promote(self, dry_run: bool = False) -> tuple[bool, str]:
        """
        Evaluate the challenger and promote it if all gates pass.
        Returns (promoted: bool, reason: str).

        dry_run=True: evaluate but do not touch disk.
        """
        # Increment shadow cycle count every call (represents one retrain cycle)
        self._state.shadow_cycles += 1
        self._state.last_shadow_at = datetime.now(timezone.utc).isoformat()
        self._save_state()

        eval_result = self.evaluate_shadow()

        if not eval_result.get("promote", False):
            reason = "shadow_gate_blocked: " + " | ".join(eval_result.get("block_reasons", ["unknown"]))
            log.info("[CC/%s] challenger NOT promoted: %s", self.family, reason)
            self._log_evaluation(eval_result, promoted=False)
            return False, reason

        if dry_run:
            reason = (
                f"dry_run — would promote challenger={self._state.challenger_version} "
                f"sortino_delta={eval_result.get('sortino_delta', float('nan')):.4f}"
            )
            log.info("[CC/%s] DRY RUN: %s", self.family, reason)
            return False, reason

        # ── Atomic promotion ──────────────────────────────────────────────
        self._install_champion(self._challenger_dir, self._state.challenger_version)

        # Archive challenger slot (keep a copy for audit)
        archive_dir = self.weights_dir / self.family / f"promoted_{self._state.challenger_version}"
        if not archive_dir.exists():
            shutil.copytree(self._challenger_dir, archive_dir)

        # Wipe shadow log — fresh cycle starts after promotion
        if self._shadow_log.exists():
            self._shadow_log.unlink(missing_ok=True)

        self._state.shadow_cycles       = 0
        self._state.shadow_trades_seen  = 0
        self._state.promoted            = True
        self._state.last_promote_at     = datetime.now(timezone.utc).isoformat()
        self._state.last_promote_reason = (
            f"sortino_delta={eval_result.get('sortino_delta', float('nan')):.4f}  "
            f"precision_delta={eval_result.get('precision_delta', float('nan')):.4f}  "
            f"n_trades={eval_result['n_resolved']}"
        )
        self._save_state()

        reason = self._state.last_promote_reason
        log.info("[CC/%s] PROMOTED challenger → champion: %s", self.family, reason)
        self._log_evaluation(eval_result, promoted=True)
        return True, reason

    def emergency_promote_if_champion_failing(self) -> tuple[bool, str]:
        """
        If the champion is in freefall (live Sortino < emergency floor),
        immediately attempt challenger promotion using a single-window gate.
        Falls through gracefully if no challenger exists or it also fails.
        """
        if self.challenger is None:
            return False, "no_challenger"

        df = self._load_shadow_resolved()
        if len(df) < 10:
            return False, f"insufficient_shadow_data (n={len(df)})"

        returns = df["outcome"].to_numpy()
        champ_buy  = df["champion_action"].astype(str) == "buy"
        chall_buy  = df["challenger_action"].astype(str) == "buy"
        s_champ = _sortino(returns[champ_buy.to_numpy()]) if champ_buy.any() else float("nan")
        s_chall = _sortino(returns[chall_buy.to_numpy()]) if chall_buy.any() else float("nan")

        if not math.isnan(s_champ) and s_champ >= self._EMERGENCY_FLOOR:
            return False, f"champion_sortino={s_champ:.4f} above emergency_floor={self._EMERGENCY_FLOOR}"

        if math.isnan(s_chall):
            return False, "challenger_sortino=nan"
        if s_chall <= s_champ:
            return False, f"challenger ({s_chall:.4f}) not better than failing champion ({s_champ:.4f})"

        log.warning("[CC/%s] EMERGENCY PROMOTION: champion sortino=%.4f < floor=%.4f",
                    self.family, s_champ, self._EMERGENCY_FLOOR)
        self._install_champion(self._challenger_dir, self._state.challenger_version)
        self._state.last_promote_reason = f"emergency: champ_sortino={s_champ:.4f}"
        self._state.last_promote_at = datetime.now(timezone.utc).isoformat()
        self._save_state()
        return True, self._state.last_promote_reason

    # -- Evaluation audit log ------------------------------------------------

    def _log_evaluation(self, eval_result: dict[str, Any], promoted: bool) -> None:
        """Append evaluation result to a running audit log."""
        audit_path = self.logs_dir / self.family / "cc_audit.csv"
        row = {
            "evaluated_at":         datetime.now(timezone.utc).isoformat(),
            "family":               self.family,
            "champion_version":     self._state.champion_version,
            "challenger_version":   self._state.challenger_version,
            "promoted":             promoted,
            "shadow_cycles":        self._state.shadow_cycles,
            "n_resolved":           eval_result.get("n_resolved", 0),
            "champion_sortino":     eval_result.get("champion_sortino", float("nan")),
            "challenger_sortino":   eval_result.get("challenger_sortino", float("nan")),
            "sortino_delta":        eval_result.get("sortino_delta", float("nan")),
            "champion_precision":   eval_result.get("champion_precision", float("nan")),
            "challenger_precision": eval_result.get("challenger_precision", float("nan")),
            "precision_delta":      eval_result.get("precision_delta", float("nan")),
            "block_reasons":        " | ".join(eval_result.get("block_reasons", [])),
        }
        try:
            existing = pd.read_csv(audit_path, low_memory=False) if audit_path.exists() else pd.DataFrame()
            updated  = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            updated.to_csv(audit_path, index=False)
        except Exception as exc:
            log.warning("[CC/%s] audit log write failed: %s", self.family, exc)

    # -- Status --------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return a human-readable status dict."""
        return {
            "family":               self.family,
            "champion_version":     self._state.champion_version,
            "challenger_version":   self._state.challenger_version,
            "champion_artifact":    str(next(self._champion_dir.glob("*.joblib"), "none")),
            "challenger_artifact":  str(next(self._challenger_dir.glob("*.joblib"), "none")),
            "shadow_cycles":        self._state.shadow_cycles,
            "shadow_trades_seen":   self._state.shadow_trades_seen,
            "min_shadow_cycles":    self._MIN_SHADOW_CYCLES,
            "min_shadow_trades":    self._MIN_SHADOW_TRADES,
            "last_shadow_at":       self._state.last_shadow_at,
            "last_promote_at":      self._state.last_promote_at,
            "last_promote_reason":  self._state.last_promote_reason,
            "challenger_sharpe":    round(self._state.challenger_sharpe, 4)
                                    if not math.isnan(self._state.challenger_sharpe) else "nan",
            "champion_sharpe":      round(self._state.champion_sharpe, 4)
                                    if not math.isnan(self._state.champion_sharpe) else "nan",
        }


# ---------------------------------------------------------------------------
# Process-level registry: one ChallengerSlot per family
# ---------------------------------------------------------------------------

_slots: dict[str, ChallengerSlot] = {}


def get_slot(
    family: str,
    weights_dir: str = "weights",
    logs_dir:    str = "logs",
) -> ChallengerSlot:
    """Return (and cache) the ChallengerSlot for a family."""
    key = f"{family}:{weights_dir}:{logs_dir}"
    if key not in _slots:
        slot = ChallengerSlot(family, weights_dir, logs_dir)
        slot.bootstrap_champion_from_weights()
        _slots[key] = slot
    return _slots[key]
