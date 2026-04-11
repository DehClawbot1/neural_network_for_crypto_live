from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from brain_paths import resolve_brain_context


logger = logging.getLogger(__name__)


_REGISTRY_COLUMNS = [
    "registered_at",
    "run_id",
    "model_kind",
    "artifact_group",
    "feature_set",
    "scaling",
    "regularization",
    "market_family",
    "regime_slice",
    "nonzero_feature_count",
    "total_feature_count",
    "n_train_rows",
    "n_test_rows",
    "accuracy",
    "precision",
    "recall",
    "rmse",
    "profit_factor",
    "replay_ev",
    "artifact_path",
    "promotion_status",
    "promotion_reason",
    "beats_champion",
    "is_champion",
    "promotion_gate_passed",
    "notes",
]

_NUMERIC_COLUMNS = {
    "nonzero_feature_count",
    "total_feature_count",
    "n_train_rows",
    "n_test_rows",
    "accuracy",
    "precision",
    "recall",
    "rmse",
    "profit_factor",
    "replay_ev",
}

_BOOLEAN_COLUMNS = {
    "beats_champion",
    "is_champion",
    "promotion_gate_passed",
}


def _safe_float(value, default=None):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except Exception:
        return default


class ModelRegistry:
    """Append-only model registry backed by a CSV file."""

    def __init__(self, logs_dir: str = "logs", *, brain_context=None, brain_id=None, market_family=None, shared_logs_dir="logs", shared_weights_dir="weights"):
        if brain_context is None and (brain_id or market_family):
            brain_context = resolve_brain_context(
                market_family,
                brain_id=brain_id,
                shared_logs_dir=shared_logs_dir,
                shared_weights_dir=shared_weights_dir,
            )
        self.logs_dir = Path(brain_context.logs_dir if brain_context is not None else logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.logs_dir / "model_registry_comparison.csv"
        self.regime_file = self.logs_dir / "regime_model_comparison.csv"

    def _read(self) -> pd.DataFrame:
        if not self.registry_file.exists():
            return pd.DataFrame(columns=_REGISTRY_COLUMNS)
        try:
            frame = pd.read_csv(self.registry_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=_REGISTRY_COLUMNS)
        for column in _REGISTRY_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        for column in _NUMERIC_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in _BOOLEAN_COLUMNS:
            if column in frame.columns:
                frame[column] = frame[column].map(
                    lambda value: value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}
                    if pd.notna(value)
                    else None
                )
        return frame[_REGISTRY_COLUMNS].copy()

    def register(self, **row) -> dict[str, Any]:
        payload = {column: None for column in _REGISTRY_COLUMNS}
        payload.update(row or {})
        payload["registered_at"] = str(payload.get("registered_at") or datetime.now(timezone.utc).isoformat())
        payload["run_id"] = str(payload.get("run_id") or "").strip()
        payload["artifact_group"] = str(payload.get("artifact_group") or payload.get("model_kind") or "").strip()
        payload["feature_set"] = str(payload.get("feature_set") or "default_tabular")
        payload["scaling"] = str(payload.get("scaling") or "none")
        payload["regularization"] = str(payload.get("regularization") or "none")
        payload["market_family"] = str(payload.get("market_family") or "all")
        payload["regime_slice"] = str(payload.get("regime_slice") or "all")
        payload["promotion_status"] = str(payload.get("promotion_status") or "candidate")
        payload["promotion_reason"] = str(payload.get("promotion_reason") or "")
        payload["notes"] = str(payload.get("notes") or "")

        existing = self._read()
        payload_frame = pd.DataFrame([payload], columns=_REGISTRY_COLUMNS)
        updated = payload_frame if existing.empty else pd.concat([existing, payload_frame], ignore_index=True, sort=False)
        updated.to_csv(self.registry_file, index=False)
        logger.info(
            "Registered model %s [%s/%s] status=%s accuracy=%s rmse=%s",
            payload.get("model_kind"),
            payload.get("market_family"),
            payload.get("regime_slice"),
            payload.get("promotion_status"),
            payload.get("accuracy"),
            payload.get("rmse"),
        )
        return payload

    def register_rows(self, rows: list[dict[str, Any]] | pd.DataFrame | None) -> pd.DataFrame:
        if rows is None:
            return pd.DataFrame(columns=_REGISTRY_COLUMNS)
        if isinstance(rows, pd.DataFrame):
            records = rows.to_dict("records")
        else:
            records = list(rows)
        registered = [self.register(**row) for row in records if isinstance(row, dict)]
        return pd.DataFrame(registered, columns=_REGISTRY_COLUMNS)

    def comparison_table(self) -> pd.DataFrame:
        return self._read()

    def current_champion(self, *, artifact_group: str, market_family: str = "all", regime_slice: str = "all") -> dict[str, Any] | None:
        table = self._read()
        if table.empty:
            return None
        work = table.copy()
        work = work[
            (work["artifact_group"].fillna("").astype(str) == str(artifact_group or ""))
            & (work["market_family"].fillna("all").astype(str) == str(market_family or "all"))
            & (work["regime_slice"].fillna("all").astype(str) == str(regime_slice or "all"))
        ].copy()
        if work.empty:
            return None
        champions = work[work["is_champion"] == True].copy()
        if not champions.empty:
            return champions.iloc[-1].to_dict()
        promoted = work[work["promotion_status"].fillna("").astype(str).str.lower() == "promoted"].copy()
        if not promoted.empty:
            return promoted.iloc[-1].to_dict()
        return None

    def write_regime_model_comparison(self) -> Path:
        table = self._read()
        out_path = self.regime_file
        if table.empty:
            pd.DataFrame(columns=["artifact_group", "market_family", "regime_slice", "model_kind"]).to_csv(out_path, index=False)
            return out_path
        work = table.copy()
        metric_value = []
        for _, row in work.iterrows():
            rmse = _safe_float(row.get("rmse"))
            accuracy = _safe_float(row.get("accuracy"))
            if rmse is not None:
                metric_value.append(-rmse)
            elif accuracy is not None:
                metric_value.append(accuracy)
            else:
                metric_value.append(None)
        work["sort_metric"] = metric_value
        work = work.sort_values(["artifact_group", "market_family", "regime_slice", "sort_metric", "registered_at"], ascending=[True, True, True, False, True], na_position="last")
        best_rows = (
            work.groupby(["artifact_group", "market_family", "regime_slice"], dropna=False, as_index=False)
            .head(1)
            .drop(columns=["sort_metric"], errors="ignore")
        )
        best_rows.to_csv(out_path, index=False)
        logger.info("Regime model comparison written to %s", out_path)
        return out_path

    def write_decision_profit_audit(self) -> Path:
        df = self._read()
        out_path = self.logs_dir / "decision_profit_audit.md"
        lines = [
            "# Decision / Profit Audit",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            f"Total registered artifacts: {len(df)}",
            "",
        ]
        if df.empty:
            lines.append("No models registered yet.")
            out_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Decision profit audit written to %s", out_path)
            return out_path

        champions = df[df["is_champion"] == True].copy()
        if champions.empty:
            champions = df[df["promotion_status"].fillna("").astype(str).str.lower() == "promoted"].copy()
        lines.extend(["## Active Champions", ""])
        if champions.empty:
            lines.append("No active champions registered.")
        else:
            for _, row in champions.iterrows():
                parts = [
                    f"`{row.get('artifact_group', '')}`",
                    f"{row.get('model_kind', '')}",
                    f"family={row.get('market_family', '')}",
                    f"regime={row.get('regime_slice', '')}",
                ]
                accuracy = _safe_float(row.get("accuracy"))
                rmse = _safe_float(row.get("rmse"))
                if accuracy is not None:
                    parts.append(f"accuracy={accuracy:.4f}")
                if rmse is not None:
                    parts.append(f"rmse={rmse:.4f}")
                profit_factor = _safe_float(row.get("profit_factor"))
                if profit_factor is not None:
                    parts.append(f"profit_factor={profit_factor:.4f}")
                lines.append(f"- {' | '.join(parts)}")

        lines.extend(["", "## Recent Registrations", "", "| artifact_group | model_kind | family | regime | status | accuracy | rmse | pf | notes |", "|---|---|---|---|---|---|---|---|---|"])
        recent = df.tail(20)
        for _, row in recent.iterrows():
            accuracy = _safe_float(row.get("accuracy"))
            rmse = _safe_float(row.get("rmse"))
            pf = _safe_float(row.get("profit_factor"))
            lines.append(
                "| {artifact_group} | {model_kind} | {family} | {regime} | {status} | {accuracy} | {rmse} | {pf} | {notes} |".format(
                    artifact_group=row.get("artifact_group", ""),
                    model_kind=row.get("model_kind", ""),
                    family=row.get("market_family", ""),
                    regime=row.get("regime_slice", ""),
                    status=row.get("promotion_status", ""),
                    accuracy=(f"{accuracy:.4f}" if accuracy is not None else ""),
                    rmse=(f"{rmse:.4f}" if rmse is not None else ""),
                    pf=(f"{pf:.4f}" if pf is not None else ""),
                    notes=str(row.get("notes", "") or "").replace("\n", " "),
                )
            )

        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Decision profit audit written to %s", out_path)
        return out_path
