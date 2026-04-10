"""Model registry — tracks every trained artifact with metadata.

Every model artifact is registered with:
  model_kind, feature_set, scaling, regularization, market_family,
  regime_slice, nonzero_feature_count, and evaluation metrics.

The registry lives at ``logs/model_registry_comparison.csv`` and is
append-only so historical comparisons are preserved.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


_REGISTRY_COLUMNS = [
    "registered_at",
    "model_kind",
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
    "notes",
]


class ModelRegistry:
    """Append-only model registry backed by a CSV file."""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.logs_dir / "model_registry_comparison.csv"

    def _read(self) -> pd.DataFrame:
        if not self.registry_file.exists():
            return pd.DataFrame(columns=_REGISTRY_COLUMNS)
        try:
            return pd.read_csv(self.registry_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=_REGISTRY_COLUMNS)

    def register(
        self,
        *,
        model_kind: str,
        feature_set: str = "default_tabular",
        scaling: str = "none",
        regularization: str = "none",
        market_family: str = "all",
        regime_slice: str = "all",
        nonzero_feature_count: Optional[int] = None,
        total_feature_count: Optional[int] = None,
        n_train_rows: Optional[int] = None,
        n_test_rows: Optional[int] = None,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        rmse: Optional[float] = None,
        profit_factor: Optional[float] = None,
        replay_ev: Optional[float] = None,
        artifact_path: Optional[str] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        row = {
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "model_kind": model_kind,
            "feature_set": feature_set,
            "scaling": scaling,
            "regularization": regularization,
            "market_family": market_family,
            "regime_slice": regime_slice,
            "nonzero_feature_count": nonzero_feature_count,
            "total_feature_count": total_feature_count,
            "n_train_rows": n_train_rows,
            "n_test_rows": n_test_rows,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "rmse": rmse,
            "profit_factor": profit_factor,
            "replay_ev": replay_ev,
            "artifact_path": artifact_path,
            "notes": notes,
        }
        existing = self._read()
        updated = pd.concat(
            [existing, pd.DataFrame([row])], ignore_index=True, sort=False
        )
        updated.to_csv(self.registry_file, index=False)
        logger.info(
            "Registered model %s (%s) — %s features, accuracy=%s",
            model_kind,
            regularization,
            nonzero_feature_count,
            accuracy,
        )
        return row

    def comparison_table(self) -> pd.DataFrame:
        return self._read()

    def write_decision_profit_audit(self) -> Path:
        """Write a short ``logs/decision_profit_audit.md`` summary."""
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
        else:
            lines.append("## Model Comparison")
            lines.append("")
            lines.append(
                "| model_kind | regularization | features | accuracy | precision | rmse | profit_factor | regime |"
            )
            lines.append(
                "|---|---|---|---|---|---|---|---|"
            )
            for _, row in df.iterrows():
                lines.append(
                    f"| {row.get('model_kind','')} "
                    f"| {row.get('regularization','')} "
                    f"| {row.get('nonzero_feature_count','')} "
                    f"| {row.get('accuracy',''):.4f}" if pd.notna(row.get('accuracy')) else f"| {row.get('accuracy','')} "
                    f"| {row.get('precision',''):.4f}" if pd.notna(row.get('precision')) else f"| {row.get('precision','')} "
                    f"| {row.get('rmse','')}"
                    f"| {row.get('profit_factor','')}"
                    f"| {row.get('regime_slice','')} |"
                )
            lines.append("")
            lines.append("## Recommendation")
            lines.append("")
            best = df.loc[df["accuracy"].idxmax()] if "accuracy" in df.columns and df["accuracy"].notna().any() else None
            if best is not None:
                lines.append(
                    f"Best accuracy: **{best.get('model_kind')}** "
                    f"({best.get('regularization')}) = {best.get('accuracy'):.4f}"
                )
            else:
                lines.append("Insufficient data for recommendation.")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Decision profit audit written to %s", out_path)
        return out_path
