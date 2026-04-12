import logging
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def drop_all_nan_features(
    df: pd.DataFrame,
    candidate_features: Iterable[str],
    *,
    context: str = "model_training",
) -> Tuple[List[str], List[str]]:
    """
    Remove features that are entirely missing in the provided frame.
    This prevents median imputer warnings and avoids training on dead columns.
    """
    existing = [col for col in candidate_features if col in df.columns]
    usable: List[str] = []
    dropped: List[str] = []

    for col in existing:
        if df[col].notna().any():
            usable.append(col)
        else:
            dropped.append(col)

    if dropped:
        logger.info("[%s] Dropping all-NaN features: %s", context, ", ".join(dropped))

    return usable, dropped


def clean_dataframe_for_training(
    df: pd.DataFrame,
    *,
    context: str = "csv_save",
) -> pd.DataFrame:
    """Apply model-clean treatment to a DataFrame before writing to CSV.

    Steps applied (in order):
    1. Coerce all numeric-looking columns to numeric dtype.
    2. Drop columns that are entirely NaN.
    3. Fill remaining NaN with per-column median (numeric) or "" (object).
    4. Clip boolean-policy features to {0, 1}.
    5. Apply log1p to log_scale-policy features.

    This mirrors what SimpleImputer(strategy='median') + feature treatment
    does in the in-memory training pipelines so the saved CSV is ready for
    direct model consumption.
    """
    from feature_treatment_policy import get_treatment

    out = df.copy()

    # 1. Coerce numeric columns
    for col in out.columns:
        if out[col].dtype == object:
            converted = pd.to_numeric(out[col], errors="coerce")
            # Only convert if at least half the non-null values are numeric
            if converted.notna().sum() >= out[col].notna().sum() * 0.5:
                out[col] = converted

    # 2. Drop all-NaN columns
    all_nan_cols = [c for c in out.columns if out[c].notna().sum() == 0]
    if all_nan_cols:
        logger.info("[%s] Dropping %d all-NaN columns before save.", context, len(all_nan_cols))
        out = out.drop(columns=all_nan_cols)

    # 3. Median imputation for numeric, empty string for object
    for col in out.columns:
        if out[col].isna().any():
            if pd.api.types.is_numeric_dtype(out[col]):
                median_val = out[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                out[col] = out[col].fillna(median_val)
            else:
                out[col] = out[col].fillna("")

    # 4 & 5. Apply feature treatment policy (boolean clip, log1p)
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        treatment = get_treatment(col)
        if treatment.kind == "boolean":
            out[col] = out[col].clip(0, 1).round().astype(int)
        elif treatment.kind == "log_scale":
            out[col] = np.log1p(out[col].clip(lower=0.0))
        elif treatment.kind == "clip01":
            out[col] = out[col].clip(0.0, 1.0)

    logger.info("[%s] Model-clean treatment applied: %d rows × %d cols.", context, len(out), len(out.columns))
    return out
