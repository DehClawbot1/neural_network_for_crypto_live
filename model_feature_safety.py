import logging
from typing import Iterable, List, Tuple

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
