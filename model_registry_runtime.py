from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from model_artifact_staging import promote_candidate_artifacts
from model_registry import ModelRegistry
from return_calibration import calibrate_return_predictions


logger = logging.getLogger(__name__)

PROMOTABLE_GROUP_TO_FILENAME = {
    "btc_tabular_classifier": "tp_classifier.joblib",
    "btc_tabular_regressor": "return_regressor.joblib",
    "weather_tabular_classifier": "tp_classifier.joblib",
    "weather_tabular_regressor": "return_regressor.joblib",
    "stage1_classifier": "stage1_tp_classifier.joblib",
    "stage1_regressor": "stage1_return_regressor.joblib",
    "weather_stage1_classifier": "stage1_tp_classifier.joblib",
    "weather_stage1_regressor": "stage1_return_regressor.joblib",
    "stage2_temporal_classifier": "stage2_temporal_classifier.joblib",
    "stage2_temporal_regressor": "stage2_temporal_regressor.joblib",
    "weather_stage2_temporal_classifier": "stage2_temporal_classifier.joblib",
    "weather_stage2_temporal_regressor": "stage2_temporal_regressor.joblib",
    "weather_temperature_classifier": "weather_temperature_model.joblib",
}

REGIME_COLUMN_CANDIDATES = [
    "btc_market_regime_label",
    "technical_regime_bucket",
    "btc_volatility_regime",
]


def _safe_read(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _safe_float(value, default=None):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except Exception:
        return default


def _load_joblib_payload(path: str | Path) -> dict | None:
    if not Path(path).exists():
        return None
    try:
        import joblib
    except Exception as exc:
        logger.warning("joblib unavailable while evaluating %s: %s", path, exc)
        return None
    try:
        payload = joblib.load(path)
    except Exception as exc:
        logger.warning("Failed to load artifact %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_regime_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def resolve_regime_column(df: pd.DataFrame) -> str | None:
    for column in REGIME_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    return None


def _profit_factor_from_returns(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    gross_profit = float(values[values > 0].sum())
    gross_loss = abs(float(values[values < 0].sum()))
    if gross_profit <= 0 and gross_loss <= 0:
        return None
    if gross_loss <= 0:
        return gross_profit
    return gross_profit / gross_loss


def _replay_metrics(df: pd.DataFrame, *, selected_mask: pd.Series, return_col: str = "forward_return_15m") -> tuple[float | None, float | None]:
    if return_col not in df.columns:
        return None, None
    selected_returns = pd.to_numeric(df.loc[selected_mask, return_col], errors="coerce").dropna()
    if selected_returns.empty:
        return None, None
    return float(selected_returns.mean()), _profit_factor_from_returns(selected_returns)


def _standard_result_row(
    *,
    run_id: str,
    model_kind: str,
    artifact_group: str,
    feature_set: str,
    scaling: str,
    regularization: str,
    market_family: str,
    regime_slice: str,
    nonzero_feature_count,
    total_feature_count,
    n_train_rows,
    n_test_rows,
    accuracy=None,
    precision=None,
    recall=None,
    rmse=None,
    profit_factor=None,
    replay_ev=None,
    artifact_path: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_kind": model_kind,
        "artifact_group": artifact_group,
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
        "artifact_path": str(artifact_path or ""),
        "promotion_status": "candidate",
        "promotion_reason": "",
        "beats_champion": None,
        "is_champion": False,
        "promotion_gate_passed": None,
        "notes": notes,
    }


def _classifier_rows_for_slice(
    *,
    payload: dict,
    test_df: pd.DataFrame,
    slice_name: str,
    run_id: str,
    artifact_group: str,
    market_family: str,
    artifact_path: str | Path,
    target_col: str,
) -> dict[str, Any] | None:
    features = list(payload.get("features") or [])
    if not features or target_col not in test_df.columns or test_df.empty:
        return None
    frame = test_df.copy()
    for feature in features:
        if feature not in frame.columns:
            frame[feature] = 0.0
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    model = payload.get("model")
    if model is None:
        return None
    try:
        preds = pd.Series(model.predict(X), index=frame.index).astype(int)
    except Exception as exc:
        logger.warning("Classifier evaluation failed for %s: %s", artifact_path, exc)
        return None
    y = pd.to_numeric(frame[target_col], errors="coerce").fillna(0).astype(int)
    if y.empty:
        return None
    accuracy = float((preds == y).mean())
    tp = int(((preds == 1) & (y == 1)).sum())
    predicted_positive = int((preds == 1).sum())
    actual_positive = int((y == 1).sum())
    precision = float(tp / predicted_positive) if predicted_positive else None
    recall = float(tp / actual_positive) if actual_positive else None
    replay_ev, profit_factor = _replay_metrics(frame, selected_mask=(preds == 1))
    return _standard_result_row(
        run_id=run_id,
        model_kind=str(payload.get("model_kind") or "classifier"),
        artifact_group=artifact_group,
        feature_set=str(payload.get("feature_set") or "default_tabular"),
        scaling=str(payload.get("scaling") or "none"),
        regularization=str(payload.get("regularization") or "none"),
        market_family=market_family,
        regime_slice=slice_name,
        nonzero_feature_count=payload.get("nonzero_feature_count"),
        total_feature_count=len(features),
        n_train_rows=None,
        n_test_rows=int(len(frame.index)),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        profit_factor=profit_factor,
        replay_ev=replay_ev,
        artifact_path=str(artifact_path),
    )


def _regressor_rows_for_slice(
    *,
    payload: dict,
    test_df: pd.DataFrame,
    slice_name: str,
    run_id: str,
    artifact_group: str,
    market_family: str,
    artifact_path: str | Path,
    target_col: str,
) -> dict[str, Any] | None:
    features = list(payload.get("features") or [])
    if not features or target_col not in test_df.columns or test_df.empty:
        return None
    frame = test_df.copy()
    for feature in features:
        if feature not in frame.columns:
            frame[feature] = 0.0
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    model = payload.get("model")
    if model is None:
        return None
    try:
        preds = pd.Series(model.predict(X), index=frame.index)
    except Exception as exc:
        logger.warning("Regressor evaluation failed for %s: %s", artifact_path, exc)
        return None
    if payload.get("return_calibration") is not None:
        preds = calibrate_return_predictions(preds, payload.get("return_calibration"), index=frame.index)
    actual = pd.to_numeric(frame[target_col], errors="coerce").fillna(0.0)
    if actual.empty:
        return None
    rmse = float(((preds - actual) ** 2).mean() ** 0.5)
    replay_ev, profit_factor = _replay_metrics(frame, selected_mask=(preds > 0))
    return _standard_result_row(
        run_id=run_id,
        model_kind=str(payload.get("model_kind") or "regressor"),
        artifact_group=artifact_group,
        feature_set=str(payload.get("feature_set") or "default_tabular"),
        scaling=str(payload.get("scaling") or "none"),
        regularization=str(payload.get("regularization") or "none"),
        market_family=market_family,
        regime_slice=slice_name,
        nonzero_feature_count=payload.get("nonzero_feature_count"),
        total_feature_count=len(features),
        n_train_rows=None,
        n_test_rows=int(len(frame.index)),
        rmse=rmse,
        profit_factor=profit_factor,
        replay_ev=replay_ev,
        artifact_path=str(artifact_path),
    )


def evaluate_artifact_against_dataset(
    *,
    run_id: str,
    dataset_file: str | Path,
    artifact_path: str | Path,
    artifact_group: str,
    market_family: str = "all",
    target_col: str,
    min_slice_rows: int = 10,
    market_family_prefix: str | None = None,
) -> pd.DataFrame:
    df = _safe_read(dataset_file)
    if df.empty:
        return pd.DataFrame()
    if market_family_prefix and "market_family" in df.columns:
        family_series = df["market_family"].fillna("").astype(str).str.lower()
        df = df[family_series.str.startswith(str(market_family_prefix).lower())].copy()
        if df.empty:
            return pd.DataFrame()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", kind="stable")
    split_idx = int(len(df.index) * 0.8)
    if split_idx <= 0 or split_idx >= len(df.index):
        return pd.DataFrame()
    test_df = df.iloc[split_idx:].copy()
    if test_df.empty:
        return pd.DataFrame()
    payload = _load_joblib_payload(artifact_path)
    if payload is None:
        return pd.DataFrame()

    mode = "regressor" if "regressor" in artifact_group else "classifier"
    builder = _regressor_rows_for_slice if mode == "regressor" else _classifier_rows_for_slice
    rows: list[dict[str, Any]] = []
    overall = builder(
        payload=payload,
        test_df=test_df,
        slice_name="all",
        run_id=run_id,
        artifact_group=artifact_group,
        market_family=market_family,
        artifact_path=artifact_path,
        target_col=target_col,
    )
    if overall is not None:
        overall["n_train_rows"] = int(split_idx)
        rows.append(overall)

    regime_col = resolve_regime_column(test_df)
    if regime_col:
        for regime_value, regime_df in test_df.groupby(regime_col, dropna=False):
            if len(regime_df.index) < int(min_slice_rows):
                continue
            slice_name = _normalize_regime_value(regime_value)
            if not slice_name:
                continue
            row = builder(
                payload=payload,
                test_df=regime_df.copy(),
                slice_name=slice_name,
                run_id=run_id,
                artifact_group=artifact_group,
                market_family=market_family,
                artifact_path=artifact_path,
                target_col=target_col,
            )
            if row is not None:
                row["n_train_rows"] = int(split_idx)
                rows.append(row)
    return pd.DataFrame(rows)


def primary_metric_name(row: dict[str, Any]) -> str:
    if _safe_float(row.get("rmse")) is not None:
        return "rmse"
    return "accuracy"


def candidate_beats_champion(candidate: dict[str, Any], champion: dict[str, Any] | None) -> bool:
    if champion is None:
        return True
    metric = primary_metric_name(candidate)
    candidate_metric = _safe_float(candidate.get(metric))
    champion_metric = _safe_float(champion.get(metric))
    if candidate_metric is None:
        return False
    if champion_metric is None:
        return True
    if metric == "rmse":
        if candidate_metric < champion_metric:
            return True
        if candidate_metric > champion_metric:
            return False
    else:
        if candidate_metric > champion_metric:
            return True
        if candidate_metric < champion_metric:
            return False
    candidate_pf = _safe_float(candidate.get("profit_factor"), default=float("-inf"))
    champion_pf = _safe_float(champion.get("profit_factor"), default=float("-inf"))
    if candidate_pf > champion_pf:
        return True
    if candidate_pf < champion_pf:
        return False
    candidate_replay = _safe_float(candidate.get("replay_ev"), default=float("-inf"))
    champion_replay = _safe_float(champion.get("replay_ev"), default=float("-inf"))
    return candidate_replay >= champion_replay


def promotion_gate_passed(row: dict[str, Any], *, min_test_rows: int = 10) -> tuple[bool, str]:
    artifact_path = str(row.get("artifact_path") or "").strip()
    if artifact_path and not Path(artifact_path).exists():
        return False, "artifact_missing"
    metric_name = primary_metric_name(row)
    if _safe_float(row.get(metric_name)) is None:
        return False, f"missing_{metric_name}"
    n_test_rows = int(_safe_float(row.get("n_test_rows"), 0) or 0)
    if n_test_rows < int(min_test_rows):
        return False, f"n_test_rows_below_{min_test_rows}"
    nonzero_feature_count = _safe_float(row.get("nonzero_feature_count"))
    regularization = str(row.get("regularization") or "").strip().lower()
    if nonzero_feature_count is not None and "l1" in regularization and nonzero_feature_count <= 0:
        return False, "degenerate_sparse_model"
    return True, ""


def register_and_promote_rows(
    *,
    registry: ModelRegistry,
    candidate_rows: pd.DataFrame | list[dict[str, Any]],
    candidate_weights_dir: str | Path | None = None,
    active_weights_dir: str | Path | None = None,
    min_test_rows: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame(candidate_rows if not isinstance(candidate_rows, pd.DataFrame) else candidate_rows.copy())
    if frame.empty:
        registry.write_regime_model_comparison()
        registry.write_decision_profit_audit()
        return frame

    promoted_filenames: list[str] = []
    for idx, row in frame.iterrows():
        row_dict = row.to_dict()
        artifact_group = str(row_dict.get("artifact_group") or "").strip()
        market_family = str(row_dict.get("market_family") or "all")
        regime_slice = str(row_dict.get("regime_slice") or "all")
        champion = registry.current_champion(
            artifact_group=artifact_group,
            market_family=market_family,
            regime_slice=regime_slice,
        )
        gate_ok, gate_reason = promotion_gate_passed(row_dict, min_test_rows=min_test_rows)
        is_promotable = artifact_group in PROMOTABLE_GROUP_TO_FILENAME and candidate_weights_dir is not None and active_weights_dir is not None
        status = "evaluation_only"
        beats = None
        reason = gate_reason
        is_champion = False
        if is_promotable:
            if not gate_ok:
                status = "blocked"
            else:
                beats = candidate_beats_champion(row_dict, champion)
                if beats:
                    status = "promoted"
                    reason = "beats_champion"
                    is_champion = True
                    promoted_filenames.append(PROMOTABLE_GROUP_TO_FILENAME[artifact_group])
                else:
                    status = "blocked"
                    reason = "did_not_beat_champion"
        frame.at[idx, "promotion_gate_passed"] = gate_ok if is_promotable else None
        frame.at[idx, "beats_champion"] = beats
        frame.at[idx, "promotion_status"] = status
        frame.at[idx, "promotion_reason"] = reason
        frame.at[idx, "is_champion"] = is_champion

    unique_filenames = tuple(sorted({name for name in promoted_filenames if name}))
    if unique_filenames and candidate_weights_dir is not None and active_weights_dir is not None:
        promote_candidate_artifacts(candidate_weights_dir, active_weights_dir, filenames=unique_filenames, backup_label="registry_promotion")
        logger.info("Promoted candidate artifacts: %s", ", ".join(unique_filenames))

    registry.register_rows(frame)
    registry.write_regime_model_comparison()
    registry.write_decision_profit_audit()
    return frame
