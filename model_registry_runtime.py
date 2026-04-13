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
    "btc_tabular_classifier": "btc_tp_classifier.joblib",
    "btc_tabular_regressor": "btc_return_regressor.joblib",
    "weather_tabular_classifier": "weather_tp_classifier.joblib",
    "weather_tabular_regressor": "weather_return_regressor.joblib",
    "stage1_classifier": "stage1_tp_classifier.joblib",
    "stage1_regressor": "stage1_return_regressor.joblib",
    "weather_stage1_classifier": "weather_stage1_tp_classifier.joblib",
    "weather_stage1_regressor": "weather_stage1_return_regressor.joblib",
    "stage2_temporal_classifier": "stage2_temporal_classifier.joblib",
    "stage2_temporal_regressor": "stage2_temporal_regressor.joblib",
    "weather_stage2_temporal_classifier": "weather_stage2_temporal_classifier.joblib",
    "weather_stage2_temporal_regressor": "weather_stage2_temporal_regressor.joblib",
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


def _brier_score_from_predictions(
    prob_preds: pd.Series, actual: pd.Series
) -> float | None:
    """Brier score: mean((prob - outcome)^2). Lower is better."""
    p = pd.to_numeric(prob_preds, errors="coerce").dropna()
    y = pd.to_numeric(actual, errors="coerce").dropna()
    common = p.index.intersection(y.index)
    if len(common) < 10:
        return None
    return float(((p.loc[common] - y.loc[common]) ** 2).mean())


def _top_decile_precision(
    prob_preds: pd.Series, actual: pd.Series
) -> float | None:
    """
    Precision in the top 10% of predicted probabilities.
    This is the metric that matters most for slippage-adjusted PnL.
    """
    p = pd.to_numeric(prob_preds, errors="coerce").dropna()
    y = pd.to_numeric(actual, errors="coerce").dropna()
    common = p.index.intersection(y.index)
    if len(common) < 20:
        return None
    threshold = p.loc[common].quantile(0.90)
    top_mask = p.loc[common] >= threshold
    if top_mask.sum() == 0:
        return None
    return float(y.loc[common][top_mask].mean())


def _replay_metrics(df: pd.DataFrame, *, selected_mask: pd.Series, return_col: str = "forward_return_15m") -> tuple[float | None, float | None]:
    if return_col not in df.columns:
        return None, None
    selected_returns = pd.to_numeric(df.loc[selected_mask, return_col], errors="coerce").dropna()
    if selected_returns.empty:
        return None, None
    return float(selected_returns.mean()), _profit_factor_from_returns(selected_returns)


def _classifier_calibration_metrics(
    payload: dict, frame: pd.DataFrame, target_col: str
) -> dict[str, Any]:
    """Compute Brier score, top-decile precision, and calibration approximations."""
    features = list(payload.get("features") or [])
    model = payload.get("model")
    if not features or model is None or target_col not in frame.columns:
        return {"brier_score": None, "top_decile_precision": None, "ece": None, "confidence_interval": None, "ev_interval": None}
    try:
        X = frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        prob_preds = pd.Series(model.predict_proba(X)[:, 1], index=frame.index)
        y = pd.to_numeric(frame[target_col], errors="coerce").fillna(0).astype(int)
        
        brier = _brier_score_from_predictions(prob_preds, y)
        tdp = _top_decile_precision(prob_preds, y)
        
        # Approximate ECE via 10 equal-width bins
        ece = None
        if len(y) > 50:
            df_cal = pd.DataFrame({'p': prob_preds, 'y': y})
            df_cal['bin'] = pd.cut(df_cal['p'], bins=10, labels=False)
            bin_stats = df_cal.groupby('bin').agg(p_mean=('p', 'mean'), count=('p', 'count'), y_mean=('y', 'mean'))
            if len(bin_stats) > 0:
                total = len(y)
                ece = float((bin_stats['count'] / total * (bin_stats['p_mean'] - bin_stats['y_mean']).abs()).sum())
                
        p_mean = float(prob_preds.mean())
        p_std = float(prob_preds.std())
        ci_lower = max(0.0, p_mean - 1.96 * p_std)
        ci_upper = min(1.0, p_mean + 1.96 * p_std)
        ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
        
        return {
            "brier_score": brier,
            "top_decile_precision": tdp,
            "ece": ece,
            "confidence_interval": ci_str,
            "ev_interval": ci_str,
        }
    except Exception:
        return {"brier_score": None, "top_decile_precision": None, "ece": None, "confidence_interval": None, "ev_interval": None}


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
    model_version: str = "",
    training_window: str = "",
    feature_schema_hash: str = "",
    target_definition: str = "",
    calibration_report: str = "",
    backtest_report: str = "",
    shadow_report: str = "",
    approval_status: str = "pending",
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
        "model_version": model_version,
        "training_window": training_window,
        "feature_schema_hash": feature_schema_hash,
        "target_definition": target_definition,
        "calibration_report": calibration_report,
        "backtest_report": backtest_report,
        "shadow_report": shadow_report,
        "approval_status": approval_status,
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
    
    calib_metrics = _classifier_calibration_metrics(payload, frame, target_col)
    import json
    
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
        model_version=str(payload.get("run_id") or run_id),
        feature_schema_hash=str(hash(tuple(features))),
        target_definition=target_col,
        calibration_report=json.dumps(calib_metrics),
        shadow_report="{}",
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
        model_version=str(payload.get("run_id") or run_id),
        feature_schema_hash=str(hash(tuple(features))),
        target_definition=target_col,
        calibration_report="{}",
        shadow_report="{}",
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
    """
    Phase 8C — challenger must beat champion on:
      1. Primary metric (accuracy or RMSE)
      2. Profit factor (replay EV quality)
      3. Replay EV
      4. Brier score (calibration — lower is better)  [Phase 8 addition]
      5. Top-decile precision                          [Phase 8 addition]
    Anti-failure: never promote if accuracy↑ but EV↓  [Phase 14]
    """
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
            pass  # candidate is better on primary, continue checks
        elif candidate_metric > champion_metric:
            return False
    else:
        if candidate_metric < champion_metric:
            return False
        # candidate_metric >= champion_metric: continue tie-breaking

    # Phase 14 anti-failure: accuracy↑ but EV↓ → block
    candidate_pf = _safe_float(candidate.get("profit_factor"), default=float("-inf"))
    champion_pf = _safe_float(champion.get("profit_factor"), default=float("-inf"))
    candidate_replay = _safe_float(candidate.get("replay_ev"), default=float("-inf"))
    champion_replay = _safe_float(champion.get("replay_ev"), default=float("-inf"))
    # Phase 14 anti-failure: accuracy↑ but EV↓ → block
    candidate_pf = _safe_float(candidate.get("profit_factor"), default=float("-inf"))
    champion_pf = _safe_float(champion.get("profit_factor"), default=float("-inf"))
    candidate_replay = _safe_float(candidate.get("replay_ev"), default=float("-inf"))
    champion_replay = _safe_float(champion.get("replay_ev"), default=float("-inf"))
    
    # Phase 4 Strict logic: MUST survive cost model and walk-forward
    if candidate_replay <= 0.0 or candidate_pf < 1.0:
        return False
        
    # Prefer higher replay EV or PF
    if candidate_replay < champion_replay - 1e-6:
        return False
    if candidate_pf < champion_pf - 1e-6:
        return False

    # Phase 8 — Brier score: lower is better (better calibration)
    candidate_brier = _safe_float(candidate.get("brier_score"), default=None)
    champion_brier = _safe_float(champion.get("brier_score"), default=None)
    if candidate_brier is not None and champion_brier is not None:
        if candidate_brier > champion_brier + 0.02:
            # Calibration worsened by >2 pp — block (Phase 14)
            return False

    # Phase 8 — Top-decile precision: higher is better
    candidate_tdp = _safe_float(candidate.get("top_decile_precision"), default=None)
    champion_tdp = _safe_float(champion.get("top_decile_precision"), default=None)
    if candidate_tdp is not None and champion_tdp is not None:
        if candidate_tdp < champion_tdp - 0.05:
            # Top-decile precision dropped >5 pp — block
            return False

    return True


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
        
    # Phase 4 strict governance: models must survive shadow.
    try:
        import json
        shadow_rpt = json.loads(str(row.get("shadow_report") or "{}"))
        shadow_samples = int(shadow_rpt.get("samples", 0))
    except Exception:
        shadow_samples = 0
        
    if shadow_samples < 50:
        # Prevent straight-to-live promotion.
        return False, f"shadow_samples_below_50_{shadow_samples}"
        
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
