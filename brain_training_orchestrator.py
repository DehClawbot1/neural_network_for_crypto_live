from __future__ import annotations

from pathlib import Path

import pandas as pd

from canonical_dataset_builder import CanonicalDatasetBuilder
from baseline_models import BaselineModels
from brain_paths import BTC_FAMILY, WEATHER_FAMILY, BrainContext, list_brain_contexts
from contract_target_builder import ContractTargetBuilder
from historical_dataset_builder import HistoricalDatasetBuilder
from learning_row_builder import LearningRowBuilder
from model_artifact_staging import build_candidate_weights_dir
from model_registry import ModelRegistry
from model_registry_runtime import evaluate_artifact_against_dataset, register_and_promote_rows
from sequence_feature_builder import SequenceFeatureBuilder
from stage1_models import Stage1Models
from stage2_temporal_models import Stage2TemporalModels
from supervised_models import SupervisedModels
from task_training_table_builder import TaskTrainingTableBuilder
from weather_temperature_trainer import WeatherTemperatureTrainer


def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _artifact_group(context: BrainContext, base_name: str) -> str:
    if context.market_family == WEATHER_FAMILY:
        mapping = {
            "tabular_classifier": "weather_tabular_classifier",
            "tabular_regressor": "weather_tabular_regressor",
            "stage1_classifier": "weather_stage1_classifier",
            "stage1_regressor": "weather_stage1_regressor",
            "stage2_temporal_classifier": "weather_stage2_temporal_classifier",
            "stage2_temporal_regressor": "weather_stage2_temporal_regressor",
            "weather_temperature_classifier": "weather_temperature_classifier",
        }
    else:
        mapping = {
            "tabular_classifier": "btc_tabular_classifier",
            "tabular_regressor": "btc_tabular_regressor",
            "stage1_classifier": "stage1_classifier",
            "stage1_regressor": "stage1_regressor",
            "stage2_temporal_classifier": "stage2_temporal_classifier",
            "stage2_temporal_regressor": "stage2_temporal_regressor",
            "weather_temperature_classifier": "weather_temperature_classifier",
        }
    return mapping[base_name]


def build_family_datasets(
    *,
    shared_logs_dir: str | Path = "logs",
    shared_weights_dir: str | Path = "weights",
    forward_minutes: int = 15,
    max_hold_minutes: int = 60,
    tp_move: float = 0.04,
    sl_move: float = 0.03,
) -> list[BrainContext]:
    contexts = list_brain_contexts(shared_logs_dir=shared_logs_dir, shared_weights_dir=shared_weights_dir)
    for context in contexts:
        HistoricalDatasetBuilder(brain_context=context).write()
        ContractTargetBuilder(brain_context=context).write(
            forward_minutes=forward_minutes,
            max_hold_minutes=max_hold_minutes,
            tp_move=tp_move,
            sl_move=sl_move,
        )
        SequenceFeatureBuilder(brain_context=context).write()
    # Build the canonical learning dataset after family-specific artifacts
    # exist, so every later training step can rely on one joined spine.
    LearningRowBuilder(logs_dir=str(shared_logs_dir)).write()
    CanonicalDatasetBuilder(logs_dir=str(shared_logs_dir)).write()
    TaskTrainingTableBuilder(logs_dir=str(shared_logs_dir)).write()
    return contexts


def train_brain_models(
    context: BrainContext,
    *,
    candidate_weights_dir: str | Path | None = None,
    candidate_prefix: str = "brain",
) -> Path:
    candidate_dir = (
        Path(candidate_weights_dir)
        if candidate_weights_dir is not None
        else build_candidate_weights_dir(context.weights_dir, prefix=f"{candidate_prefix}_{context.market_family}")
    )
    SupervisedModels(brain_context=context, weights_dir=candidate_dir).train()
    Stage1Models(brain_context=context, weights_dir=candidate_dir).train()
    SequenceFeatureBuilder(brain_context=context).write()
    Stage2TemporalModels(brain_context=context, weights_dir=candidate_dir).train()
    BaselineModels(brain_context=context, weights_dir=candidate_dir).train()
    if context.market_family == WEATHER_FAMILY:
        WeatherTemperatureTrainer(brain_context=context, weights_dir=candidate_dir).train()
    return candidate_dir


def train_execution_engine(
    context: BrainContext,
    candidate_dir: Path,
) -> dict[str, bool]:
    """
    Phase 6 Stage C — Train execution engine models from execution_feedback.csv.
    Returns dict of model_name → success.  Non-blocking on insufficient data.
    """
    try:
        from execution_engine import ExecutionEngine
        from data_integrity_layer import DataIntegrityLayer
        dil = DataIntegrityLayer(logs_dir=str(context.shared_logs_dir))
        exec_df = dil.build_execution_training_set()
        if exec_df.empty or len(exec_df) < 50:
            import logging as _logging
            _logging.info(
                "train_execution_engine[%s]: skipped (only %d execution feedback rows, need 50)",
                context.market_family, len(exec_df),
            )
            return {}
        engine = ExecutionEngine(weights_dir=str(candidate_dir), logs_dir=str(context.shared_logs_dir))
        results = engine.train(exec_df)
        import logging as _logging
        _logging.info("train_execution_engine[%s]: results=%s", context.market_family, results)
        return results
    except Exception as exc:
        import logging as _logging
        _logging.warning("train_execution_engine[%s]: failed (non-blocking): %s", context.market_family, exc)
        return {}


def train_meta_decision_engine(
    context: BrainContext,
    candidate_dir: Path,
) -> bool:
    """
    Phase 6 Stage D — Train MetaDecisionEngine from policy_feedback.csv.
    Returns True if trained successfully.  Non-blocking on insufficient data.
    """
    try:
        from data_integrity_layer import DataIntegrityLayer
        import lightgbm as lgb
        import joblib
        import pandas as pd

        dil = DataIntegrityLayer(logs_dir=str(context.shared_logs_dir))
        policy_df = dil.build_policy_training_set()
        if policy_df.empty or len(policy_df) < 50:
            import logging as _logging
            _logging.info(
                "train_meta_decision_engine[%s]: skipped (%d policy rows, need 50)",
                context.market_family, len(policy_df),
            )
            return False

        # Target: policy_success = 1
        y = (policy_df.get("policy_verdict", pd.Series("", index=policy_df.index)) == "policy_success").astype(int)
        feature_cols = [c for c in policy_df.columns if c not in {
            "policy_verdict", "attributed_at", "report_id", "token_id",
            "market_family", "signal_label", "sample_weight",
        } and pd.api.types.is_numeric_dtype(policy_df[c])]
        if not feature_cols:
            return False

        X = policy_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        sw = policy_df.get("sample_weight", pd.Series(1.0, index=policy_df.index))
        model = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, random_state=42, verbose=-1)
        model.fit(X, y, sample_weight=sw)
        path = candidate_dir / "meta_decision_engine.joblib"
        joblib.dump({"model": model, "features": feature_cols, "market_family": context.market_family}, path)

        import logging as _logging
        _logging.info("train_meta_decision_engine[%s]: trained on %d rows → %s", context.market_family, len(policy_df), path)
        return True
    except Exception as exc:
        import logging as _logging
        _logging.warning("train_meta_decision_engine[%s]: failed (non-blocking): %s", context.market_family, exc)
        return False


def evaluate_brain_candidate_rows(
    context: BrainContext,
    *,
    run_id: str,
    candidate_weights_dir: str | Path,
) -> list[dict]:
    candidate_dir = Path(candidate_weights_dir)
    contract_targets_file = context.logs_dir / "contract_targets.csv"
    sequence_dataset_file = context.logs_dir / "sequence_dataset.csv"
    rows: list[dict] = []

    rows.extend(
        evaluate_artifact_against_dataset(
            run_id=run_id,
            dataset_file=contract_targets_file,
            artifact_path=candidate_dir / "tp_classifier.joblib",
            artifact_group=_artifact_group(context, "tabular_classifier"),
            market_family=context.market_family,
            target_col="tp_before_sl_60m",
        ).to_dict("records")
    )
    rows.extend(
        evaluate_artifact_against_dataset(
            run_id=run_id,
            dataset_file=contract_targets_file,
            artifact_path=candidate_dir / "return_regressor.joblib",
            artifact_group=_artifact_group(context, "tabular_regressor"),
            market_family=context.market_family,
            target_col="forward_return_15m",
        ).to_dict("records")
    )
    rows.extend(
        evaluate_artifact_against_dataset(
            run_id=run_id,
            dataset_file=contract_targets_file,
            artifact_path=candidate_dir / "stage1_tp_classifier.joblib",
            artifact_group=_artifact_group(context, "stage1_classifier"),
            market_family=context.market_family,
            target_col="tp_before_sl_60m",
        ).to_dict("records")
    )
    rows.extend(
        evaluate_artifact_against_dataset(
            run_id=run_id,
            dataset_file=contract_targets_file,
            artifact_path=candidate_dir / "stage1_return_regressor.joblib",
            artifact_group=_artifact_group(context, "stage1_regressor"),
            market_family=context.market_family,
            target_col="forward_return_15m",
        ).to_dict("records")
    )
    rows.extend(
        evaluate_artifact_against_dataset(
            run_id=run_id,
            dataset_file=sequence_dataset_file,
            artifact_path=candidate_dir / "stage2_temporal_classifier.joblib",
            artifact_group=_artifact_group(context, "stage2_temporal_classifier"),
            market_family=context.market_family,
            target_col="tp_before_sl_60m",
        ).to_dict("records")
    )
    rows.extend(
        evaluate_artifact_against_dataset(
            run_id=run_id,
            dataset_file=sequence_dataset_file,
            artifact_path=candidate_dir / "stage2_temporal_regressor.joblib",
            artifact_group=_artifact_group(context, "stage2_temporal_regressor"),
            market_family=context.market_family,
            target_col="forward_return_15m",
        ).to_dict("records")
    )
    if context.market_family == WEATHER_FAMILY:
        rows.extend(
            evaluate_artifact_against_dataset(
                run_id=run_id,
                dataset_file=contract_targets_file,
                artifact_path=candidate_dir / "weather_temperature_model.joblib",
                artifact_group=_artifact_group(context, "weather_temperature_classifier"),
                market_family=context.market_family,
                target_col="target_up",
                market_family_prefix=WEATHER_FAMILY,
            ).to_dict("records")
        )

    baseline_df = _safe_read_csv(context.logs_dir / "baseline_eval.csv")
    if not baseline_df.empty:
        baseline_df = baseline_df.copy()
        baseline_df["run_id"] = run_id
        baseline_df["promotion_status"] = "evaluation_only"
        baseline_df["promotion_reason"] = "baseline_report_only"
        baseline_df["beats_champion"] = None
        baseline_df["is_champion"] = False
        baseline_df["promotion_gate_passed"] = None
        baseline_df["market_family"] = context.market_family
        baseline_df["notes"] = baseline_df.get("notes", pd.Series("", index=baseline_df.index)).fillna("")
        rows.extend(baseline_df.to_dict("records"))

    return rows


def register_and_promote_brain_models(
    context: BrainContext,
    *,
    candidate_rows: list[dict] | pd.DataFrame,
    candidate_weights_dir: str | Path,
    min_test_rows: int = 10,
) -> pd.DataFrame:
    registry = ModelRegistry(brain_context=context)
    return register_and_promote_rows(
        registry=registry,
        candidate_rows=candidate_rows,
        candidate_weights_dir=candidate_weights_dir,
        active_weights_dir=context.weights_dir,
        min_test_rows=min_test_rows,
    )
