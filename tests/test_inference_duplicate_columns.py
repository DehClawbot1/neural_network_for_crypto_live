import pandas as pd

from model_inference import ModelInference
from stage1_inference import Stage1Inference
from stage2_temporal_inference import Stage2TemporalInference


def _assert_prepare_matrix_handles_duplicate_columns(inference_cls):
    frame = pd.DataFrame(
        [
            [0.10, None, 0.03],
            [None, 0.25, None],
        ],
        columns=["feature_a", "feature_a", "feature_b"],
    )
    saved = {"features": ["feature_a", "feature_b"]}
    inference = inference_cls()

    matrix = inference._prepare_matrix(saved, frame)

    if hasattr(matrix, "shape"):
        assert matrix.shape[0] == 2
        assert matrix.shape[1] == 2


def test_model_inference_prepare_matrix_handles_duplicate_columns():
    _assert_prepare_matrix_handles_duplicate_columns(ModelInference)


def test_stage1_inference_prepare_matrix_handles_duplicate_columns():
    _assert_prepare_matrix_handles_duplicate_columns(Stage1Inference)


def test_stage2_temporal_inference_prepare_matrix_handles_duplicate_columns():
    _assert_prepare_matrix_handles_duplicate_columns(Stage2TemporalInference)


def test_model_inference_run_handles_duplicate_output_columns():
    inference = ModelInference()
    inference._load = lambda path: None
    frame = pd.DataFrame(
        [[0.1, 0.2, 0.3, 0.4]],
        columns=["feature_a", "p_tp_before_sl", "p_tp_before_sl", "expected_return"],
    )

    out = inference.run(frame)

    assert "edge_score" in out.columns
    assert len(out.index) == 1


def test_stage1_inference_run_handles_duplicate_output_columns():
    inference = Stage1Inference()
    inference._load = lambda path: None
    frame = pd.DataFrame(
        [[0.1, 0.2, 0.3, 0.4]],
        columns=["feature_a", "expected_return", "expected_return", "p_tp_before_sl"],
    )

    out = inference.run(frame)

    assert "edge_score" in out.columns
    assert len(out.index) == 1
