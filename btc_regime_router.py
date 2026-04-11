from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_MODEL_WEIGHTS = {
    "calm": {"legacy": 0.45, "stage1": 0.35, "stage2": 0.20, "multiplier": 1.02},
    "trend": {"legacy": 0.20, "stage1": 0.45, "stage2": 0.35, "multiplier": 1.08},
    "volatile": {"legacy": 0.15, "stage1": 0.40, "stage2": 0.45, "multiplier": 0.94},
    "chaotic": {"legacy": 0.35, "stage1": 0.25, "stage2": 0.40, "multiplier": 0.72},
}


def _safe_float(value, default=0.0):
    try:
        num = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(num):
        return float(default)
    return float(num)


def _clip01(value):
    return float(np.clip(_safe_float(value, 0.0), 0.0, 1.0))


def _safe_numeric_series(frame: pd.DataFrame, column_name: str, default=np.nan) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index)
    raw = frame.loc[:, column_name]
    if isinstance(raw, pd.DataFrame):
        picked = raw.apply(
            lambda row: next((value for value in row.tolist() if pd.notna(value)), default),
            axis=1,
        )
        return pd.to_numeric(picked, errors="coerce")
    return pd.to_numeric(raw, errors="coerce")


def _bias_alignment_score(trend_bias: str, live_bias: str) -> float:
    trend = str(trend_bias or "NEUTRAL").strip().upper()
    live = str(live_bias or "NEUTRAL").strip().upper()
    if trend in {"LONG", "SHORT"} and trend == live:
        return 1.0
    if trend == "NEUTRAL" and live == "NEUTRAL":
        return 0.5
    if "NEUTRAL" in {trend, live}:
        return 0.35
    return 0.0


def classify_btc_regime_row(row: dict | pd.Series) -> dict:
    market_structure = str((row.get("market_structure") if hasattr(row, "get") else None) or "UNKNOWN").strip().upper()
    trend_bias = str((row.get("btc_trend_bias") if hasattr(row, "get") else None) or "NEUTRAL").strip().upper()
    live_bias = str((row.get("btc_live_bias") if hasattr(row, "get") else None) or "NEUTRAL").strip().upper()

    trend_strength = abs(_safe_float(row.get("trend_score", 0.5), 0.5) - 0.5) * 2.0
    trend_confluence = _clip01(row.get("btc_trend_confluence", 0.0))
    momentum_confluence = _clip01(row.get("btc_momentum_confluence", 0.0))
    live_confluence = _clip01(row.get("btc_live_confluence", 0.0))
    bias_alignment = _bias_alignment_score(trend_bias, live_bias)

    volatility_proxy = _clip01(row.get("btc_live_volatility_proxy", 0.0))
    volatility_regime_score = _clip01(row.get("btc_volatility_regime_score", 0.0))
    basis_dislocation = _clip01(abs(_safe_float(row.get("btc_live_mark_index_basis_bps", 0.0), 0.0)) / 25.0)
    source_divergence = _clip01(abs(_safe_float(row.get("btc_live_source_divergence_bps", 0.0), 0.0)) / 35.0)
    realized_vol_component = _clip01(
        max(
            _safe_float(row.get("btc_realized_vol_1h", 0.0), 0.0) / 0.012,
            _safe_float(row.get("btc_realized_vol_4h", 0.0), 0.0) / 0.020,
        )
    )
    structure_mixed = 1.0 if market_structure == "MIXED" else 0.0
    bias_conflict = 1.0 if live_bias in {"LONG", "SHORT"} and trend_bias in {"LONG", "SHORT"} and live_bias != trend_bias else 0.0

    regime_trend_score = _clip01(
        trend_strength * 0.30
        + trend_confluence * 0.26
        + momentum_confluence * 0.20
        + live_confluence * 0.14
        + bias_alignment * 0.10
    )
    regime_volatility_score = _clip01(
        volatility_proxy * 0.38
        + volatility_regime_score * 0.30
        + basis_dislocation * 0.18
        + realized_vol_component * 0.14
    )
    regime_chaos_score = _clip01(
        source_divergence * 0.34
        + bias_conflict * 0.30
        + structure_mixed * 0.16
        + basis_dislocation * 0.20
    )
    regime_stability_score = _clip01(1.0 - ((regime_volatility_score * 0.55) + (regime_chaos_score * 0.45)))

    if regime_chaos_score >= 0.58 or (regime_volatility_score >= 0.78 and regime_trend_score <= 0.35):
        label = "chaotic"
        confidence = max(regime_chaos_score, regime_volatility_score)
    elif regime_volatility_score >= 0.55 and regime_trend_score < 0.60:
        label = "volatile"
        confidence = max(regime_volatility_score, regime_chaos_score)
    elif regime_trend_score >= 0.55 and regime_chaos_score < 0.45:
        label = "trend"
        confidence = max(regime_trend_score, regime_stability_score)
    else:
        label = "calm"
        confidence = max(regime_stability_score, 1.0 - regime_volatility_score)

    weights = REGIME_MODEL_WEIGHTS[label]
    primary_model = max(("legacy", "stage1", "stage2"), key=lambda key: weights[key])

    return {
        "btc_market_regime_label": label,
        "btc_market_regime_score": round(_clip01(confidence), 6),
        "btc_market_regime_trend_score": round(regime_trend_score, 6),
        "btc_market_regime_volatility_score": round(regime_volatility_score, 6),
        "btc_market_regime_chaos_score": round(regime_chaos_score, 6),
        "btc_market_regime_stability_score": round(regime_stability_score, 6),
        "btc_market_regime_is_calm": 1 if label == "calm" else 0,
        "btc_market_regime_is_trend": 1 if label == "trend" else 0,
        "btc_market_regime_is_volatile": 1 if label == "volatile" else 0,
        "btc_market_regime_is_chaotic": 1 if label == "chaotic" else 0,
        "btc_market_regime_primary_model": primary_model,
        "btc_market_regime_confidence_multiplier": float(weights["multiplier"]),
        "btc_market_regime_weight_legacy": float(weights["legacy"]),
        "btc_market_regime_weight_stage1": float(weights["stage1"]),
        "btc_market_regime_weight_stage2": float(weights["stage2"]),
    }


def annotate_btc_regime(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    regime_rows = pd.DataFrame([classify_btc_regime_row(row) for _, row in out.iterrows()], index=out.index)
    return pd.concat([out, regime_rows], axis=1)


def apply_regime_model_blend(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()

    out = annotate_btc_regime(frame)
    model_columns = {
        "legacy": {
            "p_tp": "legacy_p_tp_before_sl",
            "expected_return": "legacy_expected_return",
            "edge": "legacy_edge_score",
        },
        "stage1": {
            "p_tp": "stage1_p_tp_before_sl",
            "expected_return": "stage1_expected_return",
            "edge": "stage1_edge_score",
        },
        "stage2": {
            "p_tp": "temporal_p_tp_before_sl",
            "expected_return": "temporal_expected_return",
            "edge": "temporal_edge_score",
        },
    }

    for model_name, cols in model_columns.items():
        for value_col in cols.values():
            if value_col not in out.columns:
                out[value_col] = np.nan

    if "stage1_lower_confidence_bound" not in out.columns:
        out["stage1_lower_confidence_bound"] = np.nan
    if "stage1_return_std" not in out.columns:
        out["stage1_return_std"] = np.nan

    for model_name in ("legacy", "stage1", "stage2"):
        out[f"btc_market_regime_weight_{model_name}"] = _safe_numeric_series(
            out,
            f"btc_market_regime_weight_{model_name}",
            np.nan,
        ).fillna(REGIME_MODEL_WEIGHTS["calm"][model_name])

    for model_name, cols in model_columns.items():
        p_tp_series = _safe_numeric_series(out, cols["p_tp"], np.nan)
        return_series = _safe_numeric_series(out, cols["expected_return"], np.nan)
        out[cols["p_tp"]] = p_tp_series
        out[cols["expected_return"]] = return_series
        if cols["edge"] in out.columns:
            out[cols["edge"]] = _safe_numeric_series(out, cols["edge"], np.nan)
        else:
            out[cols["edge"]] = p_tp_series * return_series

    available_mask = pd.DataFrame(index=out.index)
    for model_name, cols in model_columns.items():
        p_tp_values = _safe_numeric_series(out, cols["p_tp"], np.nan)
        expected_values = _safe_numeric_series(out, cols["expected_return"], np.nan)
        edge_values = _safe_numeric_series(out, cols["edge"], np.nan)
        available_mask[model_name] = (
            (p_tp_values.notna() | expected_values.notna())
            & ~(
                p_tp_values.fillna(0.0).abs().le(1e-12)
                & expected_values.fillna(0.0).abs().le(1e-12)
                & edge_values.fillna(0.0).abs().le(1e-12)
            )
        )
    for model_name in ("legacy", "stage1", "stage2"):
        out[f"btc_market_regime_weight_{model_name}"] = out[f"btc_market_regime_weight_{model_name}"].where(
            available_mask[model_name],
            0.0,
        )

    weight_sum = (
        out["btc_market_regime_weight_legacy"]
        + out["btc_market_regime_weight_stage1"]
        + out["btc_market_regime_weight_stage2"]
    ).replace(0.0, np.nan)
    for model_name in ("legacy", "stage1", "stage2"):
        weight_col = f"btc_market_regime_weight_{model_name}"
        out[weight_col] = (out[weight_col] / weight_sum).fillna(0.0)

    p_tp_blend = pd.Series(0.0, index=out.index, dtype=float)
    return_blend = pd.Series(0.0, index=out.index, dtype=float)
    conservative_return_blend = pd.Series(0.0, index=out.index, dtype=float)
    edge_blend = pd.Series(0.0, index=out.index, dtype=float)
    return_components = []
    for model_name, cols in model_columns.items():
        weight_col = f"btc_market_regime_weight_{model_name}"
        p_tp_values = _safe_numeric_series(out, cols["p_tp"], 0.0).fillna(0.0)
        expected_values = _safe_numeric_series(out, cols["expected_return"], 0.0).fillna(0.0)
        p_tp_blend += p_tp_values * out[weight_col]
        return_blend += expected_values * out[weight_col]
        edge_blend += (p_tp_values * expected_values) * out[weight_col]
        return_components.append(expected_values.rename(model_name))

    return_matrix = pd.concat(return_components, axis=1).fillna(0.0)
    weighted_return_std = pd.Series(0.0, index=out.index, dtype=float)
    for model_name in ("legacy", "stage1", "stage2"):
        centered = return_matrix[model_name] - return_blend
        weighted_return_std += out[f"btc_market_regime_weight_{model_name}"] * (centered ** 2)
    weighted_return_std = np.sqrt(weighted_return_std.clip(lower=0.0))

    stage1_return_std = _safe_numeric_series(out, "stage1_return_std", 0.0).fillna(0.0)
    return_std = np.maximum(weighted_return_std, stage1_return_std * out["btc_market_regime_weight_stage1"].fillna(0.0))

    stage1_lcb = _safe_numeric_series(out, "stage1_lower_confidence_bound", np.nan).fillna(
        _safe_numeric_series(out, model_columns["stage1"]["expected_return"], 0.0).fillna(0.0)
    )
    conservative_return_blend += _safe_numeric_series(out, model_columns["legacy"]["expected_return"], 0.0).fillna(0.0) * out["btc_market_regime_weight_legacy"]
    conservative_return_blend += stage1_lcb * out["btc_market_regime_weight_stage1"]
    conservative_return_blend += _safe_numeric_series(out, model_columns["stage2"]["expected_return"], 0.0).fillna(0.0) * out["btc_market_regime_weight_stage2"]

    out["regime_blended_p_tp_before_sl"] = p_tp_blend.clip(0.0, 1.0)
    out["regime_blended_expected_return"] = return_blend
    out["regime_blended_edge_score"] = edge_blend
    out["regime_blended_conservative_expected_return"] = conservative_return_blend
    out["return_std"] = return_std.fillna(0.0)
    out["lower_confidence_bound"] = (conservative_return_blend - out["return_std"]).fillna(0.0)

    out["p_tp_before_sl"] = out["regime_blended_p_tp_before_sl"]
    out["expected_return"] = out["regime_blended_expected_return"]
    out["edge_score"] = out["regime_blended_edge_score"]

    return out
