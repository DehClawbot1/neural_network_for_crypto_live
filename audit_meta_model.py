import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, roc_auc_score

from meta_model_trainer import MetaModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_latest_bundle(weights_dir="weights"):
    bundles = sorted(Path(weights_dir).glob("meta_model_bundle_*.pkl"))
    if not bundles:
        raise FileNotFoundError("No meta_model_bundle_*.pkl files found.")
    bundle_path = bundles[-1]
    bundle = joblib.load(bundle_path)
    logging.info("Loaded bundle: %s", bundle_path)
    return bundle_path, bundle


def rebuild_test_split(data_path="logs/sequence_dataset.csv"):
    trainer = MetaModelTrainer(data_path=data_path)
    X_train, X_test, y_train, y_test, feature_names = trainer._prepare_data()
    return X_test, y_test, feature_names


def main():
    bundle_path, bundle = load_latest_bundle()
    model = bundle["model"]
    feature_names = bundle["features"]
    X_test, y_test, prepared_feature_names = rebuild_test_split()

    missing = [f for f in feature_names if f not in prepared_feature_names]
    if missing:
        raise ValueError(f"Prepared test split is missing model features: {missing[:10]}")

    X_test = X_test[feature_names]
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    top_k = max(1, int(len(X_test) * 0.1))
    top_idx = probs.argsort()[-top_k:]

    print(f"Bundle: {bundle_path.name}")
    print(f"Test rows: {len(X_test)}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs) if y_test.nunique() > 1 else float('nan'):.4f}")
    print(f"Precision @ Top 10%: {precision_score(y_test.iloc[top_idx], preds[top_idx], zero_division=0):.4f}")

    rf = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(rf, "feature_importances_"):
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)
        print("\n--- Top 10 Gini Importance (suspects) ---")
        print(importances.head(10).to_string(index=False))

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="precision",
    )
    perm_df = pd.DataFrame({
        "feature": feature_names,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std,
    }).sort_values("perm_mean", ascending=False)
    print("\n--- Top 10 Permutation Importance ---")
    print(perm_df.head(10).to_string(index=False))

    numeric_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True).rename("target")], axis=1)
    corr = []
    for col in X_test.columns:
        try:
            value = numeric_df[col].corr(numeric_df["target"])
            corr.append((col, value))
        except Exception:
            corr.append((col, None))
    corr_df = pd.DataFrame(corr, columns=["feature", "corr_with_target"]).sort_values("corr_with_target", ascending=False, key=lambda s: s.abs())
    print("\n--- Highest Absolute Correlation With Target ---")
    print(corr_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
