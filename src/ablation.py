"""
Ablation experiment: how much does the baseline model rely on a single
dominant feature (cards_on_device_30d, ~76% of gain importance in the
baseline model)? Retrains without it, and without it + the runner-up
(device_fingerprint_match), to see how far PR-AUC/recall fall when the
model can't lean on the top 1-2 signals.
"""
import json
import os

import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

from features import TARGET, load_dataset, split_dataset
from metrics import best_f1_threshold

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------
# Load once, build the full feature frame, reuse for every variant.
# -----------------------------------------------------------------
print("Loading data...")
df, cat_cols, num_cols = load_dataset()
y = df[TARGET]
all_features = cat_cols + num_cols

VARIANTS = {
    "full_model": [],
    "without_cards_on_device_30d": ["cards_on_device_30d"],
    "without_top2_signals": ["cards_on_device_30d", "device_fingerprint_match"],
}

# Hyperparameters are pinned here independently of train_model.py rather
# than shared -- the ablation is comparing feature sets against a fixed
# model configuration, so it should stay stable even if train_model.py's
# tuning evolves later.
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)

results = {}

for name, drop_features in VARIANTS.items():
    print("\n" + "=" * 60)
    print(f"VARIANT: {name}  (dropping: {drop_features or 'nothing'})")
    print("=" * 60)

    features = [f for f in all_features if f not in drop_features]
    X = df[features]

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    curve = best_f1_threshold(y_test, y_proba)
    y_pred_best = (y_proba >= curve["threshold"]).astype(int)

    top_features = (
        pl.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort("importance", descending=True)
        .head(5)
        .to_dicts()
    )

    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Best-F1 threshold: {curve['threshold']:.3f}")
    print(f"F1 @ best threshold: {curve['f1']:.4f}")
    print(f"Recall @ best threshold: {recall_score(y_test, y_pred_best):.4f}")
    print("Top 5 features now:")
    for row in top_features:
        print(f"  {row['feature']}: {row['importance']:.4f}")

    results[name] = {
        "dropped_features": drop_features,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "best_threshold": curve["threshold"],
        "best_f1": curve["f1"],
        "recall_at_best_f1": float(recall_score(y_test, y_pred_best)),
        "top_5_features": top_features,
    }

with open(f"{OUTPUT_DIR}/ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'variant':<30}{'PR-AUC':>10}{'ROC-AUC':>10}{'best F1':>10}{'recall':>10}")
for name, r in results.items():
    print(f"{name:<30}{r['pr_auc']:>10.4f}{r['roc_auc']:>10.4f}{r['best_f1']:>10.4f}{r['recall_at_best_f1']:>10.4f}")

print(f"\nSaved -> {OUTPUT_DIR}/ablation_results.json")
