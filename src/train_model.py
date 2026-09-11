"""
Baseline fraud classification model for the CreditTransAct dataset.
Feature selection + preprocessing with polars (memory-safe), then a
baseline XGBoost model with native categorical support, evaluated on
precision/recall and PR-AUC given the realistic class imbalance.
"""
import json
import os

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from features import TARGET, load_dataset, split_dataset
from metrics import best_f1_threshold

sns.set_theme(style="whitegrid")

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------------
# 1. Load + feature selection
# -----------------------------------------------------------------
print("=" * 60)
print("1. LOADING + FEATURE SELECTION")
print("=" * 60)

df, cat_cols, num_cols = load_dataset()

print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numeric features ({len(num_cols)}): {num_cols}")

X = df[cat_cols + num_cols]
y = df[TARGET]

fraud_rate = y.mean() * 100
print(f"\nRows: {len(df):,} | Fraud rate: {fraud_rate:.3f}%")

# -----------------------------------------------------------------
# 2. Train/test split (stratified on the imbalanced target)
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("2. TRAIN/TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = split_dataset(X, y)
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight (neg/pos): {scale_pos_weight:.2f}")

# -----------------------------------------------------------------
# 3. Train baseline XGBoost model
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("3. TRAINING BASELINE XGBOOST MODEL")
print("=" * 60)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("Training complete.")

# -----------------------------------------------------------------
# 4. Evaluation
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("4. EVALUATION")
print("=" * 60)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred_default = (y_proba >= 0.5).astype(int)

pr_auc = average_precision_score(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"PR-AUC:  {pr_auc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nClassification report @ threshold 0.5:")
print(classification_report(y_test, y_pred_default, digits=3))

curve = best_f1_threshold(y_test, y_proba)
precisions, recalls, best_idx = curve["precisions"], curve["recalls"], curve["best_idx"]
best_threshold = curve["threshold"]
y_pred_best = (y_proba >= best_threshold).astype(int)

print(f"\nBest-F1 threshold: {best_threshold:.3f}")
print("Classification report @ best-F1 threshold:")
print(classification_report(y_test, y_pred_best, digits=3))

cm = confusion_matrix(y_test, y_pred_best)
print("Confusion matrix @ best-F1 threshold:")
print(cm)

metrics = {
    "fraud_rate_pct": fraud_rate,
    "pr_auc": pr_auc,
    "roc_auc": roc_auc,
    "default_threshold": 0.5,
    "default_f1": f1_score(y_test, y_pred_default),
    "best_threshold": float(best_threshold),
    "best_f1": curve["f1"],
}
with open(f"{OUTPUT_DIR}/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved metrics -> {OUTPUT_DIR}/model_metrics.json")

# -----------------------------------------------------------------
# 5. Precision-recall curve plot
# -----------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.plot(recalls, precisions, color="#C44E52")
plt.scatter(recalls[best_idx], precisions[best_idx], color="black", zorder=5,
            label=f"best-F1 threshold = {best_threshold:.2f}")
plt.title(f"Precision-Recall Curve (PR-AUC = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_precision_recall_curve.png", dpi=150)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/06_precision_recall_curve.png")

# -----------------------------------------------------------------
# 6. Feature importance plot
# -----------------------------------------------------------------
importance = (
    pl.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    .sort("importance", descending=True)
    .head(20)
)
plt.figure(figsize=(9, 7))
sns.barplot(data=importance.to_pandas(), y="feature", x="importance",
            hue="feature", palette="Blues_r", legend=False)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_feature_importance.png", dpi=150)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/07_feature_importance.png")

# -----------------------------------------------------------------
# 7. Save model
# -----------------------------------------------------------------
model_path = f"{MODEL_DIR}/xgb_fraud_model.json"
model.save_model(model_path)
print(f"\nSaved model -> {model_path}")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE.")
print("=" * 60)
