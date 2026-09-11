"""
GenAI explanation layer demo: picks 4 illustrative cases from the known
test set (a confident catch, a borderline case, a confident clear, and a
missed fraud/false negative) and generates an analyst-readable explanation
for each, using per-transaction SHAP attribution + a local LLM.

For scoring genuinely new, unlabeled transactions, see score_new_transactions.py.

No API key required -- the model runs locally on CPU. First run downloads
the model weights (~3GB) from Hugging Face and caches them.
"""
import json
import os

import pandas as pd
import polars as pl
import shap
import xgboost as xgb

from features import DATA_PATH, TARGET, load_dataset, split_dataset
from genai import explain_transaction, load_generator, top_signals

OUTPUT_DIR = "outputs"
MODEL_PATH = "models/xgb_fraud_model.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------
# 1. Load model + reconstruct the same train/test split used to train it
# -----------------------------------------------------------------
print("=" * 60)
print("1. LOADING FRAUD MODEL + DATA")
print("=" * 60)

fraud_model = xgb.XGBClassifier()
fraud_model.load_model(MODEL_PATH)

df, cat_cols, num_cols = load_dataset()
feature_cols = cat_cols + num_cols
X = df[feature_cols]
y = df[TARGET]

_, X_test, _, y_test = split_dataset(X, y)

tx_ids = (
    pl.scan_parquet(DATA_PATH)
    .select("transaction_id")
    .collect()
    .to_pandas()["transaction_id"]
)
print(f"Test set: {len(X_test):,} rows")

# -----------------------------------------------------------------
# 2. Score the test set and pick a handful of illustrative cases
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("2. SCORING + SELECTING EXAMPLE TRANSACTIONS")
print("=" * 60)

y_proba = pd.Series(fraud_model.predict_proba(X_test)[:, 1], index=X_test.index)
fraud_mask = y_test == 1
legit_mask = y_test == 0

cases = {
    "Confidently caught fraud": y_proba[fraud_mask].idxmax(),
    "Borderline / ambiguous case": (y_proba - 0.5).abs().idxmin(),
    "Confidently legitimate": y_proba[legit_mask].idxmin(),
    "Missed fraud (false negative)": y_proba[fraud_mask].idxmin(),
}
for label, idx in cases.items():
    print(f"{label}: row {idx}, proba={y_proba.loc[idx]:.4f}, actual_fraud={y_test.loc[idx]}")

# -----------------------------------------------------------------
# 3. Load the local LLM
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("3. LOADING LOCAL LLM")
print("=" * 60)
print("First run downloads the model from Hugging Face -- this can take a while.")

generator = load_generator()

# -----------------------------------------------------------------
# 4. Per-transaction SHAP attribution + explanation
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("4. COMPUTING SHAP ATTRIBUTIONS + GENERATING EXPLANATIONS")
print("=" * 60)

explainer = shap.TreeExplainer(fraud_model)
results = []

for label, idx in cases.items():
    row = X_test.loc[[idx]]
    proba = y_proba.loc[idx]
    signals = top_signals(explainer, feature_cols, row)
    explanation = explain_transaction(generator, round(float(proba) * 100, 2), signals)

    print(f"\n{'-' * 60}")
    print(f"CASE: {label} (transaction {tx_ids.loc[idx]})")
    print(f"Model score: {proba * 100:.2f}% fraud probability | actual label: "
          f"{'fraud' if y_test.loc[idx] == 1 else 'legitimate'}")
    print(f"Top signals: {[s['feature'] for s in signals]}")
    print(f"\nSummary: {explanation['summary']}")
    print("Key signals:")
    for s in explanation["key_signals"]:
        print(f"  - {s}")
    print(f"Recommended action: {explanation['recommended_action']}")

    results.append({
        "case": label,
        "transaction_id": str(tx_ids.loc[idx]),
        "fraud_probability_pct": round(float(proba) * 100, 2),
        "actual_label": "fraud" if y_test.loc[idx] == 1 else "legitimate",
        "top_contributing_signals": signals,
        "explanation": explanation,
    })

with open(f"{OUTPUT_DIR}/sample_explanations.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print(f"Saved -> {OUTPUT_DIR}/sample_explanations.json")
print("=" * 60)
