"""
Real inference: scores genuinely new, unlabeled transactions with the trained
fraud model, and generates an analyst-readable explanation (SHAP + local LLM)
for every transaction flagged above the decision threshold.

Usage:
    python src/score_new_transactions.py path/to/new_transactions.parquet
    python src/score_new_transactions.py path/to/new_transactions.parquet --threshold 0.5
    python src/score_new_transactions.py path/to/new_transactions.parquet --max-explanations 20

The input file must have the same schema as the training data, minus
is_fraud (which won't exist yet -- that's the point).
"""
import argparse
import json
import os

import shap
import xgboost as xgb

from features import load_dataset
from genai import explain_transaction, load_generator, top_signals

OUTPUT_DIR = "outputs"
MODEL_PATH = "models/xgb_fraud_model.json"
METRICS_PATH = "outputs/model_metrics.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", help="Path to a parquet file of new, unlabeled transactions")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Fraud probability cutoff for flagging (default: the model's "
                              "best-F1 threshold from outputs/model_metrics.json)")
    parser.add_argument("--max-explanations", type=int, default=25,
                         help="Cap on how many flagged transactions get a GenAI explanation "
                              "(SHAP + local LLM generation is slow; default 25)")
    args = parser.parse_args()

    if args.threshold is not None:
        threshold = args.threshold
    else:
        with open(METRICS_PATH) as f:
            threshold = json.load(f)["best_threshold"]
    print(f"Using decision threshold: {threshold:.4f}")

    # -----------------------------------------------------------------
    # 1. Load model + new transactions
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. LOADING MODEL + NEW TRANSACTIONS")
    print("=" * 60)

    fraud_model = xgb.XGBClassifier()
    fraud_model.load_model(MODEL_PATH)

    df, cat_cols, num_cols = load_dataset(args.data_path, keep_transaction_id=True)
    feature_cols = cat_cols + num_cols
    X = df[feature_cols]
    print(f"Loaded {len(df):,} new transactions")

    # -----------------------------------------------------------------
    # 2. Score every transaction
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. SCORING")
    print("=" * 60)

    fraud_proba = fraud_model.predict_proba(X)[:, 1]
    df["fraud_probability"] = fraud_proba
    df["flagged"] = fraud_proba >= threshold

    flagged_count = int(df["flagged"].sum())
    print(f"Flagged {flagged_count:,} / {len(df):,} transactions "
          f"({flagged_count / len(df) * 100:.2f}%) at threshold {threshold:.4f}")

    # -----------------------------------------------------------------
    # 3. Explain the flagged transactions (capped, highest-probability first)
    # -----------------------------------------------------------------
    flagged = df[df["flagged"]].sort_values("fraud_probability", ascending=False)
    to_explain = flagged.head(args.max_explanations)

    results = []
    if len(to_explain) > 0:
        print("\n" + "=" * 60)
        print(f"3. GENERATING EXPLANATIONS FOR {len(to_explain)} FLAGGED TRANSACTIONS")
        print("=" * 60)

        explainer = shap.TreeExplainer(fraud_model)
        generator = load_generator()

        for idx, row_data in to_explain.iterrows():
            row = X.loc[[idx]]
            proba = row_data["fraud_probability"]
            signals = top_signals(explainer, feature_cols, row)
            explanation = explain_transaction(generator, round(float(proba) * 100, 2), signals)

            tx_id = str(df.loc[idx, "transaction_id"]) if "transaction_id" in df.columns else str(idx)
            print(f"\n{'-' * 60}")
            print(f"Transaction {tx_id}: {proba * 100:.2f}% fraud probability")
            print(f"Summary: {explanation['summary']}")
            print(f"Recommended action: {explanation['recommended_action']}")

            results.append({
                "transaction_id": tx_id,
                "fraud_probability_pct": round(float(proba) * 100, 2),
                "top_contributing_signals": signals,
                "explanation": explanation,
            })

        if flagged_count > args.max_explanations:
            print(f"\n({flagged_count - args.max_explanations} additional flagged transactions "
                  f"were not explained -- raise --max-explanations to cover more)")
    else:
        print("\nNo transactions were flagged at this threshold -- nothing to explain.")

    # -----------------------------------------------------------------
    # 4. Save output
    # -----------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scores_path = f"{OUTPUT_DIR}/new_transaction_scores.csv"
    explanations_path = f"{OUTPUT_DIR}/new_transaction_explanations.json"

    score_cols = ["fraud_probability", "flagged"]
    if "transaction_id" in df.columns:
        score_cols = ["transaction_id"] + score_cols
    df[score_cols].to_csv(scores_path, index=False)

    with open(explanations_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Saved all scores -> {scores_path}")
    print(f"Saved explanations -> {explanations_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
