"""
EDA script for the CreditTransAct fraud dataset. Uses polars for
memory-safe aggregation on 15M rows, then plots small summarized results
with matplotlib/seaborn.
"""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

DATA_PATH = "data/credit_card_fraud.parquet"
OUTPUT_DIR = "outputs"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

lazy_df = pl.scan_parquet(DATA_PATH)

# -----------------------------------------------------------------
# 1. Overall class balance
# -----------------------------------------------------------------
print("=" * 60)
print("1. OVERALL CLASS BALANCE")
print("=" * 60)
class_balance = (
    lazy_df.group_by("is_fraud")
    .agg(pl.len().alias("count"))
    .collect()
)
total = class_balance["count"].sum()
class_balance = class_balance.with_columns(
    (pl.col("count") / total * 100).alias("pct")
)
print(class_balance)

# -----------------------------------------------------------------
# 2. Missing values check
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("2. MISSING VALUES PER COLUMN")
print("=" * 60)
null_counts = lazy_df.null_count().collect()
# transpose for readability
null_counts_t = null_counts.transpose(include_header=True, header_name="column", column_names=["null_count"])
null_counts_t = null_counts_t.filter(pl.col("null_count") > 0)
if null_counts_t.height == 0:
    print("No missing values found in any column.")
else:
    print(null_counts_t)

# -----------------------------------------------------------------
# 3. Fraud rate by segment
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("3. FRAUD RATE BY SEGMENT")
print("=" * 60)
fraud_by_segment = (
    lazy_df.group_by("segment")
    .agg([
        pl.len().alias("total"),
        pl.col("is_fraud").sum().alias("fraud_count"),
    ])
    .with_columns((pl.col("fraud_count") / pl.col("total") * 100).alias("fraud_rate_pct"))
    .sort("segment")
    .collect()
)
print(fraud_by_segment)

# Chart 1: Fraud rate by segment
plt.figure(figsize=(8, 5))
pdf1 = fraud_by_segment.to_pandas()
sns.barplot(data=pdf1, x="segment", y="fraud_rate_pct", hue="segment", palette="Reds_r", legend=False)
plt.title("Fraud Rate by Customer Segment")
plt.ylabel("Fraud Rate (%)")
plt.xlabel("Segment")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_fraud_rate_by_segment.png", dpi=150)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/01_fraud_rate_by_segment.png")

# -----------------------------------------------------------------
# 4. Transaction amount: fraud vs non-fraud (summary stats, then sample for plotting)
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("4. TRANSACTION AMOUNT: FRAUD VS NON-FRAUD")
print("=" * 60)
amount_stats = (
    lazy_df.group_by("is_fraud")
    .agg([
        pl.col("transaction_amount").mean().alias("mean_amount"),
        pl.col("transaction_amount").median().alias("median_amount"),
        pl.col("transaction_amount").std().alias("std_amount"),
        pl.col("transaction_amount").min().alias("min_amount"),
        pl.col("transaction_amount").max().alias("max_amount"),
    ])
    .collect()
)
print(amount_stats)

# For the distribution chart, take a random sample (avoid loading 15M rows into matplotlib)
sample_df = (
    lazy_df.select(["transaction_amount", "is_fraud"])
    .collect()
    .sample(n=200_000, seed=42)
    .to_pandas()
)
plt.figure(figsize=(9, 5))
sns.histplot(data=sample_df, x="transaction_amount", hue="is_fraud", bins=60,
             stat="density", common_norm=False, palette=["#4C72B0", "#C44E52"], alpha=0.6)
plt.title("Transaction Amount Distribution: Fraud vs Non-Fraud (200k sample)")
plt.xlabel("Transaction Amount")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_amount_distribution.png", dpi=150)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/02_amount_distribution.png")

# -----------------------------------------------------------------
# 5. Fraud rate by merchant category
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("5. FRAUD RATE BY MERCHANT CATEGORY")
print("=" * 60)
fraud_by_merchant = (
    lazy_df.group_by("merchant_category")
    .agg([
        pl.len().alias("total"),
        pl.col("is_fraud").sum().alias("fraud_count"),
    ])
    .with_columns((pl.col("fraud_count") / pl.col("total") * 100).alias("fraud_rate_pct"))
    .sort("fraud_rate_pct", descending=True)
    .collect()
)
print(fraud_by_merchant)

plt.figure(figsize=(9, 6))
pdf2 = fraud_by_merchant.to_pandas()
sns.barplot(data=pdf2, y="merchant_category", x="fraud_rate_pct", hue="merchant_category",
            palette="Reds_r", legend=False)
plt.title("Fraud Rate by Merchant Category")
plt.xlabel("Fraud Rate (%)")
plt.ylabel("Merchant Category")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_fraud_rate_by_merchant.png", dpi=150)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/03_fraud_rate_by_merchant.png")

# -----------------------------------------------------------------
# 6. Fraud rate by cvv_match_status and three_ds_auth_result
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("6. FRAUD RATE BY CVV MATCH STATUS & 3DS AUTH RESULT")
print("=" * 60)
fraud_by_cvv = (
    lazy_df.group_by("cvv_match_status")
    .agg([pl.len().alias("total"), pl.col("is_fraud").sum().alias("fraud_count")])
    .with_columns((pl.col("fraud_count") / pl.col("total") * 100).alias("fraud_rate_pct"))
    .sort("fraud_rate_pct", descending=True)
    .collect()
)
print("By CVV match status:")
print(fraud_by_cvv)

fraud_by_3ds = (
    lazy_df.group_by("three_ds_auth_result")
    .agg([pl.len().alias("total"), pl.col("is_fraud").sum().alias("fraud_count")])
    .with_columns((pl.col("fraud_count") / pl.col("total") * 100).alias("fraud_rate_pct"))
    .sort("fraud_rate_pct", descending=True)
    .collect()
)
print("\nBy 3DS auth result:")
print(fraud_by_3ds)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(data=fraud_by_cvv.to_pandas(), x="cvv_match_status", y="fraud_rate_pct",
            hue="cvv_match_status", palette="Oranges_r", legend=False, ax=axes[0])
axes[0].set_title("Fraud Rate by CVV Match Status")
axes[0].set_ylabel("Fraud Rate (%)")
axes[0].tick_params(axis='x', rotation=30)

sns.barplot(data=fraud_by_3ds.to_pandas(), x="three_ds_auth_result", y="fraud_rate_pct",
            hue="three_ds_auth_result", palette="Purples_r", legend=False, ax=axes[1])
axes[1].set_title("Fraud Rate by 3DS Auth Result")
axes[1].set_ylabel("Fraud Rate (%)")
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_fraud_rate_cvv_3ds.png", dpi=150)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/04_fraud_rate_cvv_3ds.png")

# -----------------------------------------------------------------
# 7. Geo velocity distribution: fraud vs non-fraud
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("7. GEO VELOCITY: FRAUD VS NON-FRAUD")
print("=" * 60)
geo_stats = (
    lazy_df.group_by("is_fraud")
    .agg([
        pl.col("geo_velocity_kmh").mean().alias("mean_velocity"),
        pl.col("geo_velocity_kmh").median().alias("median_velocity"),
        pl.col("geo_velocity_kmh").max().alias("max_velocity"),
    ])
    .collect()
)
print(geo_stats)

geo_sample = (
    lazy_df.select(["geo_velocity_kmh", "is_fraud"])
    .collect()
    .sample(n=200_000, seed=42)
    .to_pandas()
)
plt.figure(figsize=(9, 5))
sns.boxplot(data=geo_sample, x="is_fraud", y="geo_velocity_kmh", hue="is_fraud",
            palette=["#4C72B0", "#C44E52"], legend=False, showfliers=False)
plt.title("Geo Velocity (km/h) by Fraud Label (200k sample, outliers hidden)")
plt.xlabel("Is Fraud")
plt.ylabel("Geo Velocity (km/h)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_geo_velocity_boxplot.png", dpi=150)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/05_geo_velocity_boxplot.png")

print("\n" + "=" * 60)
print("EDA COMPLETE. Charts saved in ./outputs/")
print("=" * 60)