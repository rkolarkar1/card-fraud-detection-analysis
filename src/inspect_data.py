import polars as pl

path = "data/credit_card_fraud.parquet"

# Lazy scan - doesn't load everything into memory
lazy_df = pl.scan_parquet(path)

print("Schema:")
print(lazy_df.collect_schema())

print("\nRow count:")
print(lazy_df.select(pl.len()).collect())

print("\nSample rows:")
print(lazy_df.limit(5).collect())

print(lazy_df.collect_schema().names())