"""
Shared feature loading for the CreditTransAct fraud dataset. Used by
train_model.py, ablation.py, explain.py, and score_new_transactions.py so
every script trains against / explains / scores an identical, reproducible
feature set and train/test split.
"""
import polars as pl
from sklearn.model_selection import train_test_split

DATA_PATH = "data/credit_card_fraud.parquet"
TARGET = "is_fraud"

# Identifiers: no predictive value, high cardinality, would just let the
# model memorize rows.
ID_COLS = ["customer_id", "transaction_id"]

# These *_bin columns are pre-binned versions of continuous features already
# present in the dataset (e.g. transaction_amount_bin derives from
# transaction_amount). A tree model learns better split points from the raw
# continuous value than from a coarse bucket, so we keep the continuous
# column and drop its bin.
REDUNDANT_BIN_COLS = [
    "transaction_amount_bin",
    "geo_velocity_bin",
    "geo_distance_bin",
    "session_duration_bin",
]

DROP_COLS = ID_COLS + REDUNDANT_BIN_COLS

# Train/test split parameters. Anything that needs to reproduce the exact
# split train_model.py used (e.g. explain.py, to find test rows the model
# never trained on) must go through split_dataset() below rather than
# calling train_test_split() directly -- otherwise the two can silently
# drift out of sync.
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(data_path: str = DATA_PATH, keep_transaction_id: bool = False):
    """Load the parquet dataset, drop identifier/redundant columns, and cast
    categorical columns to pandas 'category' dtype for XGBoost's native
    categorical support. Returns (df, cat_cols, num_cols).

    Set keep_transaction_id=True to retain transaction_id in the returned
    dataframe (excluded from cat_cols/num_cols, so it's never used as a
    model feature) -- for scripts that need to report per-transaction
    results, like score_new_transactions.py."""
    drop_cols = [c for c in DROP_COLS if c != "transaction_id"] if keep_transaction_id else DROP_COLS

    lazy_df = pl.scan_parquet(data_path).drop(drop_cols)
    schema = lazy_df.collect_schema()
    exclude = {TARGET, "transaction_id"} if keep_transaction_id else {TARGET}
    cat_cols = [name for name, dtype in schema.items() if dtype == pl.String and name not in exclude]
    num_cols = [name for name in schema.names() if name not in cat_cols and name not in exclude]

    df = lazy_df.collect().to_pandas()
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df, cat_cols, num_cols


def split_dataset(X, y):
    """The one train/test split every script shares. Centralizing this
    (rather than each script calling train_test_split() with matching
    arguments by convention) makes it structurally impossible for e.g.
    explain.py's reconstructed split to silently drift from train_model.py's
    -- which would otherwise mean explaining rows that were actually in the
    training set."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
