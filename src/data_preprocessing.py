import pandas as pd
import numpy as np
import json
from config import RAW_DATA_FILE, FEATURE_COLS, TARGET_COL, LAG_PERIODS, ROLLING_WINDOW


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Raw Data
# ─────────────────────────────────────────────────────────────────────────────
def load_raw_data(filepath: str = RAW_DATA_FILE) -> pd.DataFrame:
    
    df = pd.read_csv(filepath)
    print(f"[load_raw_data] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Dtypes  :\n{df.dtypes.to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # Sort
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # Null report
    null_counts = df.isnull().sum()
    print(f"\n[clean_data] Null counts per column:\n{null_counts.to_string()}")
    print(f"  Date range : {df['Date'].min().date()}  →  {df['Date'].max().date()}")
    print(
        f"  Stores     : {df['Store'].nunique()} unique stores ({df['Store'].min()}–{df['Store'].max()})"
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Descriptive Statistics
# ─────────────────────────────────────────────────────────────────────────────
def describe_data(df: pd.DataFrame) -> pd.DataFrame:

    desc = df.describe()
    print(f"\n[describe_data] Descriptive Statistics:\n{desc.to_string()}")
    return desc


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    # Temporal
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    # Lag features (per store to avoid cross-store leakage)
    for lag in LAG_PERIODS:
        col = f"lag{lag}"
        df[col] = df.groupby("Store")[TARGET_COL].shift(lag)

    # Rolling mean (shifted so it doesn't leak current week)
    df["rolling_mean4"] = df.groupby("Store")[TARGET_COL].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean()
    )

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_after = len(df)
    print(f"\n[engineer_features] Lag/rolling features added.")
    print(f"  Rows before dropna : {n_before:,}")
    print(
        f"  Rows after  dropna : {n_after:,}  (dropped {n_before - n_after} NaN rows)"
    )
    print(f"  Final features     : {FEATURE_COLS}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────
def train_test_split(df: pd.DataFrame, test_weeks: int = 12):
    
    split_date = df["Date"].max() - pd.Timedelta(weeks=test_weeks)

    train = df[df["Date"] <= split_date].copy()
    test = df[df["Date"] > split_date].copy()

    print(f"\n[train_test_split] Split at: {split_date.date()}")
    print(
        f"  Train : {len(train):,} rows  ({train['Date'].min().date()} → {train['Date'].max().date()})"
    )
    print(
        f"  Test  : {len(test):,}  rows  ({test['Date'].min().date()} → {test['Date'].max().date()})"
    )

    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_preprocessing(filepath: str = RAW_DATA_FILE):
    
    print("=" * 60)
    print("  WALMART SALES — DATA PREPROCESSING")
    print("=" * 60)

    df_raw = load_raw_data(filepath)
    df_clean = clean_data(df_raw)
    df_described = describe_data(df_clean)
    df_featured = engineer_features(df_clean)
    train, test = train_test_split(df_featured)

    print("\n PREPROCESSING COMPLETE.\n")
    return df_featured, train, test


if __name__ == "__main__":
    df, train, test = run_preprocessing()
    print(df.head())
