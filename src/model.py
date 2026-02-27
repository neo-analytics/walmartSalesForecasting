import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    FEATURE_COLS,
    TARGET_COL,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_RANDOM_STATE,
    RF_N_JOBS,
    MODEL_FILE,
    STATS_FILE,
    PLOTS_DIR,
    PLOT_DPI,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
)

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metric Helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, model_name: str) -> dict:

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n[model] {model_name} — Test Set Metrics:")
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  R²   : {r2:.4f}  ({r2*100:.2f}% variance explained)")
    print(f"  MAPE : {mape:.2f}%")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Baseline — Linear Regression
# ─────────────────────────────────────────────────────────────────────────────
def train_linear_regression(X_train, y_train, X_test, y_test) -> dict:
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, "Linear Regression (Baseline)")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 2. Random Forest
# ─────────────────────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train) -> RandomForestRegressor:

    print(f"\n[model] Training RandomForestRegressor ...")
    print(f"  n_estimators : {RF_N_ESTIMATORS}")
    print(f"  max_depth    : {RF_MAX_DEPTH}")
    print(f"  random_state : {RF_RANDOM_STATE}")
    print(f"  n_jobs       : {RF_N_JOBS}")
    print(f"  Training rows: {len(X_train):,}")

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=RF_N_JOBS,
    )
    rf.fit(X_train, y_train)
    print("  ✓ Training complete.")
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot — Actual vs Predicted
# ─────────────────────────────────────────────────────────────────────────────
def plot_actual_vs_predicted(test_df: pd.DataFrame, y_pred: np.ndarray) -> str:

    test_df = test_df.copy()
    test_df["Predicted"] = y_pred

    actual_by_date = test_df.groupby("Date")[TARGET_COL].sum()
    pred_by_date = test_df.groupby("Date")["Predicted"].sum()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        actual_by_date.index,
        actual_by_date.values / 1e6,
        label="Actual",
        color=COLOR_PRIMARY,
        lw=2,
    )
    ax.plot(
        pred_by_date.index,
        pred_by_date.values / 1e6,
        label="Predicted",
        color=COLOR_SECONDARY,
        lw=2,
        linestyle="--",
    )
    ax.fill_between(
        actual_by_date.index,
        actual_by_date.values / 1e6,
        pred_by_date.values / 1e6,
        alpha=0.1,
        color="gray",
        label="Error band",
    )
    ax.set_title(
        "Actual vs Predicted Total Weekly Sales (Test Period — Last 12 Weeks)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Weekly Sales (Millions $)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "08_actual_vs_predicted.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [model] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(rf: RandomForestRegressor) -> str:

    fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot.barh(ax=ax, color=COLOR_PRIMARY, edgecolor="white")
    ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score (Gini)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "09_feature_importance.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[model] Feature Importances (top 5):")
    for name, score in fi.sort_values(ascending=False).head(5).items():
        print(f"  {name:20s}: {score:.4f}")
    print(f"  Saved → {path}")

    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Save Model
# ─────────────────────────────────────────────────────────────────────────────
def save_model(rf: RandomForestRegressor, path: str = MODEL_FILE):

    with open(path, "wb") as f:
        pickle.dump(rf, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n[model] ✓ Model saved → {path}  ({size_mb:.1f} MB)")


def load_model(path: str = MODEL_FILE) -> RandomForestRegressor:

    with open(path, "rb") as f:
        rf = pickle.load(f)
    print(f"[model] Model loaded from {path}")
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# 6. Save Stats
# ─────────────────────────────────────────────────────────────────────────────
def save_stats(stats: dict, path: str = STATS_FILE):

    # Convert numpy types to native Python
    clean = {
        k: (
            float(v)
            if isinstance(v, (float, int, np.floating, np.integer))
            else str(v) if not isinstance(v, str) else v
        )
        for k, v in stats.items()
    }
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"[model] Stats saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_modeling(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    
    print("=" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 60)

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    # Baseline
    lr_metrics = train_linear_regression(X_train, y_train, X_test, y_test)

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_metrics = compute_metrics(y_test, y_pred, "Random Forest (Final Model)")

    # Plots
    plot_actual_vs_predicted(test, y_pred)
    plot_feature_importance(rf)

    # Save model
    save_model(rf)

    # Compile stats (used by report generator)
    store_sales = df.groupby("Store")[TARGET_COL].mean().sort_values(ascending=False)
    holiday_sales = df.groupby("Holiday_Flag")[TARGET_COL].mean()
    holiday_pct = (holiday_sales[1] - holiday_sales[0]) / holiday_sales[0] * 100

    all_stats = {
        **rf_metrics,
        "lr_mae": lr_metrics["mae"],
        "lr_r2": lr_metrics["r2"],
        "n_stores": int(df["Store"].nunique()),
        "date_range": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        "holiday_pct_increase": float(holiday_pct),
        "top_store": int(store_sales.index[0]),
        "bottom_store": int(store_sales.index[-1]),
        "total_rows": len(df),
    }
    save_stats(all_stats)

    print("\n[run_modeling] ✓ Modeling complete.\n")
    return rf, rf_metrics, all_stats


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import run_preprocessing

    df, train, test = run_preprocessing()
    run_modeling(df, train, test)
