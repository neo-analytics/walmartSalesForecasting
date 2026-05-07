import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from config import (
    TARGET_COL,
    ARIMA_ORDER,
    MODEL_FILE,
    STATS_FILE,
    PLOTS_DIR,
    PLOT_DPI,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
)

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Metric Helpers
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot else 0.0
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

    print(f"\n[model] {model_name} - Test Set Metrics:")
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  R^2  : {r2:.4f}  ({r2 * 100:.2f}% variance explained)")
    print(f"  MAPE : {mape:.2f}%")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def _store_series(data: pd.DataFrame, store_id: int) -> pd.Series:
    series = (
        data[data["Store"] == store_id]
        .sort_values("Date")
        .set_index("Date")[TARGET_COL]
        .astype(float)
    )

    freq = pd.infer_freq(series.index)
    if freq:
        series = series.asfreq(freq)
    if series.isna().any():
        series = series.interpolate(method="time").ffill().bfill()

    return series


# -----------------------------------------------------------------------------
# 1. Baseline - Naive Last Observed Value
# -----------------------------------------------------------------------------
def evaluate_naive_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    predictions = []

    for store_id in sorted(test["Store"].unique()):
        train_series = _store_series(train, store_id)
        test_series = _store_series(test, store_id)
        last_value = train_series.iloc[-1]
        predictions.extend([last_value] * len(test_series))

    return compute_metrics(test[TARGET_COL], predictions, "Naive Last-Value Baseline")


# -----------------------------------------------------------------------------
# 2. ARIMA Modeling
# -----------------------------------------------------------------------------
def fit_arima(series: pd.Series):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return ARIMA(
            series,
            order=ARIMA_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()


def train_arima_models(data: pd.DataFrame) -> dict:
    stores = sorted(data["Store"].unique())
    models = {}

    print(f"\n[model] Training ARIMA{ARIMA_ORDER} models ...")
    print(f"  Stores       : {len(stores)}")
    print(f"  Training rows: {len(data):,}")

    for store_id in stores:
        series = _store_series(data, store_id)
        models[int(store_id)] = fit_arima(series)

    print("  [OK] ARIMA training complete.")
    return models


def evaluate_arima_models(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    rows = []

    print(f"\n[model] Evaluating ARIMA{ARIMA_ORDER} on holdout period ...")
    for store_id in sorted(test["Store"].unique()):
        train_series = _store_series(train, store_id)
        test_series = _store_series(test, store_id)
        result = fit_arima(train_series)
        forecast = result.forecast(steps=len(test_series))
        y_pred = np.maximum(np.asarray(forecast, dtype=float), 0)

        for date, actual, pred in zip(test_series.index, test_series.values, y_pred):
            rows.append(
                {
                    "Store": int(store_id),
                    "Date": date,
                    "Actual": float(actual),
                    "Predicted": float(pred),
                }
            )

    pred_df = pd.DataFrame(rows)
    metrics = compute_metrics(
        pred_df["Actual"], pred_df["Predicted"], f"ARIMA{ARIMA_ORDER} (Final Model)"
    )
    return pred_df, metrics


# -----------------------------------------------------------------------------
# 3. Plot - Actual vs Predicted
# -----------------------------------------------------------------------------
def plot_actual_vs_predicted(pred_df: pd.DataFrame) -> str:
    actual_by_date = pred_df.groupby("Date")["Actual"].sum()
    pred_by_date = pred_df.groupby("Date")["Predicted"].sum()

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
        "Actual vs Predicted Total Weekly Sales (ARIMA Holdout)",
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
    print(f"  [model] Saved -> {path}")
    return path


# -----------------------------------------------------------------------------
# 4. Plot - Per-Store ARIMA Error
# -----------------------------------------------------------------------------
def plot_store_mape(pred_df: pd.DataFrame) -> str:
    scores = {}
    for store_id, group in pred_df.groupby("Store"):
        actual = group["Actual"].to_numpy(dtype=float)
        pred = group["Predicted"].to_numpy(dtype=float)
        nonzero = actual != 0
        scores[int(store_id)] = (
            np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100
        )

    mape_by_store = pd.Series(scores).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        mape_by_store.index.astype(str),
        mape_by_store.values,
        color=COLOR_PRIMARY,
        edgecolor="white",
    )
    ax.set_title("ARIMA Holdout MAPE by Store", fontsize=13, fontweight="bold")
    ax.set_xlabel("Store")
    ax.set_ylabel("MAPE (%)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "09_arima_store_mape.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[model] Best ARIMA store MAPE scores:")
    for store_id, score in mape_by_store.head(5).items():
        print(f"  Store {store_id:2d}: {score:.2f}%")
    print(f"  Saved -> {path}")

    return path


# -----------------------------------------------------------------------------
# 5. Save / Load Model
# -----------------------------------------------------------------------------
def save_model(models: dict, path: str = MODEL_FILE):
    with open(path, "wb") as f:
        pickle.dump(models, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n[model] [OK] ARIMA models saved -> {path}  ({size_mb:.1f} MB)")


def load_model(path: str = MODEL_FILE) -> dict:
    with open(path, "rb") as f:
        models = pickle.load(f)
    print(f"[model] ARIMA models loaded from {path}")
    return models


# -----------------------------------------------------------------------------
# 6. Save Stats
# -----------------------------------------------------------------------------
def save_stats(stats: dict, path: str = STATS_FILE):
    clean = {
        k: (
            float(v)
            if isinstance(v, (float, int, np.floating, np.integer))
            else list(v)
            if isinstance(v, tuple)
            else str(v)
            if not isinstance(v, str)
            else v
        )
        for k, v in stats.items()
    }
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"[model] Stats saved -> {path}")


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
def run_modeling(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    print("=" * 60)
    print("  ARIMA MODEL TRAINING & EVALUATION")
    print("=" * 60)

    baseline_metrics = evaluate_naive_baseline(train, test)
    pred_df, arima_metrics = evaluate_arima_models(train, test)

    plot_actual_vs_predicted(pred_df)
    plot_store_mape(pred_df)

    final_models = train_arima_models(df)
    save_model(final_models)

    store_sales = df.groupby("Store")[TARGET_COL].mean().sort_values(ascending=False)
    holiday_sales = df.groupby("Holiday_Flag")[TARGET_COL].mean()
    holiday_pct = (holiday_sales[1] - holiday_sales[0]) / holiday_sales[0] * 100

    all_stats = {
        **arima_metrics,
        "baseline_mae": baseline_metrics["mae"],
        "baseline_r2": baseline_metrics["r2"],
        "arima_order": ARIMA_ORDER,
        "n_stores": int(df["Store"].nunique()),
        "date_range": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        "holiday_pct_increase": float(holiday_pct),
        "top_store": int(store_sales.index[0]),
        "bottom_store": int(store_sales.index[-1]),
        "total_rows": len(df),
    }
    save_stats(all_stats)

    print("\n[run_modeling] [OK] Modeling complete.\n")
    return final_models, arima_metrics, all_stats


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import run_preprocessing

    df, train, test = run_preprocessing()
    run_modeling(df, train, test)
