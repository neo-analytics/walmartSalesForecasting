import os
import pickle
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    TARGET_COL,
    FORECAST_WEEKS,
    FORECAST_FILE,
    PLOTS_DIR,
    PLOT_DPI,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
)

os.makedirs(os.path.dirname(FORECAST_FILE), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Core Forecasting Logic (Per Store)
# -----------------------------------------------------------------------------
def forecast_store(
    arima_result,
    store_id: int,
    last_date: pd.Timestamp,
    n_weeks: int = FORECAST_WEEKS,
) -> list:
    forecast_dates = [last_date + pd.Timedelta(weeks=i + 1) for i in range(n_weeks)]
    forecast = arima_result.forecast(steps=n_weeks)
    predictions = np.maximum(np.asarray(forecast, dtype=float), 0)

    return [
        {
            "Store": store_id,
            "Date": forecast_date.strftime("%Y-%m-%d"),
            "Forecasted_Sales": round(float(pred), 2),
        }
        for forecast_date, pred in zip(forecast_dates, predictions)
    ]


# -----------------------------------------------------------------------------
# Forecast All Stores
# -----------------------------------------------------------------------------
def forecast_all_stores(df: pd.DataFrame, arima_models: dict) -> pd.DataFrame:
    last_date = df["Date"].max()
    print(f"[forecast] Last observed date : {last_date.date()}")
    print(f"[forecast] Forecast horizon   : {FORECAST_WEEKS} weeks")
    print(f"[forecast] Forecasting for    : {df['Store'].nunique()} stores ...")

    all_results = []
    for store_id in sorted(df["Store"].unique()):
        model = arima_models.get(int(store_id))
        if model is None:
            raise KeyError(f"No ARIMA model found for Store {store_id}")
        all_results.extend(forecast_store(model, int(store_id), last_date))

    forecast_df = pd.DataFrame(all_results)
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

    print(
        f"[forecast] [OK] Generated {len(forecast_df):,} forecast rows "
        f"({df['Store'].nunique()} stores x {FORECAST_WEEKS} weeks)"
    )
    return forecast_df


# -----------------------------------------------------------------------------
# Save Forecast CSV
# -----------------------------------------------------------------------------
def save_forecast(forecast_df: pd.DataFrame, path: str = FORECAST_FILE):
    forecast_df.to_csv(path, index=False)
    print(f"[forecast] Forecast CSV saved -> {path}")

    summary = forecast_df.groupby("Store")["Forecasted_Sales"].agg(
        ["mean", "min", "max"]
    )
    print("\n[forecast] Per-store forecast summary (first 10 stores):")
    print(summary.head(10).to_string())


# -----------------------------------------------------------------------------
# Plot - Forecast for Top 5 Stores
# -----------------------------------------------------------------------------
def plot_forecast_top5(df: pd.DataFrame, forecast_df: pd.DataFrame) -> str:
    store_avg = df.groupby("Store")[TARGET_COL].mean().sort_values(ascending=False)
    top5_stores = store_avg.head(5).index.tolist()
    last_date = df["Date"].max()

    fig, axes = plt.subplots(5, 1, figsize=(13, 16))

    for ax, sid in zip(axes, top5_stores):
        hist = (
            df[(df["Store"] == sid) & (df["Date"] >= "2012-01-01")]
            .groupby("Date")[TARGET_COL]
            .sum()
        )
        fore = forecast_df[forecast_df["Store"] == sid]

        ax.plot(
            hist.index,
            hist.values / 1e6,
            color=COLOR_PRIMARY,
            lw=1.8,
            label="Historical Sales",
        )
        ax.plot(
            fore["Date"],
            fore["Forecasted_Sales"] / 1e6,
            color=COLOR_SECONDARY,
            lw=2.2,
            linestyle="--",
            marker="o",
            ms=5,
            label="12-Week ARIMA Forecast",
        )
        ax.axvline(
            last_date, color="gray", linestyle=":", lw=1.2, label="Forecast Start"
        )
        ax.set_title(
            f"Store {sid}  (Avg: ${store_avg[sid] / 1e6:.2f}M/week)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("Sales (M$)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle(
        "12-Week ARIMA Sales Forecast - Top 5 Performing Stores",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "10_forecast_top5_stores.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[forecast] Plot saved -> {path}")
    return path


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
def run_forecasting(df: pd.DataFrame, arima_models: dict) -> pd.DataFrame:
    print("=" * 60)
    print("  12-WEEK ARIMA SALES FORECASTING")
    print("=" * 60)

    forecast_df = forecast_all_stores(df, arima_models)
    save_forecast(forecast_df)
    plot_forecast_top5(df, forecast_df)

    print("\n[run_forecasting] [OK] Forecasting complete.\n")
    return forecast_df


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import run_preprocessing
    from config import MODEL_FILE

    df, train, test = run_preprocessing()

    with open(MODEL_FILE, "rb") as f:
        arima_models = pickle.load(f)

    run_forecasting(df, arima_models)
