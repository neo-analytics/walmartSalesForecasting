import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from config import (
    FEATURE_COLS,
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
    store_df: pd.DataFrame,
    rf: RandomForestRegressor,
    store_id: int,
    last_date: pd.Timestamp,
    n_weeks: int = FORECAST_WEEKS,
) -> list:
    
    store_df = store_df.sort_values("Date")

    # Seed the rolling history with the last 4 known actual sales values
    history = store_df[TARGET_COL].values.tolist()

    # Last known static feature values
    last_row = store_df.iloc[-1]

    forecast_dates = [last_date + pd.Timedelta(weeks=i + 1) for i in range(n_weeks)]
    results = []

    for fd in forecast_dates:
        lag1 = history[-1]
        lag2 = history[-2]
        lag4 = history[-4]
        rolling_mean4 = float(np.mean(history[-4:]))

        row = {
            "Store": store_id,
            "Year": fd.year,
            "Month": fd.month,
            "Week": fd.isocalendar()[1],
            "Holiday_Flag": 0,  # assume non-holiday
            "Temperature": last_row["Temperature"],
            "Fuel_Price": last_row["Fuel_Price"],
            "CPI": last_row["CPI"],
            "Unemployment": last_row["Unemployment"],
            "lag1": lag1,
            "lag2": lag2,
            "lag4": lag4,
            "rolling_mean4": rolling_mean4,
        }

        pred = rf.predict(pd.DataFrame([row])[FEATURE_COLS])[0]
        pred = max(pred, 0)  # sales cannot be negative
        history.append(pred)

        results.append(
            {
                "Store": store_id,
                "Date": fd.strftime("%Y-%m-%d"),
                "Forecasted_Sales": round(pred, 2),
            }
        )

    return results


# -----------------------------------------------------------------------------
# Forecast All Stores
# -----------------------------------------------------------------------------
def forecast_all_stores(df: pd.DataFrame, rf: RandomForestRegressor) -> pd.DataFrame:
    
    last_date = df["Date"].max()
    print(f"[forecast] Last observed date : {last_date.date()}")
    print(f"[forecast] Forecast horizon   : {FORECAST_WEEKS} weeks")
    print(f"[forecast] Forecasting for    : {df['Store'].nunique()} stores ...")

    all_results = []
    for store_id in sorted(df["Store"].unique()):
        store_df = df[df["Store"] == store_id].copy()
        store_res = forecast_store(store_df, rf, store_id, last_date)
        all_results.extend(store_res)

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

    # Print summary stats
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
        # Historical (restrict to 2012 for readability)
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
            label="12-Week Forecast",
        )
        ax.axvline(
            last_date, color="gray", linestyle=":", lw=1.2, label="Forecast Start"
        )
        ax.set_title(
            f"Store {sid}  (Avg: ${store_avg[sid]/1e6:.2f}M/week)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("Sales (M$)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle(
        "12-Week Sales Forecast - Top 5 Performing Stores",
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
def run_forecasting(df: pd.DataFrame, rf: RandomForestRegressor) -> pd.DataFrame:
    
    print("=" * 60)
    print("  12-WEEK SALES FORECASTING")
    print("=" * 60)

    forecast_df = forecast_all_stores(df, rf)
    save_forecast(forecast_df)
    plot_forecast_top5(df, forecast_df)

    print("\n[run_forecasting] [OK] Forecasting complete.\n")
    return forecast_df


if __name__ == "__main__":
    import sys, pickle

    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import run_preprocessing
    from config import MODEL_FILE

    df, train, test = run_preprocessing()

    with open(MODEL_FILE, "rb") as f:
        rf = pickle.load(f)

    run_forecasting(df, rf)
