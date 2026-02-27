import sys
import os
import time

# Ensure src/ is on Python path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data_preprocessing import run_preprocessing
from eda import run_eda
from model import run_modeling
from forecast import run_forecasting


def main():
    overall_start = time.time()

    print("\n" + "=" * 60)
    print("  WALMART SALES FORECASTING PIPELINE")
    print("=" * 60)

    # ── Preprocessing ─────────────────────────────────────────────
    t0 = time.time()
    df, train, test = run_preprocessing()
    print(f"  Preprocessing : {time.time() - t0:.1f}s")

    # ── EDA ───────────────────────────────────────────────────────
    t0 = time.time()
    eda = run_eda(df)
    print(f"  EDA           : {time.time() - t0:.1f}s  ({len(eda)} plots)")

    # ── Modeling ──────────────────────────────────────────────────
    t0 = time.time()
    rf, rf_metrics, all_stats = run_modeling(df, train, test)
    print(f"  Modeling      : {time.time() - t0:.1f}s")

    # ── Forecasting ───────────────────────────────────────────────
    t0 = time.time()
    forecast_df = run_forecasting(df, rf)
    print(
        f"  Forecasting   : {time.time() - t0:.1f}s  ({len(forecast_df):,} forecast rows)"
    )

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time     : {elapsed:.1f}s")
    print(f"  R² Score       : {rf_metrics['r2']:.4f}")
    print(f"  MAPE           : {rf_metrics['mape']:.2f}%")
    print(f"  MAE            : ${rf_metrics['mae']:,.0f}")
    print(f"  Forecast rows  : {len(forecast_df):,} (45 stores × 12 weeks)")
    print()
    print("Output files:")
    print("   outputs/plots/                  — 10 PNG visualisations")
    print("   outputs/models/walmart_rf_model.pkl")
    print("   outputs/forecasts/walmart_12week_forecast.csv")
    print("   outputs/model_stats.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
