import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import (
    PLOTS_DIR,
    PLOT_DPI,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_ACCENT,
    COLOR_NEUTRAL,
    COLOR_HOLIDAY,
)


os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def _save(fig, filename: str):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [EDA] Saved: {path}")
    return path


# -----------------------------------------------------------------------------
# Plot 01 - Total Weekly Sales Trend
# -----------------------------------------------------------------------------
def plot_total_weekly_sales(df: pd.DataFrame) -> str:

    weekly_agg = df.groupby("Date")["Weekly_Sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        weekly_agg["Date"],
        weekly_agg["Weekly_Sales"] / 1e6,
        color=COLOR_PRIMARY,
        lw=1.8,
        label="Total Sales",
    )
    ax.fill_between(
        weekly_agg["Date"],
        weekly_agg["Weekly_Sales"] / 1e6,
        alpha=0.15,
        color=COLOR_PRIMARY,
    )
    ax.set_title(
        "Total Weekly Sales Across All Stores (2010-2012)",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Sales (Millions $)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    print("\n[EDA] Plot 01 - Total Weekly Sales")
    print(f"  Min weekly total : ${weekly_agg['Weekly_Sales'].min()/1e6:.2f}M")
    print(f"  Max weekly total : ${weekly_agg['Weekly_Sales'].max()/1e6:.2f}M")
    print(f"  Mean weekly total: ${weekly_agg['Weekly_Sales'].mean()/1e6:.2f}M")

    return _save(fig, "01_total_weekly_sales.png")


# -----------------------------------------------------------------------------
# Plot 02 - Sales by Store
# -----------------------------------------------------------------------------
def plot_sales_by_store(df: pd.DataFrame) -> str:

    store_sales = (
        df.groupby("Store")["Weekly_Sales"].mean().sort_values(ascending=False)
    )
    top10 = store_sales.head(10).index.tolist()
    bottom10 = store_sales.tail(10).index.tolist()
    colors = [
        (
            COLOR_ACCENT
            if s in top10
            else (COLOR_SECONDARY if s in bottom10 else COLOR_PRIMARY)
        )
        for s in store_sales.index
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(
        store_sales.index.astype(str),
        store_sales.values / 1e6,
        color=colors,
        edgecolor="white",
        lw=0.5,
    )
    ax.set_title(
        "Average Weekly Sales by Store", fontsize=14, fontweight="bold", pad=10
    )
    ax.set_xlabel("Store Number", fontsize=11)
    ax.set_ylabel("Avg Weekly Sales (Millions $)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    patches = [
        mpatches.Patch(color=COLOR_ACCENT, label=f"Top 10 (Best: Store {top10[0]})"),
        mpatches.Patch(
            color=COLOR_SECONDARY, label=f"Bottom 10 (Worst: Store {bottom10[-1]})"
        ),
        mpatches.Patch(color=COLOR_PRIMARY, label="Mid-tier"),
    ]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()

    print("\n[EDA] Plot 02 - Sales by Store")
    print(
        f"  Top store    : Store {top10[0]}  -> ${store_sales.iloc[0]/1e6:.3f}M avg/week"
    )
    print(
        f"  Bottom store : Store {bottom10[-1]} -> ${store_sales.iloc[-1]/1e6:.3f}M avg/week"
    )
    print(f"  Ratio (top/bottom): {store_sales.iloc[0]/store_sales.iloc[-1]:.1f}x")

    return _save(fig, "02_sales_by_store.png")


# -----------------------------------------------------------------------------
# Plot 03 - Holiday vs Non-Holiday
# -----------------------------------------------------------------------------
def plot_holiday_impact(df: pd.DataFrame) -> str:

    holiday_sales = df.groupby("Holiday_Flag")["Weekly_Sales"].mean()
    pct_increase = (holiday_sales[1] - holiday_sales[0]) / holiday_sales[0] * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Non-Holiday\n(Flag=0)", "Holiday\n(Flag=1)"],
        holiday_sales.values / 1e6,
        color=[COLOR_PRIMARY, COLOR_HOLIDAY],
        width=0.45,
        edgecolor="white",
    )
    for bar, val in zip(bars, holiday_sales.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"${val/1e6:.3f}M",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_title(
        "Average Weekly Sales:\nHoliday vs Non-Holiday Weeks",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.set_ylabel("Avg Weekly Sales (Millions $)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(holiday_sales.values / 1e6) * 1.18)

    # Annotate % increase
    ax.annotate(
        f"+{pct_increase:.1f}%",
        xy=(1, holiday_sales[1] / 1e6),
        xytext=(0.6, (holiday_sales[0] + holiday_sales[1]) / 2e6),
        fontsize=12,
        color="green",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="green"),
    )
    plt.tight_layout()

    print("\n[EDA] Plot 03 - Holiday vs Non-Holiday")
    print(f"  Non-holiday avg : ${holiday_sales[0]/1e6:.4f}M")
    print(f"  Holiday avg     : ${holiday_sales[1]/1e6:.4f}M")
    print(f"  % Increase      : {pct_increase:.2f}%")

    return _save(fig, "03_holiday_vs_nonholiday.png")


# -----------------------------------------------------------------------------
# Plot 04 - Monthly Sales
# -----------------------------------------------------------------------------
def plot_monthly_sales(df: pd.DataFrame) -> str:

    monthly = df.groupby("Month")["Weekly_Sales"].mean()
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    bar_colors = [
        (
            COLOR_SECONDARY
            if m in [11, 12]
            else (COLOR_NEUTRAL if m == 1 else COLOR_PRIMARY)
        )
        for m in range(1, 13)
    ]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(1, 13), monthly.values / 1e6, color=bar_colors, edgecolor="white")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=10)
    ax.set_title(
        "Average Weekly Sales by Month", fontsize=13, fontweight="bold", pad=10
    )
    ax.set_ylabel("Avg Weekly Sales (Millions $)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    patches = [
        mpatches.Patch(color=COLOR_SECONDARY, label="Peak (Nov-Dec)"),
        mpatches.Patch(color=COLOR_NEUTRAL, label="Post-holiday dip (Jan)"),
        mpatches.Patch(color=COLOR_PRIMARY, label="Stable baseline"),
    ]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()

    print("\n[EDA] Plot 04 - Monthly Sales")
    for m, name in enumerate(month_names, 1):
        print(f"  {name:3s}: ${monthly.get(m, 0)/1e6:.4f}M")

    return _save(fig, "04_monthly_sales.png")


# -----------------------------------------------------------------------------
# Plot 05 - Correlation Heatmap
# -----------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    
    cols = [
        "Weekly_Sales",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Holiday_Flag",
    ]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.85)
    labels = [
        "Weekly_Sales",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Holiday_Flag",
    ]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.iloc[i, j]
            color = "black" if abs(val) < 0.6 else "white"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=color,
            )

    ax.set_title("Pearson Correlation Heatmap", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()

    print("\n[EDA] Plot 05 - Correlation Heatmap")
    print("  Correlations with Weekly_Sales:")
    for col in cols[1:]:
        print(f"    {col:20s}: {corr.loc['Weekly_Sales', col]:+.4f}")

    return _save(fig, "05_correlation_heatmap.png")


# -----------------------------------------------------------------------------
# Plot 06 - Fuel Price vs Sales
# -----------------------------------------------------------------------------
def plot_fuel_vs_sales(df: pd.DataFrame) -> str:
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        df["Fuel_Price"],
        df["Weekly_Sales"] / 1e6,
        alpha=0.08,
        color=COLOR_SECONDARY,
        s=6,
    )
    ax.set_xlabel("Fuel Price ($/gallon)", fontsize=11)
    ax.set_ylabel("Weekly Sales (Millions $)", fontsize=11)
    ax.set_title("Fuel Price vs Weekly Sales", fontsize=13, fontweight="bold", pad=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    corr = df["Fuel_Price"].corr(df["Weekly_Sales"])
    print(f"\n[EDA] Plot 06 - Fuel Price vs Sales")
    print(f"  Pearson correlation: {corr:.4f}")
    print(
        f"  Fuel price range: ${df['Fuel_Price'].min():.2f} - ${df['Fuel_Price'].max():.2f}"
    )

    return _save(fig, "06_fuel_vs_sales.png")


# -----------------------------------------------------------------------------
# Plot 07 - Yearly Sales
# -----------------------------------------------------------------------------
def plot_yearly_sales(df: pd.DataFrame) -> str:
    
    yearly = df.groupby("Year")["Weekly_Sales"].mean()
    bar_colors = [COLOR_ACCENT, COLOR_PRIMARY, COLOR_NEUTRAL][: len(yearly)]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        yearly.index.astype(str),
        yearly.values / 1e6,
        color=bar_colors,
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, yearly.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"${val/1e6:.3f}M",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Average Weekly Sales by Year", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Avg Weekly Sales (Millions $)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(yearly.values / 1e6) * 1.15)
    plt.tight_layout()

    print("\n[EDA] Plot 07 - Yearly Sales")
    for yr, val in yearly.items():
        print(f"  {yr}: ${val/1e6:.4f}M avg/week")

    return _save(fig, "07_yearly_sales.png")


# -----------------------------------------------------------------------------
# Run All EDA
# -----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> dict:
    
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    paths = {
        "total_weekly_sales": plot_total_weekly_sales(df),
        "sales_by_store": plot_sales_by_store(df),
        "holiday_impact": plot_holiday_impact(df),
        "monthly_sales": plot_monthly_sales(df),
        "correlation_heatmap": plot_correlation_heatmap(df),
        "fuel_vs_sales": plot_fuel_vs_sales(df),
        "yearly_sales": plot_yearly_sales(df),
    }

    print(f"\n[run_eda] [OK] All {len(paths)} EDA plots saved to: {PLOTS_DIR}\n")
    return paths


if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import run_preprocessing

    df, _, _ = run_preprocessing()
    run_eda(df)
