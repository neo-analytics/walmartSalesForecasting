# Walmart Sales Forecasting Project

> **Retail Inventory & Demand Forecasting using Exploratory Data Analysis and Random Forest Regression**

---

## Problem Statement

A retail chain operating multiple outlets across the country faces challenges in managing inventory — matching supply with fluctuating consumer demand. This project analyzes Walmart's historical weekly sales data to surface actionable insights and forecast sales for each of the 45 stores over the next 12 weeks.

---

## Project Structure

```
walmart_project/
│
├── data/
│   └── Walmart.csv                     ← Raw dataset (6,435 rows × 8 columns)
│
├── src/
│   ├── config.py                       ← Central config: paths, hyperparameters, constants
│   ├── data_preprocessing.py           ← Load, clean, feature engineer, train/test split
│   ├── eda.py                          ← Generate all 7 EDA plots
│   ├── model.py                        ← Train LR + RF, evaluate, plot, save model
│   ├── forecast.py                     ← Generate 12-week forecasts for all 45 stores
│   ├── generate_report.py              ← Compile everything into a PDF report
│   └── main.py                         ← ⭐ Master pipeline entry point (run this)
│
├── outputs/
│   ├── plots/
│   │   ├── 01_total_weekly_sales.png
│   │   ├── 02_sales_by_store.png
│   │   ├── 03_holiday_vs_nonholiday.png
│   │   ├── 04_monthly_sales.png
│   │   ├── 05_correlation_heatmap.png
│   │   ├── 06_fuel_vs_sales.png
│   │   ├── 07_yearly_sales.png
│   │   ├── 08_actual_vs_predicted.png
│   │   ├── 09_feature_importance.png
│   │   └── 10_forecast_top5_stores.png
│   ├── models/
│   │   └── walmart_rf_model.pkl        ← Saved trained Random Forest model
│   ├── forecasts/
│   │   └── walmart_12week_forecast.csv ← 45 stores × 12 weeks = 540 forecast rows
│   ├── reports/
│   │   └── walmart_sales_report.pdf    ← Complete 12-section project report
│   └── model_stats.json                ← MAE, RMSE, R², MAPE, store stats
│
└── README.md
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt 
```

### 2. Run the Full Pipeline

```bash
cd src
python main.py
```

This runs all 4 steps in sequence and produces all outputs automatically.

OUTPUT: After completion, the final output is:
    ```
    ├── forecasts
    │   └── walmart_12week_forecast.csv
    ├── models
    │   └── walmart_rf_model.pkl
    ├── plots
    │   ├── 01_total_weekly_sales.png
    │   ├── 02_sales_by_store.png
    │   ├── 03_holiday_vs_nonholiday.png
    │   ├── 04_monthly_sales.png
    │   ├── 05_correlation_heatmap.png
    │   ├── 06_fuel_vs_sales.png
    │   ├── 07_yearly_sales.png
    │   ├── 08_actual_vs_predicted.png
    │   ├── 09_feature_importance.png
    │   └── 10_forecast_top5_stores.png
    └── model_stats.json
    ```

### 3. Run Individual Modules

```bash
# If you want to run individually:
# Step 1: Preprocessing only
python src/data_preprocessing.py

# Step 2: EDA only (preprocessing runs automatically)
python src/eda.py

# Step 3: Modeling only
python src/model.py

# Step 4: Forecasting only (requires trained model)
python src/forecast.py
```

---

## Dataset

| Feature        | Type    | Description                              |
|----------------|---------|------------------------------------------|
| Store          | Integer | Store identifier (1–45)                   |
| Date           | Date    | Week start date                          |
| Weekly_Sales   | Float   | Total weekly sales ($)                   |
| Holiday_Flag   | Binary  | 1 = Holiday week, 0 = Non-holiday week   |
| Temperature    | Float   | Temperature on sale day (°F)             |
| Fuel_Price     | Float   | Regional fuel cost ($/gallon)            |
| CPI            | Float   | Consumer Price Index                     |
| Unemployment   | Float   | Regional unemployment rate (%)           |

---

## Model Performance (Random Forest)

| Metric | Score       |
|--------|-------------|
| R²     | 0.9865      |
| MAE    | ~$42,080    |
| RMSE   | ~$61,176    |
| MAPE   | 4.14%       |

---

## Key Insights

- **Holiday weeks** generate ~7.8% higher average weekly sales
- **Store 20** is the top performer; **Store 33** is the lowest
- **Q4 (Nov–Dec)** is the peak revenue period across all stores
- **Lag features** (recent sales history) are the strongest predictors
- **CPI** shows the strongest positive correlation with weekly sales among external variables

---

## Module Descriptions

| Module                  | Purpose                                                                 |
|-------------------------|----------------------------------------------------------------------   |
| `config.py`              | Single source of truth for all paths, hyperparameters, constants        |
| `data_preprocessing.py` | Load CSV → parse dates → sort → extract features → lag/rolling → split  |
| `eda.py`                | 7 EDA plots proving insights cited in the report                        |
| `model.py`              | Train LR (baseline) + RF (final), evaluate, feature importance, save     |
| `forecast.py`           | Iterative 12-week forecast for all 45 stores, saves CSV + plot          |
| `main.py`               | Runs all modules in order, prints summary                               |