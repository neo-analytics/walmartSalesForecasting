# Walmart Sales Forecasting Project

> Retail inventory and demand forecasting using exploratory data analysis and per-store ARIMA time-series models.

## Problem Statement

A retail chain operating multiple outlets across the country needs to match inventory supply with changing weekly demand. This project analyzes Walmart historical weekly sales data, surfaces key demand patterns, trains ARIMA forecasting models, and produces 12-week sales forecasts for each of the 45 stores.

## Project Structure

```text
walmartSalesForecasting/
|
|-- data/
|   |-- Walmart.csv                      # Raw dataset: 6,435 rows x 8 columns
|
|-- notebooks/
|   |-- config.ipynb                     # Notebook view of project constants
|   |-- data_processing.ipynb            # Preprocessing notebook, maps to src/data_preprocessing.py
|   |-- eda.ipynb                        # Exploratory analysis notebook, maps to src/eda.py
|   |-- model.ipynb                      # ARIMA modeling notebook, maps to src/model.py
|   |-- forecasst.ipynb                  # Forecasting notebook, maps to src/forecast.py
|   |-- main.ipynb                       # End-to-end notebook pipeline
|
|-- src/
|   |-- config.py                        # Central paths, ARIMA settings, constants
|   |-- data_preprocessing.py            # Load, clean, add time fields, train/test split
|   |-- eda.py                           # Generate all 7 EDA plots
|   |-- model.py                         # Train/evaluate per-store ARIMA models and save artifacts
|   |-- forecast.py                      # Generate 12-week ARIMA forecasts for all 45 stores
|   |-- main.py                          # End-to-end Python pipeline entry point
|
|-- outputs/
|   |-- plots/                           # Generated PNG visualizations
|   |-- models/walmart_arima_models.pkl  # Saved per-store ARIMA models
|   |-- forecasts/walmart_12week_forecast.csv
|   |-- model_stats.json                 # MAE, RMSE, R^2, MAPE, store stats
|
|-- report/
|   |-- walmart_sales_report.pdf
|
|-- requirements.txt
|-- README.md
```

Note: `forecasst.ipynb` keeps the current filename in the repository, but it corresponds to `src/forecast.py`.

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Python Pipeline

```bash
python src/main.py
```

This runs preprocessing, EDA, ARIMA modeling, and forecasting in sequence.

### 3. Run Individual Python Modules

```bash
# Step 1: Preprocessing only
python src/data_preprocessing.py

# Step 2: EDA only; preprocessing runs automatically
python src/eda.py

# Step 3: ARIMA modeling only; preprocessing runs automatically
python src/model.py

# Step 4: Forecasting only; requires a trained ARIMA model file
python src/forecast.py
```

## Notebook Workflow

The notebooks are intended for step-by-step exploration and presentation, while the `.py` files in `src/` are the production pipeline source.

Recommended notebook execution order:

1. `notebooks/config.ipynb` - Review paths, ARIMA order, forecast horizon, and plotting constants.
2. `notebooks/data_processing.ipynb` - Load `data/Walmart.csv`, clean dates, add time fields, and create the train/test split.
3. `notebooks/eda.ipynb` - Generate exploratory plots and inspect sales patterns.
4. `notebooks/model.ipynb` - Train the naive baseline and final per-store ARIMA models, then save metrics and model artifacts.
5. `notebooks/forecasst.ipynb` - Generate 12-week ARIMA forecasts and the top-store forecast plot.
6. `notebooks/main.ipynb` - Run the complete workflow from one notebook.

Notebook-to-source mapping:

| Notebook | Source File | Purpose |
|---|---|---|
| `notebooks/config.ipynb` | `src/config.py` | Constants, paths, ARIMA settings, and plot colors |
| `notebooks/data_processing.ipynb` | `src/data_preprocessing.py` | Data loading, cleaning, time fields, split |
| `notebooks/eda.ipynb` | `src/eda.py` | Exploratory visualizations and summary insights |
| `notebooks/model.ipynb` | `src/model.py` | ARIMA model training, evaluation, artifact saving |
| `notebooks/forecasst.ipynb` | `src/forecast.py` | 12-week ARIMA forecasting for every store |
| `notebooks/main.ipynb` | `src/main.py` | End-to-end pipeline runner |

## Outputs

After a full run, the project writes these main artifacts:

```text
outputs/
|-- forecasts/
|   |-- walmart_12week_forecast.csv      # 45 stores x 12 weeks = 540 forecast rows
|-- models/
|   |-- walmart_arima_models.pkl
|-- plots/
|   |-- 01_total_weekly_sales.png
|   |-- 02_sales_by_store.png
|   |-- 03_holiday_vs_nonholiday.png
|   |-- 04_monthly_sales.png
|   |-- 05_correlation_heatmap.png
|   |-- 06_fuel_vs_sales.png
|   |-- 07_yearly_sales.png
|   |-- 08_actual_vs_predicted.png
|   |-- 09_arima_store_mape.png
|   |-- 10_forecast_top5_stores.png
|-- model_stats.json
```

## Dataset

| Feature | Type | Description |
|---|---|---|
| `Store` | Integer | Store identifier, 1-45 |
| `Date` | Date | Weekly date value |
| `Weekly_Sales` | Float | Total weekly sales in dollars |
| `Holiday_Flag` | Binary | 1 for holiday week, 0 for non-holiday week |
| `Temperature` | Float | Temperature on the sale day in degrees Fahrenheit |
| `Fuel_Price` | Float | Regional fuel cost in dollars per gallon |
| `CPI` | Float | Consumer Price Index |
| `Unemployment` | Float | Regional unemployment rate percentage |

## Modeling Approach

The previous Random Forest regression workflow has been replaced with a time-series workflow:

- A separate ARIMA model is fitted for each store's weekly sales series.
- The default ARIMA order is configured in `src/config.py` as `(1, 1, 1)`.
- The final 12 weeks are held out for evaluation.
- A naive last-observed-value forecast is used as the baseline.
- Final ARIMA models are refitted on the full history before future forecasting.

## Model Performance

Metrics are written to `outputs/model_stats.json` after running `python src/main.py`. The key metrics are:

| Metric | Meaning |
|---|---|
| `mae` | Mean Absolute Error for ARIMA holdout predictions |
| `rmse` | Root Mean Squared Error for ARIMA holdout predictions |
| `r2` | R-squared for ARIMA holdout predictions |
| `mape` | Mean Absolute Percentage Error for ARIMA holdout predictions |
| `baseline_mae` | MAE for the naive last-value baseline |
| `baseline_r2` | R-squared for the naive last-value baseline |

## Key Insights

- Holiday weeks generate higher average weekly sales than non-holiday weeks.
- Store 20 is typically the top performer; Store 33 is typically the lowest performer.
- November and December are the strongest sales months.
- Store-level weekly sales patterns are better represented as time series when using ARIMA.
- External variables such as CPI, fuel price, temperature, and unemployment remain useful for EDA, but the ARIMA forecast is based on each store's historical sales series.

## Module Descriptions

| Module | Purpose |
|---|---|
| `config.py` | Single source of truth for paths, ARIMA order, forecast horizon, constants, and plot colors |
| `data_preprocessing.py` | Loads CSV, parses dates, sorts store time series, adds time fields, and splits train/test data |
| `eda.py` | Creates seven EDA plots that explain sales trends, store differences, holidays, months, and correlations |
| `model.py` | Trains the naive baseline and per-store ARIMA models, evaluates metrics, saves models and stats |
| `forecast.py` | Produces 12-week ARIMA forecasts for all 45 stores and saves forecast outputs |
| `main.py` | Runs all modules in the correct production sequence |
