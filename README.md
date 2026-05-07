# Walmart Sales Forecasting Project

> Retail inventory and demand forecasting using exploratory data analysis and Random Forest regression.

## Problem Statement

A retail chain operating multiple outlets across the country needs to match inventory supply with changing weekly demand. This project analyzes Walmart historical weekly sales data, surfaces key demand patterns, trains forecasting models, and produces 12-week sales forecasts for each of the 45 stores.

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
|   |-- model.ipynb                      # Modeling notebook, maps to src/model.py
|   |-- forecasst.ipynb                  # Forecasting notebook, maps to src/forecast.py
|   |-- main.ipynb                       # End-to-end notebook pipeline
|
|-- src/
|   |-- config.py                        # Central paths, hyperparameters, constants
|   |-- data_preprocessing.py            # Load, clean, feature engineer, train/test split
|   |-- eda.py                           # Generate all 7 EDA plots
|   |-- model.py                         # Train Linear Regression + Random Forest, evaluate, save model
|   |-- forecast.py                      # Generate 12-week forecasts for all 45 stores
|   |-- main.py                          # End-to-end Python pipeline entry point
|
|-- outputs/
|   |-- plots/                           # 10 generated PNG visualizations
|   |-- models/walmart_rf_model.pkl      # Saved trained Random Forest model
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

This runs preprocessing, EDA, modeling, and forecasting in sequence.

### 3. Run Individual Python Modules

```bash
# Step 1: Preprocessing only
python src/data_preprocessing.py

# Step 2: EDA only; preprocessing runs automatically
python src/eda.py

# Step 3: Modeling only; preprocessing runs automatically
python src/model.py

# Step 4: Forecasting only; requires a trained model file
python src/forecast.py
```

## Notebook Workflow

The notebooks were rebuilt so their cells follow the same sequence as the matching files in `src/`. They can be used for step-by-step exploration, explanation, or presentation while the `.py` files remain the production pipeline source.

Recommended notebook execution order:

1. `notebooks/config.ipynb` - Review paths, feature columns, hyperparameters, and plotting constants.
2. `notebooks/data_processing.ipynb` - Load `data/Walmart.csv`, clean dates, create lag/rolling features, and create the train/test split.
3. `notebooks/eda.ipynb` - Generate exploratory plots and inspect sales patterns.
4. `notebooks/model.ipynb` - Train the baseline Linear Regression and final Random Forest model, then save metrics and model artifacts.
5. `notebooks/forecasst.ipynb` - Generate 12-week forecasts and the top-store forecast plot.
6. `notebooks/main.ipynb` - Run the complete workflow from one notebook.

Notebook-to-source mapping:

| Notebook | Source File | Purpose |
|---|---|---|
| `notebooks/config.ipynb` | `src/config.py` | Constants, paths, model settings, and plot colors |
| `notebooks/data_processing.ipynb` | `src/data_preprocessing.py` | Data loading, cleaning, feature engineering, split |
| `notebooks/eda.ipynb` | `src/eda.py` | Exploratory visualizations and summary insights |
| `notebooks/model.ipynb` | `src/model.py` | Model training, evaluation, feature importance, artifact saving |
| `notebooks/forecasst.ipynb` | `src/forecast.py` | Iterative 12-week forecasting for every store |
| `notebooks/main.ipynb` | `src/main.py` | End-to-end pipeline runner |

The notebook path setup supports kernels started either from the project root or from the `notebooks/` directory.

## Outputs

After a full run, the project writes these main artifacts:

```text
outputs/
|-- forecasts/
|   |-- walmart_12week_forecast.csv      # 45 stores x 12 weeks = 540 forecast rows
|-- models/
|   |-- walmart_rf_model.pkl
|-- plots/
|   |-- 01_total_weekly_sales.png
|   |-- 02_sales_by_store.png
|   |-- 03_holiday_vs_nonholiday.png
|   |-- 04_monthly_sales.png
|   |-- 05_correlation_heatmap.png
|   |-- 06_fuel_vs_sales.png
|   |-- 07_yearly_sales.png
|   |-- 08_actual_vs_predicted.png
|   |-- 09_feature_importance.png
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

## Model Performance

Current saved Random Forest metrics:

| Metric | Score |
|---|---:|
| R^2 | 0.9865 |
| MAE | ~$42,080 |
| RMSE | ~$61,176 |
| MAPE | 4.14% |

## Key Insights

- Holiday weeks generate about 7.8% higher average weekly sales.
- Store 20 is the top performer; Store 33 is the lowest performer.
- November and December are the strongest sales months.
- Lag features from recent sales history are the strongest predictors.
- CPI shows the strongest positive correlation with weekly sales among the external variables used here.

## Module Descriptions

| Module | Purpose |
|---|---|
| `config.py` | Single source of truth for paths, hyperparameters, constants, and feature columns |
| `data_preprocessing.py` | Loads CSV, parses dates, sorts store time series, creates features, and splits train/test data |
| `eda.py` | Creates seven EDA plots that explain sales trends, store differences, holidays, months, and correlations |
| `model.py` | Trains Linear Regression and Random Forest models, evaluates metrics, saves model and stats |
| `forecast.py` | Produces iterative 12-week forecasts for all 45 stores and saves forecast outputs |
| `main.py` | Runs all modules in the correct production sequence |
