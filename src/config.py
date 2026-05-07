import os

# -- Directory Paths ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
FORECASTS_DIR = os.path.join(OUTPUTS_DIR, "forecasts")

# -- File Names ---------------------------------------------------------------
RAW_DATA_FILE = os.path.join(DATA_DIR, "Walmart.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "walmart_arima_models.pkl")
FORECAST_FILE = os.path.join(FORECASTS_DIR, "walmart_12week_forecast.csv")
STATS_FILE = os.path.join(OUTPUTS_DIR, "model_stats.json")

# -- Model Hyperparameters ----------------------------------------------------
ARIMA_ORDER = (1, 1, 1)

# -- Data Columns -------------------------------------------------------------
TARGET_COL = "Weekly_Sales"

# -- Forecast Settings --------------------------------------------------------
FORECAST_WEEKS = 12  

# -- Plot Styling -------------------------------------------------------------
PLOT_DPI = 120
COLOR_PRIMARY = "#3498db"
COLOR_SECONDARY = "#e74c3c"
COLOR_ACCENT = "#2ecc71"
COLOR_NEUTRAL = "#9b59b6"
COLOR_HOLIDAY = "#e67e22"
