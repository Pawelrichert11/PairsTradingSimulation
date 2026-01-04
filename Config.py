import os
from pathlib import Path

#PATHS
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / 'data_set'
PROCESSED_DIR = BASE_DIR / 'processed_files'

DATA_SET_PATH = BASE_DIR / 'data_set'
STOCKS_PATH = DATA_SET_PATH / 'Stocks'

DB_PATH = BASE_DIR / 'trading_metadata.db'
LOGS_DIR = BASE_DIR / 'others'

PROCESSED_MARKET_DATA = PROCESSED_DIR / "processed_market_data.parquet"

# FILTERS LoadData
MIN_PRICE_EVER = 2.0
MIN_MEDIAN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 1_000_000
MIN_HISTORY_DAYS = 500

# SIMULATION
WINDOW_SIZE = 20        # Moving average window size
STD_DEV_ENTRY = 2.0     # Entry standard deviation threshold
STD_DEV_EXIT = 0.5      # Exit standard deviation threshold
STOP_LOSS_PCT = 0.05    # Stop loss percentage (5%)

NUMBER_OF_TICKERS = 200

# DATABASE CONNECTION
DB_CONNECTION_STRING = f"sqlite:///{DB_PATH}"

# DEBUGGING
DEBUG_MODE = True