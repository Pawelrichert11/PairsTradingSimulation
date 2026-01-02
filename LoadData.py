import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from DatabaseManager import DatabaseManager

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_files"
STOCKS_PATH = BASE_DIR / 'data_set' / 'Stocks'
PARQUET_FILE = PROCESSED_DIR / "processed_market_data.parquet"

MIN_PRICE_EVER = 2.0
MIN_MEDIAN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 1_000_000
MIN_HISTORY_DAYS = 500

def apply_filters(df, ticker):
    clean_df = df.dropna()
    
    if len(clean_df) < MIN_HISTORY_DAYS:
        return False, None
        
    prices = clean_df[ticker].values
    volumes = clean_df['Volume'].values
    
    if np.min(prices) < MIN_PRICE_EVER:
        return False, None
        
    if np.median(prices) < MIN_MEDIAN_PRICE:
        return False, None
        
    dollar_volume = prices * volumes
    if np.median(dollar_volume) < MIN_DOLLAR_VOLUME:
        return False, None
        
    return True, clean_df[ticker]

def load_and_merge(file_list):
    all_series = []
    skipped_filters = 0
    
    if not file_list:
        print("No files found")
        return pd.DataFrame()

    PROCESSED_DIR.mkdir(exist_ok=True)

    for file_path in tqdm(file_list, desc="Processing files..."):
        ticker = file_path.stem.upper()
        
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'], usecols=['Date', 'Close', 'Volume'])
            
            if df.empty:
                continue
                
            df.rename(columns={'Close': ticker}, inplace=True)
            df.set_index('Date', inplace=True)
            
            is_valid, filtered_series = apply_filters(df, ticker)
            
            if is_valid:
                all_series.append(filtered_series)
            else:
                skipped_filters += 1
                
        except (pd.errors.EmptyDataError, ValueError, Exception) as e:
            continue

    if not all_series:
        print("No stock passed the filters")
        return pd.DataFrame()

    print(f"\n Results:")
    print(f"Accepted: {len(all_series)}")
    print(f"Rejected: (Penny Stocks/Liquidity): {skipped_filters}")
    
    combined_data = pd.concat(all_series, axis=1)
    combined_data.sort_index(inplace=True)
    combined_data.ffill(inplace=True)

    return combined_data

if __name__ == "__main__":
    db = DatabaseManager()
    
    all_files = list(STOCKS_PATH.glob("*.txt"))
    prices_df = load_and_merge(all_files)

    if not prices_df.empty:

        PROCESSED_DIR.mkdir(exist_ok=True)
        prices_df.to_parquet(PARQUET_FILE)
        print(f"Saved prices to: {PARQUET_FILE}")

        meta_data = []
        for col in prices_df.columns:
            std_dev = prices_df[col].std()
            avg_vol = 0
            
            # Klasyfikacja zmienności (High/Medium/Low)
            if std_dev > prices_df[col].mean() * 0.05: vol_class = "HIGH"
            elif std_dev > prices_df[col].mean() * 0.02: vol_class = "MEDIUM"
            else: vol_class = "LOW"
            
            meta_data.append({
                'ticker': col,
                'volatility_class': vol_class,
                'avg_volume': avg_vol
            })
            
        df_meta = pd.DataFrame(meta_data)
        
        # 3. Zapis metadanych do SQL
        db.save_tickers_metadata(df_meta)
        
    else:
        print("\n❌ Brak danych.")