import pandas as pd
import numpy as np
import Config
from pathlib import Path
from tqdm import tqdm
from DatabaseManager import DatabaseManager

def apply_filters(df, ticker):
    clean_df = df.dropna()
    
    if len(clean_df) < Config.MIN_HISTORY_DAYS:
        return False, None
        
    prices = clean_df[ticker].values
    volumes = clean_df['Volume'].values
    
    if np.min(prices) < Config.MIN_PRICE_EVER:
        return False, None
        
    if np.median(prices) < Config.MIN_MEDIAN_PRICE:
        return False, None
        
    dollar_volume = prices * volumes
    if np.median(dollar_volume) < Config.MIN_DOLLAR_VOLUME:
        return False, None
        
    return True, clean_df[ticker]

def load_and_merge(file_list):
    all_series = []
    skipped_filters = 0
    
    if not file_list:
        print("No files found")
        return pd.DataFrame()

    Config.PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

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
                
        except (pd.errors.EmptyDataError, ValueError, Exception):
            continue

    if not all_series:
        print("No stock passed the filters")
        return pd.DataFrame()

    print(f"\nResults:")
    print(f"Accepted: {len(all_series)}")
    print(f"Rejected: {skipped_filters}")
    
    combined_data = pd.concat(all_series, axis=1)
    combined_data.sort_index(inplace=True)
    combined_data.ffill(inplace=True)

    return combined_data

if __name__ == "__main__":
    db = DatabaseManager()
    
    stock_path = Path(Config.STOCKS_PATH)
    all_files = list(stock_path.glob("*.txt"))
    
    prices_df = load_and_merge(all_files)

    if not prices_df.empty:
        Config.PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
        prices_df.to_parquet(Config.PROCESSED_MARKET_DATA)
        print(f"Saved prices to: {Config.PROCESSED_MARKET_DATA}")

        meta_data = []
        for col in prices_df.columns:
            std_dev = prices_df[col].std()
            avg_vol = 0
            mean_price = prices_df[col].mean()
            
            if std_dev > mean_price * 0.05:
                vol_class = "HIGH"
            elif std_dev > mean_price * 0.02:
                vol_class = "MEDIUM"
            else:
                vol_class = "LOW"
            
            meta_data.append({
                'ticker': col,
                'volatility_class': vol_class,
                'avg_volume': avg_vol
            })
            
        df_meta = pd.DataFrame(meta_data)
        db.save_tickers_metadata(df_meta)
        
    else:
        print("\nNo data.")