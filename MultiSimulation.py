import pandas as pd
# import itertools
from tqdm import tqdm
import Config
from Simulation import PairTradingStrategy
# from DatabaseManager import DatabaseManager
import os

def run_all_simulations():
    print("Loading market data...")
    
    try:
        data = pd.read_parquet(Config.PROCESSED_MARKET_DATA)
    except FileNotFoundError:
        print(f"Error: Data file not found at {Config.PROCESSED_MARKET_DATA}")
        print("Please run LoadData.py first.")
        return

    tickers = data.columns.tolist()
    
    # Ensure we do not take more tickers than available
    num_tickers = min(Config.NUMBER_OF_TICKERS, len(tickers))
    print(f"Generating pairs from {num_tickers} tickers...")
    
    test_tickers = tickers[0:num_tickers] 
    
    # Generate pairs based on distance (lag)
    pairs = []
    n = len(test_tickers)
    
    # Logic: k is distance, i is starting index
    for k in range(1, n):
        for i in range(n - k):
            pair = (test_tickers[i], test_tickers[i + k])
            pairs.append(pair)
            
    print(f"Total pairs to simulate: {len(pairs)}")
    
    results = []
    # db = DatabaseManager() 

    # Iterate with progress bar
    for t1, t2 in tqdm(pairs, desc="Running Simulations"):
        try:
            strategy = PairTradingStrategy(t1, t2, data)
            metrics = strategy.run_backtest()
            results.append(metrics)
                
        except Exception as e:
            # print(f"Error processing pair {t1}-{t2}: {e}") 
            continue

    if results:
        results_df = pd.DataFrame(results)
        
        # --- COLUMN ORDERING ---
        # Added 'annualized_return' to the priority list
        desired_order = [
            'pair', 'ticker_1', 'ticker_2', 
            'total_return', 'annualized_return',  # <--- NEW COLUMN HERE
            'sharpe_ratio', 'coint_pvalue', 
            'final_value'
        ]
        
        cols_to_use = [col for col in desired_order if col in results_df.columns]
        remaining_cols = [col for col in results_df.columns if col not in cols_to_use]
        
        results_df = results_df[cols_to_use + remaining_cols]
        # ----------------------------

        # Sort by Sharpe Ratio (or you can change to 'annualized_return')
        results_df.sort_values(by='sharpe_ratio', ascending=False, inplace=True)
        
        print("\nTop 5 Performing Pairs:")
        
        # Update preview columns to show the new metric
        cols_preview = ['pair', 'total_return', 'annualized_return', 'sharpe_ratio', 'coint_pvalue']
        cols_preview = [c for c in cols_preview if c in results_df.columns]
        
        print(results_df[cols_preview].head(5))
        
        Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        output_path = Config.PROCESSED_DIR / "simulation_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Saved all results to: {output_path}")
        
    else:
        print("No profitable pairs found or simulation failed.")

if __name__ == "__main__":
    run_all_simulations()