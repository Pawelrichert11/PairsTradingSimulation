from datetime import datetime
import pandas as pd
import itertools
from tqdm import tqdm
import Config
from Simulation import PairTradingStrategy
from DatabaseManager import DatabaseManager
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
    
    num_tickers = min(Config.NUMBER_OF_TICKERS, len(tickers))
    print(f"Generating pairs from {num_tickers} tickers...")
    
    test_tickers = tickers[0:num_tickers] 
    
    pairs = list(itertools.combinations(test_tickers, 2))
    
    results = []
    for t1, t2 in tqdm(pairs, desc="Running Simulations"):
        try:
            strategy = PairTradingStrategy(t1, t2, data)
            metrics = strategy.run_backtest()
            results.append(metrics)
                
        except Exception as e:
            print(f"Error processing pair {t1}-{t2}: {e}") 
            continue

    if results:
        results_df = pd.DataFrame(results)

        session_metrics = {
            'timestamp': datetime.now(),
            'window_size': Config.WINDOW_SIZE,
            'std_entry': Config.STD_DEV_ENTRY,
            'total_pairs_tested': len(results_df),
            'avg_annual_return': results_df['annualized_return'].mean(),
            'max_annual_return': results_df['annualized_return'].max(),
            'avg_sharpe': results_df['sharpe_ratio'].mean(),
            'avg_num_trades': results_df['num_trades'].mean(),
            'profitable_pairs_pct': (results_df['total_return'] > 0).mean() * 100
        }
        
        desired_order = [
            'pair', 
            'ticker_1', 
            'ticker_2', 
            'total_return', 
            'annualized_return',
            'sharpe_ratio', 
            'coint_pvalue', 
            'correlation',
            'transaction_cost_used',
            'final_value',
            'cum_return'
        ]

        cols_to_use = [col for col in desired_order if col in results_df.columns]
        remaining_cols = [col for col in results_df.columns if col not in cols_to_use]
        
        results_df = results_df[cols_to_use + remaining_cols]

        Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        output_path = Config.PROCESSED_DIR / "simulation_results.parquet"
        results_df.to_parquet(output_path)
        
        print(f"Saved all results to: {output_path}")
        print(f"Total pairs analyzed: {len(results_df)}")

        db = DatabaseManager()
        session_id = db.save_simulation_session(session_metrics)
        
    else:
        print("No profitable pairs found or simulation failed.")

if __name__ == "__main__":
    run_all_simulations()