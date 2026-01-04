import pandas as pd
import itertools
from tqdm import tqdm
import Config
from Simulation import PairTradingStrategy
from DatabaseManager import DatabaseManager

def run_all_simulations():
    print("Loading market data...")
    
    try:
        data = pd.read_parquet(Config.PROCESSED_MARKET_DATA)
    except FileNotFoundError:
        print(f"Error: Data file not found at {Config.PROCESSED_MARKET_DATA}")
        print("Please run LoadData.py first.")
        return

    tickers = data.columns.tolist()
    
    print(f"Generating pairs from {Config.NUMBER_OF_TICKERS} tickers...")
    
    test_tickers = tickers[0:Config.NUMBER_OF_TICKERS] 
    pairs = list(itertools.combinations(test_tickers, 2))
    
    print(f"Total pairs to simulate: {len(pairs)}")
    
    results = []
    db = DatabaseManager()

    # Iterate with progress bar
    for t1, t2 in tqdm(pairs, desc="Running Simulations"):
        try:
            strategy = PairTradingStrategy(t1, t2, data)
            metrics = strategy.run_backtest()
            
            results.append(metrics)
                
        except Exception as e:
            print(f"Error processing pair {t1}-{t2}: {e}")
            continue

    # Convert results to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        results_df.sort_values(by='sharpe_ratio', ascending=False, inplace=True)
        
        print("\nTop 5 Performing Pairs:")
        print(results_df.head(5))
        
        # Save results to CSV or DB
        output_path = Config.PROCESSED_DIR / "simulation_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Saved all results to: {output_path}")
    else:
        print("No profitable pairs found.")

if __name__ == "__main__":
    run_all_simulations()