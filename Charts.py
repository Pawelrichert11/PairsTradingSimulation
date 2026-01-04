import matplotlib.pyplot as plt
import pandas as pd
import argparse
import Config
from Simulation import PairTradingStrategy

# Set visual style
plt.style.use('seaborn-v0_8-darkgrid')

def plot_analysis(strategy_df, ticker1, ticker2):
    """
    Generates a 2-row plot:
    1. Strategy Cumulative Return
    2. Asset Price Comparison
    
    Automatically filters the view to the period where both stocks have data.
    """
    
    # --- FIX: Filter data to overlapping period only ---
    # We drop rows where either ticker1 or ticker2 has NaN values.
    df = strategy_df.dropna(subset=[ticker1, ticker2]).copy()

    if df.empty:
        print("Error: No overlapping data range found for these two stocks.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- PLOT 1: Strategy Performance (Cumulative Return) ---
    # We plot the 'cum_return' calculated in Simulation.py
    # Re-normalize cum_return to start at 1.0 for the visible period for better readability
    if not df['cum_return'].empty:
        # Optional: Normalize the start of the visible chart to 1.0
        # df['cum_return'] = df['cum_return'] / df['cum_return'].iloc[0]
        pass

    ax1.plot(df.index, df['cum_return'], color='green', linewidth=2, label='Strategy Equity Curve')
    
    # Draw a baseline at 1.0 (starting value)
    ax1.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax1.set_title(f"Strategy Performance: {ticker1} - {ticker2} Pair", fontsize=14)
    ax1.set_ylabel("Cumulative Return (Growth of $1)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # --- PLOT 2: Price Comparison (Dual Axis) ---
    # Primary Y-Axis (Left) for Ticker 1
    color1 = 'tab:blue'
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel(f'Price {ticker1}', color=color1, fontsize=12)
    ax2.plot(df.index, df[ticker1], color=color1, label=ticker1)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.grid(True)

    # Secondary Y-Axis (Right) for Ticker 2
    # We use twinx() to share the same X-axis but have a different Y scale
    ax3 = ax2.twinx()  
    color2 = 'tab:orange'
    ax3.set_ylabel(f'Price {ticker2}', color=color2, fontsize=12)
    ax3.plot(df.index, df[ticker2], color=color2, label=ticker2)
    ax3.tick_params(axis='y', labelcolor=color2)
    
    # Fix the grid for the secondary axis to avoid clutter
    ax3.grid(False)

    ax2.set_title(f"Price History: {ticker1} vs {ticker2}", fontsize=14)

    # Combine legends from both axes
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax3.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.tight_layout()
    plt.show()

def main():
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Generate Strategy & Price Charts")
    parser.add_argument("tickers", nargs="*", type=str, help="Two tickers to analyze (e.g. AAPL MSFT)")
    args = parser.parse_args()
    
    print("Loading market data...")
    
    try:
        # Load data
        if not Config.PROCESSED_MARKET_DATA.exists():
            print(f"Error: File {Config.PROCESSED_MARKET_DATA} not found.")
            print("Please run LoadData.py first.")
            return
            
        market_data = pd.read_parquet(Config.PROCESSED_MARKET_DATA)
        
        # Prepare tickers
        user_tickers = [t.upper() for t in args.tickers]
        
        # We need exactly 2 tickers for this view
        if len(user_tickers) != 2:
            print("Error: You must provide exactly 2 tickers to run the strategy simulation.")
            print("Example: python Charts.py KO PEP")
            
            # Fallback for testing if no arguments provided
            if len(market_data.columns) >= 2:
                print(f"\nDefaulting to first two available tickers: {market_data.columns[0]}, {market_data.columns[1]}")
                user_tickers = [market_data.columns[0], market_data.columns[1]]
            else:
                return

        t1, t2 = user_tickers

        # Validate existence
        if t1 not in market_data.columns or t2 not in market_data.columns:
            print(f"Error: One or both tickers ({t1}, {t2}) not found in the dataset.")
            return

        print(f"Running simulation for {t1} and {t2}...")

        # Initialize and run the strategy using the Simulation class
        strategy = PairTradingStrategy(t1, t2, market_data)
        results = strategy.run_backtest()

        if results is None:
            print("Error: Not enough data to run simulation.")
            return

        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

        # Plot the results
        plot_analysis(strategy.df, t1, t2)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()