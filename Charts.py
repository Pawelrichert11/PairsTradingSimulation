import matplotlib.pyplot as plt
import pandas as pd
import argparse
import Config
from Simulation import PairTradingStrategy

#File used for development purposes
plt.style.use('seaborn-v0_8-darkgrid')

def plot_analysis(strategy_df, ticker1, ticker2):

    df = strategy_df.dropna(subset=[ticker1, ticker2]).copy()

    if df.empty:
        print("Error: No overlapping data range found for these two stocks.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(df.index, df['cum_return'], color='green', linewidth=2, label='Strategy Equity Curve')
    
    ax1.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax1.set_title(f"Strategy Performance: {ticker1} - {ticker2} Pair", fontsize=14)
    ax1.set_ylabel("Cumulative Return (Growth of $1)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True)

    color1 = 'tab:blue'
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel(f'Price {ticker1}', color=color1, fontsize=12)
    ax2.plot(df.index, df[ticker1], color=color1, label=ticker1)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.grid(True)

    ax3 = ax2.twinx()  
    color2 = 'tab:orange'
    ax3.set_ylabel(f'Price {ticker2}', color=color2, fontsize=12)
    ax3.plot(df.index, df[ticker2], color=color2, label=ticker2)
    ax3.tick_params(axis='y', labelcolor=color2)
    
    ax3.grid(False)

    ax2.set_title(f"Price History: {ticker1} vs {ticker2}", fontsize=14)

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax3.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate Strategy & Price Charts")
    parser.add_argument("tickers", nargs="*", type=str, help="Two tickers to analyze (e.g. AAPL MSFT)")
    args = parser.parse_args()
    
    print("Loading market data...")
    
    try:
        if not Config.PROCESSED_MARKET_DATA.exists():
            print(f"Error: File {Config.PROCESSED_MARKET_DATA} not found.")
            print("Please run LoadData.py first.")
            return
            
        market_data = pd.read_parquet(Config.PROCESSED_MARKET_DATA)
        
        user_tickers = [t.upper() for t in args.tickers]
        
        if len(user_tickers) != 2:
            print("Error: You must provide exactly 2 tickers to run the strategy simulation.")
            print("Example: python Charts.py KO PEP")
            
            if len(market_data.columns) >= 2:
                print(f"\nDefaulting to first two available tickers: {market_data.columns[0]}, {market_data.columns[1]}")
                user_tickers = [market_data.columns[0], market_data.columns[1]]
            else:
                return

        t1, t2 = user_tickers

        if t1 not in market_data.columns or t2 not in market_data.columns:
            print(f"Error: One or both tickers ({t1}, {t2}) not found in the dataset.")
            return

        print(f"Running simulation for {t1} and {t2}...")

        strategy = PairTradingStrategy(t1, t2, market_data)
        results = strategy.run_backtest()

        if results is None:
            print("Error: Not enough data to run simulation.")
            return

        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

        plot_analysis(strategy.df, t1, t2)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()