import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import Config

class PairTradingStrategy:
    def __init__(self, ticker1, ticker2, data):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        
        # Create a copy to avoid SettingWithCopy warnings on the original dataset
        self.df = data[[ticker1, ticker2]].copy()
        
        self.window_size = Config.WINDOW_SIZE
        self.std_dev_entry = Config.STD_DEV_ENTRY
        self.std_dev_exit = Config.STD_DEV_EXIT
        self.stop_loss = Config.STOP_LOSS_PCT
        
        self.results = {}

    def check_cointegration(self):
        """
        Calculates the p-value of the cointegration test (Engle-Granger) for the entire history.
        """
        try:
            # Remove NaNs to prevent test failure
            clean_data = self.df.dropna()
            
            # Require minimum data (e.g., 30 days) for the test to be valid
            if len(clean_data) < 30:
                return 1.0
                
            s1 = clean_data[self.ticker1]
            s2 = clean_data[self.ticker2]
            
            # Engle-Granger Test. Returns: (t-stat, p-value, crit_values)
            # We take index [1] which is the p-value
            score, p_value, _ = ts.coint(s1, s2)
            return p_value
            
        except Exception as e:
            # Return 1.0 (no cointegration) in case of error
            return 1.0

    def calculate_signals(self):
        # Calculate price ratio
        self.df['ratio'] = self.df[self.ticker1] / self.df[self.ticker2]
        
        # Rolling Mean and Std Dev
        self.df['mean'] = self.df['ratio'].rolling(window=self.window_size).mean()
        self.df['std'] = self.df['ratio'].rolling(window=self.window_size).std()
        
        # Calculate Z-Score
        self.df['z_score'] = (self.df['ratio'] - self.df['mean']) / self.df['std']
        
        # Initialize signal column
        self.df['signal'] = 0 
        
        # Generate Signals
        # Short the spread
        self.df.loc[self.df['z_score'] < -self.std_dev_entry, 'signal'] = 1
        # Long the spread
        self.df.loc[self.df['z_score'] > self.std_dev_entry, 'signal'] = -1
        # Exit positions
        self.df.loc[abs(self.df['z_score']) < self.std_dev_exit, 'signal'] = 0
        
        # Fill signals forward (hold position until exit signal)
        self.df['signal'] = self.df['signal'].replace(0, np.nan).ffill().fillna(0)
        
    def run_backtest(self):
        # 1. Calculate Cointegration (Statistics)
        coint_p_value = self.check_cointegration()

        # 2. Calculate Signals (Strategy)
        self.calculate_signals()
        
        t1_ret = self.df[self.ticker1].pct_change()
        t2_ret = self.df[self.ticker2].pct_change()
        
        # Strategy returns: previous day's signal * current day's market return difference
        self.df['strategy_return'] = self.df['signal'].shift(1) * (t1_ret - t2_ret)
        self.df['cum_return'] = (1 + self.df['strategy_return']).cumprod()
        
        # Handle cases with missing data or empty cumulative returns
        if self.df['cum_return'].empty or pd.isna(self.df['cum_return'].iloc[-1]):
            total_return = 0.0
            final_val = 1.0
        else:
            total_return = self.df['cum_return'].iloc[-1] - 1
            final_val = self.df['cum_return'].iloc[-1]
            
        # --- NEW: Annualized Return Calculation (CAGR) ---
        start_date = self.df.index[0]
        end_date = self.df.index[-1]
        duration_days = (end_date - start_date).days
        years = duration_days / 365.25
        
        # Calculate CAGR only if duration is sufficient and final value is positive
        if years > 0.01 and final_val > 0:
            annualized_return = (final_val) ** (1 / years) - 1
        else:
            annualized_return = 0.0
        # -------------------------------------------------

        # Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
        std_dev = self.df['strategy_return'].std()
        if std_dev == 0 or pd.isna(std_dev):
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = self.df['strategy_return'].mean() / std_dev * np.sqrt(252)
        
        # 3. Save results including cointegration and annualized return
        self.results = {
            'pair': f"{self.ticker1}-{self.ticker2}",
            'ticker_1': self.ticker1,
            'ticker_2': self.ticker2,
            'total_return': total_return,
            'annualized_return': annualized_return,  # Added metric
            'sharpe_ratio': sharpe_ratio,
            'final_value': final_val,
            'coint_pvalue': coint_p_value
        }
        
        return self.results