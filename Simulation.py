import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import Config

class PairTradingStrategy:
    def __init__(self, ticker1, ticker2, data):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        
        self.df = data[[ticker1, ticker2]].copy()
        
        self.window_size = Config.WINDOW_SIZE
        self.std_dev_entry = Config.STD_DEV_ENTRY
        self.std_dev_exit = Config.STD_DEV_EXIT
        
        self.results = {}

    def check_cointegration(self):
        try:
            clean_data = self.df.dropna()
            
            if len(clean_data) < Config.MIN_HISTORY_DAYS:
                return 1.0
                
            s1 = clean_data[self.ticker1]
            s2 = clean_data[self.ticker2]
            
            score, p_value, _ = ts.coint(s1, s2)
            return p_value
            
        except Exception as e:
            return 1.0

    def calculate_signals(self):
        self.df['ratio'] = self.df[self.ticker1] / self.df[self.ticker2]
        
        self.df['mean'] = self.df['ratio'].rolling(window=self.window_size).mean()
        self.df['std'] = self.df['ratio'].rolling(window=self.window_size).std()
        self.df['z_score'] = (self.df['ratio'] - self.df['mean']) / self.df['std']
        
        self.df['signal'] = 0 
        
        self.df.loc[self.df['z_score'] < -self.std_dev_entry, 'signal'] = 1
        self.df.loc[self.df['z_score'] > self.std_dev_entry, 'signal'] = -1
        self.df.loc[abs(self.df['z_score']) < self.std_dev_exit, 'signal'] = 0
        
        self.df['signal'] = self.df['signal'].replace(0, np.nan).ffill().fillna(0)
        
    def run_backtest(self, transaction_cost=0.001):
        coint_p_value = self.check_cointegration()
        self.calculate_signals()
        
        t1_ret = self.df[self.ticker1].pct_change()
        t2_ret = self.df[self.ticker2].pct_change()

        self.df['strategy_return_gross'] = self.df['signal'].shift(1) * (t1_ret - t2_ret)
        trades = self.df['signal'].diff().abs().fillna(0)
        costs = trades * transaction_cost

        self.df['strategy_return'] = self.df['strategy_return_gross'] - costs

        # 6. Calculate Cumulative Returns
        self.df['cum_return'] = (1 + self.df['strategy_return']).cumprod()
        # Handle cases with missing data or empty cumulative returns
        if self.df['cum_return'].empty or pd.isna(self.df['cum_return'].iloc[-1]):
            total_return = 0.0
            final_val = 1.0
        else:
            total_return = self.df['cum_return'].iloc[-1] - 1
            final_val = self.df['cum_return'].iloc[-1]    

        # annual return
        start_date = self.df.index[0]
        end_date = self.df.index[-1]
        duration_days = (end_date - start_date).days
        years = duration_days / 365.25
        if years > 0.01 and final_val > 0:
            annualized_return = (final_val) ** (1 / years) - 1
        else:
            annualized_return = 0.0

        #check correlation
        clean_data = self.df[[self.ticker1, self.ticker2]].dropna()
        if len(clean_data) > 0:
            correlation = clean_data[self.ticker1].corr(clean_data[self.ticker2])
        else:
            correlation = 0.0

        # sharpe ratio
        std_dev = self.df['strategy_return'].std()
        if std_dev == 0 or pd.isna(std_dev):
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = self.df['strategy_return'].mean() / std_dev * np.sqrt(252)

        self.results = {
            'pair': f"{self.ticker1}-{self.ticker2}",
            'ticker_1': self.ticker1,
            'ticker_2': self.ticker2,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'final_value': final_val,
            'coint_pvalue': coint_p_value,
            'correlation': correlation
        }
        
        return self.results