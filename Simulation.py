import pandas as pd
import numpy as np
import Config

class PairTradingStrategy:
    def __init__(self, ticker1, ticker2, data):
        self.ticker1 = ticker1
        self.ticker2 = ticker2

        self.df = data[[ticker1, ticker2]].copy()
        
        self.window_size = Config.WINDOW_SIZE
        self.std_dev_entry = Config.STD_DEV_ENTRY
        self.std_dev_exit = Config.STD_DEV_EXIT
        self.stop_loss = Config.STOP_LOSS_PCT
        
        self.results = {}

    def calculate_signals(self):
        self.df['ratio'] = self.df[self.ticker1] / self.df[self.ticker2]
        self.df['mean'] = self.df['ratio'].rolling(window=self.window_size).mean()
        self.df['std'] = self.df['ratio'].rolling(window=self.window_size).std()
        self.df['z_score'] = (self.df['ratio'] - self.df['mean']) / self.df['std']
        self.df['signal'] = 0 

        # 0 = No position, 1 = Long Ratio (Buy T1, Sell T2), -1 = Short Ratio (Sell T1, Buy T2)
        self.df.loc[self.df['z_score'] < -self.std_dev_entry, 'signal'] = 1
        self.df.loc[self.df['z_score'] > self.std_dev_entry, 'signal'] = -1
        self.df.loc[abs(self.df['z_score']) < self.std_dev_exit, 'signal'] = 0
        
        # Forward fill signals to hold positions until an exit signal is generated
        self.df['signal'] = self.df['signal'].replace(0, np.nan).ffill().fillna(0)
        
    def run_backtest(self):
        self.calculate_signals()
        
        t1_ret = self.df[self.ticker1].pct_change()
        t2_ret = self.df[self.ticker2].pct_change()
        
        # If Signal 1 (Long Ratio): Buy T1, Sell T2
        # If Signal -1 (Short Ratio): Sell T1, Buy T2
        self.df['strategy_return'] = self.df['signal'].shift(1) * (t1_ret - t2_ret)
        
        # Cumulative returns
        self.df['cum_return'] = (1 + self.df['strategy_return']).cumprod()
        
        total_return = self.df['cum_return'].iloc[-1] - 1
        sharpe_ratio = self.df['strategy_return'].mean() / self.df['strategy_return'].std() * np.sqrt(252)
        
        self.results = {
            'pair': f"{self.ticker1}-{self.ticker2}",
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'final_value': self.df['cum_return'].iloc[-1]
        }
        
        return self.results