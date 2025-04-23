import pandas as pd
import numpy as np

class Trading:
    def __init__(self,initial_capital = 2000000,trade_size = 10,fee_rate = 0.001):
        self.initial_capital = initial_capital
        self.trade_size = trade_size
        self.fee_rate = fee_rate
        self.total_trade_number = 0
        self.win_trade_number = 0
        self.win_profit = 0
        self.lose_profit = 0
        self.current_capital = initial_capital
        self.current_position = 0 # (1 for bought, -1 for sold without buying, 0 for hold)
    def generate_trade_signal(self,y_predicted):
        """
        Generate trade signals based on predicted values.

        Args:
            y_predicted (np.ndarray): Predicted values from the model.

        Returns:
            pd.Series: Trade signals (1 for buy, -1 for sell, 0 for hold).
        """
        trade_signal = np.zeros(len(y_predicted))
        for i in range(1, len(y_predicted)):
            if y_predicted[i] == np.nan or y_predicted[i] == 0:
                trade_signal[i] = 0
            elif y_predicted[i] >= y_predicted[i-1] or y_predicted[i] == 1:
                trade_signal[i] = 1
            elif y_predicted[i] < y_predicted[i-1] or y_predicted[i] == -1:
                trade_signal[i] = -1
        return trade_signal
    
    def execute_trade(self,trade_signal,prices):
        """
        Execute trades based on the trade signal.

        Args:
            trade_signal (np.ndarray): Trade signals (1 for buy, -1 for sell, 0 for hold).
            prices (np.ndarray): Prices of the stock.
        """
        entry_price = 0
        exit_price = 0
        for i in range(len(trade_signal)):
            if self.current_position == 0:
                if trade_signal[i] == 1:
                    entry_price = prices[i]
                    self.current_position = 1
                if trade_signal[i] == -1:
                    entry_price = prices[i]
                    self.current_position = -1
            elif self.current_position == 1:
                if trade_signal[i] == -1:
                    exit_price = prices[i]
                    self.current_position = 0
                    profit = (exit_price - entry_price) * self.trade_size - self.fee_rate * self.trade_size*entry_price
                    if profit > 0:
                        self.win_trade_number += 1
                        self.win_profit += profit
                    else:
                        self.lose_profit += profit
                    self.current_capital += profit
                    entry_price = 0
                    self.total_trade_number += 1
            else:
                if trade_signal[i] == 1:
                    exit_price = prices[i]
                    self.current_position = 0
                    profit = (exit_price - entry_price) * self.trade_size - self.fee_rate * self.trade_size*entry_price
                    if profit > 0:
                        self.win_trade_number += 1
                        self.win_profit += profit
                    else:
                        self.lose_profit += profit
                    self.current_capital += profit
                    entry_price = 0
                    self.total_trade_number += 1
        return self.current_capital
    def performance(self):
        total_profit = self.current_capital - self.initial_capital
        win_rate = self.win_trade_number/self.total_trade_number
        profit_factor = abs(self.win_profit)/abs(self.lose_profit)
        return total_profit,self.total_trade_number,win_rate,profit_factor
        
