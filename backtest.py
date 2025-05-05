import time
import pandas as pd
from datetime import datetime, timedelta
import os
import requests
import dotenv

dotenv.load_dotenv()

class Backtest:
    def __init__(self, symbol='LTCUSDT'):
        self.symbol = symbol

    def get_history_data_from_file(self, csv_path):
        if csv_path:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
    def get_history_data_from_binance(self, start_time, end_time, interval='1h'):
        """从binance获取历史数据"""
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": 1000,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000)
        }
        response = requests.get(url, params=params)
        data = response.json()
        return data
    
    def get_history_data(self, start_time, end_time, interval='1h'):
        all_data = []
        current_start = start_time

        while current_start < end_time:
            current_end = current_start + timedelta(hours=100)
            current_end = min(current_end, end_time)
            klines = self.get_history_data_from_binance(current_start, current_end, interval)
            if not klines:
                break
            all_data.extend(klines)

            time_step = (klines[-1][0] - klines[-2][0]) / 1000
            current_start = datetime.fromtimestamp(klines[-1][0] / 1000) + timedelta(seconds=time_step)
            time.sleep(0.5)  # Avoid hitting rate limits

        # Optional: convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
        return df
    
    def save_history_data_to_file(self, df, csv_path):
        df.to_csv(csv_path, index=False)

    def get_macd_signal(self, df):
        """获取MACD信号"""
        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['prev_macd_histogram'] = df['macd_histogram'].shift(1)
        df['signal'] = 0
        df.loc[(df['macd_histogram'] > 0) & (df['prev_macd_histogram'] <= 0), 'signal'] = 1
        df.loc[(df['macd_histogram'] < 0) & (df['prev_macd_histogram'] >= 0), 'signal'] = -1
        return df[['timestamp', 'close', 'macd_histogram', 'signal']]

    def get_ema_signal(self, df):
        """获取EMA信号"""
        df['ema_long'] = df['close'].ewm(span=25, adjust=False).mean()
        df['ema_short'] = df['close'].ewm(span=7, adjust=False).mean()
        df['prev_ema_long'] = df['ema_long'].shift(1)
        df['prev_ema_short'] = df['ema_short'].shift(1)
        df['signal'] = 0
        df.loc[(df['ema_short'] > df['ema_long']) & (df['prev_ema_short'] <= df['prev_ema_long']), 'signal'] = 1
        df.loc[(df['ema_short'] < df['ema_long']) & (df['prev_ema_short'] >= df['prev_ema_long']), 'signal'] = -1
        return df[['timestamp', 'close', 'ema_short', 'ema_long', 'signal']]

    def get_active_hour(self, df):
        positive_hours = 0
        negative_hours = 0

        for index, row in df.iterrows():
            if row['macd_histogram'] > 0:
                positive_hours += 1
            else:
                negative_hours += 1
        return positive_hours, negative_hours
    
    def back_test(self, df):
        """回测函数"""
        if df is None:
            print("No data found")
            return

        # Calculate MACD
        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Loop through dataframe and find the golden crosses and death crosses
        prev_histogram = None
        df['signal'] = 0  # Initialize signal column
        for index, row in df.iterrows():
            timestamp = row['timestamp']
            if prev_histogram is not None:
                if row['macd_histogram'] > 0 and prev_histogram < 0:
                    df.at[index, 'signal'] = 1  # Golden cross
                    print(f"Golden cross at {timestamp}, Close: {row['close']:.2f}, MACD Histogram: {row['macd_histogram']:.4f}")
                elif row['macd_histogram'] < 0 and prev_histogram > 0:
                    df.at[index, 'signal'] = -1  # Death cross
                    print(f"Death cross at {timestamp}, Close: {row['close']:.2f}, MACD Histogram: {row['macd_histogram']:.4f}")
            prev_histogram = row['macd_histogram']
        
        # Simulate trading based on signals
        initial_balance = 1000.0  # Starting with 1000 USDT
        balance = initial_balance
        position = 0  # Current position in base asset
        trades = []
        
        for i in range(1, len(df)):  # Start from 1 to use previous row's signal
            prev_row = df.iloc[i-1]
            curr_row = df.iloc[i]
            
            if prev_row['signal'] == 1:  # Buy signal from previous candle
                if position == 0:  # Only buy if not already in position
                    price = curr_row['close']
                    position = (balance * 0.999) / price  # Account for 0.1% trading fee
                    balance = 0
                    trades.append({
                        'timestamp': curr_row['timestamp'],
                        'type': 'BUY',
                        'price': price,
                        'position': position,
                        'balance': balance
                    })
                    print(f"Buy at {curr_row['timestamp']}: Price={price:.2f}, Position={position:.4f}")
            elif prev_row['signal'] == -1:  # Sell signal from previous candle
                if position > 0:  # Only sell if holding position
                    price = curr_row['close']
                    balance = position * price * 0.999  # Account for 0.1% trading fee
                    trades.append({
                        'timestamp': curr_row['timestamp'],
                        'type': 'SELL',
                        'price': price,
                        'position': position,
                        'balance': balance
                    })
                    print(f"Sell at {curr_row['timestamp']}: Price={price:.2f}, Balance={balance:.2f}")
                    position = 0

        # Calculate final value and total return
        final_value = balance if position == 0 else position * df['close'].iloc[-1] * 0.999
        total_return = ((final_value - initial_balance) / initial_balance) * 100
        
        print(f"\nBacktest Results:")
        print(f"Initial Balance: {initial_balance:.2f} USDT")
        print(f"Final Value: {final_value:.2f} USDT")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {len(trades)}") 

    def calculate_trend_macd(self, df):
        """计算市场趋势"""
        # 使用MACD指标判断趋势，以最后第二个值为准，用最新的值做校验
        macd = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - macd_signal
        trend = macd_histogram.iloc[-2] # 使用倒数第二个值
        latest_trend = macd_histogram.iloc[-1] # 用当前值校验是否趋势没有变化，避免假信号
        if trend > 0 and latest_trend > 0:
            return 1
        elif trend < 0 and latest_trend < 0:
            return -1
        else:
            return 0
    
    def get_market_trend(self):
        """获取市场趋势"""
        df = self.get_market_data()
        return self.calculate_trend_macd(df)


if __name__ == "__main__":
    backtest = Backtest()
    df = backtest.get_history_data_from_file('ltc_1h_2024.csv')
    backtest.back_test(df)