import backtrader as bt
import ccxt
import pandas as pd

class EMACrossover(bt.Strategy):
    params = (('short_period', 20), ('long_period', 50), ('take_profit', 0.1), ('stop_loss', 0.05))

    def __init__(self):
        self.ema_short = bt.indicators.ExponentialMovingAverage(period=self.params.short_period)
        self.ema_long = bt.indicators.ExponentialMovingAverage(period=self.params.long_period)
        self.buy_price = None

    def next(self):
        if self.position:
            # Calculate the current profit/loss percentage
            current_price = self.data.close[0]
            profit_loss = (current_price - self.buy_price) / self.buy_price

            # Check for take profit
            if profit_loss >= self.params.take_profit:
                self.sell()
                print(f"Take profit triggered at {current_price:.2f}, profit: {profit_loss:.2%}")
                self.buy_price = None

            # Check for stop loss
            elif profit_loss <= -self.params.stop_loss:
                self.sell()
                print(f"Stop loss triggered at {current_price:.2f}, loss: {profit_loss:.2%}")
                self.buy_price = None

        if self.ema_short[0] > self.ema_long[0] and not self.position:
            self.buy()
            self.buy_price = self.data.close[0]
            print(f"Buy order executed at {self.buy_price:.2f}")
        elif self.ema_short[0] < self.ema_long[0] and self.position:
            self.sell()
            print(f"Sell order executed at {self.data.close[0]:.2f}")
            self.buy_price = None

# Fetch Binance data
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BNB/USDT', timeframe='30m', limit=2*24*10)

# Convert to DataFrame and format datetime
df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)

# Run Backtrader backtest
data = bt.feeds.PandasData(dataname=df)
cerebro = bt.Cerebro()
cerebro.addstrategy(EMACrossover)

# Set initial cash
cerebro.broker.set_cash(1000.0)

# Add analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

# Add data and run
cerebro.adddata(data)
results = cerebro.run()

# Print final portfolio value
final_value = cerebro.broker.getvalue()
print(f"Final Portfolio Value: ${final_value:.2f}")

# Print analyzers results
sharpe = results[0].analyzers.sharpe.get_analysis()
drawdown = results[0].analyzers.drawdown.get_analysis()
trades = results[0].analyzers.trades.get_analysis()

# Check if the Sharpe Ratio is available
sharpe_ratio = sharpe.get('sharperatio', None)
if sharpe_ratio is not None:
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
else:
    print("Sharpe Ratio: N/A")

print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
print(f"Total Trades: {trades.total.total}")

# Plot the results
#cerebro.plot()
