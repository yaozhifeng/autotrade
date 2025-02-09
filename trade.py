from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
import pandas as pd
#import numpy as np
import time
import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
TEST_MODE = os.getenv('TEST_MODE', 'True')
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 1000))
TRADE_SYMBOL = os.getenv('TRADE_SYMBOL', 'BNBUSDT')
TRADE_QUANTITY = float(os.getenv('TRADE_QUANTITY', 1))
SHORT_SMA = int(os.getenv('SHORT_SMA', 20))
LONG_SMA = int(os.getenv('LONG_SMA', 50))
INTERVAL = os.getenv('INTERVAL', Client.KLINE_INTERVAL_15MINUTE)
LOOKBACK = int(os.getenv('LOOKBACK', 4*24*5))
TRADING_FEE = float(os.getenv('TRADING_FEE', 0.001))

# Binance API Configuration
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

if TEST_MODE:
    BASE_URL = 'https://testnet.binance.vision'
else:
    BASE_URL = 'https://api.binance.com'

# Initialize Binance Client
client = Client(API_KEY, API_SECRET, testnet=TEST_MODE)

def fetch_historical_data(symbol, interval, lookback):
    """Fetch historical klines data from Binance"""
    klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        limit=lookback
    )
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df[['timestamp', 'close']]

def calculate_sma(df, short_period, long_period):
    """Calculate SMA indicators"""
    df['short_sma'] = df['close'].rolling(window=short_period).mean()
    df['long_sma'] = df['close'].rolling(window=long_period).mean()
    df.dropna(inplace=True)
    return df

def generate_signals(df):
    """Generate trading signals based on SMA crossover"""
    df['signal'] = 0
    df['short_prev'] = df['short_sma'].shift(1)
    df['long_prev'] = df['long_sma'].shift(1)

    # Buy signal (short SMA crosses above long SMA)
    df.loc[(df['short_sma'] > df['long_sma']) & (df['short_prev'] <= df['long_prev']), 'signal'] = 1

    # Sell signal (short SMA crosses below long SMA)
    df.loc[(df['short_sma'] < df['long_sma']) & (df['short_prev'] >= df['long_prev']), 'signal'] = -1

    return df

def backtest_strategy(df, initial_balance=INITIAL_BALANCE):
    """Backtest the SMA crossover strategy"""
    balance = initial_balance
    holdings = 0
    in_position = False
    trades = []

    for index, row in df.iterrows():
        price = row['close']

        ready_to_buy = balance >= TRADE_QUANTITY * price * (1 + TRADING_FEE)
        ready_to_sell = holdings >= TRADE_QUANTITY

        if row['signal'] == 1 and ready_to_buy and not in_position:
            # Buy
            holdings += TRADE_QUANTITY
            balance -= TRADE_QUANTITY * price * (1 + TRADING_FEE)
            in_position = True
            trades.append({
                'timestamp': row['timestamp'],
                'type': 'buy',
                'price': price,
                'value': holdings * price + balance
            })
        elif row['signal'] == -1 and ready_to_sell and in_position:
            # Sell
            balance += TRADE_QUANTITY * price * (1 - TRADING_FEE)
            holdings -= TRADE_QUANTITY
            in_position = False
            trades.append({
                'timestamp': row['timestamp'],
                'type': 'sell',
                'price': price,
                'value': balance + holdings * price
            })

    # Calculate final portfolio value
    final = balance + holdings * df.iloc[-1]['close']
    
    ## Display backtesting results
    print(f"\n{' Backtest Results ':-^40}")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Value:   ${final:.2f}")
    print(f"Return:          {(final/initial_balance-1)*100:.2f}%")
    print(f"Total Trades:    {len(trades)} trades")
    print(f"Holdings:        {holdings} coins")
    print(f"Final Balance:   ${balance:.2f}")
    # Print the latest coin price
    latest_price = df.iloc[-1]['close']
    print(f"Latest {TRADE_SYMBOL} price: ${latest_price:.2f}")

    print("\nAll trades:")
    for trade in trades:
        print(f"{trade['timestamp']} {trade['type'].upper()} at ${trade['price']:.5f} - Portfolio Value: ${trade['value']:.2f}")

def buy():
    """Place a market buy order"""
    balance = client.get_asset_balance(asset='USDT')
    if balance and float(balance['free']) >= TRADE_QUANTITY * float(client.get_symbol_ticker(symbol=TRADE_SYMBOL)['price']):
        order = client.create_order(
            symbol=TRADE_SYMBOL,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=TRADE_QUANTITY
        )
        if order['status'] == 'FILLED':
            print("Buy order executed successfully:", order)
        else:
            print("Buy order not filled:", order)
    else:
        print("Insufficient USDT balance to place buy order")

def sell():
    """Place a market sell order"""
    balance = client.get_asset_balance(asset=TRADE_SYMBOL.replace('USDT', ''))
    if balance and float(balance['free']) >= TRADE_QUANTITY:
        order = client.create_order(
            symbol=TRADE_SYMBOL,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=TRADE_QUANTITY
        )
        if order['status'] == 'FILLED':
            print("Sell order executed successfully:", order)
        else:
            print("Sell order not filled:", order)
    else:
        print(f"Insufficient {TRADE_SYMBOL.replace('USDT', '')} balance to place sell order")

def check_portfolio_value():
    """Check and print the live portfolio value including USDT and the worth of the target coin"""
    try:
        # Get USDT balance
        usdt_balance = client.get_asset_balance(asset='USDT')
        usdt_value = float(usdt_balance['free']) if usdt_balance else 0.0

        # Get target coin balance
        target_coin = TRADE_SYMBOL.replace('USDT', '')
        coin_balance = client.get_asset_balance(asset=target_coin)
        coin_value = float(coin_balance['free']) if coin_balance else 0.0

        # Get current price of the target coin in USDT
        coin_price = float(client.get_symbol_ticker(symbol=TRADE_SYMBOL)['price'])

        # Calculate total portfolio value in USDT
        total_value = usdt_value + (coin_value * coin_price)

        print(f"Portfolio Value: ${total_value:.2f} (USDT: ${usdt_value:.2f}, {target_coin}: {coin_value:.6f} @ ${coin_price:.2f})")
    except Exception as e:
        print("An error occurred while checking portfolio value:", e)

def live_trading():
    """Execute live trading based on SMA strategy"""
    print("\nStarting live trading...")
    in_position = False  # Track if we are currently holding a position

    while True:
        try:
            # Fetch historical data
            df = fetch_historical_data(TRADE_SYMBOL, INTERVAL, LONG_SMA*2)
            df = calculate_sma(df, SHORT_SMA, LONG_SMA)
            df = generate_signals(df)

            # Get latest signal
            latest_signal = df.iloc[-1]['signal']

            if latest_signal == 1 and not in_position:
                print(f"\n{datetime.datetime.now()} - BUY SIGNAL")
                # Place buy order
                buy()
                in_position = True  # Update position status

            elif latest_signal == -1 and in_position:
                print(f"\n{datetime.datetime.now()} - SELL SIGNAL")
                # Place sell order
                sell()
                in_position = False  # Update position status
            else:
                print(f"\n{datetime.datetime.now()} - NO SIGNAL or already in position")

        except Exception as e:
            print("An error occurred:", e)
        time.sleep(60*15)  # Wait for 15 minutes before checking again

def test_run():
    """Test the SMA strategy on historical data"""
    data = fetch_historical_data(TRADE_SYMBOL, INTERVAL, LOOKBACK)
    data = calculate_sma(data, SHORT_SMA, LONG_SMA)
    data = generate_signals(data)
    backtest_strategy(data, INITIAL_BALANCE)

if __name__ == "__main__":
    if TEST_MODE:
        # Backtest the strategy
        print("Running backtest for trading " + TRADE_SYMBOL)
        test_run()

    # Start live trading if not in test mode
    if not TEST_MODE:
        live_trading()
