from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
import pandas as pd
import requests
import time
import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Configuration
TEST_NET = os.getenv('TEST_NET', 'True') == 'True'
TEST_MODE = os.getenv('TEST_MODE', 'True') == 'True'
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 1000))
TRADE_SYMBOL = os.getenv('TRADE_SYMBOL', 'BNBUSDT')
TRADE_QUANTITY = float(os.getenv('TRADE_QUANTITY', 1))
SHORT_PERIOD = int(os.getenv('SHORT_PERIOD', 7))
LONG_PERIOD = int(os.getenv('LONG_PERIOD', 26))
INTERVAL = os.getenv('INTERVAL', '15m')
LOOKBACK = int(os.getenv('LOOKBACK', 5))
TRADING_FEE = float(os.getenv('TRADING_FEE', 0.001))
MANAGE_RISK = os.getenv('MANAGE_RISK', 'False') == 'True'
TAKE_PROFIT = float(os.getenv('TAKE_PROFIT', 0.1))  # 10% profit
STOP_LOSS = float(os.getenv('STOP_LOSS', 0.05))  # 5% loss

# Binance API Configuration
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

client = Client(API_KEY, API_SECRET, testnet=TEST_NET)

def fetch_historical_data(symbol, interval, limit=500):
    """Fetch historical klines data from Binance"""
    klines = client.get_historical_klines(symbol, interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    return df[['close']]

def calculate_macd(df, short_period=SHORT_PERIOD, long_period=LONG_PERIOD, signal_period=9):
    """Calculate MACD indicators"""
    df['macd'] = df['close'].ewm(span=short_period, adjust=False).mean() - df['close'].ewm(span=long_period, adjust=False).mean()
    df['signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def generate_signals(df):
    """Generate trading signals based on MACD crossover"""
    df['signal'] = 0
    df['macd_prev'] = df['macd'].shift(1)
    df['signal_prev'] = df['signal'].shift(1)

    # Buy signal (MACD crosses above signal line)
    df.loc[(df['macd'] > df['signal']) & (df['macd_prev'] <= df['signal_prev']), 'signal'] = 1

    # Sell signal (MACD crosses below signal line)
    df.loc[(df['macd'] < df['signal']) & (df['macd_prev'] >= df['signal_prev']), 'signal'] = -1

    return df

def backtest_strategy(df, initial_balance, take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS):
    balance = initial_balance
    holdings = 0
    in_position = False
    trades = []
    buy_price = None

    for index, row in df.iterrows():
        price = row['close']

        quantity = int(balance * 0.9 / price)

        ready_to_buy = balance >= quantity * price * (1 + TRADING_FEE)
        ready_to_sell = holdings >= quantity

        if in_position:
            # Calculate the current profit/loss percentage
            profit_loss = (price - buy_price) / buy_price

            if MANAGE_RISK:
                # Check for take profit
                if profit_loss >= take_profit:
                    balance += holdings * price * (1 - TRADING_FEE)
                    holdings = 0
                    in_position = False
                    trades.append({
                        'timestamp': row.name,
                        'type': 'sell',
                        'price': price,
                        'value': balance
                    })
                    print(f"Take profit triggered at {price:.2f}, profit: {profit_loss:.2%}")
                    buy_price = None

                # Check for stop loss
                elif profit_loss <= -stop_loss:
                    balance += holdings * price * (1 - TRADING_FEE)
                    holdings = 0
                    in_position = False
                    trades.append({
                        'timestamp': row.name,
                        'type': 'sell',
                        'price': price,
                        'value': balance
                    })
                    print(f"Stop loss triggered at {price:.2f}, loss: {profit_loss:.2%}")
                    buy_price = None

        if row['signal'] == 1 and ready_to_buy and not in_position:
            # Buy
            holdings += quantity
            balance -= quantity * price * (1 + TRADING_FEE)
            in_position = True
            buy_price = price
            trades.append({
                'timestamp': row.name,
                'type': 'buy',
                'price': price,
                'value': holdings * price + balance
            })
        elif row['signal'] == -1 and ready_to_sell and in_position:
            # Sell
            balance += holdings * price * (1 - TRADING_FEE)
            holdings = 0
            in_position = False
            trades.append({
                'timestamp': row.name,
                'type': 'sell',
                'price': price,
                'value': balance
            })

    # Check if DataFrame is not empty before accessing the last element
    if not df.empty:
        final = balance + holdings * df.iloc[-1]['close']
    else:
        final = balance

    ## Display backtesting results
    print(f"\n{' Backtest Results ':-^40}")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Value:   ${final:.2f}")
    print(f"Return:          {(final/initial_balance-1)*100:.2f}%")
    print(f"Total Trades:    {len(trades)} trades")
    print(f"Holdings:        {holdings} coins")
    print(f"Final Balance:   ${balance:.2f}")
    if not df.empty:
        latest_price = df.iloc[-1]['close']
        print(f"Latest {TRADE_SYMBOL} price: ${latest_price:.2f}")

    print("\nFirst 4 trades:")
    for trade in trades[:4]:
        print(f"{trade['timestamp']} {trade['type'].upper()} at ${trade['price']:.5f} - Portfolio Value: ${trade['value']:.2f}")

    print("\nLast 4 trades:")
    for trade in trades[-4:]:
        print(f"{trade['timestamp']} {trade['type'].upper()} at ${trade['price']:.5f} - Portfolio Value: ${trade['value']:.2f}")

def buy(quantity=TRADE_QUANTITY):
    """Place a market buy order"""
    balance = client.get_asset_balance(asset='USDT')
    if balance and float(balance['free']) >= quantity * float(client.get_symbol_ticker(symbol=TRADE_SYMBOL)['price']):
        try:
            order = client.create_order(
                symbol=TRADE_SYMBOL,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
        except Exception as e:
            print(f"An error occurred while placing buy order: {e}")
            return
        if order['status'] == 'FILLED':
            msg = f"Buy order executed: {order['fills'][0]['qty']} {TRADE_SYMBOL} at ${order['fills'][0]['price']}"
            send_telegram_message(msg)
            print(msg)
        else:
            print("Buy order not filled:", order)
    else:
        print("Insufficient USDT balance to place buy order")

def sell(quantity=TRADE_QUANTITY):
    """Place a market sell order"""
    balance = client.get_asset_balance(asset=TRADE_SYMBOL.replace('USDT', ''))
    if balance and float(balance['free']) >= quantity:
        try:
            order = client.create_order(
                symbol=TRADE_SYMBOL,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
        except Exception as e:
            print(f"An error occurred while placing sell order: {e}")
            return
        if order['status'] == 'FILLED':
            msg = f"Sell order executed: {order['fills'][0]['qty']} {TRADE_SYMBOL} at ${order['fills'][0]['price']}"
            send_telegram_message(msg)
            print(msg)
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

def send_telegram_message(msg):
    """Send a message to a Telegram channel"""
    try:
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={msg}"
        response = requests.get(url)
        print("Telegram message sent successfully")
    except Exception as e:
        print("An error occurred while sending Telegram message:", e)

def live_trading():
    """Execute live trading based on MACD strategy"""
    print("\nStarting live trading...")
    in_position = False  # Track if we are currently holding a position
    entry_price = 0.0  # Track the entry price of the current position

    while True:
        try:
            # Fetch historical data
            df = fetch_historical_data(TRADE_SYMBOL, INTERVAL, LONG_PERIOD*2)
            df = calculate_macd(df)
            df = generate_signals(df)

            # Get latest signal
            latest_row = df.iloc[-1]
            latest_signal = latest_row['signal']
            latest_price = latest_row['close']
            current_price = float(client.get_symbol_ticker(symbol=TRADE_SYMBOL)['price'])
            current_time = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=8)))

            if in_position:
                # Calculate the current profit/loss percentage
                profit_loss = (current_price - entry_price) / entry_price

                # Check for take profit
                if profit_loss >= TAKE_PROFIT:
                    print(f"\n{current_time} - TAKE PROFIT SIGNAL")
                    # Place sell order
                    sell()
                    in_position = False  # Update position status
                    entry_price = None
                    print(f"Take profit triggered at {current_price:.2f}, profit: {profit_loss:.2%}")

                # Check for stop loss
                elif profit_loss <= -STOP_LOSS:
                    print(f"\n{current_time} - STOP LOSS SIGNAL")
                    # Place sell order
                    sell()
                    in_position = False  # Update position status
                    entry_price = None
                    print(f"Stop loss triggered at {current_price:.2f}, loss: {profit_loss:.2%}")

            if latest_signal == 1:
                print(f"\n{current_time} - BUY SIGNAL")
                if not in_position:
                    # Place buy order
                    buy()
                    in_position = True  # Update position status
                    entry_price = current_price
                    print(f"Buy order executed at {entry_price:.2f}")
                else:
                    print(f"Already in position, skip buy order")

            elif latest_signal == -1:
                print(f"\n{current_time} - SELL SIGNAL")
                if in_position:
                    # Place sell order
                    sell()
                    in_position = False  # Update position status
                    entry_price = None
                    print(f"Sell order executed at {current_price:.2f}")
                else:
                    print(f"Not in position, skip sell order")
            else:
                print(f"\n{current_time} - NO SIGNAL - Current Price: {current_price:.2f}")

        except Exception as e:
            print("An error occurred:", e)
        time.sleep(60*30)  # Wait for 30 minutes before checking again

def test_run():
    """Test the MACD strategy on historical data"""
    data = fetch_historical_data(TRADE_SYMBOL, INTERVAL)
    data = calculate_macd(data)
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
