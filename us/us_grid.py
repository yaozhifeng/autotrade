import time
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from pytz import timezone
from dotenv import load_dotenv
import requests
import os
from decimal import Decimal

# Longport import
from longport.openapi import TradeContext, Config, OrderSide, OrderType, TimeInForceType, Market, OrderStatus
from longport.openapi import QuoteContext, TradeSession

# Load environment variables from .env file
load_dotenv()

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


class StockGridTrader:
    def __init__(self, app_key, app_secret, access_token, symbol='AAPL',
                 grid_levels=10, initial_grid_width=0.02,
                 quantity_per_grid=1):
        """
        Initialize the dynamic grid trading bot for stocks
        :param app_key: Longbridge API app key
        :param app_secret: Longbridge API app secret
        :param access_token: Longbridge API access token
        :param symbol: Stock symbol
        :param grid_levels: Number of grid levels
        :param initial_grid_width: Initial grid width (percentage)
        :param quantity_per_grid: Quantity to trade per grid
        """
        # Initialize Longbridge API clients
        config = Config(app_key=app_key, app_secret=app_secret, access_token=access_token)
        self.trade_ctx = TradeContext(config)
        self.quote_ctx = QuoteContext(config)
        
        self.symbol = symbol
        self.grid_levels = grid_levels
        self.initial_grid_width = initial_grid_width
        self.quantity = quantity_per_grid
        self.last_update_id = None  # Last telegram update id
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('stock_grid_trader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize order and grid status
        self.orders = {}
        self.current_grid = None
        self.last_adjustment_time = None
        self.last_trade_time = None  # Track the time of the last trade
        
        # Set adjustment parameters
        self.min_adjustment_interval = int(os.getenv('MIN_ADJUSTMENT_INTERVAL', 3600))  # Minimum adjustment interval (seconds)

        # Initialize daily statistics
        self.daily_stats = {
            'buy_orders': 0,
            'sell_orders': 0,
            'total_buy_price': 0.0,
            'total_sell_price': 0.0,
            'initial_balance': self.get_total_balance(),
            'initial_price': self.get_current_price(),
            'final_balance': 0.0,
            'final_price': 0.0
        }
        self.last_briefing_time = time.time()
        self.briefing_interval = int(os.getenv('BRIEFING_INTERVAL', 86400))  # Default to 24 hours
        
        # Market session
        self.market = self.get_market_from_symbol()
        
    def get_market_from_symbol(self):
        """Get the market from the symbol"""
        # This is a simple implementation - enhance based on your needs
        if self.symbol.endswith('.US'):
            return Market.US
        elif self.symbol.endswith('.HK'):
            return Market.HK
        else:
            # Default to US market
            return Market.US
        
    def get_fee(self, share=1):
        """Calculate the fee for a trade, the US fee is calculated by share instead of amount"""
        commission_fee = max(0.99, share * 0.0049)
        platform_fee = max(1.0, 0.005 * share)
        return commission_fee + platform_fee

    def is_market_open(self):
        """Check if the stock market is currently open"""
        try:
            trading_session = self.quote_ctx.trading_session()
            for session in trading_session:
                if session.market == self.market:
                    #should further check for market time from session.trade_sessions
                    for session_info in session.trade_sessions:
                        if session_info.trade_session in [TradeSession.Normal, TradeSession.Post]:
                            start_time = session_info.begin_time
                            end_time = session_info.end_time
                            current_time = datetime.now(timezone('US/Eastern')).time()
                            if start_time <= current_time <= end_time:
                                return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking market status: {str(e)}")
            return False

    def get_current_price(self):
        """Get the current price of the stock"""
        try:
            quote = self.quote_ctx.quote(symbols=[self.symbol])
            if quote and len(quote) > 0:
                return float(quote[0].last_done)
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            return 0.0

    def get_total_balance(self):
        """Get the total balance of the account"""
        available_cash, frozen_cash = self.get_cash_balance()
        total_stock_value = self.get_stock_position() * self.get_current_price()
        return available_cash + frozen_cash + total_stock_value
        
    def get_cash_balance(self, currency='USD'):
        """Get the cash balance of the account"""
        """Return available cash and frozen cash"""
        try:
            account_balance = self.trade_ctx.account_balance()
            for balance in account_balance:
                for cash_info in balance.cash_infos:
                    if cash_info.currency == currency:
                        withdraw_cash = float(cash_info.withdraw_cash)
                        frozen_cash = float(cash_info.frozen_cash)
                        return (withdraw_cash, frozen_cash)
        except Exception as e:
            self.logger.error(f"Error getting cash balance: {str(e)}")
            
        return (0.0, 0.0)
        
    def get_stock_position(self):
        """Get the current stock position"""
        try:
            positions = self.trade_ctx.stock_positions()
            for channel in positions.channels:
                for position in channel.positions:
                    if position.symbol == self.symbol:
                        stock_position = float(position.quantity)
                        available_position = float(position.available_quantity)
                        return stock_position
        except Exception as e:
            self.logger.error(f"Error getting stock position: {str(e)}")
        
        return 0

    def send_daily_briefing(self):
        """Send a daily summary"""
        self.daily_stats['final_balance'] = self.get_total_balance()
        self.daily_stats['final_price'] = self.get_current_price()

        avg_buy_price = self.daily_stats['total_buy_price'] / self.daily_stats['buy_orders'] if self.daily_stats['buy_orders'] > 0 else 0
        avg_sell_price = self.daily_stats['total_sell_price'] / self.daily_stats['sell_orders'] if self.daily_stats['sell_orders'] > 0 else 0

        gross_margin = (avg_sell_price - avg_buy_price) * min(self.daily_stats['sell_orders'], self.daily_stats['buy_orders'])
        fee = self.get_fee(self.quantity) * (self.daily_stats['sell_orders'] + self.daily_stats['buy_orders'])
        net_profit = gross_margin - fee

        briefing_msg = (
            f"Daily Summary:\n"
            f"Net Profit: {net_profit:.2f} USD\n"
            f"Gross Profit: {gross_margin:.2f} USD\n"
            f"Fees: {fee:.2f} USD\n"
            f"Buy Orders: {self.daily_stats['buy_orders']}\n"
            f"Sell Orders: {self.daily_stats['sell_orders']}\n"
            f"Average Buy Price: {avg_buy_price:.2f} USD\n"
            f"Average Sell Price: {avg_sell_price:.2f} USD\n"
            f"Initial Balance: {self.daily_stats['initial_balance']:.2f} USD\n"
            f"Final Balance: {self.daily_stats['final_balance']:.2f} USD\n"
            f"Current position: {self.get_stock_position()} shares\n"
            f"Initial Price: {self.daily_stats['initial_price']:.2f} USD\n"
            f"Final Price: {self.daily_stats['final_price']:.2f} USD\n"
            f"Period: {datetime.fromtimestamp(self.last_briefing_time).strftime('%Y-%m-%d %H:%M:%S')} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        send_telegram_message(briefing_msg)
        self.logger.info(briefing_msg)

        # Reset daily statistics
        self.daily_stats = {
            'buy_orders': 0,
            'sell_orders': 0,
            'total_buy_price': 0.0,
            'total_sell_price': 0.0,
            'initial_balance': self.get_total_balance(),
            'initial_price': self.get_current_price(),
            'final_balance': 0.0,
            'final_price': 0.0
        }
        self.last_briefing_time = time.time()

    def answer_telegram(self):
        """Respond to Telegram messages"""
        try:
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            url = f"https://api.telegram.org/bot{token}/getUpdates?offset={self.last_update_id}"
            response = requests.get(url)
            data = response.json()
            if data['ok']:
                for message in data['result']:
                    self.last_update_id = message['update_id'] + 1
                    text = message['message']['text']
                    self.logger.info(f"Received Telegram message: {text}")
                    if text == '/orders':
                        self.show_orders()
                    elif text == '/portfolio':
                        self.check_portfolio()
                    elif text == '/adjust':
                        self.cancel_all_orders()
                        self.current_grid = None
                        self.last_adjustment_time = None
        except Exception as e:
            self.logger.error(f"Failed to respond to Telegram message: {str(e)}")

    def adjust_grid_parameters(self):
        """Adjust grid parameters based on current market conditions"""
        try:
            # Use current price as market data
            current_price = self.get_current_price()
            
            grid_width = self.initial_grid_width
            
            grid_center = current_price
            
            # Calculate new grid range
            grid_range = current_price * grid_width
            lower_price = grid_center - grid_range/2
            upper_price = grid_center + grid_range/2
            
            # Calculate new grid prices
            self.current_grid = np.linspace(lower_price, upper_price, self.grid_levels)
            self.grid_gap = self.current_grid[1] - self.current_grid[0]

            profit_per_grid_percent = self.grid_gap * self.quantity - self.get_fee(self.quantity)*2

            msg = f"Grid parameters adjusted:\n"
            msg += f"Current price: {current_price:.2f} USD\n"
            msg += f"New grid range: {lower_price:.2f} - {upper_price:.2f} USD\n"
            msg += f"Grid levels: {self.grid_levels}\n"
            msg += f"Quantity per grid: {self.quantity}\n"
            msg += f"Gap per grid: {self.grid_gap:.2f} USD\n"
            msg += f"Profit per grid: {profit_per_grid_percent:.2f} USD\n"
            self.logger.info(msg)
            send_telegram_message(msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to adjust grid parameters: {str(e)}")
            return False

    def get_active_orders(self):
        """Get all active orders"""
        try:
            active_status = [OrderStatus.New, OrderStatus.PartialFilled, OrderStatus.NotReported]
            orders = self.trade_ctx.today_orders(symbol=self.symbol, status=active_status)
            return orders
        except Exception as e:
            self.logger.error(f"Error getting active orders: {str(e)}")
            return []
        
    def get_filled_orders(self):
        """Get all filled orders"""
        try:
            orders = self.trade_ctx.today_orders(symbol=self.symbol, status=[OrderStatus.Filled])
            return orders
        except Exception as e:
            self.logger.error(f"Error getting filled orders: {str(e)}")
            return []

    def cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            # Get all open orders
            orders = self.get_active_orders()
            
            for order in orders:
                self.trade_ctx.cancel_order(order.order_id)
            
            self.orders.clear()
            self.logger.info("All orders have been cancelled")
        except Exception as e:
            self.logger.error(f"Failed to cancel orders: {str(e)}")

    def place_grid_orders(self, buy_only=False, sell_only=False):
        """Place grid orders"""
        if not self.is_market_open():
            self.logger.info("Market is closed. Not placing orders.")
            return

        if self.current_grid is None:
            self.adjust_grid_parameters()
        
        current_price = self.get_current_price()
        
        # Get available balances and positions
        try:
            buying_power, _ = self.get_cash_balance()         
            stock_position = self.get_stock_position()
            
            buy_orders_placed = 0
            sell_orders_placed = 0
            
            for price in self.current_grid:
                try:
                    if price < current_price and not sell_only:
                        # Check if there is enough cash to place a buy order
                        required_cash = self.quantity * price
                        if buying_power >= required_cash:
                            # Place buy order
                            order = self.trade_ctx.submit_order(
                                symbol=self.symbol,
                                order_type=OrderType.LO,
                                side=OrderSide.Buy,
                                submitted_price=Decimal(f"{price:.2f}"),
                                submitted_quantity=self.quantity,
                                time_in_force=TimeInForceType.Day
                            )
                            
                            self.orders[order.order_id] = {
                                'price': price,
                                'side': 'BUY',
                                'status': 'OPEN'
                            }
                            self.logger.info(f"Order placed: BUY {price:.2f} USD, Quantity: {self.quantity}")
                            buying_power -= required_cash
                            buy_orders_placed += 1
                        else:
                            self.logger.warning(f"Insufficient buying power to place buy order at {price:.2f} USD")
                    
                    elif price > current_price and not buy_only:
                        # Check if there is enough stock to place a sell order
                        if stock_position >= self.quantity:
                            # Place sell order
                            order = self.trade_ctx.submit_order(
                                symbol=self.symbol,
                                order_type=OrderType.LO,
                                side=OrderSide.Sell,
                                submitted_price=Decimal(f"{price:.2f}"),
                                submitted_quantity=self.quantity,
                                time_in_force=TimeInForceType.Day
                            )
                            
                            self.orders[order.order_id] = {
                                'price': price,
                                'side': 'SELL',
                                'status': 'OPEN'
                            }
                            self.logger.info(f"Order placed: SELL {price:.2f} USD, Quantity: {self.quantity}")
                            stock_position -= self.quantity
                            sell_orders_placed += 1
                        else:
                            self.logger.warning(f"Insufficient stock position to place sell order at {price:.2f} USD")
                                
                except Exception as e:
                    self.logger.error(f"Failed to place order: {str(e)}")
            
            self.logger.info(f"Buy orders: {buy_orders_placed}, Sell orders: {sell_orders_placed}")
            send_telegram_message(f"Buy orders: {buy_orders_placed}, Sell orders: {sell_orders_placed}")
            
        except Exception as e:
            self.logger.error(f"Error getting account information: {str(e)}")

    def should_adjust_grid(self):
        """Determine if the grid should be adjusted"""
        if self.current_grid is None or self.last_adjustment_time is None:
            return True
            
        current_time = time.time()
        
        # Check if market is open
        if not self.is_market_open():
            return False

        # Check time interval
        if current_time - self.last_adjustment_time < self.min_adjustment_interval:
            return False
        
        # Check price changes
        try:
            current_price = self.get_current_price()
            active_orders = self.get_active_orders()
            
            if active_orders:
                high_price = max([float(order.price) for order in active_orders])
                low_price = min([float(order.price) for order in active_orders])
                
                if current_price - self.grid_gap * 2 > high_price or current_price + self.grid_gap * 2 < low_price:
                    return True
            else:
                if time.time() - self.last_adjustment_time >= 600:  # 10 minutes
                    return True
        except Exception as e:
            self.logger.error(f"Error checking orders: {str(e)}")
            
        return False

    def show_orders(self):
        """Display current orders"""
        try:
            active_orders = self.get_active_orders()
            sorted_orders = sorted(active_orders, key=lambda x: float(x.price), reverse=True)
            
            for order in sorted_orders:
                side = "BUY" if order.side == OrderSide.Buy else "SELL"
                self.logger.info(f"Order: {side} {float(order.price):.2f} USD, Quantity: {order.quantity}")
                send_telegram_message(f"Order: {side} {float(order.price):.2f} USD, Quantity: {order.quantity}")
                
        except Exception as e:
            self.logger.error(f"Error showing orders: {str(e)}")

    def check_portfolio(self):
        """Check portfolio status"""
        try:
            cash_info = self.trade_ctx.cash_info()
            available_cash = float(cash_info.available_cash)
            frozen_cash = float(cash_info.frozen_cash)
            
            positions = self.trade_ctx.positions()
            stock_position = 0
            for position in positions:
                if position.symbol == self.symbol:
                    stock_position = float(position.quantity)
                    break
            
            self.logger.info(f"Cash: {available_cash:.2f} USD (Available), {frozen_cash:.2f} USD (Frozen)")
            send_telegram_message(f"Cash: {available_cash:.2f} USD (Available), {frozen_cash:.2f} USD (Frozen)")
            self.logger.info(f"{self.symbol}: {stock_position} shares")
            send_telegram_message(f"{self.symbol}: {stock_position} shares")

            # Calculate total balance
            total_balance = float(cash_info.total_cash)
            current_price = self.get_current_price()
            stock_value = stock_position * current_price
            total_value = total_balance + stock_value
            
            self.logger.info(f"Total value: {total_value:.2f} USD")
            send_telegram_message(f"Total value: {total_value:.2f} USD")
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio: {str(e)}")

    def evaluate_risk(self):
        """Evaluate risk and determine if we should only sell"""
        try:
            stock_position = self.get_stock_position()
                    
            max_stock_position = float(os.getenv('MAX_STOCK_POSITION', 100))  # Maximum allowed stock position
            
            sell_only = False
            
            if stock_position > max_stock_position:
                msg = f"{self.symbol} position {stock_position} exceeds maximum allowed {max_stock_position}, only selling"
                self.logger.warning(msg)
                send_telegram_message(msg)
                sell_only = True
                
            return sell_only
            
        except Exception as e:
            self.logger.error(f"Error evaluating risk: {str(e)}")
            return False
    
    def run(self):
        """Run the dynamic grid trading bot"""
        self.logger.info("Starting stock grid trading bot...")
        self.logger.info(f"Symbol: {self.symbol}")
        
        while True:
            try:
                # Skip processing if market is closed
                if not self.is_market_open():
                    # Check if daily briefing is needed
                    if time.time() - self.last_briefing_time >= self.briefing_interval:
                        self.send_daily_briefing()
                     
                    time.sleep(60)  # Check every minute when market is closed
                    continue
                
                # Check order status
                try:
                    for order_id, order_info in list(self.orders.items()):
                        order_detail = self.trade_ctx.order_detail(order_id=order_id)

                        if order_detail and order_detail.status == OrderStatus.Expired:
                            self.logger.info(f"Order expired: {order_info['side']} {order_info['price']:.2f} USD")
                            send_telegram_message(f"Order expired: {order_info['side']} {order_info['price']:.2f} USD")
                            self.orders.pop(order_id)
                            continue

                        if order_detail and order_detail.status == OrderStatus.Filled:
                            self.logger.info(f"Order filled: {order_info['side']} {order_info['price']:.2f} USD")
                            send_telegram_message(f"Order filled: {order_info['side']} {order_info['price']:.2f} USD")
                            self.orders.pop(order_id)
                            self.last_trade_time = time.time()  # Update the last trade time
                            
                            # Update daily stats
                            if order_info['side'] == 'BUY':
                                self.daily_stats['buy_orders'] += 1
                                self.daily_stats['total_buy_price'] += order_info['price'] * self.quantity
                            else:
                                self.daily_stats['sell_orders'] += 1
                                self.daily_stats['total_sell_price'] += order_info['price'] * self.quantity
                            
                            # Place reverse order
                            new_side = 'SELL' if order_info['side'] == 'BUY' else 'BUY'
                            new_price = order_info['price'] - self.grid_gap if new_side == 'BUY' else order_info['price'] + self.grid_gap
                            
                            # Check balance before placing reverse order
                            if new_side == 'BUY':
                                available_cash, _ = self.get_cash_balance()
                                required_cash = self.quantity * new_price
                                
                                if available_cash >= required_cash:
                                    new_order = self.trade_ctx.submit_order(
                                        symbol=self.symbol,
                                        order_type=OrderType.LO,
                                        side=OrderSide.Buy,
                                        submitted_price=Decimal(f"{new_price:.2f}"),
                                        submitted_quantity=self.quantity,
                                        time_in_force=TimeInForceType.Day
                                    )
                                    
                                    self.orders[new_order.order_id] = {
                                        'price': new_price,
                                        'side': new_side,
                                        'status': 'OPEN'
                                    }
                                    self.logger.info(f"New order created: {new_side} {new_price:.2f} USD")
                                else:
                                    self.logger.warning(f"Insufficient cash to place buy order at {new_price:.2f} USD")
                            else:
                                stock_position = self.get_stock_position()
                                        
                                if stock_position >= self.quantity:
                                    new_order = self.trade_ctx.submit_order(
                                        symbol=self.symbol,
                                        order_type=OrderType.LO,
                                        side=OrderSide.Sell,
                                        submitted_price=Decimal(f"{new_price:.2f}"),
                                        submitted_quantity=self.quantity,
                                        time_in_force=TimeInForceType.Day
                                    )
                                    
                                    self.orders[new_order.order_id] = {
                                        'price': new_price,
                                        'side': new_side,
                                        'status': 'OPEN'
                                    }
                                    self.logger.info(f"New order created: {new_side} {new_price:.2f} USD")
                                else:
                                    self.logger.warning(f"Insufficient stock position to place sell order at {new_price:.2f} USD")
                                
                except Exception as e:
                    self.logger.error(f"Failed to check order status: {str(e)}")

                # Check if grid adjustment is needed
                if self.should_adjust_grid():
                    self.logger.info("Starting grid adjustment...")
                    self.cancel_all_orders()
                    self.adjust_grid_parameters()
                    sell_only = self.evaluate_risk()
                    self.place_grid_orders(sell_only=sell_only)
                    self.last_adjustment_time = time.time()
                
                # Answer Telegram messages
                self.answer_telegram()
                   
                time.sleep(10)
            except KeyboardInterrupt:
                self.logger.info("Ctrl+C detected, exiting...")
                self.cancel_all_orders()
                self.send_daily_briefing()
                break
                
            except Exception as e:
                self.logger.error(f"Runtime error: {str(e)}")
                time.sleep(5)


def get_bot():
    # Configuration parameters
    config = {
        'app_key': os.getenv('LB_APP_KEY'),
        'app_secret': os.getenv('LB_APP_SECRET'),
        'access_token': os.getenv('LB_ACCESS_TOKEN'),
        'symbol': os.getenv('TRADE_SYMBOL', 'AAPL.US'),  # Load trading symbol from environment variable, default AAPL.US
        'grid_levels': int(os.getenv('GRID_LEVELS', 10)),  # Load grid levels from environment variable, default 10
        'initial_grid_width': float(os.getenv('INITIAL_GRID_WIDTH', 0.02)),  # Load initial grid width from environment variable, default 2%
        'quantity_per_grid': float(os.getenv('QUANTITY_PER_GRID', 1.0)),  # Quantity to trade per grid
    }
    
    # Create and run the dynamic grid trading bot
    bot = StockGridTrader(**config)
    return bot

if __name__ == "__main__":
    bot = get_bot()
    bot.run()