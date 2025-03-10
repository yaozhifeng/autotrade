import time
import numpy as np
import pandas as pd
import logging
import requests
import os
from datetime import datetime, timedelta
from pytz import timezone
from dotenv import load_dotenv

# Longbridge SDK imports
from longbridge.openapi import QuoteContext, TradeContext, Config
from longbridge.openapi import OrderSide, OrderType, TimeInForceType, OrderStatus

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
    def __init__(self, app_key, app_secret, access_token, 
                 symbol='AAPL.US',
                 grid_levels=10, initial_grid_width=0.01,
                 quantity_per_grid=1):
        """
        Initialize dynamic grid trader for stocks
        :param app_key: Longbridge App Key
        :param app_secret: Longbridge App Secret
        :param access_token: Longbridge Access Token
        :param symbol: Stock symbol (with exchange suffix)
        :param grid_levels: Number of grid levels
        :param initial_grid_width: Initial grid width as percentage
        :param quantity_per_grid: Quantity of shares per grid
        """
        # Configure Longbridge clients
        self.config = Config(app_key=app_key, app_secret=app_secret, access_token=access_token)
        self.quote_ctx = QuoteContext(self.config)
        self.trade_ctx = TradeContext(self.config)
        
        self.symbol = symbol
        self.grid_levels = grid_levels
        self.initial_grid_width = initial_grid_width
        self.quantity = quantity_per_grid
        self.last_update_id = None  # Last telegram update id
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('stock_grid_trader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize order and grid state
        self.orders = {}
        self.current_grid = None
        self.last_adjustment_time = None
        self.last_trade_time = None  # Track the time of the last trade
        
        # Set adjustment parameters
        self.min_adjustment_interval = int(os.getenv('MIN_ADJUSTMENT_INTERVAL', 3600))  # Minimum adjustment interval (seconds)

        # Market hours configuration (default US market: 9:30AM - 4:00PM Eastern)
        self.market_tz = timezone('US/Eastern')
        self.market_open_hour = int(os.getenv('MARKET_OPEN_HOUR', 9))
        self.market_open_minute = int(os.getenv('MARKET_OPEN_MINUTE', 30))
        self.market_close_hour = int(os.getenv('MARKET_CLOSE_HOUR', 16))
        self.market_close_minute = int(os.getenv('MARKET_CLOSE_MINUTE', 0))

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

    def is_market_open(self):
        """Check if the market is currently open"""
        now = datetime.now(self.market_tz)
        market_open = now.replace(hour=self.market_open_hour, minute=self.market_open_minute, second=0)
        market_close = now.replace(hour=self.market_close_hour, minute=self.market_close_minute, second=0)
        
        # Check if it's a weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Check if within trading hours
        if market_open <= now <= market_close:
            return True
        
        return False

    def get_current_price(self):
        """Get current price of the stock"""
        try:
            quote = self.quote_ctx.quote(symbols=[self.symbol])
            return float(quote[0].last_done)
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            # Return last known price or raise exception
            raise

    def get_total_balance(self):
        """Get total balance (cash + stock value)"""
        try:
            # Get cash balance
            account_balance = self.trade_ctx.account_balance()
            cash_balance = float(account_balance.net_cash_balance)
            
            # Get stock positions
            positions = self.trade_ctx.positions()
            stock_value = 0.0
            
            for position in positions:
                if position.symbol == self.symbol:
                    current_price = self.get_current_price()
                    stock_value += position.quantity * current_price
            
            total_balance = cash_balance + stock_value
            return total_balance
        except Exception as e:
            self.logger.error(f"Error getting total balance: {str(e)}")
            return 0.0

    def send_daily_briefing(self):
        """Send daily performance report"""
        self.daily_stats['final_balance'] = self.get_total_balance()
        self.daily_stats['final_price'] = self.get_current_price()

        avg_buy_price = self.daily_stats['total_buy_price'] / self.daily_stats['buy_orders'] if self.daily_stats['buy_orders'] > 0 else 0
        avg_sell_price = self.daily_stats['total_sell_price'] / self.daily_stats['sell_orders'] if self.daily_stats['sell_orders'] > 0 else 0

        gross_margin = (avg_sell_price - avg_buy_price) * min(self.daily_stats['sell_orders'], self.daily_stats['buy_orders'])
        fee = (self.daily_stats['total_buy_price'] + self.daily_stats['total_sell_price']) * float(os.getenv('FEE_RATE', 0.0025))
        net_profit = gross_margin - fee

        # Format the briefing message
        briefing_msg = (
            f"Daily Report:\n"
            f"Gross Profit: ${gross_margin:.2f}\n"
            f"Fees: ${fee:.2f}\n"
            f"Net Profit: ${net_profit:.2f}\n"
            f"Buy Orders: {self.daily_stats['buy_orders']}\n"
            f"Sell Orders: {self.daily_stats['sell_orders']}\n"
            f"Avg Buy Price: ${avg_buy_price:.2f}\n"
            f"Avg Sell Price: ${avg_sell_price:.2f}\n"
            f"Initial Balance: ${self.daily_stats['initial_balance']:.2f}\n"
            f"Final Balance: ${self.daily_stats['final_balance']:.2f}\n"
            f"Initial Price: ${self.daily_stats['initial_price']:.2f}\n"
            f"Final Price: ${self.daily_stats['final_price']:.2f}\n"
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
                    self.logger.info(f"Telegram message received: {text}")
                    if text == '/orders':
                        self.show_orders()
                    elif text == '/portfolio':
                        self.check_portfolio()
                    elif text == '/adjust':
                        self.cancel_all_orders()
                        self.current_grid = None
                        self.last_adjustment_time = None
                    elif text == '/status':
                        self.send_status_update()
        except Exception as e:
            self.logger.error(f"Error responding to Telegram: {str(e)}")

    def send_status_update(self):
        """Send current trading status via Telegram"""
        try:
            current_price = self.get_current_price()
            total_balance = self.get_total_balance()
            market_status = "OPEN" if self.is_market_open() else "CLOSED"
            
            status_msg = (
                f"Status Update:\n"
                f"Market: {market_status}\n"
                f"Current Price: ${current_price:.2f}\n"
                f"Current Balance: ${total_balance:.2f}\n"
                f"Open Orders: {len(self.orders)}\n"
                f"Last Adjustment: {datetime.fromtimestamp(self.last_adjustment_time).strftime('%Y-%m-%d %H:%M:%S') if self.last_adjustment_time else 'Never'}"
            )
            send_telegram_message(status_msg)
            self.logger.info(status_msg)
        except Exception as e:
            self.logger.error(f"Error sending status update: {str(e)}")

    def adjust_grid_parameters(self):
        """Adjust grid parameters based on current price"""
        try:
            # Use current price for market data
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

            # Calculate potential profit per grid
            profit_per_grid_percent = (self.initial_grid_width/(self.grid_levels-1) - float(os.getenv('FEE_RATE', 0.0025))*2) * 100

            # Format the adjustment message
            msg = f"Grid parameters adjusted:\n"
            msg += f"Current Price: ${current_price:.2f}\n"
            msg += f"Grid Range: ${lower_price:.2f} - ${upper_price:.2f}\n"
            msg += f"Grid Levels: {self.grid_levels}\n"
            msg += f"Grid Gap: ${self.grid_gap:.2f}\n"
            msg += f"Profit Per Grid: {profit_per_grid_percent:.2f}%"
            self.logger.info(msg)
            send_telegram_message(msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adjusting grid parameters: {str(e)}")
            return False

    def cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            # Get all open orders
            open_orders = self.trade_ctx.today_orders()
            
            for order in open_orders:
                if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                    self.trade_ctx.cancel_order(order_id=order.order_id)
            
            self.orders.clear()
            self.logger.info("All orders cancelled")
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {str(e)}")

    def place_grid_orders(self, buy_only=False, sell_only=False):
        """Place grid orders based on current parameters"""
        if not self.is_market_open():
            self.logger.info("Market is closed. No orders will be placed.")
            send_telegram_message("Market is closed. No orders will be placed.")
            return
            
        if self.current_grid is None:
            self.adjust_grid_parameters()
        
        current_price = self.get_current_price()
        
        # Get available balances
        account_balance = self.trade_ctx.account_balance()
        cash_balance = float(account_balance.cash_balance)
        
        # Get current position
        stock_quantity = 0
        positions = self.trade_ctx.positions()
        for position in positions:
            if position.symbol == self.symbol:
                stock_quantity = position.quantity
                break
        
        buy_orders_placed = 0
        sell_orders_placed = 0
        
        for price in self.current_grid:
            try:
                price_formatted = round(price, 2)  # Format to 2 decimal places
                
                if price < current_price and not sell_only:
                    # Check if there is enough cash to place a buy order
                    required_cash = self.quantity * price_formatted
                    if cash_balance >= required_cash:
                        # Place buy order
                        order_result = self.trade_ctx.submit_order(
                            symbol=self.symbol,
                            order_type=OrderType.LO,  # Limit Order
                            side=OrderSide.BUY,
                            submitted_price=price_formatted,
                            submitted_quantity=self.quantity,
                            time_in_force=TimeInForceType.DAY
                        )
                        
                        self.orders[order_result.order_id] = {
                            'price': price_formatted,
                            'side': 'BUY',
                            'status': 'OPEN'
                        }
                        self.logger.info(f"Order placed: BUY ${price_formatted}, Quantity: {self.quantity}")
                        cash_balance -= required_cash  # Update available cash
                        buy_orders_placed += 1
                    else:
                        self.logger.warning(f"Insufficient cash balance to place buy order at ${price_formatted}")
                
                elif price > current_price and not buy_only:
                    # Check if there are enough shares to place a sell order
                    if stock_quantity >= self.quantity:
                        # Place sell order
                        order_result = self.trade_ctx.submit_order(
                            symbol=self.symbol,
                            order_type=OrderType.LO,  # Limit Order
                            side=OrderSide.SELL,
                            submitted_price=price_formatted,
                            submitted_quantity=self.quantity,
                            time_in_force=TimeInForceType.DAY
                        )
                        
                        self.orders[order_result.order_id] = {
                            'price': price_formatted,
                            'side': 'SELL',
                            'status': 'OPEN'
                        }
                        self.logger.info(f"Order placed: SELL ${price_formatted}, Quantity: {self.quantity}")
                        stock_quantity -= self.quantity  # Update available shares
                        sell_orders_placed += 1
                    else:
                        self.logger.warning(f"Insufficient stock quantity to place sell order at ${price_formatted}")
                                
            except Exception as e:
                self.logger.error(f"Error placing order: {str(e)}")
        
        summary = f"Orders placed: {buy_orders_placed} buy orders, {sell_orders_placed} sell orders"
        self.logger.info(summary)
        send_telegram_message(summary)

    def should_adjust_grid(self):
        """Determine if grid needs adjustment"""
        if self.current_grid is None or self.last_adjustment_time is None:
            return True
            
        if not self.is_market_open():
            return False
            
        current_time = time.time()
        current_price = self.get_current_price()
        
        # Check time interval
        if current_time - self.last_adjustment_time < self.min_adjustment_interval:
            return False
        
        # Check if price has moved significantly compared to order placement range
        orders = self.trade_ctx.today_orders()
        active_orders = [order for order in orders if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]]
        
        if active_orders:
            high_price = max([order.submitted_price for order in active_orders])
            low_price = min([order.submitted_price for order in active_orders])
            
            # If price has moved out of range plus buffer, adjust the grid
            if current_price - self.grid_gap * 2 > high_price or current_price + self.grid_gap * 2 < low_price:
                return True
        else:
            # If no active orders, adjust grid every 5 minutes
            if current_time - self.last_adjustment_time >= 300:
                return True
            
        return False

    def show_orders(self):
        """Display current orders"""
        try:
            orders = self.trade_ctx.today_orders()
            active_orders = [order for order in orders if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]]
            
            if not active_orders:
                msg = "No active orders"
                self.logger.info(msg)
                send_telegram_message(msg)
                return
                
            sorted_orders = sorted(active_orders, key=lambda x: x.submitted_price, reverse=True)
            messages = []
            
            for order in sorted_orders:
                side = "BUY" if order.side == OrderSide.BUY else "SELL"
                messages.append(f"Order: {side} ${order.submitted_price:.2f}, Quantity: {order.submitted_quantity}")
            
            full_message = "\n".join(messages)
            self.logger.info(full_message)
            send_telegram_message(full_message)
        except Exception as e:
            self.logger.error(f"Error showing orders: {str(e)}")

    def check_portfolio(self):
        """Check portfolio status"""
        try:
            # Get cash balance
            account_balance = self.trade_ctx.account_balance()
            cash_balance = float(account_balance.cash_balance)
            
            # Get stock positions
            positions = self.trade_ctx.positions()
            stock_quantity = 0
            stock_value = 0
            
            for position in positions:
                if position.symbol == self.symbol:
                    stock_quantity = position.quantity
                    stock_value = position.market_value
                    break
                    
            current_price = self.get_current_price()
            
            # Calculate total portfolio value
            total_value = cash_balance + stock_value
            
            # Format the portfolio message
            msg = (
                f"Portfolio Status:\n"
                f"Cash: ${cash_balance:.2f}\n"
                f"Stock: {stock_quantity} shares of {self.symbol}\n"
                f"Stock Value: ${stock_value:.2f}\n"
                f"Current Price: ${current_price:.2f}\n"
                f"Total Value: ${total_value:.2f}"
            )
            self.logger.info(msg)
            send_telegram_message(msg)
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio: {str(e)}")

    def evaluate_risk(self):
        """Evaluate risk and determine if we should limit trading to sell orders"""
        try:
            # Check portfolio to see if we're holding too much stock
            positions = self.trade_ctx.positions()
            stock_quantity = 0
            
            for position in positions:
                if position.symbol == self.symbol:
                    stock_quantity = position.quantity
                    break
                    
            max_stock_quantity = int(os.getenv('MAX_STOCK_QUANTITY', 100))  # Maximum allowed stock quantity
            
            sell_only = False
            
            if stock_quantity > max_stock_quantity:
                msg = f"Stock holdings ({stock_quantity} shares) exceed maximum allowed ({max_stock_quantity}). Sell orders only."
                self.logger.warning(msg)
                send_telegram_message(msg)
                sell_only = True

            return sell_only
            
        except Exception as e:
            self.logger.error(f"Error evaluating risk: {str(e)}")
            return False
    
    def run(self):
        """Run the dynamic grid trading bot"""
        self.logger.info("Starting dynamic grid trading bot...")
        self.logger.info(f"Trading symbol: {self.symbol}")
        send_telegram_message(f"Grid trader started for {self.symbol}")
        
        while True:
            try:
                # Skip processing during market closed hours
                if not self.is_market_open():
                    time.sleep(60)  # Check every minute if market is open
                    continue
                
                # Check order status
                orders = self.trade_ctx.today_orders()
                
                for order in orders:
                    order_id = order.order_id
                    
                    if order_id in self.orders and order.status == OrderStatus.FILLED:
                        order_info = self.orders[order_id]
                        self.logger.info(f"Order filled: {order_info['side']} ${order_info['price']:.2f}")
                        send_telegram_message(f"Order filled: {order_info['side']} ${order_info['price']:.2f}")
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
                        new_price = round(new_price, 2)  # Format to 2 decimal places
                        
                        try:
                            # Check balance before placing reverse order
                            if new_side == 'BUY':
                                account_balance = self.trade_ctx.account_balance()
                                cash_balance = float(account_balance.cash_balance)
                                required_cash = self.quantity * new_price
                                
                                if cash_balance >= required_cash:
                                    order_result = self.trade_ctx.submit_order(
                                        symbol=self.symbol,
                                        order_type=OrderType.LO,
                                        side=OrderSide.BUY,
                                        submitted_price=new_price,
                                        submitted_quantity=self.quantity,
                                        time_in_force=TimeInForceType.DAY
                                    )
                                    
                                    self.orders[order_result.order_id] = {
                                        'price': new_price,
                                        'side': new_side,
                                        'status': 'OPEN'
                                    }
                                    self.logger.info(f"New order created: {new_side} ${new_price:.2f}")
                                else:
                                    self.logger.warning(f"Insufficient cash balance to place buy order at ${new_price:.2f}")
                            else:
                                positions = self.trade_ctx.positions()
                                stock_quantity = 0
                                
                                for position in positions:
                                    if position.symbol == self.symbol:
                                        stock_quantity = position.quantity
                                        break
                                        
                                if stock_quantity >= self.quantity:
                                    order_result = self.trade_ctx.submit_order(
                                        symbol=self.symbol,
                                        order_type=OrderType.LO,
                                        side=OrderSide.SELL,
                                        submitted_price=new_price,
                                        submitted_quantity=self.quantity,
                                        time_in_force=TimeInForceType.DAY
                                    )
                                    
                                    self.orders[order_result.order_id] = {
                                        'price': new_price,
                                        'side': new_side,
                                        'status': 'OPEN'
                                    }
                                    self.logger.info(f"New order created: {new_side} ${new_price:.2f}")
                                else:
                                    self.logger.warning(f"Insufficient stock quantity to place sell order at ${new_price:.2f}")
                        except Exception as e:
                            self.logger.error(f"Error placing reverse order: {str(e)}")

                # Check if grid needs adjustment
                if self.should_adjust_grid():
                    self.logger.info("Adjusting grid...")
                    self.cancel_all_orders()
                    self.adjust_grid_parameters()
                    sell_only = self.evaluate_risk()
                    self.place_grid_orders(sell_only=sell_only)
                    self.last_adjustment_time = time.time()
                
                # Answer Telegram messages
                self.answer_telegram()

                # Check if daily briefing is due
                briefing_interval = int(os.getenv('BRIEFING_INTERVAL', 86400))  # Default to 24 hours
                if time.time() - self.last_briefing_time >= briefing_interval:
                    self.send_daily_briefing()
                    
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
        'symbol': os.getenv('TRADE_SYMBOL', 'AAPL.US'),  # Default to AAPL
        'grid_levels': int(os.getenv('GRID_LEVELS', 10)),  # Default 10 grid levels
        'initial_grid_width': float(os.getenv('INITIAL_GRID_WIDTH', 0.01)),  # Default 1% grid width
        'quantity_per_grid': int(os.getenv('QUANTITY_PER_GRID', 1)),  # Default 1 share per grid
    }
    
    # Create and return the stock grid trading bot
    bot = StockGridTrader(**config)
    return bot

if __name__ == "__main__":
    bot = get_bot()
    bot.run()