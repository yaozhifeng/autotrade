import time
from binance.client import Client
from binance.enums import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from pytz import timezone
from dotenv import load_dotenv
import requests
import os
import math

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


class DynamicGridTrader:
    def __init__(self, api_key, api_secret, symbol='LTCUSDT'):
        """
        初始化动态网格交易机器人
        :param api_key: 币安API密钥
        :param api_secret: 币安API密钥
        :param symbol: 交易对
        """
        self.client = Client(api_key, api_secret, testnet=os.getenv('TEST_NET', 'False').lower() == 'true')
        self.symbol = symbol

        # 集中策略参数
        self.strategy = {
            'grid_levels': int(os.getenv('GRID_LEVELS', 20)), # 最多网格数量
            'quantity_per_grid': int(os.getenv('QUANTITY_PER_GRID', 18)), # 每个网格的交易数量
            'grid_gain': float(os.getenv('GRID_GAIN', 0.005)), # 单网格利润, 默认 0.5%
            'max_base_asset_grids': int(os.getenv('MAX_BASE_ASSET_GRIDS', 10)), # 最大持仓网格数量
            'initial_position': float(os.getenv('INITIAL_POSITION', 4)), # 初始持仓网格数量
            'adjustment_factor': 1.0 # 网格间距调整系数
        }

        # 初始化订单和网格状态
        self.orders = {}
        self.last_adjustment_time = None
        self.last_trade_time = None  # Track the time of the last trade
        self.last_briefing_time = time.time()
        self.last_update_id = None #Last telegram update id
        self.last_check_time = None
        self.market_trend = 0
        self.in_bull_market = True # 是否处于牛市
        self.enable_trading = True # 是否开启交易

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dynamic_grid_trader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化每日统计数据
        price = self.get_current_price()
        self.daily_stats = {
            'buy_orders': 0,
            'sell_orders': 0,
            'total_buy_price': 0.0,
            'total_sell_price': 0.0,
            'initial_balance': self.get_total_balance(), # 本周期初始余额
            'initial_price': price, # 本周期初始价格
            'last_price': price, # 上一周期初始价格
            'final_balance': 0.0, # 本周期最终余额
            'final_price': 0.0, # 本周期最终价格
            'highest_price_24h': self.get_24h_highest_price() # 24小时内最高价格
        }
        self.stop_loss_price = self.daily_stats['highest_price_24h'] * float(os.getenv('STOP_LOSS', 0.92))

        # 连续买卖单跟踪
        self.consecutive_buy_orders = 0   # 连续买单数
        self.consecutive_sell_orders = 0  # 连续卖单数
        self.max_consecutive_buy_orders = 0   # 最大连续买单数
        self.max_consecutive_sell_orders = 0  # 最大连续卖单数

    def get_current_price(self):
        """获取当前价格"""
        return float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])

    def get_total_balance(self):
        """获取总余额"""
        usdt_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
        usdt_locked = float(self.client.get_asset_balance(asset='USDT')['locked'])
        base_asset = self.symbol.replace('USDT', '')
        base_asset_balance = float(self.client.get_asset_balance(asset=base_asset)['free'])
        base_asset_locked = float(self.client.get_asset_balance(asset=base_asset)['locked'])
        
        total_balance = usdt_balance + usdt_locked
        current_price = self.get_current_price()
        total_balance += base_asset_balance * current_price
        total_balance += base_asset_locked * current_price
        return total_balance

    def send_daily_briefing(self):
        """发送每日简报"""
        self.daily_stats['final_balance'] = self.get_total_balance()
        self.daily_stats['final_price'] = self.get_current_price()

        avg_buy_price = self.daily_stats['total_buy_price'] / self.daily_stats['buy_orders'] / self.strategy['quantity_per_grid'] if self.daily_stats['buy_orders'] > 0 else 0
        avg_sell_price = self.daily_stats['total_sell_price'] / self.daily_stats['sell_orders'] / self.strategy['quantity_per_grid'] if self.daily_stats['sell_orders'] > 0 else 0

        #gross_margin = (avg_sell_price - avg_buy_price) * self.strategy['quantity_per_grid'] * min(self.daily_stats['sell_orders'], self.daily_stats['buy_orders'])
        gross_margin = self.daily_stats['final_balance'] - self.daily_stats['initial_balance']
        fee = (self.daily_stats['total_buy_price'] + self.daily_stats['total_sell_price']) * float(os.getenv('FEE_RATE', 0.001))
        net_profit = gross_margin - fee

        briefing_msg = (
            f"每日简报:\n"
            f"利润: {net_profit:.2f} USDT\n"
            f"毛利: {gross_margin:.2f} USDT\n"
            f"手续费: {fee:.2f} USDT\n"
            f"买单数量: {self.daily_stats['buy_orders']}\n"
            f"卖单数量: {self.daily_stats['sell_orders']}\n"
            f"平均买入价格: {avg_buy_price:.2f} USDT\n"
            f"平均卖出价格: {avg_sell_price:.2f} USDT\n"
            f"初始价格: {self.daily_stats['initial_price']:.2f} USDT\n"
            f"最终价格: {self.daily_stats['final_price']:.2f} USDT\n"
            f"24小时最高价: {self.daily_stats['highest_price_24h']:.2f} USDT\n"
            f"止损价格: {self.stop_loss_price:.2f} USDT\n"
            f"初始余额: {self.daily_stats['initial_balance']:.2f} USDT\n"
            f"最终余额: {self.daily_stats['final_balance']:.2f} USDT\n"
            f"最大连续买单: {self.max_consecutive_buy_orders}\n"
            f"最大连续卖单: {self.max_consecutive_sell_orders}\n"
            f"统计周期: {datetime.fromtimestamp(self.last_briefing_time).strftime('%Y-%m-%d %H:%M:%S')} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        send_telegram_message(briefing_msg)
        self.logger.info(briefing_msg)

        # 重置每日统计数据
        last_price = self.daily_stats['initial_price']
        self.daily_stats = {
            'buy_orders': 0,
            'sell_orders': 0,
            'total_buy_price': 0.0,
            'total_sell_price': 0.0,
            'last_price': last_price,
            'initial_balance': self.get_total_balance(),
            'initial_price': self.get_current_price(),
            'final_balance': 0.0,
            'final_price': 0.0,
            'highest_price_24h': self.get_24h_highest_price()  # 重新获取真正的24小时最高价
        }
        self.stop_loss_price = self.daily_stats['highest_price_24h'] * float(os.getenv('STOP_LOSS', 0.92))
        self.last_briefing_time = time.time()
        # 重置最大连续买卖单数
        self.max_consecutive_buy_orders = 0
        self.max_consecutive_sell_orders = 0

    def answer_telegram(self):
        """回答Telegram消息"""
        try:
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            url = f"https://api.telegram.org/bot{token}/getUpdates?offset={self.last_update_id}"
            response = requests.get(url)
            data = response.json()
            if data['ok']:
                for message in data['result']:
                    self.last_update_id = message['update_id'] + 1
                    text = message['message']['text']
                    self.logger.info(f"收到Telegram消息: {text}")
                    if text == '/orders': # 查看当前订单
                        self.show_orders()
                    elif text == '/portfolio': # 查看投资组合
                        self.check_portfolio()
                    elif text.startswith('/adjust'): # 调整网格参数
                        factor = float(text.split(' ')[1]) if len(text.split(' ')) > 1 else 1.0
                        self.strategy['adjustment_factor'] = factor
                        self.enable_trading = True
                        self.in_bull_market = self.market_trend >= 0
                        self.cancel_all_orders()
                        self.prepare_position(self.strategy['initial_position'])
                        self.adjust_grid_parameters()
                        self.place_grid_orders()
                    elif text == '/chase': # 手动追高
                        self.chase_grid()
                    elif text == '/close': # 手动平仓
                        self.close_grid()
                    elif text == '/restart': # 重启交易
                        self.restart_trading()
        except Exception as e:
            self.logger.error(f"回答Telegram消息失败: {str(e)}")

    def get_market_data(self, interval='1h'):
        """获取市场数据, 默认间隔1小时，可以传入其他间隔，如15m, 1h, 4h, 1d"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=48) # 获取48小时数据
        
        # 获取K线数据
        klines = self.client.get_historical_klines(
            self.symbol,
            interval,
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # 转换为DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['open'] = pd.to_numeric(df['open'])
        return df

    def calculate_trend(self, df):
        """计算市场趋势"""
        # 使用EMA指标判断趋势
        ema_short = df['close'].ewm(span=7, adjust=False).mean()
        ema_long = df['close'].ewm(span=25, adjust=False).mean()
        
        # 计算趋势强度
        trend_strength = (ema_short - ema_long) / ema_long
        return trend_strength.iloc[-2] # 使用倒数第二个值，因为最后一个值是当前值，

    def calculate_trend_macd(self, df):
        """计算市场趋势"""
        # 使用MACD指标判断趋势，以最后第二个值为准，用最新的值做校验
        macd = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - macd_signal
        trend = float(macd_histogram.iloc[-2]) # 使用倒数第二个值
        latest_trend = float(macd_histogram.iloc[-1]) # 用当前值校验是否趋势没有变化，避免假信号
        self.logger.info(f"MACD柱状值: {trend:.6f}")
        self.logger.info(f"最新MACD柱状值: {latest_trend:.6f}")
        if (trend > 0) and (latest_trend > 0):
            return 1
        elif (trend < 0) and (latest_trend < 0):
            return -1
        else:
            return 0
    
    def get_market_trend(self):
        """获取市场趋势"""
        df = self.get_market_data()
        trend = self.calculate_trend_macd(df) # 使用EMA指标判断趋势, 可以尝试使用MACD指标判断趋势
        self.logger.info(f"市场趋势: {trend:.6f}")
        self.last_check_time = time.time()
        self.market_trend = trend
        return trend

    def get_market_atr(self, interval='15m'):
        """计算 15m K线 的ATR（真实波动幅度）"""
        df = self.get_market_data(interval)
        # Calculate True Range (TR)
        tr = pd.DataFrame()
        tr['h_l'] = df['high'] - df['low']
        tr['h_pc'] = abs(df['high'] - df['close'].shift(1))
        tr['l_pc'] = abs(df['low'] - df['close'].shift(1))
        tr['tr'] = tr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        
        # Calculate ATR using exponential moving average
        atr = tr['tr'].ewm(alpha=1/14, adjust=False).mean()
        return atr.iloc[-2]

    def get_24h_highest_price(self):
        """获取真正的24小时最高价"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            # 获取24小时K线数据
            klines = self.client.get_historical_klines(
                self.symbol,
                Client.KLINE_INTERVAL_1HOUR,
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            if not klines:
                return self.get_current_price()
            
            # 转换为DataFrame并获取最高价
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                             'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
            df['high'] = pd.to_numeric(df['high'])
            
            return float(df['high'].max())
            
        except Exception as e:
            self.logger.error(f"获取24小时最高价失败: {str(e)}")
            return self.daily_stats.get('highest_price_24h', self.get_current_price())

    def adjust_grid_parameters(self):
        """调整网格参数"""
        # 计算当前价格上下各一半网格价格
        try:
            # 使用当前价格作为市场数据
            current_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])
            self.current_grid = []
            
            # 根据系数调整网格间距
            grid_gap = self.strategy['grid_gain'] * current_price  #  获取预设的单网格利润
            atr_gap = self.get_market_atr(os.getenv('ATR_INTERVAL', '15m'))
            self.grid_gap = max(grid_gap, atr_gap) * self.strategy['adjustment_factor']  # 按系数调整网格间距

            grids = self.strategy['grid_levels']//2

            for i in range(grids):
                self.current_grid.append(current_price + self.grid_gap * (i+1))
                self.current_grid.append(current_price - self.grid_gap * (i+1))
            
            profit_per_grid_percent = self.grid_gap/current_price - float(os.getenv('FEE_RATE', 0.001))*2  # 每格利润（扣除手续费）
            profit_per_grid = profit_per_grid_percent * current_price * self.strategy['quantity_per_grid']

            msg = f"调整网格\n"
            msg += f"当前价格: {current_price:.2f} USDT\n"
            msg += f"网格间隔: {self.grid_gap:.2f} USDT\n"
            msg += f"每格利润: {profit_per_grid:.2f} USDT ({profit_per_grid_percent*100:.2f}%)"
            self.logger.info(msg)
            send_telegram_message(msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"调整网格参数失败: {str(e)}")
            return False

    def cancel_all_orders(self):
        """取消所有未成交订单"""
        try:
            open_orders = self.client.get_open_orders(symbol=self.symbol)
            for order in open_orders:
                self.client.cancel_order(
                    symbol=self.symbol,
                    orderId=order['orderId']
                )
            self.orders.clear()
            self.logger.info("所有订单已取消")
        except Exception as e:
            self.logger.error(f"取消订单失败: {str(e)}")

    def place_grid_orders(self):
        """放置网格订单"""
        if self.current_grid is None:
            self.adjust_grid_parameters()
        
        current_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])

        buy_prices = [price for price in self.current_grid if price < current_price]
        sell_prices = [price for price in self.current_grid if price > current_price]
        
        # Get available balances
        base_asset = self.symbol.replace('USDT', '')
        usdt_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
        base_asset_balance = float(self.client.get_asset_balance(asset=base_asset)['free'])

        # Calculate the max buy orders we can place, based on the max_base_asset to avoid buying too much
        max_base_asset = self.strategy['max_base_asset_grids'] * self.strategy['quantity_per_grid']  # Maximum allowed base asset balance
        max_buy_orders = math.ceil((max_base_asset - base_asset_balance) / self.strategy['quantity_per_grid'])  # Maximum number of buy orders we can place

        buy_orders_placed = 0
        sell_orders_placed = 0
        # Check if consecutive buy orders exceed threshold
        if self.consecutive_buy_orders >= 5:
            self.logger.warning(f"连续买单数量({self.consecutive_buy_orders})超过阈值(5)，暂停买单")
            send_telegram_message(f"连续买单数量({self.consecutive_buy_orders})超过阈值(5)，暂停买单")
            buy_prices = []  # Clear buy prices to prevent placing buy orders

        # Place buy orders from highest price to lowest price
        for price in sorted(buy_prices, reverse=True):
            try:
                required_usdt = self.strategy['quantity_per_grid'] * price
                if usdt_balance >= required_usdt and buy_orders_placed < max_buy_orders:
                    # Place buy order
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side=SIDE_BUY,
                        type=ORDER_TYPE_LIMIT,
                        timeInForce=TIME_IN_FORCE_GTC,
                        quantity=self.strategy['quantity_per_grid'],
                        price=f"{price:.2f}"
                        )
                    self.orders[order['orderId']] = {
                        'price': price,
                        'side': 'BUY',
                        'status': 'OPEN'
                        }
                    self.logger.info(f"下单成功: {order['side']} {order['price']} USDT, 数量: {order['origQty']}")
                    usdt_balance -= required_usdt  # Update available USDT balance
                    buy_orders_placed += 1
                else:
                    self.logger.warning(f"买单未放置 {price:.2f} USDT")
            except Exception as e:
                self.logger.error(f"下单失败: {str(e)}")
        # Place sell orders from lowest price to highest price
        for price in sorted(sell_prices):
            try:
                if base_asset_balance >= self.strategy['quantity_per_grid']:
                    # Place sell order
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side=SIDE_SELL,
                        type=ORDER_TYPE_LIMIT,
                        timeInForce=TIME_IN_FORCE_GTC,
                        quantity=self.strategy['quantity_per_grid'],
                        price=f"{price:.2f}"
                        )
                    self.orders[order['orderId']] = {
                        'price': price,
                        'side': 'SELL',
                        'status': 'OPEN'
                        }
                    self.logger.info(f"下单成功: {order['side']} {order['price']} USDT, 数量: {order['origQty']}")
                    base_asset_balance -= self.strategy['quantity_per_grid']  # Update available base asset balance
                    sell_orders_placed += 1
                else:
                    self.logger.warning(f"卖单未放置 {price:.2f} USDT")
            except Exception as e:
                self.logger.error(f"下单失败: {str(e)}")

        lowest_price = min(order_info['price'] for order_info in self.orders.values())
        highest_price = max(order_info['price'] for order_info in self.orders.values())

        self.last_adjustment_time = time.time()

        msg = f"网格订单已放置:\n"
        msg += f"最低价格: {lowest_price:.2f} USDT\n"
        msg += f"最高价格: {highest_price:.2f} USDT\n"
        msg += f"买单数量: {buy_orders_placed}\n"
        msg += f"卖单数量: {sell_orders_placed}"
        self.logger.info(msg)
        send_telegram_message(msg)

    def should_adjust_grid(self):
        """判断是否需要调整网格"""
        if self.current_grid is None or self.last_adjustment_time is None:
            return True
                    
        # 检查价格变化,如果超过当前订单2个网格，就调整网格
        current_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])
        open_orders = self.client.get_open_orders(symbol=self.symbol)
        if len(open_orders) > 0:
            high_price = max([float(order['price']) for order in open_orders])
            low_price = min([float(order['price']) for order in open_orders])
            if current_price - self.grid_gap * 2 > high_price or current_price + self.grid_gap * 2 < low_price:
                return True
            
        return False
    
    def get_sell_order_count(self):
        """获取卖单数量"""
        sell_orders = [order for order in self.orders.values() if order['side'] == 'SELL']
        return len(sell_orders)
    
    def get_buy_order_count(self):
        """获取买单数量"""
        buy_orders = [order for order in self.orders.values() if order['side'] == 'BUY']
        return len(buy_orders)

    def show_orders(self):
        """显示当前订单"""
        open_orders = self.client.get_open_orders(symbol=self.symbol)
        sorted_orders = sorted(open_orders, key=lambda x: float(x['price']), reverse=True)
        for order in sorted_orders:
            self.logger.info(f"订单: {order['side']} {order['price']} USDT, 数量: {order['origQty']}")
            send_telegram_message(f"订单: {order['side']} {order['price']} USDT, 数量: {order['origQty']}")

    def check_portfolio(self):
        """检查投资组合"""
        usdt_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
        usdt_locked = float(self.client.get_asset_balance(asset='USDT')['locked'])
        base_asset = self.symbol.replace('USDT', '')
        base_asset_balance = float(self.client.get_asset_balance(asset=base_asset)['free'])
        base_asset_locked = float(self.client.get_asset_balance(asset=base_asset)['locked'])
        
        self.logger.info(f"USDT: {usdt_balance:.2f} (Free), {usdt_locked:.2f} (Locked)")
        send_telegram_message(f"USDT: {usdt_balance:.2f} (Free), {usdt_locked:.2f} (Locked)")
        self.logger.info(f"{base_asset}: {base_asset_balance:.6f} (Free), {base_asset_locked:.6f} (Locked)")
        send_telegram_message(f"{base_asset}: {base_asset_balance:.6f} (Free), {base_asset_locked:.6f} (Locked)")

        # calculate total balance in USDT, with base asset converted to USDT
        total_balance = usdt_balance + usdt_locked
        current_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])
        total_balance += base_asset_balance * current_price
        total_balance += base_asset_locked * current_price
        self.logger.info(f"Total balance: {total_balance:.2f} USDT or {total_balance/current_price:.8f} in {base_asset}")
        send_telegram_message(f"Total balance: {total_balance:.2f} USDT or {total_balance/current_price:.8f} in {base_asset}")

    def close_position(self, keep_grids=0):
        """平仓函数：全部卖掉"""
        # keep_grids 保留的网格数量
        base_asset = self.symbol.replace('USDT', '')
        base_asset_balance = float(self.client.get_asset_balance(asset=base_asset)['free'])
        sell_amount = (base_asset_balance - keep_grids * self.strategy['quantity_per_grid']) * 0.99 # 保留1%，避免手续费问题
        
        if sell_amount <= 0.01: # 如果卖出数量小于0.01，直接返回
            msg = f"无需平仓，当前{base_asset}持仓({base_asset_balance:.6f})"
            self.logger.info(msg)
            send_telegram_message(msg)
            return
        
        try:
            # 市价单全部卖出多余部分
            order = self.client.create_order(
                symbol=self.symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=round(sell_amount, 2)  # 保留2位小数，防止精度问题
            )
            # Get the actual executed price from the order fills
            executed_price = float(order['fills'][0]['price'])
            msg = f"已平仓，卖出 {sell_amount:.2f} {base_asset}，价格: {executed_price:.2f} USDT"
            self.logger.info(msg)
            send_telegram_message(msg)
        except Exception as e:
            msg = f"平仓失败: {str(e)}"
            self.logger.error(msg)
            send_telegram_message(msg)

    def prepare_position(self, grid_count=2):
        """准备仓位：市价买入指定网格数量的现货
        
        Args:
            grid_count: 需要准备的网格数量，默认2个网格
        """
        base_asset = self.symbol.replace('USDT', '')
        base_asset_balance = float(self.client.get_asset_balance(asset=base_asset)['free'])
        target_amount = self.strategy['quantity_per_grid'] * grid_count
        
        if base_asset_balance >= target_amount:
            msg = f"无需买入，当前{base_asset}持仓({base_asset_balance:.6f})已满足{grid_count}个网格({target_amount:.6f})"
            self.logger.info(msg)
            send_telegram_message(msg)
            return
        
        buy_amount = target_amount - base_asset_balance
        
        try:
            # 检查USDT余额是否足够
            current_price = self.get_current_price()
            required_usdt = buy_amount * current_price
            usdt_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
            
            if usdt_balance < required_usdt: #calculate how much we can afford
                buy_amount = usdt_balance / current_price
                msg = f"USDT余额不足，需要{required_usdt:.2f} USDT，当前余额{usdt_balance:.2f} USDT，只能买入{buy_amount:.2f} {base_asset}"
                self.logger.error(msg)
                send_telegram_message(msg)
                
            # 市价单买入
            order = self.client.create_order(
                symbol=self.symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=math.ceil(buy_amount * 100) / 100  # 保留2位小数，向上进位
            )
            # Get the actual executed price from the order fills
            executed_price = float(order['fills'][0]['price'])
            msg = f"已买入 {buy_amount:.2f} {base_asset}，总持仓达到 {target_amount:.2f} {base_asset}，价格: {executed_price:.2f} USDT"
            self.logger.info(msg)
            send_telegram_message(msg)
        except Exception as e:
            msg = f"买入失败: {str(e)}"
            self.logger.error(msg)
            send_telegram_message(msg)

    def chase_grid(self):
        """追高，买入两个网格的现货，并重新调整网络"""
        try:
            self.cancel_all_orders() # 取消所有未成交订单
            self.prepare_position(self.strategy['initial_position']) 
            # 调整网格参数
            self.adjust_grid_parameters()
            self.place_grid_orders()
            msg = f"追高成功，当前价格: {self.get_current_price():.2f} USDT"
            self.logger.info(msg)
            send_telegram_message(msg)
        except Exception as e:
            self.logger.error(f"追高失败: {str(e)}")
            send_telegram_message(f"追高失败: {str(e)}")

    def close_grid(self):
        """平掉当前网格的现货，停止交易"""
        try:
            self.enable_trading = False
            self.in_bull_market = False
            self.cancel_all_orders()
            self.close_position()
            self.send_daily_briefing()
        except Exception as e:
            self.logger.error(f"平掉当前网格的现货失败: {str(e)}")
            send_telegram_message(f"平掉当前网格的现货失败: {str(e)}")

    def process_grid(self):
        """处理网格，检查订单状态并更新"""
        for order_id, order_info in list(self.orders.items()):
            try:
                order_status = self.client.get_order(
                    symbol=self.symbol,
                    orderId=order_id
                )
                
                if order_status['status'] == 'FILLED':
                    self.logger.info(f"订单已成交: {order_info['side']} {order_info['price']:.2f} USDT")
                    send_telegram_message(f"订单已成交: {order_info['side']} {order_info['price']:.2f} USDT")
                    self.orders.pop(order_id)
                    self.last_trade_time = time.time()  # Update the last trade time
                    
                    # Update daily stats
                    if order_info['side'] == 'BUY':
                        self.daily_stats['buy_orders'] += 1
                        self.daily_stats['total_buy_price'] += order_info['price'] * self.strategy['quantity_per_grid']
                        # 连续买单+1，卖单归零
                        self.consecutive_buy_orders += 1
                        self.consecutive_sell_orders = 0
                        if self.consecutive_buy_orders > self.max_consecutive_buy_orders:
                            self.max_consecutive_buy_orders = self.consecutive_buy_orders
                    else:
                        self.daily_stats['sell_orders'] += 1
                        self.daily_stats['total_sell_price'] += order_info['price'] * self.strategy['quantity_per_grid']
                        # 连续卖单+1，买单归零
                        self.consecutive_sell_orders += 1
                        self.consecutive_buy_orders = 0
                        if self.consecutive_sell_orders > self.max_consecutive_sell_orders:
                            self.max_consecutive_sell_orders = self.consecutive_sell_orders
                    
                    # 放置反向订单
                    new_side = 'SELL' if order_info['side'] == 'BUY' else 'BUY'
                    new_price = order_info['price'] - self.grid_gap if new_side == 'BUY' else order_info['price'] + self.grid_gap
                    
                    # Check balance before placing reverse order
                    if new_side == 'BUY':
                        usdt_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
                        required_usdt = self.strategy['quantity_per_grid'] * new_price
                        if usdt_balance >= required_usdt:
                            new_order = self.client.create_order(
                                symbol=self.symbol,
                                side=new_side,
                                type=ORDER_TYPE_LIMIT,
                                timeInForce=TIME_IN_FORCE_GTC,
                                quantity=self.strategy['quantity_per_grid'],
                                price=f"{new_price:.2f}"
                            )
                            self.orders[new_order['orderId']] = {
                                'price': new_price,
                                'side': new_side,
                                'status': 'OPEN'
                            }
                            self.logger.info(f"新订单已创建: {new_side} {new_price:.2f} USDT")
                        else:
                            self.logger.warning(f"Insufficient USDT balance to place buy order at {new_price:.2f} USDT")
                    else:
                        base_asset = self.symbol.replace('USDT', '')
                        base_asset_balance = float(self.client.get_asset_balance(asset=base_asset)['free'])
                        if base_asset_balance >= self.strategy['quantity_per_grid']:
                            new_order = self.client.create_order(
                                symbol=self.symbol,
                                side=new_side,
                                type=ORDER_TYPE_LIMIT,
                                timeInForce=TIME_IN_FORCE_GTC,
                                quantity=self.strategy['quantity_per_grid'],
                                price=f"{new_price:.2f}"
                            )
                            self.orders[new_order['orderId']] = {
                                'price': new_price,
                                'side': new_side,
                                'status': 'OPEN'
                            }
                            self.logger.info(f"新订单已创建: {new_side} {new_price:.2f} USDT")
                        else:
                            self.logger.warning(f"Insufficient {base_asset} balance to place sell order at {new_price:.2f} USDT")
                    
            except Exception as e:
                self.logger.error(f"检查订单状态失败: {str(e)}")

    def calc_drawdown(self, N):
        # 计算N个网格的回撤
        Q = self.strategy['quantity_per_grid']
        delta_P = self.grid_gap
        return Q * delta_P * N * (N - 1) / 2

    def restart_trading(self):
        """重启交易"""
        self.logger.info("重启交易")
        send_telegram_message("重启交易")
        self.enable_trading = True

        market_trend = self.get_market_trend()
        self.in_bull_market = market_trend >= 0
        self.last_check_time = time.time()
        self.last_briefing_time = time.time()
        self.max_consecutive_buy_orders = 0
        self.max_consecutive_sell_orders = 0
        self.consecutive_buy_orders = 0
        self.consecutive_sell_orders = 0

        price = self.get_current_price()
        self.daily_stats = {
            'buy_orders': 0,
            'sell_orders': 0,
            'total_buy_price': 0.0,
            'total_sell_price': 0.0,
            'initial_balance': self.get_total_balance(), # 本周期初始余额
            'initial_price': price, # 本周期初始价格
            'last_price': price, # 上一周期初始价格
            'final_balance': 0.0, # 本周期最终余额
            'final_price': 0.0, # 本周期最终价格
            'highest_price_24h': self.get_24h_highest_price() # 获取真正的24小时最高价格
        }
        self.stop_loss_price = self.daily_stats['highest_price_24h'] * float(os.getenv('STOP_LOSS', 0.92))
        
        self.cancel_all_orders()
        self.prepare_position(self.strategy['initial_position'])
        self.adjust_grid_parameters()
        self.place_grid_orders()

    def check_stop_loss(self):
        """检查是否要止损"""
        if self.enable_trading:
            current_price = self.get_current_price()
            if current_price < self.stop_loss_price:
                self.in_bull_market = False # 标记熊市
                self.logger.info("价格下跌超过止损比例，停止交易，等待市场回升")
                send_telegram_message(f"价格下跌超过止损比例，停止交易，当前价格: {current_price:.2f} USDT")
                self.enable_trading = False # 停止交易
                self.cancel_all_orders()
                self.close_position()
                send_telegram_message(f"已停止交易，需手动恢复！")

    def run(self):
        """运行动态网格交易机器人"""
        self.logger.info("启动动态网格交易机器人...")
        self.logger.info(f"交易对: {self.symbol}")
        send_telegram_message(f"启动动态网格交易机器人... 交易对: {self.symbol}")

        self.cancel_all_orders() # 取消可能遗留的所有未成交订单

        # 初始化网格参数
        market_trend = self.get_market_trend()
        self.in_bull_market = market_trend >= 0
        self.enable_trading = True
        self.prepare_position(self.strategy['initial_position']) # 初始仓位
        self.adjust_grid_parameters()
        self.place_grid_orders()

        while True:
            try:
                # 检查是否要止损
                self.check_stop_loss()
                
                # 检查订单状态
                self.process_grid()
                
                # 回答Telegram消息
                self.answer_telegram()

                if self.enable_trading:
                    if self.get_sell_order_count() == 0:
                        # 判断要不要追高, 如果当前没有卖单，不管牛熊，都可以追高
                        self.logger.info("卖单耗尽，追高")
                        send_telegram_message("卖单耗尽，追高")
                        self.cancel_all_orders()
                        self.prepare_position(2) # 及时追高，只买入2个网格
                        self.adjust_grid_parameters()
                        self.place_grid_orders()

                    # 每半小时检查市场趋势，看是否需要平仓止损，或调整交易规则
                    if self.last_check_time is None or time.time() - self.last_check_time >= 1800:
                        market_trend = self.get_market_trend()
                    
                        # 更新24小时最高价格
                        self.daily_stats['highest_price_24h'] = self.get_24h_highest_price()
                        self.stop_loss_price = self.daily_stats['highest_price_24h'] * float(os.getenv('STOP_LOSS', 0.92))
                        self.logger.info(f"24小时最高价: {self.daily_stats['highest_price_24h']:.2f} USDT, 止损价格: {self.stop_loss_price:.2f} USDT")
                    
                        # 追低检查每半小时进行一次
                        if self.get_buy_order_count() == 0 and self.should_adjust_grid():
                            # 买单耗尽，且需要调整网格(超过 2 个网格)，则追低
                            self.logger.info("买单耗尽，追低")
                            send_telegram_message("买单耗尽，追低")
                            self.cancel_all_orders()
                            self.close_position(8) # 平仓保留8个网格, 只加 2 个买单网格
                            self.adjust_grid_parameters()
                            self.place_grid_orders()
                        # 市场趋势判断，暂时不调整交易策略
                        if market_trend > 0: # 如果市场趋势向上，牛市交易
                            if not self.in_bull_market: # 之前是熊市，转到牛市交易规则
                                self.in_bull_market = True # 标记牛市
                                self.logger.info("熊转牛，继续交易")
                                send_telegram_message("熊转牛，继续交易")
                        elif market_trend < 0: # 如果市场趋势向下，平仓保留少量网格，继续交易
                            if self.in_bull_market: #刚从牛市转换到熊市
                                self.in_bull_market = False # 标记熊市
                                self.logger.info("牛转熊，继续交易")
                                send_telegram_message("牛转熊，继续交易")


                    # 检查是否需要发送每日简报
                    briefing_interval = int(os.getenv('BRIEFING_INTERVAL', 86400))  # Default to 24 hours
                    if time.time() - self.last_briefing_time >= briefing_interval:
                        self.send_daily_briefing()
                    
                time.sleep(10)
            except KeyboardInterrupt:
                self.logger.info("检测到Ctrl+C，正在退出...")
                self.cancel_all_orders()
                self.send_daily_briefing()
                break
                
            except Exception as e:
                self.logger.error(f"运行错误: {str(e)}")
                time.sleep(5)


def get_bot():
    # 配置参数
    config = {
        'api_key': os.getenv('API_KEY'),
        'api_secret': os.getenv('API_SECRET'),
        'symbol': os.getenv('TRADE_SYMBOL', 'LTCUSDT')  # 从环境变量加载交易对，默认LTCUSDT
    }
    
    # 创建并运行动态网格交易机器人
    bot = DynamicGridTrader(**config)
    return bot

if __name__ == "__main__":
    bot = get_bot()
    bot.run()