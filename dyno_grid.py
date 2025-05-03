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
    def __init__(self, api_key, api_secret, symbol='LTCUSDT',
                 grid_levels=10, initial_grid_width=0.1,
                 quantity_per_grid=0.1,
                 short_period=7,
                 long_period=25):
        """
        初始化动态网格交易机器人
        :param api_key: 币安API密钥
        :param api_secret: 币安API密钥
        :param symbol: 交易对
        :param grid_levels: 网格数量
        :param initial_grid_width: 初始网格宽度（百分比）
        :param quantity_per_grid: 每个网格的交易数量
        """
        self.client = Client(api_key, api_secret, testnet=os.getenv('TEST_NET', 'False').lower() == 'true')
        self.symbol = symbol
        self.grid_levels = grid_levels
        self.initial_grid_width = initial_grid_width
        self.quantity = quantity_per_grid
        self.last_update_id = None #Last telegram update id
        self.adjustment_factor = 1.0
        self.enable_trading = True
        self.last_check_time = None
        self.short_period = short_period
        self.long_period = long_period
        
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
        
        # 初始化订单和网格状态
        self.orders = {}
        self.current_grid = None
        self.last_adjustment_time = None
        self.last_trade_time = None  # Track the time of the last trade
        
        # 初始化每日统计数据
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

        avg_buy_price = self.daily_stats['total_buy_price'] / self.daily_stats['buy_orders'] / self.quantity if self.daily_stats['buy_orders'] > 0 else 0
        avg_sell_price = self.daily_stats['total_sell_price'] / self.daily_stats['sell_orders'] / self.quantity if self.daily_stats['sell_orders'] > 0 else 0

        gross_margin = (avg_sell_price - avg_buy_price) * self.quantity * min(self.daily_stats['sell_orders'], self.daily_stats['buy_orders'])
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
            f"初始余额: {self.daily_stats['initial_balance']:.2f} USDT\n"
            f"最终余额: {self.daily_stats['final_balance']:.2f} USDT\n"
            f"初始价格: {self.daily_stats['initial_price']:.2f} USDT\n"
            f"最终价格: {self.daily_stats['final_price']:.2f} USDT\n"
            f"统计周期: {datetime.fromtimestamp(self.last_briefing_time).strftime('%Y-%m-%d %H:%M:%S')} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        send_telegram_message(briefing_msg)
        self.logger.info(briefing_msg)

        # 根据上一周期买卖次数，计算下一周期的网格宽度调整系数
        buy_sell_threshold_low = int(os.getenv('BUY_SELL_THRESHOLD_LOW', 18))
        buy_sell_threshold_high = int(os.getenv('BUY_SELL_THRESHOLD_HIGH', 36))

        adjustment_factor = self.adjustment_factor
        if (self.daily_stats['buy_orders'] + self.daily_stats['sell_orders']) < buy_sell_threshold_low:
            if self.adjustment_factor >= 1.2:
                adjustment_factor = 1.0
            else:
                adjustment_factor = 0.8
        elif (self.daily_stats['buy_orders'] + self.daily_stats['sell_orders']) > buy_sell_threshold_high:
            if self.adjustment_factor <= 0.8:
                adjustment_factor = 1.0
            else:
                adjustment_factor = 1.2

        # 重置每日统计数据
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

        return adjustment_factor


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
                    if text == '/orders':
                        self.show_orders()
                    elif text == '/portfolio':
                        self.check_portfolio()
                    elif text.startswith('/adjust'):
                        factor = float(text.split(' ')[1]) if len(text.split(' ')) > 1 else 1.0
                        self.cancel_all_orders()
                        self.adjust_grid_parameters(factor)
                        sell_only = self.evaluate_risk()
                        self.place_grid_orders(sell_only=sell_only)
                        self.last_adjustment_time = time.time()
        except Exception as e:
            self.logger.error(f"回答Telegram消息失败: {str(e)}")

    def get_market_data(self):
        """获取市场数据, 1小时K线"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=self.long_period*1*2)
        
        # 获取K线数据
        klines = self.client.get_historical_klines(
            self.symbol,
            Client.KLINE_INTERVAL_1HOUR,
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # 转换为DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        df['close'] = pd.to_numeric(df['close'])
        return df

    def calculate_trend(self, df):
        """计算市场趋势"""
        # 使用EMA指标判断趋势
        ema_short = df['close'].ewm(span=self.short_period, adjust=False).mean()
        ema_long = df['close'].ewm(span=self.long_period, adjust=False).mean()
        
        # 计算趋势强度
        trend_strength = (ema_short - ema_long) / ema_long
        return trend_strength.iloc[-2] # 使用倒数第二个值，因为最后一个值是当前值，

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
        return self.calculate_trend_macd(df) # 使用EMA指标判断趋势, 可以尝试使用MACD指标判断趋势

    def adjust_grid_parameters(self, adjust_factor=1.0):
        """调整网格参数"""
        try:
            # 使用当前价格作为市场数据, 忽略波动率和趋势
            current_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])
            self.current_grid = []
            
            # 根据波动率调整网格宽度
            self.adjustment_factor = adjust_factor
            grid_gain = float(os.getenv('GRID_GAIN', 0.005)) * adjust_factor  #  单网格利润, 默认 0.5%, 并按系数调整
            self.grid_gap = grid_gain * current_price

            for i in range(self.grid_levels//2):
                self.current_grid.append(current_price + self.grid_gap * (i+1))
                self.current_grid.append(current_price - self.grid_gap * (i+1))
            
            profit_per_grid_percent = (grid_gain - float(os.getenv('FEE_RATE', 0.001))*2) * 100  # 每格利润（扣除手续费）

            msg = f"调整网格(系数 {adjust_factor:.1f}):\n"
            msg += f"当前价格: {current_price:.2f} USDT\n"
            msg += f"新网格范围: {min(self.current_grid):.2f} - {max(self.current_grid):.2f} USDT\n"
            msg += f"网格数量: {self.grid_levels}\n"
            msg += f"网格间隔: {self.grid_gap:.2f} USDT\n"
            msg += f"每格利润: {profit_per_grid_percent:.2f}%"
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

    def place_grid_orders(self, buy_only = False, sell_only = False):
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
        
        buy_orders_placed = 0
        sell_orders_placed = 0

        # Place buy orders from highest price to lowest price
        for price in sorted(buy_prices, reverse=True):
            try:
                required_usdt = self.quantity * price
                if usdt_balance >= required_usdt and not sell_only:
                    # Place buy order
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side=SIDE_BUY,
                        type=ORDER_TYPE_LIMIT,
                        timeInForce=TIME_IN_FORCE_GTC,
                        quantity=self.quantity,
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
                if base_asset_balance >= self.quantity and not buy_only:
                    # Place sell order
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side=SIDE_SELL,
                        type=ORDER_TYPE_LIMIT,
                        timeInForce=TIME_IN_FORCE_GTC,
                        quantity=self.quantity,
                        price=f"{price:.2f}"
                        )
                    self.orders[order['orderId']] = {
                        'price': price,
                        'side': 'SELL',
                        'status': 'OPEN'
                        }
                    self.logger.info(f"下单成功: {order['side']} {order['price']} USDT, 数量: {order['origQty']}")
                    base_asset_balance -= self.quantity  # Update available base asset balance
                    sell_orders_placed += 1
                else:
                    self.logger.warning(f"卖单未放置 {price:.2f} USDT")
            except Exception as e:
                self.logger.error(f"下单失败: {str(e)}")

        lowest_price = min(order_info['price'] for order_info in self.orders.values())
        highest_price = max(order_info['price'] for order_info in self.orders.values())

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

    def evaluate_risk(self):
        """评估风险"""
        # 检查投资组合, 看是否吃进了太多的资产，如果持有的加密币资产超过了预定的数量，就只卖出
        base_asset = self.symbol.replace('USDT', '')
        base_asset_balance = float(self.client.get_asset_balance(asset=base_asset)['free'])
        max_base_asset = float(os.getenv('MAX_BASE_ASSET', 10.0))  # Maximum allowed base asset balance

        sell_only = False
        
        if base_asset_balance > max_base_asset:
            msg = f"{base_asset} 存货 {base_asset_balance:.2f} 超过了最大允许值 {max_base_asset:.2f}，只卖出"
            self.logger.warning(msg)
            send_telegram_message(msg)
            sell_only = True

        return sell_only
    
    def close_position(self):
        """平仓函数：全部卖掉"""
        base_asset = self.symbol.replace('USDT', '')
        base_asset_balance = float(self.client.get_asset_balance(asset=base_asset)['free'])
        sell_amount = base_asset_balance
        
        if sell_amount <= 0:
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
            msg = f"已平仓，卖出 {sell_amount:.2f} {base_asset}"
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
        target_amount = self.quantity * grid_count
        
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
                quantity=round(buy_amount, 2)  # 保留2位小数，防止精度问题
            )
            msg = f"已买入 {buy_amount:.2f} {base_asset}，总持仓达到 {target_amount:.2f} {base_asset}"
            self.logger.info(msg)
            send_telegram_message(msg)
        except Exception as e:
            msg = f"买入失败: {str(e)}"
            self.logger.error(msg)
            send_telegram_message(msg)

    def run(self):
        """运行动态网格交易机器人"""
        self.logger.info("启动动态网格交易机器人...")
        self.logger.info(f"交易对: {self.symbol}")

        # 初始化网格参数
        if self.get_market_trend() > 0:
            self.enable_trading = True
            self.prepare_position(2) # 初始化时，准备2个网格
            self.adjust_grid_parameters()
            sell_only = self.evaluate_risk()
            self.place_grid_orders(sell_only=sell_only)
            self.last_adjustment_time = time.time()
        else:
            self.enable_trading = False
            self.logger.info("市场趋势向下，不启动交易")
        
        while True:
            try:
                # 检查订单状态
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
                                self.daily_stats['total_buy_price'] += order_info['price'] * self.quantity
                            else:
                                self.daily_stats['sell_orders'] += 1
                                self.daily_stats['total_sell_price'] += order_info['price'] * self.quantity
                            
                            # 放置反向订单
                            new_side = 'SELL' if order_info['side'] == 'BUY' else 'BUY'
                            new_price = order_info['price'] - self.grid_gap if new_side == 'BUY' else order_info['price'] + self.grid_gap
                            
                            # Check balance before placing reverse order
                            if new_side == 'BUY':
                                usdt_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
                                required_usdt = self.quantity * new_price
                                if usdt_balance >= required_usdt:
                                    new_order = self.client.create_order(
                                        symbol=self.symbol,
                                        side=new_side,
                                        type=ORDER_TYPE_LIMIT,
                                        timeInForce=TIME_IN_FORCE_GTC,
                                        quantity=self.quantity,
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
                                if base_asset_balance >= self.quantity:
                                    new_order = self.client.create_order(
                                        symbol=self.symbol,
                                        side=new_side,
                                        type=ORDER_TYPE_LIMIT,
                                        timeInForce=TIME_IN_FORCE_GTC,
                                        quantity=self.quantity,
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
                
                # 回答Telegram消息
                self.answer_telegram()

                # 每半小时检查市场趋势，看是否需要终止交易并平仓
                if self.last_check_time is None or time.time() - self.last_check_time >= 1800:
                    self.last_check_time = time.time()
                    market_trend = self.get_market_trend()
                    if market_trend < 0:
                        if self.enable_trading: # 如果市场趋势向下，则平仓
                            self.logger.info("市场趋势向下，平仓")
                            send_telegram_message("市场趋势向下，平仓")
                            self.enable_trading = False
                            self.cancel_all_orders()
                            self.close_position()
                            self.send_daily_briefing()
                    elif market_trend > 0: # 如果市场趋势向上，则恢复交易
                        if not self.enable_trading:
                            self.logger.info("市场趋势向上，恢复交易")
                            send_telegram_message("市场趋势向上，恢复交易")
                            self.enable_trading = True
                            self.prepare_position(6) # 恢复交易时，准备6个网格，新的行情来了
                            self.adjust_grid_parameters(1.0)
                            self.place_grid_orders()
                            self.last_adjustment_time = time.time()
                            self.send_daily_briefing()
                # 检查是否需要发送每日简报
                briefing_interval = int(os.getenv('BRIEFING_INTERVAL', 86400))  # Default to 24 hours
                if self.enable_trading and (time.time() - self.last_briefing_time >= briefing_interval):
                    adjust_factor = self.send_daily_briefing()
                    if (not adjust_factor == self.adjustment_factor or self.should_adjust_grid()):
                        self.cancel_all_orders()
                        self.adjust_grid_parameters(adjust_factor)
                        sell_only = self.evaluate_risk()
                        self.place_grid_orders(sell_only=sell_only)
                        self.last_adjustment_time = time.time()
                    
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
        'symbol': os.getenv('TRADE_SYMBOL', 'LTCUSDT'),  # 从环境变量加载交易对，默认LTCUSDT
        'grid_levels': int(os.getenv('GRID_LEVELS', 10)),  # 从环境变量加载网格数量，默认10
        'initial_grid_width': float(os.getenv('INITIAL_GRID_WIDTH', 0.1)),  # 从环境变量加载初始网格宽度，默认10%
        'quantity_per_grid': float(os.getenv('QUANTITY_PER_GRID', 1.0)),   # 每个网格的交易数量
        'short_period': int(os.getenv('SHORT_PERIOD', 7)),
        'long_period': int(os.getenv('LONG_PERIOD', 25))
    }
    
    # 创建并运行动态网格交易机器人
    bot = DynamicGridTrader(**config)
    return bot

if __name__ == "__main__":
    bot = get_bot()
    bot.run()