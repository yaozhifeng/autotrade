import time
from binance.client import Client
from binance.enums import *
import numpy as np
import pandas as pd
import talib
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(override=True)


class AdaptiveGridTrader:
    def __init__(self, api_key, api_secret, symbol='LTCUSDT',
                 grid_levels=10, initial_grid_width=0.1,
                 base_quantity=0.1, leverage=1):
        """
        初始化自适应网格交易机器人
        :param api_key: 币安API密钥
        :param api_secret: 币安API密钥
        :param symbol: 交易对
        :param grid_levels: 网格数量
        :param initial_grid_width: 初始网格宽度（百分比）
        :param base_quantity: 基础交易数量
        :param leverage: 杠杆倍数
        """
        self.client = Client(api_key, api_secret, testnet=os.getenv('TEST_NET'))
        self.symbol = symbol
        self.grid_levels = grid_levels
        self.initial_grid_width = initial_grid_width
        self.base_quantity = base_quantity
        self.leverage = leverage
        
        # 趋势判断参数
        self.trend_params = {
            'short_window': 7,   # 短期均线
            'medium_window': 25,  # 中期均线
            'long_window': 50,    # 长期均线
            'rsi_period': 14,     # RSI周期
            'macd_fast': 12,      # MACD快线
            'macd_slow': 26,      # MACD慢线
            'macd_signal': 9      # MACD信号线
        }
        
        # 策略参数
        self.strategy_params = {
            'trend_threshold': 0.02,     # 趋势判断阈值
            'strong_trend': 0.05,        # 强趋势阈值
            'volume_scale': 1.5,         # 顺势交易量倍数
            'grid_asymmetry': 0.3,       # 网格不对称度
            'min_profit': 0.01          # 最小利润率
        }
        
        self._setup_logging()
        self.orders = {}
        self.position = {}
        self.current_trend = None
        self.trend_strength = 0
        
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('adaptive_grid_trader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_market_data(self, interval='30m', limit=100):
        """获取市场数据并计算技术指标"""
        klines = self.client.get_klines(
            symbol=self.symbol,
            interval=interval,
            limit=limit
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        df = df.astype({
            'close': 'float64',
            'high': 'float64',
            'low': 'float64',
            'volume': 'float64'
        })
        
        # 计算技术指标
        df['ema_short'] = talib.EMA(df['close'], timeperiod=self.trend_params['short_window'])
        df['ema_medium'] = talib.EMA(df['close'], timeperiod=self.trend_params['medium_window'])
        df['ema_long'] = talib.EMA(df['close'], timeperiod=self.trend_params['long_window'])
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.trend_params['rsi_period'])
        
        # 计算MACD
        macd, signal, hist = talib.MACD(
            df['close'],
            fastperiod=self.trend_params['macd_fast'],
            slowperiod=self.trend_params['macd_slow'],
            signalperiod=self.trend_params['macd_signal']
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        return df

    def analyze_trend(self, df):
        """分析市场趋势和强度"""
        latest = df.iloc[-1]
        
        # 趋势信号计算
        trend_signals = {
            'ema_trend': (
                1 if latest['ema_short'] > latest['ema_medium'] > latest['ema_long']
                else -1 if latest['ema_short'] < latest['ema_medium'] < latest['ema_long']
                else 0
            ),
            'rsi_trend': (
                1 if latest['rsi'] > 60
                else -1 if latest['rsi'] < 40
                else 0
            ),
            'macd_trend': (
                1 if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0
                else -1 if latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0
                else 0
            )
        }
        
        # 综合趋势判断
        trend_score = sum(trend_signals.values()) / len(trend_signals)
        
        # 计算趋势强度
        strength = abs(trend_score)
        
        # 更新趋势状态
        self.current_trend = (
            'STRONG_UP' if trend_score > self.strategy_params['strong_trend']
            else 'UP' if trend_score > self.strategy_params['trend_threshold']
            else 'STRONG_DOWN' if trend_score < -self.strategy_params['strong_trend']
            else 'DOWN' if trend_score < -self.strategy_params['trend_threshold']
            else 'NEUTRAL'
        )
        self.trend_strength = strength
        
        self.logger.info(f"趋势分析 - 方向: {self.current_trend}, 强度: {strength:.3f}")
        return self.current_trend, strength

    def calculate_grid_parameters(self, current_price):
        """根据趋势计算网格参数"""
        base_width = self.initial_grid_width * current_price
        
        if self.current_trend in ['STRONG_UP', 'STRONG_DOWN']:
            # 强趋势时增加网格宽度
            grid_width = base_width * (1 + self.trend_strength)
            # 调整网格分布的不对称性
            asymmetry = self.strategy_params['grid_asymmetry'] * (
                1 if self.current_trend == 'STRONG_UP' else -1
            )
        else:
            grid_width = base_width
            asymmetry = 0
        
        # 计算网格范围
        grid_center = current_price * (1 + asymmetry)
        lower_price = grid_center - grid_width/2
        upper_price = grid_center + grid_width/2
        
        return lower_price, upper_price

    def calculate_order_quantity(self, price, side):
        """根据趋势强度调整订单数量"""
        base_qty = self.base_quantity
        
        # 根据趋势调整交易量
        if self.current_trend in ['STRONG_UP', 'STRONG_DOWN']:
            if (self.current_trend == 'STRONG_UP' and side == 'BUY') or \
               (self.current_trend == 'STRONG_DOWN' and side == 'SELL'):
                # 顺势加倍
                qty = base_qty * self.strategy_params['volume_scale']
            else:
                # 逆势减半
                qty = base_qty * 0.5
        else:
            qty = base_qty
        
        return qty

    def place_dynamic_grid_orders(self):
        """放置动态网格订单"""
        current_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])
        lower_price, upper_price = self.calculate_grid_parameters(current_price)
        
        # 取消现有订单
        self.cancel_all_orders()
        
        # 计算网格价格
        grid_prices = np.linspace(lower_price, upper_price, self.grid_levels)
        
        for price in grid_prices:
            try:
                if price < current_price:
                    # 买单
                    qty = self.calculate_order_quantity(price, 'BUY')
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side=SIDE_BUY,
                        type=ORDER_TYPE_LIMIT,
                        timeInForce=TIME_IN_FORCE_GTC,
                        quantity=f"{qty:.4f}",
                        price=f"{price:.2f}"
                    )
                    self.orders[order['orderId']] = {
                        'price': price,
                        'quantity': qty,
                        'side': 'BUY',
                        'status': 'OPEN'
                    }
                else:
                    # 卖单
                    qty = self.calculate_order_quantity(price, 'SELL')
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side=SIDE_SELL,
                        type=ORDER_TYPE_LIMIT,
                        timeInForce=TIME_IN_FORCE_GTC,
                        quantity=f"{qty:.4f}",
                        price=f"{price:.2f}"
                    )
                    self.orders[order['orderId']] = {
                        'price': price,
                        'quantity': qty,
                        'side': 'SELL',
                        'status': 'OPEN'
                    }
                
                self.logger.info(f"下单成功: {price:.2f} USDT, 数量: {qty:.4f}")
                
            except Exception as e:
                self.logger.error(f"下单失败: {str(e)}")

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

    def run(self):
        """运行自适应网格交易机器人"""
        self.logger.info("启动自适应网格交易机器人...")
        self.logger.info(f"交易对: {self.symbol}")
        
        while True:
            try:
                # 获取和分析市场数据
                market_data = self.get_market_data()
                self.analyze_trend(market_data)
                
                # 更新网格订单
                if self.trend_strength > self.strategy_params['trend_threshold']:
                    self.place_dynamic_grid_orders()
                
                # 检查订单状态
                self.check_and_handle_filled_orders()
                
                # 等待更新
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"运行错误: {str(e)}")
                time.sleep(5)

    def check_and_handle_filled_orders(self):
        """检查和处理已成交订单"""
        for order_id, order_info in list(self.orders.items()):
            try:
                order_status = self.client.get_order(
                    symbol=self.symbol,
                    orderId=order_id
                )
                
                if order_status['status'] == 'FILLED':
                    self.logger.info(f"订单已成交: {order_info['price']:.2f} USDT")
                    self.orders.pop(order_id)
                    
                    # 计算新的反向订单
                    new_side = 'SELL' if order_info['side'] == 'BUY' else 'BUY'
                    new_price = order_info['price'] * (
                        1 + self.strategy_params['min_profit'] if new_side == 'SELL'
                        else 1 - self.strategy_params['min_profit']
                    )
                    new_qty = self.calculate_order_quantity(new_price, new_side)
                    
                    # 放置反向订单
                    new_order = self.client.create_order(
                        symbol=self.symbol,
                        side=new_side,
                        type=ORDER_TYPE_LIMIT,
                        timeInForce=TIME_IN_FORCE_GTC,
                        quantity=f"{new_qty:.4f}",
                        price=f"{new_price:.2f}"
                    )
                    
                    self.orders[new_order['orderId']] = {
                        'price': new_price,
                        'quantity': new_qty,
                        'side': new_side,
                        'status': 'OPEN'
                    }
                    
            except Exception as e:
                self.logger.error(f"检查订单状态失败: {str(e)}")

def main():
    # 配置参数

    config = {
        'api_key': os.getenv('API_KEY'),
        'api_secret': os.getenv('API_SECRET'),
        'symbol': os.getenv('TRADE_SYMBOL', 'LTCUSDT'),  # 从环境变量加载交易对，默认LTCUSDT
        'grid_levels': 10,
        'initial_grid_width': 0.1,  # 初始网格宽度为10%
        'base_quantity': 0.1,       # 基础交易数量
        'leverage': 1               # 杠杆倍数
    }
    
    # 创建并运行自适应网格交易机器人
    bot = AdaptiveGridTrader(**config)
    bot.run()

if __name__ == "__main__":
    main()