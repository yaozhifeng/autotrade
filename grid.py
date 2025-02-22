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

TEST_NET = os.getenv('TEST_NET', 'True') == 'True'


class DynamicGridTrader:
    def __init__(self, api_key, api_secret, symbol='LTCUSDT',
                 grid_levels=10, initial_grid_width=0.1,
                 quantity_per_grid=0.1, volatility_window=24,
                 trend_window=12):
        """
        初始化动态网格交易机器人
        :param api_key: 币安API密钥
        :param api_secret: 币安API密钥
        :param symbol: 交易对
        :param grid_levels: 网格数量
        :param initial_grid_width: 初始网格宽度（百分比）
        :param quantity_per_grid: 每个网格的交易数量
        :param volatility_window: 波动率计算窗口（小时）
        :param trend_window: 趋势计算窗口（小时）
        """
        self.client = Client(api_key, api_secret, testnet=TEST_NET)
        self.symbol = symbol
        self.grid_levels = grid_levels
        self.initial_grid_width = initial_grid_width
        self.quantity = quantity_per_grid
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        
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
        
        # 设置调整参数
        self.adjustment_threshold = 0.05  # 5%价格变化触发调整
        self.min_adjustment_interval = 3600  # 最小调整间隔（秒）

    def get_market_data(self):
        """获取市场数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=max(self.volatility_window, self.trend_window))
        
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

    def calculate_volatility(self, df):
        """计算市场波动率"""
        returns = np.log(df['close'] / df['close'].shift(1))
        return returns.std() * np.sqrt(24)  # 年化波动率

    def calculate_trend(self, df):
        """计算市场趋势"""
        # 使用EMA指标判断趋势
        ema_short = talib.EMA(df['close'], timeperiod=self.trend_window)
        ema_long = talib.EMA(df['close'], timeperiod=self.trend_window * 2)
        
        # 计算趋势强度
        trend_strength = (ema_short - ema_long) / ema_long
        return trend_strength.iloc[-1]

    def adjust_grid_parameters(self):
        """调整网格参数"""
        try:
            # 获取市场数据
            df = self.get_market_data()
            current_price = float(df['close'].iloc[-1])
            volatility = self.calculate_volatility(df)
            trend = self.calculate_trend(df)
            
            # 根据波动率调整网格宽度
            grid_width = self.initial_grid_width * (1 + volatility)
            
            # 根据趋势调整网格中心
            grid_center = current_price * (1 + trend * 0.1)
            
            # 计算新的网格范围
            grid_range = current_price * grid_width
            lower_price = grid_center - grid_range/2
            upper_price = grid_center + grid_range/2
            
            # 计算新的网格价格
            self.current_grid = np.linspace(lower_price, upper_price, self.grid_levels)
            
            self.logger.info(f"网格参数已调整 - 波动率: {volatility:.4f}, 趋势: {trend:.4f}")
            self.logger.info(f"新网格范围: {lower_price:.2f} - {upper_price:.2f} USDT")
            
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
        
        for price in self.current_grid:
            try:
                if price < current_price:
                    # 在当前价格下方放置买单
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
                else:
                    # 在当前价格上方放置卖单
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
                
                self.logger.info(f"下单成功: {price:.2f} USDT")
                
            except Exception as e:
                self.logger.error(f"下单失败: {str(e)}")

    def should_adjust_grid(self):
        """判断是否需要调整网格"""
        if self.current_grid is None or self.last_adjustment_time is None:
            return True
            
        current_time = time.time()
        current_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])
        
        # 检查时间间隔
        if current_time - self.last_adjustment_time < self.min_adjustment_interval:
            return False
            
        # 检查价格变化
        grid_center = np.mean(self.current_grid)
        price_change = abs(current_price - grid_center) / grid_center
        
        return price_change > self.adjustment_threshold

    def run(self):
        """运行动态网格交易机器人"""
        self.logger.info("启动动态网格交易机器人...")
        self.logger.info(f"交易对: {self.symbol}")
        
        while True:
            try:
                # 检查是否需要调整网格
                if self.should_adjust_grid():
                    self.logger.info("开始调整网格...")
                    self.cancel_all_orders()
                    self.adjust_grid_parameters()
                    self.place_grid_orders()
                    self.last_adjustment_time = time.time()
                
                # 检查订单状态
                for order_id, order_info in list(self.orders.items()):
                    try:
                        order_status = self.client.get_order(
                            symbol=self.symbol,
                            orderId=order_id
                        )
                        
                        if order_status['status'] == 'FILLED':
                            self.logger.info(f"订单已成交: {order_info['price']:.2f} USDT")
                            self.orders.pop(order_id)
                            
                            # 放置反向订单
                            new_side = 'SELL' if order_info['side'] == 'BUY' else 'BUY'
                            new_order = self.client.create_order(
                                symbol=self.symbol,
                                side=new_side,
                                type=ORDER_TYPE_LIMIT,
                                timeInForce=TIME_IN_FORCE_GTC,
                                quantity=self.quantity,
                                price=f"{order_info['price']:.2f}"
                            )
                            
                            self.orders[new_order['orderId']] = {
                                'price': order_info['price'],
                                'side': new_side,
                                'status': 'OPEN'
                            }
                            
                    except Exception as e:
                        self.logger.error(f"检查订单状态失败: {str(e)}")
                
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"运行错误: {str(e)}")
                time.sleep(5)

def main():
    # 配置参数
    # Load environment variables from .env file
    load_dotenv()

    config = {
        'api_key': os.getenv('API_KEY'),
        'api_secret': os.getenv('API_SECRET'),
        'symbol': 'LTCUSDT',
        'grid_levels': 10,
        'initial_grid_width': 0.1,  # 初始网格宽度为10%
        'quantity_per_grid': 0.1,   # 每个网格的交易数量
        'volatility_window': 24,    # 24小时波动率窗口
        'trend_window': 12          # 12小时趋势窗口
    }
    
    # 创建并运行动态网格交易机器人
    bot = DynamicGridTrader(**config)
    bot.run()

if __name__ == "__main__":
    main()