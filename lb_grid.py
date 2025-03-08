import time
import logging
from decimal import Decimal, ROUND_HALF_UP
from longbridge.openapi import QuoteContext, TradeContext, Config, SecurityType, TradeStatus, OrderType, OrderSide, TimeInForce

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
class GridConfig:
    # 长桥证券API配置
    APP_KEY = "YOUR_APP_KEY"  # 替换为您的App Key
    APP_SECRET = "YOUR_APP_SECRET"  # 替换为您的App Secret
    ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"  # 替换为您的Access Token
    
    # 交易配置
    SYMBOL = "NVDA"  # 交易标的
    GRID_NUM = 10  # 网格数量
    GRID_PERCENT = 0.01  # 网格间距(以当前价格的百分比表示)
    SHARES_PER_GRID = 1  # 每个网格的股数
    
    # 风控配置
    MAX_POSITION = 100  # 最大持仓数量
    MAX_INVESTMENT = 100000  # 最大投资额度
    
    # 其他配置
    CHECK_INTERVAL = 60  # 检查间隔(秒)


class GridTrading:
    def __init__(self, config):
        self.config = config
        # 初始化长桥API配置
        self.lb_config = Config(app_key=config.APP_KEY, 
                               app_secret=config.APP_SECRET, 
                               access_token=config.ACCESS_TOKEN)
        # 初始化行情和交易上下文
        self.quote_ctx = QuoteContext(self.lb_config)
        self.trade_ctx = TradeContext(self.lb_config)
        # 网格信息
        self.grids = []
        self.current_price = None
        self.grid_orders = {}  # 记录网格订单

    def round_price(self, price):
        """四舍五入价格到合适的精度"""
        return float(Decimal(str(price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    async def get_current_price(self):
        """获取当前股票价格"""
        try:
            quote = await self.quote_ctx.subscribe_quote([self.config.SYMBOL], True)
            snapshot = await self.quote_ctx.snapshot([self.config.SYMBOL])
            if snapshot and len(snapshot) > 0:
                self.current_price = snapshot[0].last_done
                logger.info(f"Current price for {self.config.SYMBOL}: ${self.current_price}")
                return self.current_price
            else:
                logger.error("Failed to get snapshot")
                return None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None

    async def initialize_grids(self):
        """初始化网格"""
        # 获取当前价格
        price = await self.get_current_price()
        if not price:
            logger.error("Cannot initialize grids without current price")
            return False
        
        # 计算网格价格
        grid_span = price * self.config.GRID_PERCENT
        self.grids = []
        
        # 生成网格 (N/2个格子在当前价格上方，N/2个在下方)
        half_grid = self.config.GRID_NUM // 2
        
        for i in range(-half_grid, half_grid + self.config.GRID_NUM % 2):
            grid_price = self.round_price(price + i * grid_span)
            grid = {
                "price": grid_price,
                "buy_order_id": None,  # 买单ID
                "sell_order_id": None,  # 卖单ID
                "status": "init"  # 状态：init, buy_placed, buy_filled, sell_placed, sell_filled
            }
            self.grids.append(grid)
        
        # 按价格排序网格
        self.grids.sort(key=lambda x: x["price"])
        logger.info(f"Initialized {len(self.grids)} grids from ${self.grids[0]['price']} to ${self.grids[-1]['price']}")
        return True

    async def get_account_info(self):
        """获取账户信息"""
        try:
            account_balance = await self.trade_ctx.account_balance()
            positions = await self.trade_ctx.positions(symbol=self.config.SYMBOL)
            
            # 计算当前持仓
            current_position = 0
            if positions:
                for position in positions:
                    if position.symbol == self.config.SYMBOL:
                        current_position = position.quantity
            
            return {
                "cash_balance": account_balance.net_asset,
                "current_position": current_position
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    async def place_order(self, symbol, price, quantity, side):
        """下单函数"""
        try:
            # 确定订单类型
            order_type = OrderType.LO  # 限价单
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            
            # 下单
            submit_result = await self.trade_ctx.submit_order(
                symbol=symbol,
                order_type=order_type,
                order_side=order_side,
                submitted_price=price,
                submitted_quantity=quantity,
                time_in_force=TimeInForce.DAY
            )
            
            if submit_result and submit_result.order_id:
                logger.info(f"Placed {side} order for {quantity} shares of {symbol} at ${price} - Order ID: {submit_result.order_id}")
                return submit_result.order_id
            else:
                logger.error(f"Failed to place {side} order: {submit_result}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing {side} order: {e}")
            return None

    async def cancel_order(self, order_id):
        """取消订单"""
        try:
            cancel_result = await self.trade_ctx.cancel_order(order_id=order_id)
            if cancel_result and cancel_result.status == "success":
                logger.info(f"Canceled order: {order_id}")
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}: {cancel_result}")
                return False
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    async def check_order_status(self, order_id):
        """检查订单状态"""
        try:
            order_detail = await self.trade_ctx.order_detail(order_id)
            if order_detail:
                status = order_detail.status
                logger.info(f"Order {order_id} status: {status}")
                return status
            else:
                logger.error(f"Failed to get order detail for {order_id}")
                return None
        except Exception as e:
            logger.error(f"Error checking order status for {order_id}: {e}")
            return None

    async def execute_grid_strategy(self):
        """执行网格策略"""
        # 获取账户信息
        account_info = await self.get_account_info()
        if not account_info:
            logger.error("Cannot execute strategy without account info")
            return
        
        # 获取当前价格
        current_price = await self.get_current_price()
        if not current_price:
            logger.error("Cannot execute strategy without current price")
            return
        
        # 检查每个网格
        for i, grid in enumerate(self.grids):
            # 检查订单状态
            if grid["status"] == "buy_placed" and grid["buy_order_id"]:
                status = await self.check_order_status(grid["buy_order_id"])
                if status == TradeStatus.FILLED:
                    grid["status"] = "buy_filled"
                    logger.info(f"Grid {i} buy order filled at ${grid['price']}")
            
            if grid["status"] == "sell_placed" and grid["sell_order_id"]:
                status = await self.check_order_status(grid["sell_order_id"])
                if status == TradeStatus.FILLED:
                    grid["status"] = "sell_filled"
                    logger.info(f"Grid {i} sell order filled at ${grid['price']}")


            # 根据网格价格和当前价格判断操作
            if current_price > grid["price"] and grid["status"] in ["init", "sell_filled"]:
                # 当前价格下面的网格，设置买入单
                if account_info["cash_balance"] > grid["price"] * self.config.SHARES_PER_GRID:
                    order_id = await self.place_order(
                        self.config.SYMBOL, 
                        grid["price"], 
                        self.config.SHARES_PER_GRID, 
                        "buy"
                    )
                    if order_id:
                        grid["buy_order_id"] = order_id
                        grid["status"] = "buy_placed"
            
            elif current_price < grid["price"] and grid["status"] in ["init", "buy_filled"]:
                # 当前价格上面的网格，设置卖出单
                if account_info["current_position"] >= self.config.SHARES_PER_GRID:
                    order_id = await self.place_order(
                        self.config.SYMBOL, 
                        grid["price"], 
                        self.config.SHARES_PER_GRID, 
                        "sell"
                    )
                    if order_id:
                        grid["sell_order_id"] = order_id
                        grid["status"] = "sell_placed"
            

    async def run(self):
        """运行网格交易策略"""
        logger.info("Starting NVDA Grid Trading Strategy")
        
        # 初始化网格
        if not await self.initialize_grids():
            logger.error("Failed to initialize grids. Exiting.")
            return
        
        # 输出初始网格信息
        logger.info("Grid prices:")
        for i, grid in enumerate(self.grids):
            logger.info(f"Grid {i+1}: ${grid['price']}")
        
        # 主循环
        try:
            while True:
                await self.execute_grid_strategy()
                time.sleep(self.config.CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Strategy stopped by user")
        except Exception as e:
            logger.error(f"Strategy error: {e}")
        finally:
            # 关闭API连接
            self.quote_ctx.close()
            self.trade_ctx.close()
            logger.info("Grid Trading Strategy terminated")

# 主函数
async def main():
    config = GridConfig()
    strategy = GridTrading(config)
    await strategy.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())