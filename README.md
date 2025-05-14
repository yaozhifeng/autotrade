## Crypto Trading Bot

cp sample.env .env

- Config binance API and telegram API
- Config parameters in gird settings
- Config telegram bot

python dyno_grid.py

Have fun!

## The Strategy

- 动态网格交易，使用 30m ATR（Average True Range）作为网格间距，每次调整网格时计算

- 最多20格，每格交易数量一致

- 设置止损价格，当两个周期内（一个周期 8 小时），价格下落超过预设比例（默认 6%），则立即平仓停止交易。等待人工恢复。

- 及时追高：卖单耗尽时立即补仓追高

- 延迟追低：30 分钟检查，买单耗尽并价格低于网格超过 2 格，追低（平仓保留 4 格现货）

## Telegram Commands

```
adjust - 调整网格
portfolio - 查看资产组合
orders - 查看订单
chase - 手动追高
close - 手动平仓
restart - 重新启动交易
```