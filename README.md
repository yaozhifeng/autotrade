## Crypto Trading Bot

cp sample.env .env

- Config binance API and telegram API
- Config parameters in gird settings

python dyno_grid.py

Have fun!

## The Strategy

- Grid trading is effective for crypto markets, but it can lead to significant losses during sharp price drops. This happens because the strategy accumulates too many positions, which may then be sold at lower prices as the grid continues to follow the downtrend.

- To mitigate this risk, we set a stop-loss price to exit market. If the price drop reaches the stop-loss price, we promptly close positions and disable further trading, until a favorable market is detected again.

- Technical indicators like EMA and MACD can help identify market trends. MACD appears to perform better for this purpose. We use MACD golden cross signal for recovery trading.

## The Details

- Use 1-hour candlesticks for faster market responsiveness.

- Set each grid to target a 0.5% profit to maximize trading opportunities.

- Analyze market price and trends every hour and respond immediately if conditions change.

- Review the grid setup every 8 hours to determine if adjustments are needed.

- When a favorable trend is detected, pre-fill positions for 4 grid levels to ensure active participation.

## Telegram Commands

```
adjust - 调整网格
portfolio - 查看资产组合
orders - 查看订单
chase - 手动追高
close - 手动平仓
```