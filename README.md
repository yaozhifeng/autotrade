## Crypto Trading Bot

cp sample.env .env

- Config binance API and telegram API
- Config parameters in gird settings

python dyno_grid.py

Have fun!

## The Strategy

- Grid trading is effective for crypto markets, but it can lead to significant losses during sharp price drops. This happens because the strategy accumulates too many positions, which may then be sold at lower prices as the grid continues to follow the downtrend.

- To mitigate this risk, we only activate trading in favorable market conditions. If a market downturn is detected, we promptly close existing positions.

- Technical indicators like EMA and MACD can help identify market trends. MACD appears to perform better for this purpose.

- Therefore, grid trading is enabled when a MACD golden cross is detected, and disabled—with all positions closed—when a dead cross occurs.

## The Details

- Use 1-hour candlesticks for faster market responsiveness.

- Set each grid to target a 0.5% profit to maximize trading opportunities.

- Analyze market trends every hour and respond immediately if conditions change.

- Review the grid setup every 8 hours to determine if adjustments are needed.

- When a favorable trend is detected, pre-fill positions for 6 grid levels to ensure active participation.

