# QUANT-X AI PRO — Final

داشبورد پژوهشی اسکالپ Binance Futures با داده عمومی و بدون نیاز به API Key برای مشاهده بازار.

## امکانات
- Binance Futures REST market data
- قیمت و کندل زنده با refresh
- EMA20/50/200, RSI, MACD, ATR, Volume Z-score
- Order Book imbalance + top liquidity
- Funding + Open Interest + OI change
- Long/Short ratio + Taker ratio
- Multi-timeframe matrix
- Rule-based Quant score
- Small ML research model (HistGradientBoosting) با train split زمانی
- ترکیب Rule + ML probability
- Backtest ساده و شفاف
- Telegram signal alerts
- Live order execution عمداً غیرفعال است

## اجرا
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Telegram
در Sidebar توکن Bot و Chat ID را وارد کن و Send signal alerts را روشن کن.

## نکته مهم
این سیستم تضمین سود یا درصد برد نمی‌دهد. مدل ML فقط یک مدل تحقیقاتی کوچک است. قبل از پول واقعی باید walk-forward، out-of-sample، fee/slippage، latency و paper trading انجام شود.

Binance در سال 2026 ساختار WebSocket Futures را به endpointهای Public/Market/Private تفکیک کرده است. برای نسخه‌های بعدی WebSocket زنده باید از endpoint جدید Market/Public استفاده شود.
