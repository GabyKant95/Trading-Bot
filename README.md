# Trading Bot — BTC/ETH Auto | Spot Binance

## Qué hace
- Selecciona automáticamente BTC o ETH según las mejores condiciones
- Estrategia pullback + EMA200 + RSI + ADX en timeframe 4h
- Se desactiva automáticamente si detecta caída mayor al 8%
- Se reactiva solo cuando el mercado se recupera
- Reinvierte ganancias automáticamente

## Configuración en Render.com
1. Subir este proyecto a GitHub
2. Conectar repositorio en Render.com
3. Agregar variables de entorno:
   - BINANCE_API_KEY
   - BINANCE_API_SECRET
4. Deploy

## Parámetros ajustables en bot.py
- CRASH_THRESHOLD: % de caída para desactivarse (default -8%)
- ATR_SL / ATR_TP: multiplicadores de stop loss y take profit
- ADX_MIN: fuerza mínima de tendencia requerida
- TIMEFRAME: temporalidad (default 4h)
