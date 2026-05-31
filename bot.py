import ccxt
import pandas as pd
import numpy as np
import time
import os
import logging
from datetime import datetime

# ============================================================
# CONFIGURACION
# ============================================================
API_KEY    = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")

CAPITAL_INICIAL = 40        # USDT
RIESGO_POR_TRADE = 0.95     # Usa 95% del capital disponible
TIMEFRAME = "4h"
PARES = ["ETH/USDT", "BTC/USDT"]

# Parametros estrategia
EMA_TREND   = 200
EMA_ENTRY   = 50
RSI_OB      = 65
RSI_OS      = 48
ADX_MIN     = 20
ATR_SL      = 1.5
ATR_TP      = 3.0

# Auto-desactivacion: si el precio cae mas de este % en 24h
CRASH_THRESHOLD = -8.0

# Reinversion: reinvierte todo automaticamente
REINVERTIR = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ============================================================
# CONEXION BINANCE
# ============================================================
exchange = ccxt.binance({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "spot"}
})

# ============================================================
# INDICADORES
# ============================================================
def calcular_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def calcular_atr(df, period=14):
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calcular_adx(df, period=14):
    up   = df["high"].diff()
    down = -df["low"].diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    atr = calcular_atr(df, period)
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    dx  = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

# ============================================================
# OBTENER DATOS
# ============================================================
def obtener_datos(par, timeframe=TIMEFRAME, limit=300):
    ohlcv = exchange.fetch_ohlcv(par, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

# ============================================================
# SELECCIONAR MEJOR PAR
# ============================================================
def seleccionar_par():
    mejor_par   = None
    mejor_score = -1

    for par in PARES:
        try:
            df = obtener_datos(par)
            df["ema200"] = calcular_ema(df["close"], EMA_TREND)
            df["ema50"]  = calcular_ema(df["close"], EMA_ENTRY)
            df["rsi"]    = calcular_rsi(df["close"])
            df["adx"]    = calcular_adx(df)
            df["atr"]    = calcular_atr(df)

            last = df.iloc[-1]

            # Score basado en condiciones alcistas
            score = 0
            if last["close"] > last["ema200"]: score += 2
            if last["adx"] > ADX_MIN:          score += 2
            if last["rsi"] < RSI_OB:           score += 1
            if last["close"] <= last["ema50"] * 1.002: score += 2

            # Penalizar si hubo crash reciente
            cambio_24h = ((last["close"] - df.iloc[-7]["close"]) / df.iloc[-7]["close"]) * 100
            if cambio_24h < CRASH_THRESHOLD:
                score -= 5
                log.warning(f"{par} penalizado por caida de {cambio_24h:.2f}%")

            log.info(f"{par} score: {score} | RSI: {last['rsi']:.1f} | ADX: {last['adx']:.1f}")

            if score > mejor_score:
                mejor_score = score
                mejor_par   = par

        except Exception as e:
            log.error(f"Error evaluando {par}: {e}")

    return mejor_par, mejor_score

# ============================================================
# DETECTAR SEÑAL
# ============================================================
def detectar_senal(par):
    df = obtener_datos(par)
    df["ema200"] = calcular_ema(df["close"], EMA_TREND)
    df["ema50"]  = calcular_ema(df["close"], EMA_ENTRY)
    df["rsi"]    = calcular_rsi(df["close"])
    df["adx"]    = calcular_adx(df)
    df["atr"]    = calcular_atr(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend_up     = last["close"] > last["ema200"]
    pullback     = last["close"] <= last["ema50"] * 1.002
    rsi_ok       = last["rsi"] < RSI_OS
    vela_alcista = last["close"] > last["open"]
    adx_ok       = last["adx"] > ADX_MIN

    # Crash check
    cambio_24h = ((last["close"] - df.iloc[-7]["close"]) / df.iloc[-7]["close"]) * 100
    crash = cambio_24h < CRASH_THRESHOLD

    buy = trend_up and pullback and rsi_ok and vela_alcista and adx_ok and not crash

    sl = last["close"] - last["atr"] * ATR_SL
    tp = last["close"] - last["atr"] * ATR_TP  # Se calcula pero se usa el de abajo
    tp = last["close"] + last["atr"] * ATR_TP

    return {
        "buy":    buy,
        "crash":  crash,
        "price":  last["close"],
        "sl":     sl,
        "tp":     tp,
        "atr":    last["atr"],
        "rsi":    last["rsi"],
        "adx":    last["adx"],
        "trend":  trend_up,
        "cambio_24h": cambio_24h
    }

# ============================================================
# OBTENER CAPITAL DISPONIBLE
# ============================================================
def obtener_capital():
    balance = exchange.fetch_balance()
    usdt = balance["free"].get("USDT", 0)
    return usdt

def obtener_posicion(par):
    moneda = par.split("/")[0]
    balance = exchange.fetch_balance()
    return balance["free"].get(moneda, 0)

# ============================================================
# EJECUTAR ORDEN
# ============================================================
def ejecutar_compra(par, capital):
    try:
        precio   = exchange.fetch_ticker(par)["last"]
        cantidad = (capital * RIESGO_POR_TRADE) / precio
        info     = exchange.market(par)
        cantidad = exchange.amount_to_precision(par, cantidad)
        orden    = exchange.create_market_buy_order(par, cantidad)
        log.info(f"✅ COMPRA ejecutada: {cantidad} {par} a ${precio:.2f}")
        return orden, precio
    except Exception as e:
        log.error(f"❌ Error en compra: {e}")
        return None, None

def ejecutar_venta(par):
    try:
        moneda   = par.split("/")[0]
        cantidad = obtener_posicion(par)
        if cantidad > 0:
            cantidad = exchange.amount_to_precision(par, cantidad)
            orden    = exchange.create_market_sell_order(par, cantidad)
            precio   = exchange.fetch_ticker(par)["last"]
            log.info(f"✅ VENTA ejecutada: {cantidad} {par} a ${precio:.2f}")
            return orden
    except Exception as e:
        log.error(f"❌ Error en venta: {e}")
    return None

# ============================================================
# LOOP PRINCIPAL
# ============================================================
def main():
    log.info("🤖 Bot iniciado — BTC/ETH Auto Selector | Spot Binance")
    log.info(f"Capital inicial: ${CAPITAL_INICIAL} USDT | Timeframe: {TIMEFRAME}")

    en_posicion   = False
    par_actual    = None
    precio_entrada = None
    sl_actual     = None
    tp_actual     = None
    bot_activo    = True

    while True:
        try:
            ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.info(f"--- Ciclo: {ahora} ---")

            # ------------------------------------------------
            # VERIFICAR SI ESTAMOS EN POSICION
            # ------------------------------------------------
            if en_posicion and par_actual:
                senal   = detectar_senal(par_actual)
                precio  = senal["price"]
                cambio  = ((precio - precio_entrada) / precio_entrada) * 100

                log.info(f"Posicion abierta en {par_actual} | Entrada: ${precio_entrada:.2f} | Actual: ${precio:.2f} | PnL: {cambio:.2f}%")

                # Cerrar por TP
                if precio >= tp_actual:
                    log.info(f"🎯 TAKE PROFIT alcanzado! +{cambio:.2f}%")
                    ejecutar_venta(par_actual)
                    en_posicion = False
                    par_actual  = None

                # Cerrar por SL
                elif precio <= sl_actual:
                    log.info(f"🛑 STOP LOSS activado. {cambio:.2f}%")
                    ejecutar_venta(par_actual)
                    en_posicion = False
                    par_actual  = None

                # Cerrar por crash
                elif senal["crash"]:
                    log.warning(f"⚠️ CRASH detectado ({senal['cambio_24h']:.2f}%). Cerrando posicion y desactivando bot.")
                    ejecutar_venta(par_actual)
                    en_posicion = False
                    par_actual  = None
                    bot_activo  = False

            # ------------------------------------------------
            # REACTIVAR BOT SI EL MERCADO SE RECUPERO
            # ------------------------------------------------
            if not bot_activo:
                for par in PARES:
                    try:
                        df     = obtener_datos(par, limit=50)
                        cambio = ((df["close"].iloc[-1] - df["close"].iloc[-7]) / df["close"].iloc[-7]) * 100
                        ema200 = calcular_ema(df["close"], 200).iloc[-1]
                        if cambio > 3.0 and df["close"].iloc[-1] > ema200:
                            log.info(f"✅ Mercado recuperado en {par} (+{cambio:.2f}%). Reactivando bot.")
                            bot_activo = True
                            break
                    except:
                        pass

                if not bot_activo:
                    log.info("😴 Bot en pausa por crash. Esperando recuperacion...")
                    time.sleep(3600)
                    continue

            # ------------------------------------------------
            # BUSCAR NUEVA ENTRADA
            # ------------------------------------------------
            if not en_posicion and bot_activo:
                par, score = seleccionar_par()

                if par and score >= 4:
                    senal = detectar_senal(par)

                    if senal["buy"]:
                        capital = obtener_capital()
                        log.info(f"💰 Capital disponible: ${capital:.2f} USDT")

                        if capital >= 5:
                            orden, precio = ejecutar_compra(par, capital)
                            if orden:
                                en_posicion    = True
                                par_actual     = par
                                precio_entrada = precio
                                sl_actual      = senal["sl"]
                                tp_actual      = senal["tp"]
                                log.info(f"📊 SL: ${sl_actual:.2f} | TP: ${tp_actual:.2f}")
                        else:
                            log.warning(f"Capital insuficiente: ${capital:.2f}")
                    else:
                        log.info(f"Sin señal en {par} | RSI: {senal['rsi']:.1f} | ADX: {senal['adx']:.1f} | Trend: {senal['trend']}")
                else:
                    log.info(f"Score insuficiente ({score}). Esperando mejor condicion de mercado.")

            # Esperar al proximo ciclo (1 hora)
            log.info("⏳ Esperando 1 hora para el proximo ciclo...")
            time.sleep(3600)

        except KeyboardInterrupt:
            log.info("Bot detenido manualmente.")
            break
        except Exception as e:
            log.error(f"Error en ciclo principal: {e}")
            time.sleep(300)

if __name__ == "__main__":
    main()
