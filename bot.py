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

PAPER_TRADING = os.environ.get("PAPER_TRADING", "true").lower() == "true"

# Capital fijo en USDT
CAPITAL_USDT = 40.0

# SOL reservado para operar (mitad del acumulado)
SOL_RESERVA = 0.1866

# Par y timeframe
PAR       = "SOL/USDT"
TIMEFRAME = "1h"

# Parametros estrategia (Trend Breakout ATR v3 optimizado)
BREAKOUT_LEN = 9
ATR_SL       = 1.5
ATR_TP       = 13.5
VOL_MULT     = 1.4
RSI_MAX      = 69
ADX_MIN      = 12
CRASH_THRESHOLD = -8.0

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
# ESTADO PAPER TRADING
# ============================================================
paper_usdt        = CAPITAL_USDT
paper_sol         = SOL_RESERVA
paper_en_posicion = False
paper_entrada     = None
paper_sl          = None
paper_tp          = None
paper_cantidad    = None
paper_trades      = []

def paper_resumen():
    if not paper_trades:
        log.info("📋 Sin trades completados aún.")
        return
    ganados   = [t for t in paper_trades if t["pnl"] > 0]
    perdidos  = [t for t in paper_trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in paper_trades)
    win_rate  = len(ganados) / len(paper_trades) * 100
    log.info("=" * 55)
    log.info(f"📊 RESUMEN PAPER TRADING")
    log.info(f"   Capital USDT inicial:  ${CAPITAL_USDT:.2f}")
    log.info(f"   Capital USDT actual:   ${paper_usdt:.2f}")
    log.info(f"   SOL reserva inicial:   {SOL_RESERVA:.4f}")
    log.info(f"   SOL reserva actual:    {paper_sol:.4f}")
    log.info(f"   PnL total:             ${total_pnl:.2f}")
    log.info(f"   Trades totales:        {len(paper_trades)}")
    log.info(f"   Ganados/Perdidos:      {len(ganados)}/{len(perdidos)}")
    log.info(f"   Win rate:              {win_rate:.1f}%")
    if ganados:
        log.info(f"   Ganancia promedio:     ${sum(t['pnl'] for t in ganados)/len(ganados):.2f}")
    if perdidos:
        log.info(f"   Pérdida promedio:      ${sum(t['pnl'] for t in perdidos)/len(perdidos):.2f}")
    log.info("=" * 55)

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
    up       = df["high"].diff()
    down     = -df["low"].diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    atr      = calcular_atr(df, period)
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    dx       = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

# ============================================================
# DATOS Y SEÑAL
# ============================================================
def obtener_datos(limit=300):
    ohlcv = exchange.fetch_ohlcv(PAR, TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def detectar_senal():
    df = obtener_datos()
    df["ema50"]  = calcular_ema(df["close"], 50)
    df["ema200"] = calcular_ema(df["close"], 200)
    df["rsi"]    = calcular_rsi(df["close"])
    df["adx"]    = calcular_adx(df)
    df["atr"]    = calcular_atr(df)

    # Filtro diario
    df_d = obtener_datos(limit=500)
    ema200d = calcular_ema(df_d["close"], 200).iloc[-1]

    last     = df.iloc[-1]
    prev     = df.iloc[-2]
    vol_ma   = df["volume"].rolling(20).mean().iloc[-1]

    # Condiciones Trend Breakout
    trend_up     = last["ema50"] > last["ema200"]
    trend_htf    = last["close"] > ema200d
    highest      = df["high"].iloc[-BREAKOUT_LEN-1:-1].max()
    breakout     = last["close"] > highest
    volatility   = last["atr"] > df["atr"].rolling(50).mean().iloc[-1]
    vol_ok       = last["volume"] > vol_ma * VOL_MULT
    rsi_ok       = last["rsi"] < RSI_MAX
    adx_ok       = last["adx"] > ADX_MIN
    cuerpo       = abs(last["close"] - last["open"])
    rango        = last["high"] - last["low"]
    vela_fuerte  = cuerpo > rango * 0.5

    cambio_24h   = ((last["close"] - df["close"].iloc[-7]) / df["close"].iloc[-7]) * 100
    crash        = cambio_24h < CRASH_THRESHOLD

    buy = trend_up and trend_htf and breakout and volatility and vol_ok and rsi_ok and adx_ok and vela_fuerte and not crash

    return {
        "buy":     buy,
        "crash":   crash,
        "price":   last["close"],
        "sl":      last["close"] - last["atr"] * ATR_SL,
        "tp":      last["close"] + last["atr"] * ATR_TP,
        "atr":     last["atr"],
        "rsi":     last["rsi"],
        "adx":     last["adx"],
        "trend":   trend_up and trend_htf,
        "breakout": breakout,
        "cambio_24h": cambio_24h
    }

# ============================================================
# CAPITAL DISPONIBLE
# ============================================================
def obtener_balance_usdt():
    balance = exchange.fetch_balance()
    return balance["free"].get("USDT", 0)

def obtener_balance_sol():
    balance = exchange.fetch_balance()
    return balance["free"].get("SOL", 0)

def calcular_capital_total(precio_sol):
    """Capital total = USDT fijo + valor de SOL reserva"""
    return CAPITAL_USDT + (SOL_RESERVA * precio_sol)

# ============================================================
# ORDENES REALES
# ============================================================
def ejecutar_compra_real(capital_usdt):
    try:
        precio   = exchange.fetch_ticker(PAR)["last"]
        cantidad = exchange.amount_to_precision(PAR, (capital_usdt * 0.95) / precio)
        orden    = exchange.create_market_buy_order(PAR, cantidad)
        log.info(f"✅ COMPRA REAL: {cantidad} SOL a ${precio:.2f}")
        return float(cantidad), precio
    except Exception as e:
        log.error(f"❌ Error compra: {e}")
        return None, None

def ejecutar_venta_real(cantidad):
    try:
        cantidad_str = exchange.amount_to_precision(PAR, cantidad)
        exchange.create_market_sell_order(PAR, cantidad_str)
        precio = exchange.fetch_ticker(PAR)["last"]
        log.info(f"✅ VENTA REAL: {cantidad_str} SOL a ${precio:.2f}")
        return precio
    except Exception as e:
        log.error(f"❌ Error venta: {e}")
        return None

# ============================================================
# LOOP PRINCIPAL
# ============================================================
def main():
    global paper_usdt, paper_sol, paper_en_posicion
    global paper_entrada, paper_sl, paper_tp, paper_cantidad

    modo = "📝 PAPER TRADING" if PAPER_TRADING else "💰 REAL"
    log.info(f"🤖 Bot SOL/USDT iniciado — {modo}")
    log.info(f"Capital USDT: ${CAPITAL_USDT} | SOL reserva: {SOL_RESERVA} | TF: {TIMEFRAME}")

    en_posicion    = False
    precio_entrada = None
    sl_actual      = None
    tp_actual      = None
    cantidad_pos   = None
    bot_activo     = True
    ciclo          = 0

    while True:
        try:
            ciclo += 1
            log.info(f"--- Ciclo #{ciclo} | {datetime.now().strftime('%Y-%m-%d %H:%M')} ---")

            # Estado actual
            if PAPER_TRADING:
                precio_actual = exchange.fetch_ticker(PAR)["last"]
                capital_total = calcular_capital_total(precio_actual)
                log.info(f"💼 USDT: ${paper_usdt:.2f} | SOL reserva: {paper_sol:.4f} | Total: ${capital_total:.2f}")
            
            # ------------------------------------------------
            # VERIFICAR POSICION ABIERTA
            # ------------------------------------------------
            if en_posicion:
                precio = exchange.fetch_ticker(PAR)["last"]
                cambio = ((precio - precio_entrada) / precio_entrada) * 100
                log.info(f"📈 Posicion abierta | Entrada: ${precio_entrada:.2f} | Actual: ${precio:.2f} | PnL: {cambio:.2f}%")

                cerrar = False
                razon  = ""

                if precio >= tp_actual:
                    cerrar, razon = True, f"🎯 TAKE PROFIT +{cambio:.2f}%"
                elif precio <= sl_actual:
                    cerrar, razon = True, f"🛑 STOP LOSS {cambio:.2f}%"
                else:
                    senal = detectar_senal()
                    if senal["crash"]:
                        cerrar, razon = True, f"⚠️ CRASH {senal['cambio_24h']:.2f}%"
                        bot_activo = False

                if cerrar:
                    log.info(razon)
                    if PAPER_TRADING:
                        pnl = cantidad_pos * (precio - precio_entrada)
                        # Reinvertir: si ganó agrega al USDT, si perdió resta
                        paper_usdt += pnl
                        # Devolver SOL al balance de reserva
                        paper_sol  += cantidad_pos * (precio / precio_entrada) - cantidad_pos
                        paper_trades.append({
                            "par": PAR, "entrada": precio_entrada,
                            "salida": precio, "pnl": pnl, "razon": razon
                        })
                        log.info(f"💼 USDT actualizado: ${paper_usdt:.2f}")
                        paper_resumen()
                    else:
                        ejecutar_venta_real(cantidad_pos)

                    en_posicion = False

            # ------------------------------------------------
            # REACTIVAR TRAS CRASH
            # ------------------------------------------------
            if not bot_activo:
                try:
                    df     = obtener_datos(limit=50)
                    cambio = ((df["close"].iloc[-1] - df["close"].iloc[-7]) / df["close"].iloc[-7]) * 100
                    ema200 = calcular_ema(df["close"], 200).iloc[-1]
                    if cambio > 3.0 and df["close"].iloc[-1] > ema200:
                        log.info(f"✅ Mercado recuperado (+{cambio:.2f}%). Reactivando bot.")
                        bot_activo = True
                except:
                    pass

                if not bot_activo:
                    log.info("😴 Bot pausado por crash. Revisando en 1h...")
                    time.sleep(3600)
                    continue

            # ------------------------------------------------
            # BUSCAR ENTRADA
            # ------------------------------------------------
            if not en_posicion and bot_activo:
                senal = detectar_senal()
                log.info(f"RSI: {senal['rsi']:.1f} | ADX: {senal['adx']:.1f} | Trend: {senal['trend']} | Breakout: {senal['breakout']}")

                if senal["buy"]:
                    precio = senal["price"]
                    log.info(f"🟢 SEÑAL DE COMPRA | ${precio:.2f} | SL: ${senal['sl']:.2f} | TP: ${senal['tp']:.2f}")

                    if PAPER_TRADING:
                        # Usar USDT disponible + valor SOL reserva
                        capital_total = calcular_capital_total(precio)
                        cantidad      = (capital_total * 0.95) / precio
                        en_posicion   = True
                        precio_entrada = precio
                        sl_actual     = senal["sl"]
                        tp_actual     = senal["tp"]
                        cantidad_pos  = cantidad
                        paper_usdt   -= CAPITAL_USDT
                        paper_sol    -= SOL_RESERVA
                        log.info(f"📝 COMPRA SIMULADA: {cantidad:.4f} SOL a ${precio:.2f}")
                        log.info(f"   Capital usado: ${capital_total:.2f} (${CAPITAL_USDT} USDT + {SOL_RESERVA} SOL)")
                    else:
                        # Real: comprar con USDT + convertir SOL reserva
                        balance_usdt = obtener_balance_usdt()
                        balance_sol  = obtener_balance_sol()
                        capital_usdt = min(CAPITAL_USDT, balance_usdt)

                        # Vender SOL reserva primero si hay disponible
                        sol_disponible = min(SOL_RESERVA, balance_sol)
                        if sol_disponible > 0:
                            log.info(f"🔄 Convirtiendo {sol_disponible:.4f} SOL a USDT...")
                            ejecutar_venta_real(sol_disponible)
                            time.sleep(2)
                            capital_usdt = min(obtener_balance_usdt(), CAPITAL_USDT + sol_disponible * senal["price"])

                        if capital_usdt >= 5:
                            cantidad, precio_real = ejecutar_compra_real(capital_usdt)
                            if cantidad:
                                en_posicion    = True
                                precio_entrada = precio_real
                                sl_actual      = senal["sl"]
                                tp_actual      = senal["tp"]
                                cantidad_pos   = cantidad
                        else:
                            log.warning(f"Capital insuficiente: ${capital_usdt:.2f}")
                else:
                    log.info("⏳ Sin señal. Esperando próximo ciclo.")

            log.info("⏳ Próximo ciclo en 1 hora...")
            time.sleep(3600)

        except KeyboardInterrupt:
            log.info("Bot detenido.")
            paper_resumen()
            break
        except Exception as e:
            log.error(f"Error: {e}")
            time.sleep(300)

if __name__ == "__main__":
    main()
