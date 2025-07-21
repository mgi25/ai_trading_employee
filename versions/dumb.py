
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, time, timezone
import time as ptime
import logging
import sys
from statsmodels.tsa.stattools import adfuller
import os
from datetime import datetime, timedelta
import ta
from ta.volatility import AverageTrueRange
from datetime import timezone
from decimal import Decimal
sys.stdout.reconfigure(encoding='utf-8')

# === STRATEGY SETTINGS ===
LOT_SIZE = 0.01
PRICE_STEP = 0.01
SKEW_THRESHOLD = 0.05
VOL_SPIKE_FACTOR = 0.8
MAGIC = 123456
HEDGE_TRIGGER_PERCENT = 2
SESSION_START = time(0, 0)   # 00:00 UTC -> 5:30 AM IST
SESSION_END = time(23, 59)   # 23:59 UTC -> 5:29 AM IST next day
# Global Day session range -> 9:30 AM to 8:30 PM IST (04:00 to 15:00 UTC)
DAY_SESSION_START = time(0, 0)
DAY_SESSION_END = time(23, 59)
PIP_SIZE        = 0.01   # 1 pip on XAUUSD micro-pip pricing
HEDGE_STEP_PIPS = 100     # want 20 pips per opposite-side hedge
STEP_PRICE      = PIP_SIZE * HEDGE_STEP_PIPS  # -> 0.20
GRID_TOLERANCE  = 0.05
# === RECOVERY SNIPER SETTINGS ===
last_sniper_time = datetime.min
recovery_snipers_fired = 0
MAX_RECOVERY_SNIPERS = 3
current_leg = 1
entry_sequence = []
hedge_mode = False
last_price_snapshot = None
last_entry_side = None
price_left_zone = True
last_sniper_time = datetime.min
recovery_snipers_fired = 0
locked_active = False
locked_loss = 0.0
recovery_attempts = 0
post_lock_recovery = False
base_entry_price = None  # 🔑 This tracks the fixed starting price for all hedge legs
hedged_tickets = set()
restacked_snipers        = set()
recovery_hedged_tickets  = set()
recovery_sniper_count = 0
recovery_hedge_count = 0
lock_order_ticket = None  

# === CONFIGURATION ===
SPREAD_LIMIT = 1.5
PROFIT_SCALP_TARGET = 0.25
HEDGE_TRIGGER_PIPS = 100  # <--- 🔥 Add this
post_lock_recovery_pnl = 0.0
recovery_hedge_map = {}
# === LOGGING ===
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('live_bot.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[console_handler, file_handler]
)

# === GLOBAL STATE ===
entry_sequence = []
current_leg = 1
hedge_mode = False
last_price_snapshot = 0
last_entry_side = None
price_left_zone = True

locked_active = False
locked_loss = 0.0

LOGIN = 204215535
PASSWORD = "Mgi@2005"
SERVER = "Exness-MT5Trial7"
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M1
# === CONSOLIDATION DETECTION ===
def hurst_exponent(prices):
    prices = np.asarray(prices)
    N = len(prices)
    if N < 20:
        return 0.5
    var1 = np.var(np.diff(prices))
    Hs = []
    for k in [2, 5, 10]:
        if k < N:
            var_k = np.var(prices[k:] - prices[:-k])
            if var1 > 0 and var_k > 0:
                Hs.append(0.5 * np.log(var_k / var1) / np.log(k))
    return max(0.0, min(1.0, np.mean(Hs))) if Hs else 0.5

def detect_consolidation(prices, window=100):
    prices = np.asarray(prices)
    if len(prices) < window:
        return 0.0, False

    # ATR Compression (low volatility)
    atrs = np.abs(np.diff(prices))
    atr_current = np.mean(atrs[-14:])
    atr_prev = np.mean(atrs[-50:-14]) + 1e-9
    atr_compression = 1.0 - min(1.0, atr_current / atr_prev)

    # Bollinger Band Width
    rolling_mean = pd.Series(prices).rolling(20).mean()
    rolling_std = pd.Series(prices).rolling(20).std()
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    bbw = np.mean((upper - lower) / (rolling_mean + 1e-9))
    bbw_score = 1.0 - min(1.0, bbw / (np.std(prices) + 1e-9))

    # Volume profile flatness - simulated using price range compactness
    price_range = max(prices[-window:]) - min(prices[-window:])
    price_std = np.std(prices[-window:])
    vp_flatness = 1.0 - min(1.0, price_std / (price_range + 1e-9))

    # Hurst exponent
    hurst = hurst_exponent(prices)
    hurst_score = 1.0 - min(1.0, abs(hurst - 0.5) / 0.5)

    # ADF stationarity
    try:
        adf_p = adfuller(prices)[1]
    except:
        adf_p = 1.0
    adf_score = max(0.0, 1.0 - adf_p)

    # Final Score
    score = np.mean([atr_compression, bbw_score, vp_flatness, hurst_score, adf_score])
    return score, score >= 0.7  # Now uses stricter 0.7 threshold

# === UTILITIES ===
def get_effective_balance():
    info = mt5.account_info()
    return 100 if info is None or info.balance < 110 else info.balance

def is_momentum_candle(closes, highs, lows, volumes):
    if len(closes) < 20:
        return False

    closes = np.array(closes)
    highs = np.array(highs)
    lows = np.array(lows)
    volumes = np.array(volumes)

    opens = closes[:-1]
    body_size = abs(closes[-1] - opens[-1])
    full_size = highs[-1] - lows[-1] + 1e-9
    body_ratio = body_size / full_size

    atr = np.mean(np.maximum(
        highs[-14:] - lows[-14:],
        np.maximum(
            np.abs(highs[-14:] - closes[-15:-1]),
            np.abs(lows[-14:] - closes[-15:-1])
        )
    ))

    volume_now = volumes[-1]
    avg_volume = np.mean(volumes[-20:])

    return (body_ratio > 0.6) and (full_size > atr) and (volume_now > avg_volume * 1.1)

def calc_skew(df_ticks, return_poc=False, min_tick_volume=1):
    if df_ticks.empty:
        return (None, None) if return_poc else None

    df = df_ticks.copy()
    df['price'] = (df['bid'] + df['ask']) / 2
    df['volume'] = df['volume'].replace(0, min_tick_volume)
    df['bin'] = (df['price'] / PRICE_STEP).round() * PRICE_STEP

    vp = df.groupby('bin')['volume'].sum().sort_index()
    if vp.empty:
        return (None, None) if return_poc else None

    poc = vp.idxmax()
    total_volume = vp.sum()
    cum_volume = 0.0
    va_bins = []

    for p, v in vp.sort_values(ascending=False).items():
        cum_volume += v
        va_bins.append(p)
        if cum_volume >= 0.7 * total_volume:
            break

    val, vah = min(va_bins), max(va_bins)
    width = vah - val
    mid = (val + vah) / 2
    skew = (poc - mid) / width if width > 0 else 0

    # Additional: POC drift signal
    last_poc = getattr(calc_skew, "last_poc", None)
    calc_skew.last_poc = poc
    poc_drift = abs(poc - last_poc) if last_poc else 0

    if return_poc:
        return skew, poc, poc_drift
    return skew

def is_trap_zone_expanding(entry_sequence, threshold=3.0):
    if len(entry_sequence) < 3:
        return False
    prices = [p[0] for p in entry_sequence]
    trap_range = max(prices) - min(prices)
    return trap_range > threshold
def get_higher_tf_bias():
    bars = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 30)
    if not bars or len(bars) < 20:
        return None

    closes = np.array([b['close'] for b in bars])
    ema_short = pd.Series(closes).ewm(span=5).mean().iloc[-1]
    ema_long = pd.Series(closes).ewm(span=20).mean().iloc[-1]

    return 'BULLISH' if ema_short > ema_long else 'BEARISH'

# === MT5 FUNCTIONS ===
def initialize():
    if not mt5.initialize(server=SERVER, login=LOGIN, password=PASSWORD):
        logging.error(f"MT5 init failed: {mt5.last_error()}")
        raise SystemExit
    mt5.symbol_select(SYMBOL, True)
    logging.info("Bot initialized")

def shutdown():
    mt5.shutdown()
    logging.info("MT5 shutdown")

def should_block_hedge_final(entry_sequence, bars, df_ticks, 
                                 drift_threshold_factor=1.5, max_legs_in_chop=3):
    """
    Block hedging if:
    - Still in consolidation
    - Not enough price drift outside VA
    - Already has too many legs
    """
    if not entry_sequence or len(bars) < 30 or df_ticks.empty:
        return False

    entry_price = entry_sequence[-1][0]
    latest_price = df_ticks['ask'].iloc[-1] if 'ask' in df_ticks.columns else df_ticks['price'].iloc[-1]

    # === 1. Consolidation confirmation using volatility ratio
    highs = [b['high'] for b in bars[-30:]]
    lows = [b['low'] for b in bars[-30:]]
    total_range = max(highs) - min(lows)
    avg_range = np.mean([h - l for h, l in zip(highs, lows)])
    chop_score = avg_range / total_range if total_range != 0 else 1

    is_choppy = chop_score < 0.2  # tighter range = more consolidation

    # === 2. Volume profile drift calc
    df_ticks['price'] = (df_ticks['bid'] + df_ticks['ask']) / 2
    df_ticks['volume'] = df_ticks['volume'].replace(0, 1)
    df_ticks['bin'] = (df_ticks['price'] / 0.01).round() * 0.01
    vp = df_ticks.groupby('bin')['volume'].sum().sort_index()

    if vp.empty:
        return False

    total_volume = vp.sum()
    cum_volume = 0
    va_bins = []
    for price, vol in vp.sort_values(ascending=False).items():
        cum_volume += vol
        va_bins.append(price)
        if cum_volume >= 0.7 * total_volume:
            break

    val, vah = min(va_bins), max(va_bins)
    va_width = vah - val
    price_drift = abs(latest_price - entry_price)
    drifted = price_drift > drift_threshold_factor * va_width

    # === 3. Final Decision
    if is_choppy and len(entry_sequence) >= max_legs_in_chop and not drifted:
        logging.info("[BLOCKED] Too many legs in consolidation with no breakout")
        return True
    return False

def place_entry(side, price, volume=LOT_SIZE, comment='VP_ENTRY'):
    order_type = mt5.ORDER_TYPE_BUY if side == 'BUY' else mt5.ORDER_TYPE_SELL
    req = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': SYMBOL,
        'volume': volume,
        'type': order_type,
        'price': price,
        'deviation': 10,
        'magic': MAGIC,
        'comment': comment,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(req)
    logging.info(f"{comment} -> {side} at {price:.3f}, vol={volume}, retcode={result.retcode}")
    return result

def close_position(pos, comment=''):
    tick = mt5.symbol_info_tick(SYMBOL)
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
    req = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': SYMBOL,
        'volume': pos.volume,
        'type': close_type,
        'position': pos.ticket,
        'price': close_price,
        'deviation': 10,
        'magic': MAGIC,
        'comment': comment,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC
    }
    mt5.order_send(req)
    logging.warning(f"{comment} @ {close_price:.2f} for ticket {pos.ticket}")

def close_everything():
    for pos in mt5.positions_get(symbol=SYMBOL) or []:
        close_position(pos, comment='AUTO_EXIT')

def rebuild_state():
    global entry_sequence, current_leg, hedge_mode, last_entry_side, price_left_zone, base_entry_price
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        logging.info("No open positions. Fresh start.")
        return
    entry_sequence.clear()
    for pos in sorted(positions, key=lambda x: x.time):
        side = 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL'
        entry_sequence.append((pos.price_open, side, pos.volume))
    current_leg = len(entry_sequence)
    last_entry_side = entry_sequence[-1][1]
    hedge_mode = len(entry_sequence) > 1
    price_left_zone = True
    base_entry_price = entry_sequence[0][0] if entry_sequence else None


def log_live_equity(spread=0.0, skew=0.0, vol_check=False, momentum_ok=False, 
                    consolidation_score=0.0, is_consolidating=False, 
                    action_taken="HOLD"):
    account = mt5.account_info()
    positions = mt5.positions_get(symbol=SYMBOL) or []
    lot_size = positions[-1].volume if positions else 0
    pnl = sum(p.profit for p in positions)
    timestamp = datetime.now(timezone.utc)

    file_exists = os.path.isfile("ml_trading_log.csv")
    with open("ml_trading_log.csv", "a") as f:
        if not file_exists:
            f.write("timestamp,equity,balance,lot_size,pnl,spread,skew,vol_check,momentum_ok,consolidation_score,is_consolidating,hedge_mode,current_leg,action_taken\n")
        f.write(f"{timestamp},{account.equity:.2f},{account.balance:.2f},{lot_size:.2f},{pnl:.2f},{spread:.5f},{skew:.5f},{int(vol_check)},{int(momentum_ok)},{consolidation_score:.5f},{int(is_consolidating)},{int(hedge_mode)},{current_leg},{action_taken}\n")

def init_csv_file():
    """Ensure the ML logging file exists and has the correct header."""
    filepath = "ml_trading_log.csv"
    if not os.path.isfile(filepath):
        with open(filepath, "w") as f:
            f.write("timestamp,equity,balance,lot_size,pnl,spread,skew,vol_check,momentum_ok,consolidation_score,is_consolidating,hedge_mode,current_leg,action_taken\n")
def is_consolidation(bars, atr_period=14, window=20, atr_contract_thresh=0.8):
    if len(bars) < atr_period + 1 or len(bars) < window:
        return False

    tr = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]['high'], bars[i]['low'], bars[i - 1]['close']
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))

    curr_atr = sum(tr[-atr_period:]) / atr_period
    long_term = sum(tr) / len(tr)
    vol_contracted = (long_term > 0) and (curr_atr / long_term < atr_contract_thresh)

    highs = [b['high'] for b in bars[-window:]]
    lows = [b['low'] for b in bars[-window:]]
    band = max(highs) - min(lows)
    tight_band = band < 1.5 * curr_atr

    return vol_contracted and tight_band

def should_force_hedge_consolidation(tick_price, entry_price, df_ticks, bars, entry_sequence,
                                     min_breakout_factor=2.0, min_volume_ratio=1.2, momentum_lookback=2):
    if df_ticks.empty or len(bars) < 20 or not entry_sequence:
        return False

    # === Volume Profile Setup
    df_ticks['price'] = (df_ticks['bid'] + df_ticks['ask']) / 2
    df_ticks['volume'] = df_ticks['volume'].replace(0, 1)
    df_ticks['bin'] = (df_ticks['price'] / PRICE_STEP).round() * PRICE_STEP
    vp = df_ticks.groupby('bin')['volume'].sum().sort_index()
    if vp.empty:
        return False

    total_vol = vp.sum()
    cum_vol = 0
    va_bins = []
    for price, vol in vp.sort_values(ascending=False).items():
        cum_vol += vol
        va_bins.append(price)
        if cum_vol >= 0.7 * total_vol:
            break
    val, vah = min(va_bins), max(va_bins)
    va_width = vah - val + 1e-9
    price_drift = abs(tick_price - entry_price)
    breakout_factor = price_drift / va_width

    # === Trap Rejection Check
    trap_low, trap_high = get_trap_zone(entry_sequence)
    if trap_low and trap_high and trap_low < tick_price < trap_high:
        logging.info("[TRAP] Price still inside trap bounds. No hedge.")
        return False

    # === Volume Confirmation
    recent_volumes = [b['tick_volume'] for b in bars[-20:]]
    vol_now = recent_volumes[-1]
    avg_vol = np.mean(recent_volumes[:-1])
    volume_ok = vol_now > min_volume_ratio * avg_vol

    # === Momentum Check
    closes = [b['close'] for b in bars[-(momentum_lookback + 1):]]
    if len(closes) < momentum_lookback + 1:
        return False
    momentum_ok = abs(closes[-1] - closes[-2]) > 0.1

    # === Strong Candle Confirmation
    strong_candle = is_strong_candle([b['close'] for b in bars], [b['high'] for b in bars], [b['low'] for b in bars])

    # === Final Logic
    if breakout_factor >= min_breakout_factor and volume_ok and momentum_ok and strong_candle:
        logging.info(f"[HEDGE CONFIRMED] BreakoutFactor={breakout_factor:.2f} | VolOK={volume_ok} | Momentum={momentum_ok} | CandleStrong={strong_candle}")
        return True

    logging.info(f"[HEDGE BLOCKED] BreakoutFactor={breakout_factor:.2f} | VolOK={volume_ok} | Momentum={momentum_ok} | CandleStrong={strong_candle}")
    return False


def handle_recovery_trap(entry_sequence, bars, df_ticks, place_entry_func,
                         trap_zone_width=2.0, momentum_lookback=2, sniper_cooldown_minutes=5,
                         max_snipers=2, max_volume=0.16):
    global last_sniper_time, recovery_snipers_fired

    if not entry_sequence or len(bars) < 20 or df_ticks.empty:
        return False

    trap_prices = [entry[0] for entry in entry_sequence]
    trap_center = np.mean(trap_prices)
    trap_upper = trap_center + trap_zone_width
    trap_lower = trap_center - trap_zone_width
    tick_price = df_ticks['ask'].iloc[-1] if 'ask' in df_ticks.columns else df_ticks['price'].iloc[-1]

    if trap_lower <= tick_price <= trap_upper:
        return False  # Still in trap, wait for breakout

    if datetime.now() - last_sniper_time < timedelta(minutes=sniper_cooldown_minutes):
        return False
    if recovery_snipers_fired >= max_snipers:
        return False

    closes = [b['close'] for b in bars[-(momentum_lookback + 1):]]
    if len(closes) < momentum_lookback + 1:
        return False
    direction = closes[-1] - closes[-2]
    if abs(direction) < 0.1:
        return False

    if not is_strong_candle([b['close'] for b in bars], [b['high'] for b in bars], [b['low'] for b in bars]):
        return False

    if not value_area_breakout(df_ticks, tick_price):
        return False

    recent_volumes = [b['tick_volume'] for b in bars[-20:]]
    vol_now = recent_volumes[-1]
    avg_vol = np.mean(recent_volumes[:-1])
    if vol_now < 1.2 * avg_vol:
        return False

    # 🔁 Exponential lot size with cap
    volume = min(max_volume, round(0.01 * (2 ** recovery_snipers_fired), 2))
    side = 'BUY' if direction > 0 else 'SELL'
    price = tick_price

    logging.info(f"[SNIPER READY] Side={side}, Price={price:.2f}, Vol={vol_now:.2f}, Dir={direction:.2f}, Lot={volume}")

    result = place_entry_func(side, price, volume, comment='TRAP_RECOVERY_SNIPER_V3')
    if hasattr(result, 'retcode') and result.retcode == 10009:
        recovery_snipers_fired += 1
        last_sniper_time = datetime.now()
        entry_sequence.append((price, side, volume))
        return True

    return False


def is_strong_candle(closes, highs, lows):
    if len(closes) < 2:
        return False
    open_price = closes.iloc[-2] if hasattr(closes, 'iloc') else closes[-2]
    close_price = closes.iloc[-1] if hasattr(closes, 'iloc') else closes[-1]
    high = highs.iloc[-1] if hasattr(highs, 'iloc') else highs[-1]
    low = lows.iloc[-1] if hasattr(lows, 'iloc') else lows[-1]

    body = abs(close_price - open_price)
    wick = (high - low) - body
    if (body > wick) and (body > (np.std(closes[-20:]) * 0.5)):
        return True
    return False

def detect_trap_escape(entry_sequence, current_price, recent_volumes, threshold=1.5):
    if not entry_sequence or len(recent_volumes) < 10:
        return False
    trap_prices = [p[0] for p in entry_sequence]
    trap_center = np.mean(trap_prices)
    trap_range = max(abs(current_price - min(trap_prices)), abs(current_price - max(trap_prices)))

    # Check volume breakout
    avg_vol = np.mean(recent_volumes[:-1])
    vol_now = recent_volumes[-1]
    escaped = (trap_range > 1.5) and (vol_now > threshold * avg_vol)
    return escaped

def value_area_breakout(df_ticks, price, factor=1.5):
    if df_ticks.empty:
        return False

    df_ticks['price'] = (df_ticks['bid'] + df_ticks['ask']) / 2
    df_ticks['volume'] = df_ticks['volume'].replace(0, 1)
    df_ticks['bin'] = (df_ticks['price'] / PRICE_STEP).round() * PRICE_STEP

    vp = df_ticks.groupby('bin')['volume'].sum().sort_index()
    if vp.empty:
        return False

    total_vol = vp.sum()
    cum_vol = 0
    va_bins = []
    for bin_price, vol in vp.sort_values(ascending=False).items():
        cum_vol += vol
        va_bins.append(bin_price)
        if cum_vol >= 0.7 * total_vol:
            break

    val = min(va_bins)
    vah = max(va_bins)
    va_width = vah - val
    outside_va = price < val or price > vah
    distance = abs(price - ((val + vah) / 2))
    return outside_va and distance > (factor * va_width)

def is_high_confidence_entry(skew, spread, vol_check, momentum_ok, is_consolidating, strong_candle, poc_drift=0.0):
    confirmations = [
        abs(skew) >= SKEW_THRESHOLD,
        spread <= SPREAD_LIMIT,
        vol_check,
        momentum_ok,
        strong_candle,
        not is_consolidating,
        poc_drift > 0.02  # new confirmation: is POC moving?
    ]
    score = sum(confirmations)
    logging.info(f"[CONFIRMATION SCORE] {score}/7 -> {'PASS' if score >= 4 else 'FAIL'}")
    return score >= 4
def should_avoid_hedge_on_chop(entry_sequence, bars, df_ticks, skew, max_range=1.0):
    if len(bars) < 20 or not entry_sequence:
        return False

    closes = [b['close'] for b in bars]
    price_range = max(closes) - min(closes)
    recent_skew_flat = abs(skew) < 0.02

    trap_score = get_trap_score(entry_sequence)
    return (price_range < max_range) and recent_skew_flat and trap_score > 0.6


def get_value_area_bounds(df_ticks):
    if df_ticks.empty:
        return None, None

    df_ticks['price'] = (df_ticks['bid'] + df_ticks['ask']) / 2
    df_ticks['volume'] = df_ticks['volume'].replace(0, 1)
    df_ticks['bin'] = (df_ticks['price'] / PRICE_STEP).round() * PRICE_STEP

    vp = df_ticks.groupby('bin')['volume'].sum().sort_index()
    if vp.empty:
        return None, None

    total_vol = vp.sum()
    cum_vol = 0
    va_bins = []
    for bin_price, vol in vp.sort_values(ascending=False).items():
        cum_vol += vol
        va_bins.append(bin_price)
        if cum_vol >= 0.7 * total_vol:
            break

    return min(va_bins), max(va_bins)

def get_trap_zone(entry_sequence):
    if not entry_sequence:
        return None, None
    prices = [p[0] for p in entry_sequence]
    center = np.mean(prices)
    width = max(abs(center - min(prices)), abs(center - max(prices)))
    return center - width, center + width

def log_entry_criteria(skew, spread, vol_check, momentum_ok, is_consolidating, strong_candle):
    log_str = "[CONFIRMATION] -> "
    log_str += f"Skew={abs(skew) >= SKEW_THRESHOLD}, "
    log_str += f"Spread={spread <= SPREAD_LIMIT}, "
    log_str += f"Volume={vol_check}, "
    log_str += f"Momentum={momentum_ok}, "
    log_str += f"StrongCandle={strong_candle}, "
    log_str += f"NoConsolidation={not is_consolidating}"
    logging.info(log_str)

def get_trade_bias(closes, skew):
    if skew > 0.05 and closes[-1] > closes[-2]:
        return "BULLISH"
    elif skew < -0.05 and closes[-1] < closes[-2]:
        return "BEARISH"
    return "NEUTRAL"
def draw_console_debug_info(price, va_low, va_high, trap_low, trap_high):
    logging.info(f"[ZONE] Price={price:.2f}, VA=[{va_low:.2f}, {va_high:.2f}], Trap=[{trap_low:.2f}, {trap_high:.2f}]")

def is_stack_allowed(entry_sequence, bars, df_ticks):
    if len(entry_sequence) < 2:
        return True

    # Check if we’re in consolidation and already have too many entries
    if is_consolidation(bars) and len(entry_sequence) >= 3:
        logging.warning("[STACK BLOCK] Too many entries inside consolidation")
        return False

    # Check if trap zone is too wide
    trap_prices = [entry[0] for entry in entry_sequence]
    trap_width = max(trap_prices) - min(trap_prices)
    if trap_width > 5.0:
        logging.warning(f"[STACK BLOCK] Trap zone width too high: {trap_width:.2f}")
        return False

    return True

def is_entry_bias_valid(skew, closes):
    bias = get_trade_bias(closes, skew)
    if bias == "BULLISH" and skew < 0:
        logging.info("[BIAS FILTER] Bias is BULLISH but skew is SELL -> Blocked")
        return False
    elif bias == "BEARISH" and skew > 0:
        logging.info("[BIAS FILTER] Bias is BEARISH but skew is BUY -> Blocked")
        return False
    return True

def should_throttle_due_to_drawdown():
    #acc = mt5.account_info()
    #if not acc:
     #   return False
    #drawdown = acc.balance - acc.equity
    #if drawdown >= 5:  # tweak as needed
     #   logging.warning(f"[THROTTLE] Drawdown too high: {drawdown:.2f} -> throttle entries")
      #  return True
    return False

def is_night_session():
    hour = datetime.now(timezone.utc).hour
    return not (DAY_SESSION_START <= datetime.now(timezone.utc).time() <= DAY_SESSION_END)

def is_strong_body_candle(close, open_, high, low, min_body_ratio=0.6):
    body = abs(close - open_)
    full_range = high - low + 1e-9
    return (body / full_range) >= min_body_ratio

def is_sniper_entry(closes, opens, highs, lows, volumes, skew, spread, is_consolidating):
    if len(closes) < 20:
        return False

    close = closes[-1]
    open_ = opens[-1]
    high = highs[-1]
    low = lows[-1]
    volume_now = volumes[-1]
    avg_volume = np.mean(volumes[-20:])

    body_ok = is_strong_body_candle(close, open_, high, low)
    vol_ok = volume_now > avg_volume * VOL_SPIKE_FACTOR
    momentum_ok = is_momentum_candle(closes, highs, lows, volumes)
    strong_candle = is_strong_candle(closes, highs, lows)
    skew_ok = abs(skew) >= SKEW_THRESHOLD
    spread_ok = spread <= SPREAD_LIMIT
    no_consolidation = not is_consolidating

    # ✅ HTF Bias filter
    bias = get_higher_tf_bias()
    bias_valid = True
    if bias:
        bias_valid = (bias == 'BULLISH' and skew > 0) or (bias == 'BEARISH' and skew < 0)

    confirmations = [
        body_ok,
        vol_ok,
        momentum_ok,
        strong_candle,
        skew_ok,
        spread_ok,
        no_consolidation
    ]

    passed = sum(confirmations)
    logging.info(f"[SNIPER CHECK] Body={body_ok} | Vol={vol_ok} | Momentum={momentum_ok} | Strong={strong_candle} | Skew={skew_ok} | Spread={spread_ok} | NoConsolidation={no_consolidation} | BiasMatch={bias_valid} -> Score: {passed}/7")

    return passed >= 6 and bias_valid

def get_higher_tf_bias():
    bars_15m = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 50)
    if not bars_15m or len(bars_15m) < 2:
        return None
    close_now = bars_15m[-1]['close']
    close_prev = bars_15m[-2]['close']
    return 'BULLISH' if close_now > close_prev else 'BEARISH'

def trap_breakout_confirmed(entry_sequence, current_price, recent_prices, confirmation_candles=3, min_distance=0.5):
    if not entry_sequence or len(recent_prices) < confirmation_candles:
        return False

    trap_low, trap_high = get_trap_zone(entry_sequence)
    if not trap_low or not trap_high:
        return False

    # Price must close outside trap for confirmation_candles straight
    outside_count = 0
    for p in recent_prices[-confirmation_candles:]:
        if p < trap_low - min_distance or p > trap_high + min_distance:
            outside_count += 1

    return outside_count >= confirmation_candles

def get_trap_score(entry_sequence, max_allowed_entries=5, max_zone_width=3.0):
    if not entry_sequence:
        return 0.0

    prices = [p[0] for p in entry_sequence]
    zone_width = max(prices) - min(prices)
    entry_count = len(entry_sequence)

    width_score = min(1.0, zone_width / max_zone_width)
    entry_score = min(1.0, entry_count / max_allowed_entries)

    trap_score = 0.5 * width_score + 0.5 * entry_score
    return trap_score  # closer to 1 = stale, dangerous trap
# ─── Lux-Algo consolidation detector (single source-of-truth) ────────────────
LUX_LEN      = 8
LUX_ATR_LEN  = 300
LUX_MULT     = 0.9
LUX_MIN_DUR  = 3          # candles that must persist before locking
LUX_MIN_BARS = 100        # how many M1 bars we need for a reading

def get_lux_ranges(df_bars, length=LUX_LEN, mult=LUX_MULT,
                   atr_len=LUX_ATR_LEN, min_duration=LUX_MIN_DUR):
    closes = df_bars['close'].values
    highs  = df_bars['high'].values
    lows   = df_bars['low'].values
    times  = pd.to_datetime(df_bars['time'], unit='s', utc=True)

    ma  = pd.Series(closes).rolling(length).mean().values
    atr = ta.volatility.average_true_range(
        pd.Series(highs), pd.Series(lows), pd.Series(closes), atr_len
    ).values * mult

    ranges, cur, dur = [], None, 0
    for i in range(length, len(closes)):
        if np.isnan(ma[i]) or np.isnan(atr[i]):
            continue
        top = ma[i] + atr[i]
        bot = ma[i] - atr[i]
        # “compressed” if ≤1 candle closes outside ±ATR band
        if np.sum(np.abs(closes[i-length:i] - ma[i]) > atr[i]) <= 1:
            if cur is None:                         # start a new box
                cur = dict(start=times[i-length],
                           end=times[i],
                           top=top,
                           bottom=bot)
                dur = 1
            else:                                   # extend the box
                cur['end']    = times[i]
                cur['top']    = max(cur['top'], top)
                cur['bottom'] = min(cur['bottom'], bot)
                dur += 1
        elif cur is not None:                       # box just broke
            if dur >= min_duration:
                ranges.append(cur)
            cur, dur = None, 0
    if cur is not None and dur >= min_duration:
        ranges.append(cur)
    return ranges

def current_lux_box(ranges, now):
    for r in ranges:
        if r['start'] <= now <= r['end']:
            return r
    return None


def update_last_action():
    global last_action_time
    last_action_time = datetime.now()
def is_100_percent_trade(skew, spread, vol_check, momentum_ok, is_consolidating, strong_candle, poc_drift):
    confirmations = [
        abs(skew) >= SKEW_THRESHOLD,
        spread <= SPREAD_LIMIT,
        vol_check,
        momentum_ok,
        strong_candle,
        not is_consolidating,
        poc_drift > 0.02
    ]
    return sum(confirmations) == 7
def price_left_lux_range(price, locked_ranges, current_time):
    """Check if price has moved outside any locked LuxAlgo range."""
    for r in locked_ranges:
        if r['start'] <= current_time <= r['end']:
            if price < r['bottom'] or price > r['top']:
                return True
    return False

def get_confirmation_score(skew, spread, vol_ok, momentum_ok, is_consolidating, strong_candle, poc_drift):
    confirmations = [
        abs(skew) >= SKEW_THRESHOLD,
        spread <= SPREAD_LIMIT,
        vol_ok,
        momentum_ok,
        strong_candle,
        not is_consolidating,
        poc_drift > 0.02
    ]
    score = sum(confirmations)
    logging.info(f"[CONFIRMATION SCORE] {score}/7 -> {'PASS' if score >= 4 else 'FAIL'}")
    return score

def get_trade_direction_based_on_skew_or_momentum(skew, closes):
    """Smart direction chooser for recovery"""
    if skew > 0.05 and closes[-1] > closes[-2]:
        return 'BUY'
    elif skew < -0.05 and closes[-1] < closes[-2]:
        return 'SELL'
    return 'BUY' if closes[-1] > closes[-2] else 'SELL'

def calculate_cumulative_hedge_lot(entry_sequence):
    base_lot = Decimal("0.01")
    hedge_lots = next_hedge_lot = round(LOT_SIZE * (current_leg + 1), 2)
    return round(base_lot + sum(hedge_lots), 2) if hedge_lots else base_lot * 2
# ─────────────────────────────────────────────────────────────
# RESET STATE – drop this in place of your old one
# ─────────────────────────────────────────────────────────────
def reset_state():
    global entry_sequence, locked_active, locked_loss, post_lock_recovery, recovery_attempts
    global base_entry_price, current_leg, post_lock_recovery_pnl
    global recovery_snipers_fired, recovery_hedge_count, recovery_sniper_count
    global restacked_snipers, recovery_hedged_tickets, hedged_tickets
    global lock_order_ticket
    post_lock_recovery_pnl   = 0.0
    entry_sequence           = []
    locked_active            = False
    locked_loss              = 0.0
    post_lock_recovery       = False
    recovery_attempts        = 0
    base_entry_price         = None
    current_leg              = 0
    recovery_snipers_fired   = 0
    recovery_hedge_count     = 0
    recovery_sniper_count    = 0
    restacked_snipers.clear()
    recovery_hedged_tickets.clear()
    hedged_tickets.clear()
    lock_order_ticket = None





def log_hedge_debug(price_now, expected_price, base_entry_price, current_leg):
    logging.info(f"[HEDGE DEBUG] Now={price_now:.2f}, Expected={expected_price:.2f}, Base={base_entry_price:.2f}, Leg={current_leg}")

def log_trade(entry_sequence):
    import csv
    from datetime import datetime
    import os

    if not entry_sequence:
        return

    filename = "Log.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Group_ID", "Timestamp", "Price", "Side", "Volume", "Type"])

        group_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.now(timezone.utc).isoformat()

        for trade in entry_sequence:
            row = [
                group_id,
                timestamp,
                trade[0],  # Price
                trade[1],  # Side
                trade[2],  # Volume
                trade[3] if len(trade) > 3 else ("LOCK" if locked_active else "ENTRY")
            ]
            writer.writerow(row)
def update_locked_loss_with_profit(pnl_gain):
    global locked_loss
    locked_loss = round(locked_loss + pnl_gain, 2)
    logging.info(f"[RECOVERY UPDATE] Locked Loss Adjusted: New Locked Loss = {locked_loss:.2f}")
def get_last_position():
    pos = mt5.positions_get(symbol=SYMBOL)
    return pos[-1] if pos else None

# ─────────────────────────────────────────────────────────────
# GRID SETTINGS (put these right after your other config section)
# ─────────────────────────────────────────────────────────────
HEDGE_TRIGGER_PIPS_MAIN = 20   # 20-pip grid for normal legs
HEDGE_TRIGGER_PIPS_REC  = 50  # 100-pip loss trigger for recovery
GRID_PIPS   =20   # spacing between legs
REC_PIPS    = 50   # loss that spawns a recovery hedge

# ─────────────────────────────────────────────────────────────
# Recovery helper: how many RECOVERY trades are currently live
# ─────────────────────────────────────────────────────────────
def count_recovery_trades(positions) -> int:
    """
    Returns the total number of open positions whose comment begins
    with 'RECOVERY'.  Works even if the broker truncates comments.
    """
    return sum(
        1 for p in positions
        if p.comment and p.comment.upper().startswith("RECOVERY")
    )


# ─────────────────────────────────────────────────────────────
# Grid math helper: constant 20-pip spacing for each hedge leg
# ─────────────────────────────────────────────────────────────

def next_grid_price(base_price: float, leg: int, hedge_side: str) -> float:
    """
    leg 1,3,5... = opposite-side hedge  (± STEP_PRICE)
    leg 2,4,6... = same-side re-entry   (back to base price)

    Example: base BUY @ 2000
      leg 1 (SELL)  -> 1999.80
      leg 2 (BUY)   -> 2000.00
      leg 3 (SELL)  -> 1999.80
      ...
    """
    opposite_leg = (leg % 2 == 1)
    if opposite_leg:
        sign = -1 if hedge_side == 'SELL' else 1  # SELL hedge wants lower price
        return round(base_price + sign * STEP_PRICE, 3)
    return round(base_price, 3)  # same-side stack
# utils ───────────────────────────────────────────
def price_is_close(target, price, tolerance=PIP_SIZE * 3):  # ≈3 pips
    return abs(target - price) <= tolerance
# ================================================
# PANIC-EXIT when price reaches next grid level
# ================================================
def panic_if_next_grid_hit(tick, base_price, current_leg, last_side):
    """
    If we already have 3 recovery trades open (sniper + hedge + restack)
    and price is sitting right where the 4-th hedge *would* go,
    close the lot instead of opening a new position.
    """
    if current_leg < 3:                 # need at least 3 legs first
        return False

    # next leg is OPPOSITE the last one
    next_side = 'SELL' if last_side == 'BUY' else 'BUY'
    target    = next_grid_price(base_price, current_leg + 1, next_side)
    quote     = tick.bid if next_side == 'SELL' else tick.ask

    if price_is_close(target, quote):
        logging.warning("[PANIC-EXIT] price hit 4th-grid level "
                        f"{target:.3f} -> closing stack")
        log_trade(entry_sequence)
        close_everything()
        reset_state()
        return True

    return False

def close_everything():
    """
    Close every open position AND cancel every working order
    on the current symbol.
    """
    # 1) positions
    for pos in mt5.positions_get(symbol=SYMBOL) or []:
        close_position(pos, comment='AUTO_EXIT')

    # 2) pending orders
    for o in mt5.orders_get(symbol=SYMBOL) or []:
        mt5.order_delete(o.ticket)
# ─────────────────────────────────────────────────────────────
# Recovery kill-switch: no 4-th recovery trade
# ─────────────────────────────────────────────────────────────
def must_panic_exit_recovery(tick) -> bool:
    """
    When we already have 3 RECOVERY trades, if price reaches the
    exact level where the 4-th would be placed, flatten the stack.
    Returns True if an exit was performed.
    """
    if not locked_active or not post_lock_recovery:
        return False

    # recovery positions only
    rec_pos = [
        p for p in (mt5.positions_get(symbol=SYMBOL) or [])
        if p.comment and p.comment.upper().startswith("RECOVERY")
    ]
    if len(rec_pos) < 3:          # need 3 legs first
        return False

    last_side = 'BUY' if rec_pos[-1].type == mt5.POSITION_TYPE_BUY else 'SELL'
    next_side = 'SELL' if last_side == 'BUY' else 'BUY'

    # base_entry_price is the pre-lock anchor
    if base_entry_price is None:
        return False

    target  = next_grid_price(base_entry_price, 4, next_side)
    quote   = tick.bid if next_side == 'SELL' else tick.ask

    if price_is_close(target, quote):
        logging.warning("[RECOVERY PANIC EXIT] price hit 4-th grid → closing stack")
        log_trade(entry_sequence)
        close_everything()        # ← cancels orders too
        reset_state()
        return True
    return False

def main():
    global current_leg, entry_sequence, hedge_mode, last_entry_side
    global base_entry_price, locked_active, locked_loss, post_lock_recovery
    global recovery_sniper_count, recovery_hedge_count
    global hedged_tickets, recovery_hedged_tickets, restacked_snipers
    global lock_order_ticket  

    # ── INIT ────────────────────────────────────────────────────────────────
    initialize()
    init_csv_file()
    rebuild_state()
    logging.info("Bot started")

    # ── HOT LOOP ────────────────────────────────────────────────────────────
    while True:
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            ptime.sleep(1)
            continue

        # —— Session & spread gates
        if not (SESSION_START <= datetime.now(timezone.utc).time() <= SESSION_END):
            ptime.sleep(1)
            continue

        spread = tick.ask - tick.bid
        if spread > SPREAD_LIMIT:
            ptime.sleep(1)
            continue

        price     = tick.ask                       # use ask for mid-logic
        positions = mt5.positions_get(symbol=SYMBOL) or []
        total_pnl = sum(p.profit for p in positions)
        # ── did the pending lock order fill? ─────────────────────────
        if lock_order_ticket is not None:
            orders = mt5.orders_get(ticket=lock_order_ticket)   # ← change here

            if orders is None:
                logging.error(f"[MT5 ERROR] orders_get failed → {mt5.last_error()}")
                # if it fails we can’t make a decision, so stay in the loop
                pass
            elif len(orders) == 0:                              # order vanished → filled or cancelled
                pos = next((p for p in positions
                            if p.comment == "PENDING_LOCK_4TH"), None)
                if pos:
                    side = 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL'
                    entry_sequence.append(
                        (pos.price_open, side, pos.volume, "AUTO_LOCK_4TH_ENTRY")
                    )

                    locked_active      = True
                    post_lock_recovery = True
                    locked_loss        = abs(sum(p.profit for p in positions))
                    current_leg        = 0
                    logging.info(f"[LOCK FILLED] {side} {pos.volume:.2f} @ {pos.price_open:.3f}")

                lock_order_ticket = None     # reset handle


        if positions and not locked_active and total_pnl >= PROFIT_SCALP_TARGET:
            log_trade(entry_sequence)
            close_everything()
            reset_state()
            continue

        # —— Get M1 bars & Lux consolidation box
        bars = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 600)
        if bars is None or len(bars) < LUX_MIN_BARS:
            ptime.sleep(1)
            continue

        df_bars           = pd.DataFrame(bars)
        df_bars['time']   = pd.to_datetime(df_bars['time'], unit='s', utc=True)
        closes            = df_bars['close'].tolist()
        time_now_utc      = df_bars['time'].iloc[-1].to_pydatetime()

        lux_ranges        = get_lux_ranges(df_bars)           # ⇦ tuned helper
        lux_box           = current_lux_box(lux_ranges, time_now_utc)
        in_lux_box        = bool(lux_box and lux_box['bottom'] <= price <= lux_box['top'])

        # ─────────────────────────────────────────────────────
        # 0)  RECOVERY KILL-SWITCH (before placing new trades)
        # ─────────────────────────────────────────────────────
        if must_panic_exit_recovery(tick):
            ptime.sleep(1)
            continue



       # ─────────────────────────────────────────────────────────────
        # 1) GRID-LOCK after 3-rd leg  → place a pending order
        # ─────────────────────────────────────────────────────────────
        positions = mt5.positions_get(symbol=SYMBOL) or []

        if not locked_active and lock_order_ticket is None and len(positions) == 3:
            # total volume per side
            buy_lot  = sum(p.volume for p in positions if p.type == mt5.POSITION_TYPE_BUY)
            sell_lot = sum(p.volume for p in positions if p.type == mt5.POSITION_TYPE_SELL)

            if buy_lot == sell_lot:                 # already balanced
                pass                                # nothing to do
            else:
                lock_side = 'SELL' if buy_lot > sell_lot else 'BUY'
                lock_lot  = round(abs(buy_lot - sell_lot), 2)

                # first opposite-side hedge price  (= leg-1 price)
                hedge_price = next(
                    (pr for pr, sd, *_ in entry_sequence if sd == lock_side),
                    None
                )
                if hedge_price is None:             # safety fallback
                    base_price = entry_sequence[0][0]
                    sign       = 1 if lock_side == 'BUY' else -1
                    hedge_price = round(base_price + sign * STEP_PRICE, 3)

                # select order type: LIMIT if price is beyond current quote, STOP otherwise
                bid, ask = tick.bid, tick.ask
                if lock_side == 'BUY':
                    order_type = (mt5.ORDER_TYPE_BUY_LIMIT 
                                if hedge_price < ask else mt5.ORDER_TYPE_BUY_STOP)
                    price_used = hedge_price
                else:
                    order_type = (mt5.ORDER_TYPE_SELL_LIMIT
                                if hedge_price > bid else mt5.ORDER_TYPE_SELL_STOP)
                    price_used = hedge_price

                req = {
                    "action":      mt5.TRADE_ACTION_PENDING,
                    "symbol":      SYMBOL,
                    "volume":      lock_lot,
                    "type":        order_type,
                    "price":       price_used,
                    "deviation":   10,
                    "magic":       MAGIC,
                    "comment":     "PENDING_LOCK_4TH",
                    "type_time":   mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN
                }
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    lock_order_ticket = res.order
                    logging.info(f"[LOCK-ORDER] {lock_side} {lock_lot:.2f} pending @ {price_used:.3f}")



         # ─────────────────────────────────────────────────────
        # 2)  QUICK-SCALP EXIT (flat-stack profit)
                # ─────────────────────────────────────────────────────
        if positions and not locked_active and total_pnl >= PROFIT_SCALP_TARGET:
            log_trade(entry_sequence)
            close_everything()
            reset_state()
            continue      

        # ─────────────────────────────────────────────────────
        # 3)  Instant LOCK if price falls back inside Lux box
        # ─────────────────────────────────────────────────────
        if positions and not locked_active and in_lux_box:
            buy_lot  = sum(p.volume for p in positions if p.type == mt5.POSITION_TYPE_BUY)
            sell_lot = sum(p.volume for p in positions if p.type == mt5.POSITION_TYPE_SELL)
            diff     = round(abs(buy_lot - sell_lot), 2)

            if diff:                                        # need hedge to balance
                lock_side  = 'SELL' if buy_lot > sell_lot else 'BUY'
                lock_price = tick.bid if lock_side == 'SELL' else tick.ask
                res        = place_entry(lock_side, lock_price, diff, "LOCK_LUX_BOX")
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    entry_sequence.append((lock_price, lock_side, diff, "LOCK_LUX_BOX"))

            locked_active, post_lock_recovery = True, True
            locked_loss   = abs(total_pnl)
            current_leg   = 0
            recovery_hedge_count = 0
            logging.info("[LOCK] Stack frozen inside Lux box")
            continue

        # ————————————————————————————————————————————————
        # 4) RECOVERY MODE
        # ————————————————————————————————————————————————
        if locked_active and post_lock_recovery:
            # 4-A: hard-cap
            snipers = [p for p in positions if p.comment == "RECOVERY_SNIPER"]
            hedges  = [p for p in positions if p.comment == "RECOVERY_HEDGE"]
            total_recovery_trades = len(snipers) + len(hedges)
            if total_recovery_trades >= 4:
                logging.warning("[LIMIT] 4 recovery trades reached -> closing stack")
                log_trade(entry_sequence)
                close_everything()
                reset_state()
                continue

            # 4-B: group & solo TP
            for sniper in snipers:
                # group TP
                group_hedges = [h for h in hedges if h.type != sniper.type and h.volume >= sniper.volume]
                combined_pnl = sniper.profit + sum(h.profit for h in group_hedges)
                if combined_pnl >= PROFIT_SCALP_TARGET and group_hedges:
                    close_position(sniper, comment="[GROUP TP - SNIPER]")
                    for h in group_hedges:
                        close_position(h, comment="[GROUP TP - HEDGE]")
                    locked_loss -= combined_pnl
                    logging.info(f"[RECOVERY] Group TP +{combined_pnl:.2f}$ • Remaining lock {locked_loss:.2f}")
                    recovery_sniper_count = recovery_hedge_count = 0
                    continue
                # solo TP
                if sniper.profit >= PROFIT_SCALP_TARGET:
                    all_rec = [p for p in positions if "RECOVERY" in p.comment]
                    total_rec_pnl = sum(p.profit for p in all_rec)
                    if total_rec_pnl >= PROFIT_SCALP_TARGET:
                        for p in all_rec:
                            close_position(p, comment="[RECOVERY TP EXIT]")
                        locked_loss = round(max(0.0, locked_loss - total_rec_pnl), 2)
                        logging.info(f"[RECOVERY EXIT] Full TP +{total_rec_pnl:.2f}$ • Remaining lock {locked_loss:+.2f}")
                        if locked_loss <= 0:
                            logging.warning("[FULL RECOVERY] Lock cleared -> flat")
                            log_trade(entry_sequence)
                            close_everything()
                            reset_state()
                        continue

            # 4-C: full account recovery
            acc = mt5.account_info()
            if acc and (acc.balance - acc.equity) <= 0:
                log_trade(entry_sequence)
                close_everything()
                reset_state()
                continue

            # 4-C-ii: FIRST RECOVERY SNIPER
            if not snipers and not in_lux_box:
                from_time = datetime.now(timezone.utc) - timedelta(minutes=5)
                ticks = mt5.copy_ticks_from(SYMBOL, from_time, 3000, mt5.COPY_TICKS_ALL)
                df_ticks = pd.DataFrame(ticks)

                if not df_ticks.empty:
                    skew, _, poc_drift = calc_skew(df_ticks, return_poc=True)
                    vol_ok   = df_bars['tick_volume'].iloc[-1] > df_bars['tick_volume'].mean()
                    mom_ok   = is_momentum_candle(closes, df_bars['high'], df_bars['low'], df_bars['tick_volume'])
                    strong   = is_strong_candle(closes, df_bars['high'], df_bars['low'])
                    score    = get_confirmation_score(skew, spread, vol_ok, mom_ok, False, strong, poc_drift)

                    logging.info(f"[SNIPER_CHECK] Score={score} in_range={in_lux_box}")

                    if score >= 4:
                        side = 'BUY' if skew > 0 else 'SELL'
                        res  = place_entry(side, price, 0.01, comment="RECOVERY_SNIPER")
                        if res.retcode == mt5.TRADE_RETCODE_DONE:
                            entry_sequence.append((price, side, 0.01, "RECOVERY_SNIPER"))
                            current_leg += 1
                            recovery_sniper_count += 1
                            logging.info(f"[RECOVERY_SNIPER] {side} 0.01 @ {price:.3f}")

            # 4-D: RESTACK (3rd leg)
            pip_tol   = 0.05  # 5-pip tolerance
            mid_price = (tick.bid + tick.ask) / 2

            for sniper in snipers:
                if sniper.ticket in restacked_snipers:
                    continue
                opp = [h for h in hedges if h.type != sniper.type]
                if len(opp) != 1:
                    continue

                expected_price = sniper.price_open
                current_price = tick.bid if sniper.type == mt5.POSITION_TYPE_BUY else tick.ask

                if abs(current_price - expected_price) <= 0.01:
                    lot  = round(sniper.volume + opp[0].volume, 2)
                    side = 'BUY' if sniper.type == mt5.POSITION_TYPE_BUY else 'SELL'
                    res  = place_entry(side, current_price, lot, comment="RECOVERY_RESTACK")

                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        entry_sequence.append((mid_price, side, lot, "RECOVERY_RESTACK"))
                        current_leg += 1
                        restacked_snipers.add(sniper.ticket)
                        logging.info(f"[RE-STACK] {side} {lot:.2f} @ {mid_price:.3f}")

            # 4-E: extreme revisit -> close all
            if restacked_snipers and len(snipers) >= 1 and len(hedges) == 1:
                hedge_price = hedges[0].price_open
                tick_price  = tick.bid if hedges[0].type == mt5.POSITION_TYPE_SELL else tick.ask
                if abs(tick_price - hedge_price) <= pip_tol:
                    positions_now = mt5.positions_get(symbol=SYMBOL) or []
                    instant_pnl = sum(p.profit for p in positions_now)

                    if instant_pnl >= 0:
                        logging.warning("[EXIT] Instant TP Triggered -> Closing All")
                        log_trade(entry_sequence)
                        close_everything()
                        reset_state()
                        continue

            # 4-F: single recovery hedge (instant)
            for sniper in snipers:
                if sniper.ticket in recovery_hedged_tickets:
                    continue

                hedge_side  = 'SELL' if sniper.type == mt5.POSITION_TYPE_BUY else 'BUY'
                current_price = tick.bid if hedge_side == 'SELL' else tick.ask
                loss_pips = abs(sniper.price_open - current_price) * 100

                if loss_pips >= HEDGE_TRIGGER_PIPS:
                    lot = round(sniper.volume + 0.01, 2)
                    res = place_entry(hedge_side, current_price, lot, comment="RECOVERY_HEDGE")
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        entry_sequence.append((current_price, hedge_side, lot, "RECOVERY_HEDGE"))
                        recovery_hedged_tickets.add(sniper.ticket)
                        current_leg += 1
                        logging.info(f"[INSTANT HEDGE] {hedge_side} {lot:.2f} @ {current_price:.3f} — Loss={loss_pips:.1f}p")

        # ———————————————————————————————————————————
        # 5) Normal grid-hedge (fixed ticket tracking)
        # ———————————————————————————————————————————
        if not locked_active and positions:
            hedged_this_tick = False

            for p in positions:
                if p.ticket in hedged_tickets:
                    continue

                hedge_side = 'SELL' if p.type == mt5.POSITION_TYPE_BUY else 'BUY'
                target_price = next_grid_price(base_entry_price, current_leg, hedge_side)
                quote = tick.ask if hedge_side == 'BUY' else tick.bid

                # Only trigger hedge if price is within GRID_TOLERANCE of target grid level
                if abs(quote - target_price) <= GRID_TOLERANCE:
                    lot = round((current_leg + 1) * LOT_SIZE, 2)
                    res = place_entry(hedge_side, quote, lot, comment="HEDGE_LAYER")
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        new_ticket = getattr(res, 'order', None) or getattr(res, 'deal', None)
                        hedged_tickets.update({p.ticket, new_ticket})
                        entry_sequence.append((quote, hedge_side, lot, "HEDGE_LAYER"))
                        current_leg += 1
                        hedged_this_tick = True
                        logging.info(f"[GRID HEDGE] {hedge_side} {lot} at {quote:.3f} | Target: {target_price:.3f}")
                        break  # Only one hedge per loop

            # 5-b) quick re-stack if we revisited base price
            if not hedged_this_tick and len(entry_sequence) == 2:
                base_price, base_side, base_lot, _ = entry_sequence[0]
                revisit_quote = tick.ask if base_side == 'BUY' else tick.bid

                # Only stack if *exactly* at base price (strict tolerance)
                if price_is_close(revisit_quote, base_price, GRID_TOLERANCE):
                    lot = LOT_SIZE * (current_leg + 1)  # e.g., 0.03 for 3rd leg
                    res = place_entry(base_side, base_price, lot, comment="REENTRY_STACK")
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        entry_sequence.append((base_price, base_side, lot, "REENTRY_STACK"))
                        current_leg += 1
                        hedged_tickets.add(getattr(res, 'order', None) or getattr(res, 'deal', None))
                        logging.info(f"[RE-STACK] {base_side} {lot:.2f} at {base_price:.3f}")


                # ─────────────────────────────────────────────────────
        # 6)  HIGH-CONFIDENCE VP ENTRY (flat only)
        # ─────────────────────────────────────────────────────
        if not positions and not locked_active:

            # hard-block entries while price is inside the Lux box
            if in_lux_box:
                logging.info("[ENTRY] blocked – price still inside Lux consolidation box")
                ptime.sleep(1)
                continue

            # pull the last 5 minutes of ticks for skew / POC
            from_time = datetime.now(timezone.utc) - timedelta(minutes=5)
            ticks     = mt5.copy_ticks_from(SYMBOL, from_time, 3000, mt5.COPY_TICKS_ALL)
            df_ticks  = pd.DataFrame(ticks)
            if df_ticks.empty:
                ptime.sleep(1)
                continue

            # — confirmation metrics —
            skew, _, poc_drift = calc_skew(df_ticks, return_poc=True)

            vol_now   = df_bars['tick_volume'].iloc[-1]
            vol_avg20 = df_bars['tick_volume'].tail(20).mean()
            vol_ok    = vol_now > vol_avg20                      # fresh volume spike

            closes_lst = df_bars['close'].tolist()
            highs_lst  = df_bars['high'].tolist()
            lows_lst   = df_bars['low'].tolist()
            vols_lst   = df_bars['tick_volume'].tolist()

            mom_ok    = is_momentum_candle(closes_lst, highs_lst, lows_lst, vols_lst)
            strong    = is_strong_candle(closes_lst, highs_lst, lows_lst)
            is_consol = detect_consolidation(closes_lst)[1]      # True = consolidating

            if is_high_confidence_entry(
                    skew, spread, vol_ok, mom_ok, is_consol, strong, poc_drift):
                side        = 'BUY' if skew > 0 else 'SELL'
                entry_price = tick.ask if side == 'BUY' else tick.bid
                res = place_entry(side, entry_price, LOT_SIZE, comment="VP_ENTRY")
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    entry_sequence.append((entry_price, side, LOT_SIZE, "VP_ENTRY"))
                    base_entry_price = entry_price
                    last_entry_side  = side
                    current_leg      = 1


        # —— loop pacing
        ptime.sleep(1)



if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception("Bot crashed")
    finally:
        shutdown()
        
        #this one mate


        # —— loop pacing
        ptime.sleep(0.1)
