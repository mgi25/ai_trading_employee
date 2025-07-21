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
volume = LOT_SIZE
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
GRID_TOLERANCE  = PIP_SIZE * 3 
# === RECOVERY SNIPER SETTINGS ===
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

# === GLOBAL STATE (Complete list, keep these at the top, right after logging setup) ===
entry_sequence = []
current_leg = 1
hedge_mode = False
last_price_snapshot = 0
last_entry_side = None
price_left_zone = True

locked_active = False
locked_loss = 0.0
lock_order_ticket = None
base_entry_price = None

hedged_tickets = set()
recovery_hedged_tickets = set()
restacked_snipers = set()

post_lock_recovery = False
recovery_attempts = 0
post_lock_recovery_pnl = 0.0

recovery_sniper_count = 0
recovery_hedge_count = 0
recovery_snipers_fired = 0
last_sniper_time = datetime.now()

pending_watchlist = {}
base_entry_side = None

lock_anchor_price = None
lock_anchor_side = None

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
    base_entry_side = entry_sequence[0][1] if entry_sequence else None


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
def pending_in_watchlist(comment, side):
    return any(
        info['comment'] == comment and info['side'] == side
        for info in pending_watchlist.values()
    )


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

def get_lux_ranges(df_bars, length=10, mult=1.0, atr_len=500, min_duration=3):
    closes = df_bars['close'].values
    highs = df_bars['high'].values
    lows = df_bars['low'].values
    times = pd.to_datetime(df_bars['time'], unit='s')

    # Safety: Adjust atr_len if not enough bars
    atr_len = min(atr_len, len(closes) - 1)

    ma = pd.Series(closes).rolling(length).mean().values
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
def cancel_all_pendings():
    """Cancel ALL pending orders for the symbol."""
    for o in mt5.orders_get(symbol=SYMBOL) or []:
        delete_order(o.ticket)
# ─────────────────────────────────────────────────────────────
# RESET STATE – drop this in place of your old one
# ─────────────────────────────────────────────────────────────
def reset_state():
    global entry_sequence, locked_active, locked_loss, post_lock_recovery, recovery_attempts
    global base_entry_price, current_leg, post_lock_recovery_pnl
    global recovery_snipers_fired, recovery_hedge_count, recovery_sniper_count
    global restacked_snipers, recovery_hedged_tickets, hedged_tickets
    global lock_order_ticket, post_lock_recovery, pending_watchlist, base_entry_side
    global lock_anchor_price, lock_anchor_side

    post_lock_recovery_pnl   = 0.0
    entry_sequence           = []
    locked_active            = False
    locked_loss              = 0.0
    post_lock_recovery       = False
    recovery_attempts        = 0
    base_entry_price         = None
    base_entry_side          = None   # <--- add this
    current_leg              = 0
    recovery_snipers_fired   = 0
    recovery_hedge_count     = 0
    recovery_sniper_count    = 0
    lock_order_ticket        = None
    pending_watchlist.clear()
    restacked_snipers.clear()
    recovery_hedged_tickets.clear()
    hedged_tickets.clear()
    cancel_all_pendings()
    lock_anchor_price = None
    lock_anchor_side = None
 # <-- Ensures that no pending grid, hedge or recovery orders remain

 # <-- Ensures that no pending grid, hedge or recovery orders remain




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

def next_grid_price(base_price: float, leg: int, last_side: str) -> float:
    """Grid price for given leg and side (anchor on base)."""
    opposite = (leg % 2 == 1)
    sign = -1 if (opposite and last_side == 'BUY') or (not opposite and last_side == 'SELL') else 1
    return round(base_price + sign * (leg // 2 + 1) * GRID_PIPS * PIP_SIZE, 3) if opposite else round(base_price, 3)

# utils ───────────────────────────────────────────
def price_is_close(target, price, tolerance=PIP_SIZE * 3):  # ≈3 pips
    return abs(target - price) <= tolerance
def place_all_grid_orders(base_entry_price, base_entry_side, num_legs=4):
    for leg in range(2, num_legs+1):
        side = next_grid_side(base_entry_side, leg)
        sign = 1 if side == 'BUY' else -1
        grid_price = round(base_entry_price + sign * (leg-1) * GRID_PIPS * PIP_SIZE, 3)
        lot = round(LOT_SIZE * leg, 2)
        if not already_pending("HEDGE_LAYER", side):
            place_pending(side, grid_price, lot, f"HEDGE_LAYER_{leg}")

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
        logging.warning("[RECOVERY PANIC EXIT] price hit 4-th grid -> closing stack")
        log_trade(entry_sequence)
        close_everything()        # ← cancels orders too
        reset_state()
        return True
    return False
# ──────────────────────────────────────────────────────────────
#  Universal pending-order helpers
# ──────────────────────────────────────────────────────────────
def pending_type_for(side: str, target: float, bid: float, ask: float):
    """
    Decide whether we need a LIMIT or a STOP order so that the
    target price is *inside* the spread when the order triggers.
    """
    if side == 'BUY':
        return mt5.ORDER_TYPE_BUY_LIMIT  if target < ask else mt5.ORDER_TYPE_BUY_STOP
    else:
        return mt5.ORDER_TYPE_SELL_LIMIT if target > bid else mt5.ORDER_TYPE_SELL_STOP


def place_pending(side: str, price: float, volume: float,
                  comment: str = "PENDING_ENTRY") -> 'mt5.TradeResult':
    """
    Wrapper that sends a LIMIT / STOP order at price.
    """
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        logging.error("[PLACE_PENDING] No tick – abort")
        return None

    order_type = pending_type_for(side, price, tick.bid, tick.ask)
    req = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       SYMBOL,
        "volume":       volume,
        "type":         order_type,
        "price":        price,
        "deviation":    10,
        "magic":        MAGIC,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    res = mt5.order_send(req)
    logging.info(f"{comment} -> {side}  {volume:.2f} @ {price:.3f}  ->  ret={res.retcode}")
    return res
def delete_order(ticket: int):
    """
    Replacement for the removed mt5.order_delete().
    """
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order":  ticket,
    }
    return mt5.order_send(request)



def close_everything():
    # close positions …
    for pos in mt5.positions_get(symbol=SYMBOL) or []:
        close_position(pos, comment='AUTO_EXIT')
    # … and wipe pendings
    cancel_all_pendings()
pending_watchlist: dict[int, dict] = {}      # <-- add near other globals
WATCH_RET_CODES = (
    mt5.TRADE_RETCODE_PLACED,   # 10023  “order placed”
    mt5.TRADE_RETCODE_DONE,     # 10009  “request accepted”  (Exness)
)
SUCCESS = WATCH_RET_CODES
# ………………………………………………………………………………………………………………………………
# helper – call right after *every* successful place_pending / order_send
def calculate_cumulative_hedge_lot(entry_sequence):
    base_lot   = Decimal("0.01")
    next_lot   = Decimal(str(round(LOT_SIZE * (current_leg + 1), 2)))
    return base_lot + next_lot
# …………………………………………………………………………………………………………
# helper – call right after every successful order_send / place_pending
def track_pending(res: 'mt5.TradeResult', comment: str, side: str) -> None:
    """Remember LIMIT/STOP orders so the bot doesn't re-submit them every loop."""
    if res and res.retcode in WATCH_RET_CODES:
        pending_watchlist[res.order] = {
            "timestamp": datetime.now(timezone.utc),
            "comment":   comment,
            "side":      side,
        }
def already_pending(comment: str, side: str) -> bool:
    return any(
        o.comment == comment and
        ((side == 'BUY'  and o.type in (mt5.ORDER_TYPE_BUY_STOP,  mt5.ORDER_TYPE_BUY_LIMIT)) or
         (side == 'SELL' and o.type in (mt5.ORDER_TYPE_SELL_STOP, mt5.ORDER_TYPE_SELL_LIMIT)))
        for o in (mt5.orders_get(symbol=SYMBOL) or [])
    )
# After your initial VP_ENTRY order is filled:
def place_initial_grid(base_price, base_side, base_lot=LOT_SIZE, grid_pips=20, grid_levels=3):
    for leg in range(1, grid_levels + 1):
        side = next_grid_side(base_side, leg)
        sign = 1 if side == 'BUY' else -1
        grid_price = round(base_price + sign * leg * grid_pips * PIP_SIZE, 3)
        lot = round(0.01 * leg, 2)  # 0.01, 0.02, 0.03, ...
        if not already_pending("HEDGE_LAYER", side):
            place_pending(side, grid_price, lot, "HEDGE_LAYER")

def place_recovery_grid(base_price, sniper_side, sniper_volume, rec_pips=50, max_rec_legs=3):
    for leg in range(1, max_rec_legs + 1):
        # Alternate hedge side each leg
        side = 'SELL' if (leg % 2 == 1 and sniper_side == 'BUY') or (leg % 2 == 0 and sniper_side == 'SELL') else 'BUY'
        sign = 1 if side == 'BUY' else -1
        grid_price = round(base_price + sign * leg * rec_pips * PIP_SIZE, 3)
        lot = round(sniper_volume + 0.01 * leg, 2)
        if not already_pending("RECOVERY_HEDGE", side):
            place_pending(side, grid_price, lot, "RECOVERY_HEDGE")

def next_grid_side(base_side, leg):
    """
    Returns the correct side (BUY/SELL) for any leg number in the grid,
    always alternating starting from the original entry.
    Leg 1: base_side
    Leg 2: opposite
    Leg 3: base_side
    Leg 4: opposite
    ...
    """
    if leg % 2 == 1:
        return base_side
    else:
        return 'BUY' if base_side == 'SELL' else 'SELL'


UNICODE_SAFE = {'PLUS': '[+]', 'ARROW': '->'}   # keep cp1252 consoles happy
def place_grid_pending_orders(base_entry, base_side, start_leg=2, end_leg=4):
    """
    Places pending grid orders from leg=start_leg to leg=end_leg (inclusive).
    Leg 2: Opposite side, base_entry ± 1*step, 0.02
    Leg 3: Same side,    base_entry,          0.03
    Leg 4: Opposite side,base_entry ± 2*step, 0.04
    """
    # Leg 2: Opposite, ±1*step
    opp_side = 'SELL' if base_side == 'BUY' else 'BUY'
    step_sign = -1 if base_side == 'BUY' else 1
    leg_prices = [
        (2, opp_side, round(base_entry + step_sign * STEP_PRICE, 3), 0.02, "GRID_LEG_2"),
        (3, base_side, base_entry, 0.03, "GRID_LEG_3"),
        (4, opp_side, round(base_entry + step_sign * 2 * STEP_PRICE, 3), 0.04, "GRID_LEG_4"),
    ]
    for leg, side, price, lot, comment in leg_prices:
        if not already_pending(comment, side):
            place_pending(side, price, lot, comment)
def price_between_last_buy_sell(entry_sequence, price):
    buys  = [e[0] for e in entry_sequence if e[1] == 'BUY']
    sells = [e[0] for e in entry_sequence if e[1] == 'SELL']
    if not buys or not sells:
        return False
    min_buy  = min(buys)
    max_sell = max(sells)
    low  = min(min_buy, max_sell)
    high = max(min_buy, max_sell)
    return low < price < high
def sync_pendings():
    """Check if any pending orders in watchlist have been filled."""
    positions = mt5.positions_get(symbol=SYMBOL) or []
    to_remove = []
    for oid, info in pending_watchlist.items():
        # If order is no longer in orders_get, it's filled or canceled
        order = mt5.orders_get(ticket=oid)
        if not order:  # Not found = filled or canceled
            # Try to match with position (by comment and side)
            for pos in positions:
                pos_side = 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL'
                if pos_side == info['side'] and info['comment'] in (pos.comment or ''):
                    # Pending filled, now it's a position
                    entry_sequence.append((pos.price_open, pos_side, pos.volume, info['comment']))
                    break
            to_remove.append(oid)
    for oid in to_remove:
        pending_watchlist.pop(oid, None)
    # Also, after any new fill, rebuild state to update current_leg
    rebuild_state()
def ensure_recovery_grid(lock_anchor_price, lock_anchor_side, current_num_rec):
    # Places the next required RECOVERY pending order if needed
    # Always keeps RECOVERY_2 and RECOVERY_3 grid orders working if possible

    # RECOVERY_2
    if lock_anchor_price is None or lock_anchor_side is None:
        logging.error("[ensure_recovery_grid] Called with None anchor price or side! Skipping recovery grid placement.")
        return
    if current_num_rec == 1 and not already_pending("RECOVERY_2", lock_anchor_side):
        rec_side = lock_anchor_side
        rec_lot = 0.02
        sign = 1 if rec_side == 'BUY' else -1
        grid_price = round(lock_anchor_price + sign * STEP_PRICE, 3)
        comment = "RECOVERY_2"
        res = place_pending(rec_side, grid_price, rec_lot, comment)
        if hasattr(res, 'retcode') and res.retcode in SUCCESS:
            pending_watchlist[res.order] = {"comment": comment, "side": rec_side}

    # RECOVERY_3
    elif current_num_rec == 2 and not already_pending("RECOVERY_3", 'SELL' if lock_anchor_side == 'BUY' else 'BUY'):
        rec_side = 'SELL' if lock_anchor_side == 'BUY' else 'BUY'
        rec_lot = 0.03
        sign = 1 if rec_side == 'BUY' else -1
        grid_price = round(lock_anchor_price + sign * 2 * STEP_PRICE, 3)
        comment = "RECOVERY_3"
        res = place_pending(rec_side, grid_price, rec_lot, comment)
        if hasattr(res, 'retcode') and res.retcode in SUCCESS:
            pending_watchlist[res.order] = {"comment": comment, "side": rec_side}
def move_all_stops_to_lock_price(lock_price):
    for pos in mt5.positions_get(symbol=SYMBOL) or []:
        ticket = pos.ticket
        volume = pos.volume
        side = 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL'

        # SL should be *below* for BUY, *above* for SELL
        if side == 'BUY' and lock_price < pos.price_open:
            sl = lock_price
        elif side == 'SELL' and lock_price > pos.price_open:
            sl = lock_price
        else:
            continue  # skip if lock price would instantly stop out

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": 0.0,  # No TP
        }
        result = mt5.order_send(req)
        logging.info(f"Moved SL for ticket {ticket} to {sl}. Retcode={result.retcode}")
def recovery_grid_price(lock_anchor_price, lock_anchor_side, leg):
    """
    For recovery after lock:
    leg 1: market, opposite side, 0.01 (done in logic below)
    leg 2: lock_anchor_side, 0.02, at lock_anchor_price ± step
    leg 3: opposite to lock_anchor_side, 0.03, at lock_anchor_price
    """
    if leg == 2:
        sign = 1 if lock_anchor_side == 'BUY' else -1
        return round(lock_anchor_price + sign * STEP_PRICE, 3)
    elif leg == 3:
        return round(lock_anchor_price, 3)
    else:
        raise ValueError("Invalid recovery leg (should be 2 or 3)")

def main():
    global current_leg, entry_sequence, locked_active, lock_order_ticket
    global base_entry_price, last_entry_side, post_lock_recovery, pending_watchlist
    global lock_anchor_price, lock_anchor_side



    # ── INIT ────────────────────────────────────────────────────────────────
    initialize()
    init_csv_file()
    reset_state()
    logging.info("Bot started")

    # ── HOT LOOP ────────────────────────────────────────────────────────────
    while True:
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            ptime.sleep(1)
            continue

        # ── Session & spread gate
        now = datetime.now(timezone.utc).time()
        if not (SESSION_START <= now <= SESSION_END):
            ptime.sleep(1); continue
        spread = tick.ask - tick.bid
        if spread > SPREAD_LIMIT:
            ptime.sleep(1); continue

        # ── Sync fills for our pending orders
        sync_pendings()


        positions = mt5.positions_get(symbol=SYMBOL) or []
        total_pnl = sum(p.profit for p in positions)
        # --- FETCH LuxAlgo box (consolidation range) ---
        bars_df = pd.DataFrame(mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 120))
        now_utc = pd.Timestamp.utcnow()
        lux_ranges = get_lux_ranges(bars_df)
        lux_box = current_lux_box(lux_ranges, now_utc)
        in_lux_box = False
        if lux_box:
            in_lux_box = (lux_box['bottom'] <= tick.bid <= lux_box['top']) or (lux_box['bottom'] <= tick.ask <= lux_box['top'])

        # --- Trap zone: price between last BUY/SELL? ---
        trap_zone = price_between_last_buy_sell(entry_sequence, tick.bid)

        # --- LOCK LOGIC: If inside Lux box AND inside trap zone ---
        if positions and not locked_active and in_lux_box and trap_zone:
            last_side = entry_sequence[-1][1] if entry_sequence else None
            lock_side = 'SELL' if last_side == 'BUY' else 'BUY'
            lock_lot = sum(e[2] for e in entry_sequence if e[1] != lock_side)
            lock_price = tick.bid if lock_side == 'SELL' else tick.ask
            if lock_lot > 0.0:
                res = place_entry(lock_side, lock_price, lock_lot, "LUX_LOCK")
                if hasattr(res, 'retcode') and res.retcode in SUCCESS:
                    locked_active = True
                    lock_order_ticket = getattr(res, 'order', None)
                    lock_anchor_price = lock_price
                    lock_anchor_side = lock_side
                    logging.info("Locked: waiting for lock to fill, not resetting state yet")
                    # Do not call reset_state() here! Wait until stack is closed.
                    continue





        # ── Profit‐target exit
        if positions and not locked_active and total_pnl >= PROFIT_SCALP_TARGET:
            log_trade(entry_sequence)
            close_everything()
            reset_state()
            continue

        # ── Handle lock fill (leg 4)
        # Handle lock fill (leg 4)
        if lock_order_ticket is not None:
            orders = mt5.orders_get(ticket=lock_order_ticket)
            if orders is not None and len(orders) == 0:
                rebuild_state()
                locked_active = True
                post_lock_recovery = True
                logging.info("[LOCK FILLED] Entering recovery mode")

                # --- Fix: Set anchor vars if not set ---
                if lock_anchor_price is None or lock_anchor_side is None:
                    # Find the lock position just opened (should have "LUX_LOCK" in comment)
                    lock_pos = [
                        p for p in (mt5.positions_get(symbol=SYMBOL) or [])
                        if p.comment and "LUX_LOCK" in p.comment
                    ]
                    if lock_pos:
                        lock_anchor_price = lock_pos[0].price_open
                        lock_anchor_side = 'BUY' if lock_pos[0].type == mt5.POSITION_TYPE_BUY else 'SELL'
                    else:
                        # Fallback: use tick
                        lock_anchor_price = tick.bid if last_entry_side == 'BUY' else tick.ask
                        lock_anchor_side = 'SELL' if last_entry_side == 'BUY' else 'BUY'

                lock_order_ticket = None


        # ── MARKET & GRID SEQUENCE ────────────────────────────────────────────

        # 1) Initial Market Entry → Leg 2 Pending
        if not positions and not locked_active:
            # compute entry signal
            from_time = datetime.now(timezone.utc) - timedelta(minutes=5)
            ticks = mt5.copy_ticks_from(SYMBOL, from_time, 3000, mt5.COPY_TICKS_ALL)
            df_ticks = pd.DataFrame(ticks)
            if not df_ticks.empty:
                skew, _, poc_drift = calc_skew(df_ticks, return_poc=True)
                vol_ok = df_ticks['volume'].iloc[-1] > df_ticks['volume'].mean()
                bars = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)
                closes = [b['close'] for b in bars]
                highs = [b['high'] for b in bars]
                lows = [b['low'] for b in bars]
                mom_ok = is_momentum_candle(closes, highs, lows, [b['tick_volume'] for b in bars])
                is_consol = detect_consolidation(closes)[1]
                strong = is_strong_candle(closes, highs, lows)
                if is_high_confidence_entry(skew, spread, vol_ok, mom_ok, is_consol, strong, poc_drift):
                    # In your first entry logic:
                    side = 'BUY' if skew > 0 else 'SELL'
                    entry_price = tick.ask if side == 'BUY' else tick.bid
                    res = place_entry(side, entry_price, LOT_SIZE, "GRID_1_MARKET")
                    if hasattr(res, 'retcode') and res.retcode == mt5.TRADE_RETCODE_DONE:
                        entry_sequence.append((entry_price, side, LOT_SIZE))
                        base_entry_price = entry_price
                        base_entry_side = side  # <--- SET ONCE HERE!
                        last_entry_side = side
                        current_leg = 1
                        # Leg 2 (opposite side) pending
                        opp = next_grid_side(base_entry_side, 2)
                        leg2_price = entry_price - STEP_PRICE if base_entry_side == 'BUY' else entry_price + STEP_PRICE
                        res2 = place_pending(opp, leg2_price, LOT_SIZE*2, "GRID_2_PENDING")
                        if hasattr(res2, 'retcode') and res2.retcode in SUCCESS:
                            pending_watchlist[res2.order] = {"comment":"GRID_2_PENDING","side":opp}


        # 2) After Leg 2 fills → Leg 3 pending
        if current_leg == 2 and not any(v["comment"]=="GRID_3_PENDING" for v in pending_watchlist.values()):
            side_leg3 = next_grid_side(base_entry_side, 3)
            if side_leg3 == base_entry_side:
                leg3_price = base_entry_price
            else:
                sign = 1 if side_leg3 == 'BUY' else -1
                leg3_price = round(base_entry_price + sign * STEP_PRICE, 3)
            res3 = place_pending(side_leg3, leg3_price, LOT_SIZE*3, "GRID_3_PENDING")
            if hasattr(res3,'retcode') and res3.retcode in SUCCESS:
                pending_watchlist[res3.order] = {"comment":"GRID_3_PENDING","side":side_leg3}


        # 3) After Leg 3 fills → Lock (Leg 4) pending at Leg 2 price
        if current_leg == 3 and lock_order_ticket is None:
            leg2_price = entry_sequence[1][0]
            lot2 = entry_sequence[1][2]
            lock_side = 'SELL' if last_entry_side=='BUY' else 'BUY'
            res4 = place_pending(lock_side, leg2_price, lot2, "PENDING_LOCK_4TH")
            if hasattr(res4,'retcode') and res4.retcode in SUCCESS:
                lock_order_ticket = res4.order
                logging.info(f"[LOCK PENDING] {lock_side} {lot2:.2f} @ {leg2_price:.3f}")

        # ── RECOVERY SEQUENCE ──────────────────────────────────────────────────
        if locked_active and post_lock_recovery:
            rec_positions = [
                p for p in positions if p.comment and p.comment.startswith("RECOVERY")
            ]
            num_rec = len(rec_positions)

            # 1st: Market recovery, opposite side to lock
            if num_rec == 0:
                rec_side = 'SELL' if lock_anchor_side == 'BUY' else 'BUY'
                rec_lot = 0.01
                rec_price = tick.ask if rec_side == 'BUY' else tick.bid
                comment = "RECOVERY_1"
                res = place_entry(rec_side, rec_price, rec_lot, comment)
                if hasattr(res, 'retcode') and res.retcode == mt5.TRADE_RETCODE_DONE:
                    entry_sequence.append((rec_price, rec_side, rec_lot, comment))

            # 2nd: Pending, lock_anchor_side, 0.02, at anchor ± step
            elif num_rec == 1 and not already_pending("RECOVERY_2", lock_anchor_side):
                rec_side = lock_anchor_side
                rec_lot = 0.02
                rec_price = recovery_grid_price(lock_anchor_price, lock_anchor_side, 2)
                comment = "RECOVERY_2"
                res = place_pending(rec_side, rec_price, rec_lot, comment)
                if hasattr(res, 'retcode') and res.retcode in SUCCESS:
                    pending_watchlist[res.order] = {"comment": comment, "side": rec_side}

            # 3rd: Pending, opposite to lock_anchor_side, 0.03, at anchor
            elif num_rec == 2 and not already_pending("RECOVERY_3", 'SELL' if lock_anchor_side == 'BUY' else 'BUY'):
                rec_side = 'SELL' if lock_anchor_side == 'BUY' else 'BUY'
                rec_lot = 0.03
                rec_price = recovery_grid_price(lock_anchor_price, lock_anchor_side, 3)
                comment = "RECOVERY_3"
                res = place_pending(rec_side, rec_price, rec_lot, comment)
                if hasattr(res, 'retcode') and res.retcode in SUCCESS:
                    pending_watchlist[res.order] = {"comment": comment, "side": rec_side}

            # After all 3 filled, close all (panic exit)
            elif num_rec >= 3:
                logging.warning("[RECOVERY PANIC EXIT] 3 recoveries placed, closing stack.")
                log_trade(entry_sequence)
                close_everything()
                reset_state()
                continue

            # c) No 4th recovery → rely on must_panic_exit_recovery()

        # ── Loop pacing ───────────────────────────────────────────────────────
        ptime.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception("Bot crashed")
    finally:
        shutdown()