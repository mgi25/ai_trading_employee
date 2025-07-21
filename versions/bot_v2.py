#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────
#  XAUUSD – Advanced 1-Minute EMA/ATR Micro-Scalper
#     • One trade at a time
#     • Dynamic TP/SL trail
#     • Daily loss / win guard
#     • Spread + volatility filters
# ──────────────────────────────────────────────────────────────
import time, logging, os
from datetime import datetime, timezone, date, time as dtime, timedelta

import numpy  as np
import pandas as pd
import MetaTrader5 as mt5

# ───── ACCOUNT ───────────────────────────────────────────────
LOGIN      = 79939940
PASSWORD   = "Mgi@2005"
SERVER     = "Exness-MT5Trial8"
SYMBOL     = "XAUUSD"          # <- update if broker suffix differs
MAGIC      = 555555

# ───── STRATEGY SETTINGS ─────────────────────────────────────
TIMEFRAME         = mt5.TIMEFRAME_M1
FAST_EMA, SLOW_EMA = 14, 28
ATR_PERIOD        = 14
INIT_SL_MULT      = 2.0          # base stop  = 2×ATR
INIT_TP_MULT      = 2.0          # base target= 2×ATR
TRAIL_STEP_ATR    = 0.5          # after price moves +0.5 ATR, slide TP
TRAIL_SL_ATR      = 0.5          # SL to breakeven+0.5 ATR when in profit ≥1 ATR
VOL_ATR_LOW       = 0.10         # skip if ATR < $0.10  (dead market)
VOL_ATR_HIGH      = 1.00         # skip if ATR > $1.00 (news spike)

RISK_PCT          = 0.01         # 1 % equity per trade.  Set FIXED_LOT to override
FIXED_LOT         = None         # e.g. 0.05 ; leave None for %-risk sizing

SPREAD_CAP        = 0.20         # dollars
SESSION_WINDOWS   = [(dtime(0,0), dtime(23,59))]

DAILY_STOP_PCT    = -0.06        # -6 % equity = stop for day
DAILY_TARGET_PCT  =  200       # +10 % profit = stop for day

LOG_PATH          = "xau_scalper.log"

# ───── LOGGING ───────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH),
              logging.StreamHandler()])

# ───── HELPERS ───────────────────────────────────────────────
def in_session(t: dtime):
    return any(a <= t <= b for a,b in SESSION_WINDOWS)

def compute_atr(df: pd.DataFrame, period=ATR_PERIOD):
    hi, lo, cl = df['high'], df['low'], df['close']
    prev_cl = cl.shift(1).fillna(cl)
    tr = np.maximum.reduce([hi-lo, abs(hi-prev_cl), abs(lo-prev_cl)])
    return pd.Series(tr).rolling(period).mean().iloc[-1]

def percent_equity_lot(sl_dist):
    info = mt5.symbol_info(SYMBOL)
    if info is None or sl_dist <= 0:
        return 0.01
    eq = mt5.account_info().equity
    risk_usd  = eq * RISK_PCT
    ticks     = sl_dist / info.trade_tick_size
    usd_tick  = info.trade_tick_value
    raw_lot   = risk_usd / (ticks * usd_tick + 1e-9)
    step      = info.volume_step
    lot = max(info.volume_min, (raw_lot // step) * step)
    return round(min(lot, info.volume_max), 2)

def send_order(side:str, lot:float, sl:float, tp:float, tag:str):
    tick  = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if side=="BUY" else tick.bid
    req = {
        "action":mt5.TRADE_ACTION_DEAL,
        "symbol":SYMBOL,
        "volume":lot,
        "type":mt5.ORDER_TYPE_BUY if side=="BUY" else mt5.ORDER_TYPE_SELL,
        "price":price,
        "sl":sl, "tp":tp, "deviation":20,
        "magic":MAGIC, "comment":tag,
        "type_time":mt5.ORDER_TIME_GTC,
        "type_filling":mt5.ORDER_FILLING_IOC
    }
    res = mt5.order_send(req)
    logging.info(f"{tag} {side} {lot}@{price:.3f} SL{sl:.3f} TP{tp:.3f} → {res.retcode}")
    return res

def modify_sl_tp(ticket:int, new_sl:float=None, new_tp:float=None):
    req = {"action":mt5.TRADE_ACTION_SLTP,
           "position":ticket,
           "sl":new_sl if new_sl else 0.0,
           "tp":new_tp if new_tp else 0.0}
    mt5.order_send(req)

# ───── MAIN ──────────────────────────────────────────────────
def main():
    if not mt5.initialize(server=SERVER, login=LOGIN, password=PASSWORD):
        logging.error("MT5 init failed"); return
    if not mt5.symbol_select(SYMBOL, True):
        logging.error(f"{SYMBOL} not found"); return
    logging.info("🚀  EMA-ATR Scalper running")

    day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    day_pl = 0.0
    open_ticket = None          # track our single position
    last_move_tp = None         # last price when TP trailed

    while True:
        now = datetime.now(timezone.utc).replace(tzinfo=timezone.utc)
        # reset daily counters
        if now.date() != day_start.date():
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            day_pl, open_ticket = 0.0, None
            last_move_tp = None

        # daily stop / target
        if day_pl <= DAILY_STOP_PCT*mt5.account_info().balance:
            logging.warning("Daily loss hit, pausing until tomorrow")
            time.sleep(60); continue
        if day_pl >= DAILY_TARGET_PCT*mt5.account_info().balance:
            logging.info("Daily target reached, pausing until tomorrow")
            time.sleep(60); continue

        # session & spread
        tick = mt5.symbol_info_tick(SYMBOL)
        if not in_session(now.time()) or tick is None:
            time.sleep(5); continue
        spread = tick.ask - tick.bid
        if spread > SPREAD_CAP:
            time.sleep(1); continue

        # pull 1-min bars
        bars = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 300)
        if bars is None or len(bars) < SLOW_EMA + 2:
            time.sleep(1); continue
        df = pd.DataFrame(bars)

        # indicators
        df['ema_fast'] = df['close'].ewm(span=FAST_EMA).mean()
        df['ema_slow'] = df['close'].ewm(span=SLOW_EMA).mean()
        atr = compute_atr(df)
        if np.isnan(atr): time.sleep(1); continue
        if atr < VOL_ATR_LOW or atr > VOL_ATR_HIGH:
            time.sleep(1); continue   # volatility filter

        # crossover signal (use bar-1 closed)
        fast_now, slow_now   = df.iloc[-2][['ema_fast','ema_slow']]
        fast_prev, slow_prev = df.iloc[-3][['ema_fast','ema_slow']]
        signal = None
        if fast_now > slow_now and fast_prev <= slow_prev:
            signal = "BUY"
        elif fast_now < slow_now and fast_prev >= slow_prev:
            signal = "SELL"

        # update open position trailing TP/SL
        if open_ticket:
            pos = next((p for p in mt5.positions_get(symbol=SYMBOL) or []
                        if p.ticket == open_ticket), None)
            if pos:
                direction = "BUY" if pos.type==mt5.POSITION_TYPE_BUY else "SELL"
                price = tick.bid if direction=="BUY" else tick.ask
                entry = pos.price_open
                open_profit = (price - entry) if direction=="BUY" else (entry - price)

                # move SL to BE + TRAIL_SL_ATR after profit >= 1 ATR
                if open_profit >= atr and (direction=="BUY" and pos.sl < entry or
                                           direction=="SELL" and pos.sl > entry):
                    new_sl = entry + TRAIL_SL_ATR*atr if direction=="BUY" else entry - TRAIL_SL_ATR*atr
                    modify_sl_tp(pos.ticket, new_sl, pos.tp)
                    logging.info("SL → BE+trail")

                # trail TP every 0.5 ATR gained
                if last_move_tp is None: last_move_tp = entry
                if abs(price - last_move_tp) >= TRAIL_STEP_ATR*atr:
                    if direction=="BUY":
                        new_tp = pos.tp + TRAIL_STEP_ATR*atr
                    else:
                        new_tp = pos.tp - TRAIL_STEP_ATR*atr
                    modify_sl_tp(pos.ticket, pos.sl, new_tp)
                    last_move_tp = price
                    logging.info("TP trailed further")

            # update day P/L after close
            if pos is None:
                open_ticket, last_move_tp = None, None
                # update daily pl
                deals = mt5.history_deals_get(day_start, now, group=f"*{MAGIC}")
                day_pl = sum(d.profit for d in deals)
                continue

        # ---------- new entry ----------
        if signal and open_ticket is None:
            sl_dist = INIT_SL_MULT * atr
            tp_dist = INIT_TP_MULT * atr
            lot = FIXED_LOT if FIXED_LOT else percent_equity_lot(sl_dist)
            if lot <= 0: time.sleep(1); continue

            if signal == "BUY":
                price = tick.ask
                sl = price - sl_dist
                tp = price + tp_dist
            else:
                price = tick.bid
                sl = price + sl_dist
                tp = price - tp_dist

            res = send_order(signal, lot, sl, tp, "EMA_SCALP")
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                open_ticket = res.order or res.deal
                last_move_tp = price

        # quick sleep
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error")
    finally:
        mt5.shutdown()
