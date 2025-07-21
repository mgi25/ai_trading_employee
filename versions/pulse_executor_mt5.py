#!/usr/bin/env python3
# =============================================================
#   Pulse-Executor v3   –  single-position / smart trailing
#   (listens to Redis *or* falls back to a local noop-queue)
# =============================================================
"""
Usage
-----
1.  Start your signal-producer (or any process that pushes JSON:
        {"ts":  ... , "side": "BUY|SELL", "atr": 0.xx}
    onto a Redis list/stream … front name matches QUEUE.)

2.  Run *this* file – it will:
    •  wait for Redis if absent (poll every 15 s)
    •  passively pop one signal at a time
    •  place **one** net position, manage it with chandelier-ATR stop
       & various safety exits.

Everything is logged to **logs/executor.log** AND to stdout.
"""

import os, time, json, math, logging
from datetime import datetime, timezone, timedelta, time as dtime
from typing import Optional

import numpy  as np
import pandas as pd
import MetaTrader5 as mt5

try:
    import redis
    _redis_ok = True
except ImportError:
    redis, _redis_ok = None, False            # keep scalper runnable

# ╭───────────────────  LOGIN / BROKER  ───────────────────╮
LOGIN, PASSWORD, SERVER = 204215535, "Mgi@2005", "Exness-MT5Trial7"
SYMBOL                  = "XAUUSDm"         # exact MT5 market-watch name
MAGIC                   = 909090            # EA identifier
# ╰─────────────────────────────────────────────────────────╯

# ╭────────────────  RUNTIME / QUEUE SETTINGS  ────────────╮
REDIS_HOST  = "localhost"
REDIS_PORT  = 6379
QUEUE       = "PULSE:CMD"                   # ← producer pushes here
REDIS_RETRY = 15                            # seconds between attempts
# ╰─────────────────────────────────────────────────────────╯

# ╭──────────────────  STRATEGY PARAMS  ────────────────────╮
ATR_LEN            = 14
INIT_SL_ATR        = 1.5
INIT_TP_ATR        = 3.0
CHAN_K             = 1.3     # chandelier gap
STEP_EXT_ATR       = 0.25    # bar must beat last extreme by this
VOL_TRAIL_MIN      = 0.80    # 80 % of 20-bar avg vol
BE_TRIGGER_ATR     = 0.50
BE_BUFFER_ATR      = 0.25
TIME_STOP_MIN      = 20

FAST_EMA           = 9       # used for “soft” volume exit

SOFT_VOL_DROP      = 0.65    # <65 % avg volume + back-below EMA ⇒ quit
SPREAD_CAP         = 0.20

SESSION            = [(dtime(0,0), dtime(23,59))]
# ╰─────────────────────────────────────────────────────────╯
# pulse_executor_mt5.py
USE_REDIS = False          # ← add / change this line near the top


# ╭──────────────────────  LOGGING  ────────────────────────╮
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)-4s %(message)s",
    handlers=[logging.FileHandler("logs/executor.log", encoding="utf-8"),
              logging.StreamHandler()]
)
# ╰─────────────────────────────────────────────────────────╯


# ╭────────────────────  MT5 HELPERS  ──────────────────────╮
def ensure_mt5() -> bool:
    """Reconnect to MT5 if disconnected. True when terminal ready."""
    if mt5.account_info() and mt5.terminal_info():
        return True

    mt5.shutdown()
    ok = mt5.initialize(server=SERVER, login=LOGIN, password=PASSWORD)
    if ok:
        mt5.symbol_select(SYMBOL, True)
        logging.info("MT5 re-connect OK")
    else:
        logging.error("MT5 connect FAIL %s", mt5.last_error())
    return ok


tick   = lambda: mt5.symbol_info_tick(SYMBOL)
bars   = lambda n=200: mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, n)


def atr(df: pd.DataFrame) -> float:
    h, l, c = df['high'], df['low'], df['close']
    prev    = c.shift(1).fillna(c)
    tr = np.maximum.reduce([h-l, (h-prev).abs(), (l-prev).abs()])
    return pd.Series(tr).rolling(ATR_LEN).mean().iloc[-1]


def lot_for(sl_points: float) -> float:
    info  = mt5.symbol_info(SYMBOL)
    acc   = mt5.account_info()
    if not info or not acc: return 0.0
    risk  = acc.equity * 0.01          # 1 % per trade
    ticks = sl_points / info.trade_tick_size
    raw   = risk / (ticks*info.trade_tick_value + 1e-9)
    step  = info.volume_step
    lot   = math.floor(raw/step)*step
    return round(max(info.volume_min, min(lot, info.volume_max)), 2)


def send(side: str, lot: float, sl: float, tp: float) -> Optional[int]:
    tk = tick()
    price = tk.ask if side == "BUY" else tk.bid
    req = dict(
        action       = mt5.TRADE_ACTION_DEAL,
        symbol       = SYMBOL,
        volume       = lot,
        type         = mt5.ORDER_TYPE_BUY if side=="BUY" else mt5.ORDER_TYPE_SELL,
        price        = price,
        sl           = sl, tp = tp,
        deviation    = 20,
        magic        = MAGIC,
        comment      = "PULSEv3",
        type_time    = mt5.ORDER_TIME_GTC,
        type_filling = mt5.ORDER_FILLING_IOC
    )
    r = mt5.order_send(req)
    logging.info("[ENT] %s %.2f@%.3f SL%.3f TP%.3f → %s",
                 side, lot, price, sl, tp, r.retcode)
    return (r.order or r.deal) if r.retcode == 0 else None


def modify(ticket: int, new_sl: float|None = None, new_tp: float|None = None):
    mt5.order_send(dict(
        action   = mt5.TRADE_ACTION_SLTP,
        position = ticket,
        sl       = new_sl or 0.0,
        tp       = new_tp or 0.0
    ))


def market_close(pos):
    """Immediate opposite-side market exit."""
    side = "SELL" if pos.type == mt5.POSITION_TYPE_BUY else "BUY"
    px   = tick().bid if side == "SELL" else tick().ask
    mt5.order_send(dict(
        action   = mt5.TRADE_ACTION_DEAL,
        symbol   = SYMBOL,
        position = pos.ticket,
        volume   = pos.volume,
        type     = mt5.ORDER_TYPE_SELL if side=="SELL" else mt5.ORDER_TYPE_BUY,
        price    = px,
        deviation= 20,
        magic    = MAGIC,
        comment  = "TIME-EXIT",
        type_time= mt5.ORDER_TIME_GTC,
        type_filling = mt5.ORDER_FILLING_IOC
    ))
    logging.info("[EXIT] market-close %s", pos.ticket)
# ╰─────────────────────────────────────────────────────────╯


# ╭────────────────────  REDIS HANDLE  ──────────────────────╮
def redis_or_none():
    if not _redis_ok: return None
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2)
        r.ping()
        return r
    except redis.exceptions.ConnectionError:
        return None
# ╰─────────────────────────────────────────────────────────╯


def in_session(t: dtime):
    return any(a <= t <= b for a, b in SESSION)


def main():
    if not ensure_mt5(): return

    R: Optional["redis.Redis"] = None
    live, opened, hi, lo = None, None, None, None
    logging.info("Executor started – waiting for signals")

    while True:
        # (re)connect Redis if needed
        if R is None:
            R = redis_or_none()
            if R is None:
                logging.warning("Redis not available – retry in %ss", REDIS_RETRY)
                time.sleep(REDIS_RETRY); continue
            logging.info("Redis connected")

        if not ensure_mt5(): time.sleep(5); continue

        # ─── signal poll ─────────────────────────────────────────-
        raw = R.lpop(QUEUE) if R else None
        if raw and not live:
            try:
                sig = json.loads(raw)
                side = sig["side"]; sig_atr = max(float(sig["atr"]), 0.01)
            except Exception as e:
                logging.error("Bad signal %s – %s", raw, e)
                side = None

            if side:
                sl_d = INIT_SL_ATR*sig_atr
                tp_d = INIT_TP_ATR*sig_atr
                lt   = lot_for(sl_d)
                tk   = tick()
                if lt and tk.ask - tk.bid < SPREAD_CAP:
                    if side == "BUY":
                        live = send("BUY", lt, tk.ask - sl_d, tk.ask + tp_d)
                    else:
                        live = send("SELL", lt, tk.bid + sl_d, tk.bid - tp_d)
                    if live:
                        opened = datetime.now(timezone.utc)
                        hi = tk.ask; lo = tk.bid
                continue  # allow management loop next tick

        # ─── manage 1 live pos ──────────────────────────────────
        if live:
            pos = next((p for p in mt5.positions_get(symbol=SYMBOL) or []
                        if p.ticket == live), None)
            if pos is None:
                live = None; continue

            df = pd.DataFrame(bars(40))
            cur_atr = atr(df)
            last    = df.iloc[-2]
            vol_avg = df['tick_volume'].iloc[-22:-2].mean()

            tk = tick()
            if pos.type == mt5.POSITION_TYPE_BUY:
                hi  = max(hi, last['high'])
                ch  = hi - CHAN_K*cur_atr
                gained = tk.bid - pos.price_open

                # move to BE
                if gained >= BE_TRIGGER_ATR*cur_atr and pos.sl < pos.price_open:
                    modify(live, new_sl=pos.price_open + BE_BUFFER_ATR*cur_atr)

                # chandelier trail
                if hi - last['low'] >= STEP_EXT_ATR*cur_atr and \
                   last['tick_volume'] >= VOL_TRAIL_MIN * vol_avg and ch > pos.sl:
                    modify(live, new_sl=round(ch,3))

                # soft-exit on vol collapse
                fast = df['close'].ewm(span=FAST_EMA).mean().iloc[-1]
                if last['tick_volume'] < SOFT_VOL_DROP*vol_avg and last['close'] < fast:
                    market_close(pos); live=None; continue

            else:   # SELL
                lo  = min(lo, last['low'])
                ch  = lo + CHAN_K*cur_atr
                gained = pos.price_open - tk.ask

                if gained >= BE_TRIGGER_ATR*cur_atr and pos.sl > pos.price_open:
                    modify(live, new_sl=pos.price_open - BE_BUFFER_ATR*cur_atr)

                if last['high'] - lo >= STEP_EXT_ATR*cur_atr and \
                   last['tick_volume'] >= VOL_TRAIL_MIN * vol_avg and ch < pos.sl:
                    modify(live, new_sl=round(ch,3))

                fast = df['close'].ewm(span=FAST_EMA).mean().iloc[-1]
                if last['tick_volume'] < SOFT_VOL_DROP*vol_avg and last['close'] > fast:
                    market_close(pos); live=None; continue

            # absolute time-out
            if datetime.now(timezone.utc) - opened >= timedelta(minutes=TIME_STOP_MIN):
                market_close(pos); live=None; continue

        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        mt5.shutdown()
