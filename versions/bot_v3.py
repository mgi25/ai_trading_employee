# #!/usr/bin/env python3
# # ──────────────────────────────────────────────────────────────
# #  XAUUSD “Pulse-Scalper” — 1-Minute EMA / Volume / ATR engine
# #  • Many micro-entries per session
# #  • Step–trailing SL & TP
# #  • Auto-reconnect, safe wrappers, rich logging
# # ──────────────────────────────────────────────────────────────
# import os, time, logging
# from datetime import datetime, timezone, timedelta, time as dtime
# import numpy as np, pandas as pd
# import MetaTrader5 as mt5

# # ╭──────────────────  BROKER / LOGIN  ──────────────────╮
# LOGIN, PASSWORD, SERVER = 79939940, "Mgi@2005", "Exness-MT5Trial8"
# SYMBOL                  = "XAUUSD"            # check suffix!
# MAGIC                   = 909090               # EA identifier
# # ╰──────────────────────────────────────────────────────╯

# # ╭──────────────────  STRATEGY PARAMS  ─────────────────╮
# TIMEFRAME        = mt5.TIMEFRAME_M1
# EMA_LEN          = 9
# ATR_LEN          = 14

# SL_INIT_MULT     = 1.5      # initial SL  distance  (×ATR)
# TP_INIT_MULT     = 3.0      # initial TP  distance  (×ATR)

# BE_TRIGGER_ATR   = 0.5      # profit ≥0.5 ATR → SL break-even+buffer
# TRAIL_STEP_ATR   = 0.25     # each +0.25 ATR → ratchet SL & TP
# TRAIL_SL_ATR     = 0.25     # buffer added to break-even SL

# VOL_SPIKE        = 1.0      # bar volume ≥ 1.1×20-bar avg
# BODY_RATIO_MIN   = 0.40     # candle body ≥60 % of range

# VOL_ATR_LOW      = 0.10     # ignore dead <0.10 ATR
# VOL_ATR_HIGH     = 1.00     # ignore crazy >1   ATR

# MAX_CONCURRENT   = 4        # simultaneous positions
# RISK_PCT         = 0.01     # 1 % equity risk per trade
# SPREAD_CAP       = 0.20     # $0.20 micro-pips

# DAILY_STOP_PCT   = -0.08    # stop trading –8 % of balance
# DAILY_TARGET_PCT =  0.12    # pause after +12 %

# SESSION          = [(dtime(0,0), dtime(23,59))]   # main liquidity UTC
# # ╰──────────────────────────────────────────────────────╯

# # ╭─────────────────────  LOGGING  ──────────────────────╮
# LOG_PATH = "pulse_scalper.log"
# os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s %(levelname)-7s %(message)s",
#     handlers=[
#         logging.FileHandler(LOG_PATH, encoding="utf-8"),
#         logging.StreamHandler()
#     ])
# # ╰──────────────────────────────────────────────────────╯


# def ensure_mt5() -> bool:
#     """
#     Return True when MT5 is connected.
#     Re-initialise if   • mt5.account_info() is None
#                        • terminal_info() is None
#     """
#     if mt5.account_info() is None or mt5.terminal_info() is None:
#         mt5.shutdown()
#         ok = mt5.initialize(server=SERVER, login=LOGIN, password=PASSWORD)
#         if ok:
#             mt5.symbol_select(SYMBOL, True)
#             logging.info("[CORE]  Re-connected to MT5")
#         else:
#             logging.error(f"[CORE]  init failed → {mt5.last_error()}")
#         return ok
#     return True


# def in_session(t: dtime):
#     return any(a<=t<=b for a,b in SESSION)

# def tick():   return mt5.symbol_info_tick(SYMBOL)
# def acc():    return mt5.account_info()
# def rates(n): return mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n)

# def atr(df: pd.DataFrame) -> float:
#     """14-period Average-True-Range for the DataFrame provided."""
#     h, l, c = df['high'], df['low'], df['close']
#     prev    = c.shift(1).fillna(c)
#     tr_raw  = np.maximum.reduce([h - l,
#                                  (h - prev).abs(),
#                                  (l - prev).abs()])

#     # convert to Series ⬇️ so 'rolling' is available
#     tr = pd.Series(tr_raw, index=df.index)
#     return tr.rolling(ATR_LEN).mean().iloc[-1]


# def lot_by_risk(sl_dist: float) -> float:
#     info = mt5.symbol_info(SYMBOL)
#     account = acc()
#     if not info or not account: return 0.0
#     risk_usd = account.equity * RISK_PCT
#     ticks = sl_dist / info.trade_tick_size
#     raw = risk_usd / (ticks*info.trade_tick_value + 1e-9)
#     step=info.volume_step
#     lot=max(info.volume_min, (raw//step)*step)
#     return round(min(lot, info.volume_max), 2)

# def send(side, lot, sl, tp, tag):
#     tk = tick()
#     if tk is None:                       # <— linter-friendly
#         return None

#     px  = tk.ask if side == "BUY" else tk.bid
#     res = mt5.order_send({
#         "action":  mt5.TRADE_ACTION_DEAL,
#         "symbol":  SYMBOL,
#         "volume":  lot,
#         "type":    mt5.ORDER_TYPE_BUY if side=="BUY" else mt5.ORDER_TYPE_SELL,
#         "price":   px,
#         "sl":      sl,
#         "tp":      tp,
#         "deviation": 20,
#         "magic":     MAGIC,
#         "comment":   tag,
#         "type_time":  mt5.ORDER_TIME_GTC,
#         "type_filling": mt5.ORDER_FILLING_IOC,
#     })
#     logging.info(f"[ENTRY] {tag} {side} {lot}@{px:.3f} SL{sl:.3f} TP{tp:.3f} → {res.retcode}")
#     return res

# def modify(ticket, new_sl=None, new_tp=None):
#     mt5.order_send({"action":mt5.TRADE_ACTION_SLTP,
#                     "position":ticket,
#                     "sl":new_sl or 0.0,
#                     "tp":new_tp or 0.0})
# # ╰──────────────────────────────────────────────────────╯


# def main():
#     if not ensure_mt5(): return
#     logging.info("[CORE]  Pulse-Scalper started")

#     day0 = datetime.now(timezone.utc).date()
#     daily_pl = 0.0
#     trail_anchor = {}           # ticket -> anchor price for next trail

#     while True:
#         if not ensure_mt5(): time.sleep(5); continue
#         now = datetime.now(timezone.utc)

#         # ─── Daily rollover
#         if now.date() != day0:
#             day0, daily_pl, trail_anchor = now.date(), 0.0, {}
#             logging.info("[CORE]  New trading day")

#         # ─── Daily guardrails
#         balance = acc().balance
#         if daily_pl <= DAILY_STOP_PCT*balance:
#             logging.warning("[RISK] Daily loss limit hit – pausing 30 min")
#             time.sleep(1800); continue
#         if daily_pl >= DAILY_TARGET_PCT*balance:
#             logging.info("[TARGET] Daily profit target reached – pausing 30 min")
#             time.sleep(1800); continue

#         # ─── Session / spread gates
#         if not in_session(now.time()): time.sleep(10); continue
#         tk = tick()
#         if not tk or tk.ask==0 or tk.bid==0 or tk.ask-tk.bid > SPREAD_CAP:
#             time.sleep(1); continue

#         # ─── Market data fetch
#         bars = rates(200)
#         if bars is None: time.sleep(1); continue
#         df = pd.DataFrame(bars)
#         df['ema'] = df['close'].ewm(span=EMA_LEN).mean()
#         cur_atr = atr(df)
#         if np.isnan(cur_atr) or cur_atr<VOL_ATR_LOW or cur_atr>VOL_ATR_HIGH:
#             time.sleep(1); continue

#         last = df.iloc[-2]       # last closed candle
#         body = abs(last['close']-last['open'])
#         rng  = last['high']-last['low']+1e-9
#         body_ok = body/rng >= BODY_RATIO_MIN
#         vol_ok = last['tick_volume'] >= VOL_SPIKE*df['tick_volume'].iloc[-22:-2].mean()

#         side = None
#         if body_ok and vol_ok:
#             if last['close'] > last['ema']+0.05: side="BUY"
#             elif last['close'] < last['ema']-0.05: side="SELL"

#         logging.debug(f"[CHK] {now.time()} "
#                       f"body_ok={body_ok} vol_ok={vol_ok} side={side} "
#                       f"ATR={cur_atr:.3f}")

#         # ─── Open new position
#         open_pos = [p for p in mt5.positions_get(symbol=SYMBOL) or [] if p.magic==MAGIC]
#         if side and len(open_pos) < MAX_CONCURRENT:
#             sl_d = SL_INIT_MULT*cur_atr
#             tp_d = TP_INIT_MULT*cur_atr
#             lot  = lot_by_risk(sl_d)
#             if lot>0:
#                 if side=="BUY":
#                     sl=tk.ask-sl_d; tp=tk.ask+tp_d
#                 else:
#                     sl=tk.bid+sl_d; tp=tk.bid-tp_d
#                 res = send(side, lot, sl, tp, "PULSE")
#                 if res and res.retcode==mt5.TRADE_RETCODE_DONE:
#                     ticket = res.order or res.deal
#                     trail_anchor[ticket] = tk.ask if side=="BUY" else tk.bid

#         # ─── Manage open positions
#         for p in open_pos:
#             side = "BUY" if p.type==mt5.POSITION_TYPE_BUY else "SELL"
#             price_now = tk.bid if side=="BUY" else tk.ask
#             entry     = p.price_open
#             gain      = price_now-entry if side=="BUY" else entry-price_now
#             ticket    = p.ticket

#             # move SL → BE+buffer
#             if gain >= BE_TRIGGER_ATR*cur_atr and \
#                ((side=="BUY" and p.sl<entry) or (side=="SELL" and p.sl>entry)):
#                 new_sl = entry + (TRAIL_SL_ATR*cur_atr if side=="BUY"
#                                   else -TRAIL_SL_ATR*cur_atr)
#                 modify(ticket, new_sl, p.tp)
#                 logging.info(f"[MAN] {ticket} SL → BE+{TRAIL_SL_ATR}ATR")

#             # step-trail
#             if abs(price_now - trail_anchor.get(ticket, entry)) >= TRAIL_STEP_ATR*cur_atr:
#                 new_tp = p.tp + (TRAIL_STEP_ATR*cur_atr if side=="BUY"
#                                  else -TRAIL_STEP_ATR*cur_atr)
#                 new_sl = p.sl + (TRAIL_STEP_ATR*cur_atr if side=="BUY"
#                                  else -TRAIL_STEP_ATR*cur_atr)
#                 modify(ticket, new_sl, new_tp)
#                 trail_anchor[ticket] = price_now
#                 logging.info(f"[TRAIL] {ticket} +{TRAIL_STEP_ATR}ATR")

#         # ─── Update daily P/L every 30 s
#         if now.second % 30 == 0:
#             deals = mt5.history_deals_get(
#                 datetime.combine(day0, dtime.min, tzinfo=timezone.utc),
#                 now, group=f"*{MAGIC}")
#             daily_pl = sum(d.profit for d in deals)

#         time.sleep(1)

# # ─────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     try:
#         main()
#     except Exception:
#         logging.exception("Fatal")
#     finally:
#         mt5.shutdown()


#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────
#  XAUUSD “Pulse-Scalper” (single-position version)
#  • M1 EMA / Volume / ATR engine
#  • One open ticket at a time
#  • Step-trailing SL & TP, daily guard-rails, auto reconnect
# ──────────────────────────────────────────────────────────────
# import os, time, logging
# from datetime import datetime, timezone, timedelta, time as dtime
# import numpy   as np
# import pandas  as pd
# import MetaTrader5 as mt5

# # ╭──────────────────  BROKER / LOGIN  ──────────────────╮
# LOGIN, PASSWORD, SERVER = 204215535, "Mgi@2005", "Exness-MT5Trial7"
# SYMBOL                  = "XAUUSDm"          # check suffix!
# MAGIC                   = 909090            # EA identifier
# # ╰──────────────────────────────────────────────────────╯

# # ╭──────────────────  STRATEGY PARAMS  ─────────────────╮
# TIMEFRAME        = mt5.TIMEFRAME_M1
# EMA_LEN          = 9
# ATR_LEN          = 14

# SL_INIT_MULT     = 1.5      # initial SL  distance  (×ATR)
# TP_INIT_MULT     = 3.0      # initial TP  distance  (×ATR)

# BE_TRIGGER_ATR   = 0.5      # profit ≥0.5 ATR → SL break-even + buffer
# TRAIL_STEP_ATR   = 0.25     # every +0.25 ATR → ratchet SL & TP
# TRAIL_SL_ATR     = 0.25     # extra buffer beyond break-even

# VOL_SPIKE        = 1.00     # bar vol ≥ 1.0×20-bar avg
# BODY_RATIO_MIN   = 0.40     # candle body ≥ 40 % of range

# ATR_LOWER        = 0.10     # skip if ATR below / above these
# ATR_UPPER        = 1.00

# RISK_PCT         = 0.01     # 1 % equity risk
# SPREAD_CAP       = 0.20     # $0.20  (= 2 micro-pips on gold)

# DAILY_STOP_PCT   = -0.08    # pause if ≤ −8 %
# DAILY_TARGET_PCT =  0.12    # pause if ≥ +12 %

# SESSION          = [(dtime(0,0), dtime(23,59))]   # trade all day UTC
# # ╰──────────────────────────────────────────────────────╯

# # ╭─────────────────────  LOGGING  ──────────────────────╮
# os.makedirs("logs", exist_ok=True)
# logging.basicConfig(
#     level   = logging.DEBUG,
#     format  = "%(asctime)s %(levelname)-7s %(message)s",
#     handlers=[
#         logging.FileHandler("logs/pulse_scalper_single.log", encoding="utf-8"),
#         logging.StreamHandler()
#     ])
# # ╰──────────────────────────────────────────────────────╯


# # ───── helpers ──────────────────────────────────────────
# def ensure_mt5() -> bool:
#     """(Re)connect to MT5 if needed – returns True when ready"""
#     if mt5.account_info() and mt5.terminal_info():
#         return True
#     mt5.shutdown()
#     ok = mt5.initialize(server=SERVER, login=LOGIN, password=PASSWORD)
#     if ok:
#         mt5.symbol_select(SYMBOL, True)
#         logging.info("[CORE]  Re-connected to MT5")
#     else:
#         logging.error(f"[CORE]  init failed {mt5.last_error()}")
#     return ok


# def in_session(t: dtime): return any(a <= t <= b for a, b in SESSION)
# def tick():   return mt5.symbol_info_tick(SYMBOL)
# def acc():    return mt5.account_info()
# def rates(n): return mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n)


# def atr(df: pd.DataFrame) -> float:
#     h, l, c = df['high'], df['low'], df['close']
#     prev    = c.shift(1).fillna(c)
#     tr_raw  = np.maximum.reduce([h - l,
#                                  (h - prev).abs(),
#                                  (l - prev).abs()])
#     tr = pd.Series(tr_raw, index=df.index)
#     return tr.rolling(ATR_LEN).mean().iloc[-1]


# def lot_by_risk(sl_dist: float) -> float:
#     info = mt5.symbol_info(SYMBOL);  acct = acc()
#     if not info or not acct: return 0.0
#     risk = acct.equity * RISK_PCT
#     ticks = sl_dist / info.trade_tick_size
#     raw   = risk / (ticks * info.trade_tick_value + 1e-9)
#     step  = info.volume_step
#     lot   = max(info.volume_min, (raw // step) * step)
#     return round(min(lot, info.volume_max), 2)


# def send(side, lot, sl, tp, tag):
#     tk = tick()
#     if not tk: return None
#     px  = tk.ask if side == "BUY" else tk.bid
#     res = mt5.order_send(dict(
#         action       = mt5.TRADE_ACTION_DEAL,
#         symbol       = SYMBOL,
#         volume       = lot,
#         type         = mt5.ORDER_TYPE_BUY if side=="BUY" else mt5.ORDER_TYPE_SELL,
#         price        = px,
#         sl           = sl,
#         tp           = tp,
#         deviation    = 20,
#         magic        = MAGIC,
#         comment      = tag,
#         type_time    = mt5.ORDER_TIME_GTC,
#         type_filling = mt5.ORDER_FILLING_IOC))
#     logging.info(f"[ENTRY] {tag} {side} {lot}@{px:.3f} SL{sl:.3f} TP{tp:.3f} → {res.retcode}")
#     return res


# def modify(ticket, new_sl=None, new_tp=None):
#     mt5.order_send({"action": mt5.TRADE_ACTION_SLTP,
#                     "position": ticket,
#                     "sl": new_sl or 0.0,
#                     "tp": new_tp or 0.0})

# # ───── main loop ─────────────────────────────────────────
# def main():
#     if not ensure_mt5(): return
#     logging.info("[CORE]  Pulse-Scalper single-pos started")

#     trade_lock  = None          # ticket of the current position
#     last_bar_ts = None          # to fire only once per M1
#     trail_anchor = {}

#     day0       = datetime.now(timezone.utc).date()
#     daily_pl   = 0.0

#     while True:
#         if not ensure_mt5(): time.sleep(5); continue
#         now = datetime.now(timezone.utc)

#         # rollover P/L
#         if now.date() != day0:
#             day0, daily_pl, trail_anchor = now.date(), 0.0, {}
#             logging.info("[CORE]  New trading day")

#         bal = acc().balance
#         if daily_pl <= DAILY_STOP_PCT*bal or daily_pl >= DAILY_TARGET_PCT*bal:
#             time.sleep(30); continue

#         if not in_session(now.time()): time.sleep(10); continue
#         tk = tick()
#         if not tk or tk.ask-tk.bid > SPREAD_CAP: time.sleep(1); continue

#         bars = rates(200)
#         if bars is None: time.sleep(1); continue
#         df = pd.DataFrame(bars)
#         if last_bar_ts == df.iloc[-2]['time']:   # already processed
#             pass
#         else:
#             last_bar_ts = df.iloc[-2]['time']

#             df['ema'] = df['close'].ewm(span=EMA_LEN).mean()
#             cur_atr = atr(df)
#             if np.isnan(cur_atr) or cur_atr<ATR_LOWER or cur_atr>ATR_UPPER:
#                 continue

#             last = df.iloc[-2]
#             body = abs(last['close']-last['open'])
#             rng  = last['high']-last['low']+1e-9
#             body_ok = body/rng >= BODY_RATIO_MIN
#             vol_ok  = last['tick_volume'] >= VOL_SPIKE*df['tick_volume'].iloc[-22:-2].mean()
#             side = None
#             if body_ok and vol_ok:
#                 if last['close'] > last['ema']+0.05: side="BUY"
#                 elif last['close'] < last['ema']-0.05: side="SELL"

#             logging.debug(f"[BAR] body_ok={body_ok} vol_ok={vol_ok} side={side} ATR={cur_atr:.3f}")

#             # Only open if **no** live ticket
#             live = [p for p in (mt5.positions_get(symbol=SYMBOL) or []) if p.magic==MAGIC]
#             if side and not live:
#                 sl_d = SL_INIT_MULT*cur_atr
#                 tp_d = TP_INIT_MULT*cur_atr
#                 lot  = lot_by_risk(sl_d)
#                 if lot>0:
#                     if side=="BUY":
#                         sl=tk.ask-sl_d; tp=tk.ask+tp_d
#                     else:
#                         sl=tk.bid+sl_d; tp=tk.bid-tp_d
#                     res = send(side, lot, sl, tp, "PULSE")
#                     if res and res.retcode==mt5.TRADE_RETCODE_DONE:
#                         trade_lock = res.order or res.deal
#                         trail_anchor[trade_lock] = tk.ask if side=="BUY" else tk.bid

#         # ─── manage the one open trade (if any)
#         if trade_lock:
#             pos = next(
#                 (p for p in mt5.positions_get(symbol=SYMBOL) or [] if p.ticket == trade_lock),
#                 None
#             )
#             if not pos:
#                 trade_lock = None  # closed externally
#             else:
#                 side = "BUY" if pos.type==mt5.POSITION_TYPE_BUY else "SELL"
#                 price_now = tk.bid if side=="BUY" else tk.ask
#                 entry     = pos.price_open
#                 gain      = price_now-entry if side=="BUY" else entry-price_now

#                 cur_atr = atr(df)
#                 # 1) move SL → BE+buffer
#                 if gain >= BE_TRIGGER_ATR*cur_atr and \
#                    ((side=="BUY" and pos.sl<entry) or (side=="SELL" and pos.sl>entry)):
#                     new_sl = entry + (TRAIL_SL_ATR*cur_atr if side=="BUY" else -TRAIL_SL_ATR*cur_atr)
#                     modify(trade_lock, new_sl, pos.tp)
#                     logging.info("SL → BE+trail")

#                 # 2) step-trail
#                 anchor = trail_anchor.get(trade_lock, entry)
#                 if abs(price_now-anchor) >= TRAIL_STEP_ATR*cur_atr:
#                     new_tp = pos.tp + (TRAIL_STEP_ATR*cur_atr if side=="BUY" else -TRAIL_STEP_ATR*cur_atr)
#                     new_sl = pos.sl + (TRAIL_STEP_ATR*cur_atr if side=="BUY" else -TRAIL_STEP_ATR*cur_atr)
#                     modify(trade_lock, new_sl, new_tp)
#                     trail_anchor[trade_lock] = price_now
#                     logging.info("TP/SL trailed further")

#         # update P/L every 30 s
#         if now.second % 30 == 0:
#             from_ = datetime.combine(day0, dtime.min, tzinfo=timezone.utc)
#             deals = mt5.history_deals_get(from_, now, group=f"*{MAGIC}")
#             daily_pl = sum(d.profit for d in deals)

#         time.sleep(1)


# # ─────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     try:
#         main()
#     except Exception:
#         logging.exception("Fatal")
#     finally:
#         mt5.shutdown()



# ──────────────────────────────────────────────────────────────

# #!/usr/bin/env python3
# # =============================================================
# #  XAUUSD  “Pulse-Scalper v4”  – single-position, smart trailing
# #  © 2025 – educational code  (no live-trade guarantee!)
# # =============================================================
# import os, time, math, logging
# from datetime import datetime, timezone, timedelta, time as dtime

# import numpy as np
# import pandas as pd
# import MetaTrader5 as mt5

# # ─────────────────────────────────────────────────────────────
# #               BROKER / LOGIN (← replace yours)              
# # ─────────────────────────────────────────────────────────────
# LOGIN, PASSWORD, SERVER = 204215535, "Mgi@2005", "Exness-MT5Trial7"
# SYMBOL                  = "XAUUSDm"
# MAGIC                   = 909090           # EA identifier

# # ─────────────────────────────────────────────────────────────
# #                 STRATEGY — tweakables                        
# # ─────────────────────────────────────────────────────────────
# TF                = mt5.TIMEFRAME_M1
# FAST_EMA          = 9
# ATR_LEN           = 14

# INIT_SL_ATR       = 1.5        # initial protective stop
# INIT_TP_ATR       = 3.0        # initial hard target (rarely hit)

# CHAN_K            = 1.3        # chandelier coefficient
# STEP_EXT_ATR      = 0.25       # new extreme ≥ 0.25-ATR → trail
# VOL_TRAIL_MIN     = 0.80       # bar vol must be ≥ 0.8×20-bar avg

# BE_TRIGGER_ATR    = 0.50       # profit ≥ 0.5-ATR   → move SL to BE
# BE_BUFFER_ATR     = 0.25       #   …  plus this buf

# SOFT_VOL_DROP     = 0.40       # if bar vol < 0.4×avg   → soft exit
# SOFT_EMA_FLIP     = True       # exit if close crosses fast-EMA

# TIME_STOP_MIN     = 20         # maximum channel-hold minutes

# BODY_MIN          = 0.40       # bar body ≥ 40 % of range
# VOL_SPIKE         = 1.00       # bar vol ≥ 1×avg to consider entry
# ATR_MIN, ATR_MAX  = 0.10, 1.00

# RISK_PCT          = 0.01       # 1 % equity per trade
# SPREAD_CAP        = 0.20       # $0.20  (≈2 micropips on gold)

# DAY_STOP_PCT      = -0.08      # kill-switch
# DAY_TARGET_PCT    =  0.12

# SESSION           = [(dtime(0,0), dtime(23,59))]     # trade 24 h UTC

# # ─────────────────────────────────────────────────────────────
# #                         LOGGING                             
# # ─────────────────────────────────────────────────────────────
# os.makedirs("logs", exist_ok=True)
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s %(levelname)-4s %(message)s",
#     handlers=[
#         logging.FileHandler("logs/pulse_scalper_v4.log", encoding="utf-8"),
#         logging.StreamHandler()
#     ])

# # ──────────────────── MT5 Convenience ───────────────────────
# def ensure_mt5() -> bool:
#     if mt5.account_info() and mt5.terminal_info():
#         return True
#     mt5.shutdown()
#     ok = mt5.initialize(server=SERVER, login=LOGIN, password=PASSWORD)
#     if ok:
#         mt5.symbol_select(SYMBOL, True)
#         logging.info("[CORE] re-connected")
#     else:
#         logging.error("[CORE] init failed %s", mt5.last_error())
#     return ok

# tick    = lambda: mt5.symbol_info_tick(SYMBOL)
# account = lambda: mt5.account_info()
# bars    = lambda n=200: mt5.copy_rates_from_pos(SYMBOL, TF, 0, n)

# # ─────────────────── Indicators / math ──────────────────────
# def atr(df: pd.DataFrame) -> float:
#     h, l, c = df['high'], df['low'], df['close']
#     prev    = c.shift(1).fillna(c)
#     tr_raw  = np.maximum.reduce([h-l, (h-prev).abs(), (l-prev).abs()])
#     return pd.Series(tr_raw).rolling(ATR_LEN).mean().iloc[-1]

# def lot_for(sl_points: float) -> float:
#     info = mt5.symbol_info(SYMBOL); acct = account()
#     if not info or not acct: return 0.0
#     risk_usd = acct.equity * RISK_PCT
#     ticks    = sl_points / info.trade_tick_size
#     raw_lot  = risk_usd / (ticks*info.trade_tick_value + 1e-9)
#     step     = info.volume_step
#     lot      = max(info.volume_min, math.floor(raw_lot/step)*step)
#     return round(min(lot, info.volume_max), 2)

# def send_deal(side:str, lot:float, price:float, sl:float, tp:float, tag:str):
#     order_type = mt5.ORDER_TYPE_BUY if side=="BUY" else mt5.ORDER_TYPE_SELL
#     res = mt5.order_send({
#         "action": mt5.TRADE_ACTION_DEAL,
#         "symbol": SYMBOL,
#         "volume": lot,
#         "type":   order_type,
#         "price":  price,
#         "sl": sl, "tp": tp,
#         "deviation": 20,
#         "magic": MAGIC,
#         "comment": tag,
#         "type_time": mt5.ORDER_TIME_GTC,
#         "type_filling": mt5.ORDER_FILLING_IOC})
#     logging.info("[ENT] %s %.2f @%.3f SL %.3f TP %.3f → %s",
#                  side, lot, price, sl, tp, res.retcode)
#     return (res.order or res.deal) if res.retcode==mt5.TRADE_RETCODE_DONE else None

# def market_close(pos):
#     opp_type = mt5.ORDER_TYPE_SELL if pos.type==mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
#     px       = tick().bid if pos.type==mt5.POSITION_TYPE_BUY else tick().ask
#     mt5.order_send({
#         "action":   mt5.TRADE_ACTION_DEAL,
#         "symbol":   SYMBOL,
#         "position": pos.ticket,
#         "type":     opp_type,
#         "volume":   pos.volume,
#         "price":    px,
#         "deviation": 20,
#         "magic":    MAGIC,
#         "comment":  "FORCED_EXIT",
#         "type_time":  mt5.ORDER_TIME_GTC,
#         "type_filling": mt5.ORDER_FILLING_IOC})
#     logging.info("[EXIT] market close ticket=%s @%.3f", pos.ticket, px)

# def modify(ticket:int, new_sl:float|None=None, new_tp:float|None=None):
#     mt5.order_send({"action": mt5.TRADE_ACTION_SLTP,
#                     "position": ticket,
#                     "sl": new_sl or 0.0,
#                     "tp": new_tp or 0.0})

# # ───────────────────── Main Engine ──────────────────────────
# def in_session(t: dtime): return any(a<=t<=b for a,b in SESSION)

# def main():
#     if not ensure_mt5(): return
#     logging.info("[CORE] Pulse-Scalper v4 running")

#     live_ticket : int | None = None
#     opened_at   : datetime | None = None
#     hi_water = lo_water = None
#     day0   = datetime.now(timezone.utc).date()
#     day_pl = 0.0

#     while True:
#         if not ensure_mt5(): time.sleep(5); continue
#         now = datetime.now(timezone.utc)

#         # ─── daily metrics
#         if now.date()!=day0:
#             day0, day_pl = now.date(), 0.0

#         bal = account().balance
#         if day_pl <= DAY_STOP_PCT*bal or day_pl >= DAY_TARGET_PCT*bal:
#             logging.warning("[DAY] target/stop reached → 30 m nap")
#             time.sleep(1800); continue

#         if not in_session(now.time()): time.sleep(10); continue
#         tk = tick()
#         if not tk or tk.ask==0 or tk.bid==0 or tk.ask-tk.bid > SPREAD_CAP:
#             time.sleep(1); continue

#         # fetch once per finished minute
#         df = pd.DataFrame(bars(200))
#         if df.empty: time.sleep(1); continue
#         df['ema'] = df['close'].ewm(span=FAST_EMA).mean()
#         cur_atr = atr(df)
#         if not ATR_MIN < cur_atr < ATR_MAX:
#             time.sleep(1); continue

#         last = df.iloc[-2]                     # closed bar
#         rng  = last['high']-last['low']+1e-9
#         body = abs(last['close']-last['open'])
#         body_ok = body/rng >= BODY_MIN
#         vol_avg = df['tick_volume'].iloc[-22:-2].mean()
#         vol_ok  = last['tick_volume'] >= VOL_SPIKE*vol_avg

#         ema_slope_ok = last['ema'] > df['ema'].iloc[-3]   # simple 2-bar slope

#         side = None
#         if body_ok and vol_ok and ema_slope_ok:
#             if last['close'] > last['ema']: side="BUY"
#             elif last['close'] < last['ema']: side="SELL"

#         # ====== ENTRY (one at a time) =========================
#         if side and live_ticket is None:
#             sl_dist = INIT_SL_ATR*cur_atr
#             tp_dist = INIT_TP_ATR*cur_atr
#             lot     = lot_for(sl_dist)
#             if lot>0:
#                 if side=="BUY":
#                     sl = tk.ask - sl_dist
#                     tp = tk.ask + tp_dist
#                     price = tk.ask
#                 else:
#                     sl = tk.bid + sl_dist
#                     tp = tk.bid - tp_dist
#                     price = tk.bid
#                 live_ticket = send_deal(side, lot, price, sl, tp, "PULSEv4")
#                 if live_ticket:
#                     opened_at = now
#                     hi_water  = last['high']
#                     lo_water  = last['low']
#             time.sleep(1); continue

#         # ====== MANAGE OPEN POSITION =========================
#         if live_ticket:
#             pos = next((p for p in mt5.positions_get(symbol=SYMBOL) or []
#                         if p.ticket==live_ticket), None)
#             if pos is None:
#                 live_ticket=None; opened_at=None
#             else:
#                 side = "BUY" if pos.type==mt5.POSITION_TYPE_BUY else "SELL"
#                 price_now = tk.bid if side=="BUY" else tk.ask
#                 entry     = pos.price_open
#                 gain_atr  = (price_now-entry) / cur_atr if side=="BUY" \
#                             else (entry-price_now)/cur_atr

#                 # update chandelier extreme
#                 hi_water = max(hi_water, last['high']) if side=="BUY" else hi_water
#                 lo_water = min(lo_water, last['low'])  if side=="SELL" else lo_water
#                 extreme  = hi_water if side=="BUY" else lo_water
#                 chdl_sl  = (extreme - CHAN_K*cur_atr) if side=="BUY" \
#                            else (extreme + CHAN_K*cur_atr)

#                 # MOVE SL → BE once +0.5-ATR
#                 if gain_atr >= BE_TRIGGER_ATR:
#                     be_sl = entry + BE_BUFFER_ATR*cur_atr if side=="BUY" \
#                             else entry - BE_BUFFER_ATR*cur_atr
#                     if (side=="BUY" and be_sl>pos.sl) or (side=="SELL" and be_sl<pos.sl):
#                         modify(pos.ticket, new_sl=round(be_sl, 3))
#                         logging.info("[BE ] ticket=%s SL→BE+%.2fATR", pos.ticket, BE_BUFFER_ATR)

#                 # CHANDELIER trail on new extreme
#                 extend = (price_now - extreme) if side=="BUY" else (extreme - price_now)
#                 if extend >= STEP_EXT_ATR*cur_atr and last['tick_volume'] >= VOL_TRAIL_MIN*vol_avg:
#                     if (side=="BUY" and chdl_sl>pos.sl) or (side=="SELL" and chdl_sl<pos.sl):
#                         modify(pos.ticket, new_sl=round(chdl_sl,3))
#                         logging.info("[CHDL] ticket=%s newSL %.3f", pos.ticket, chdl_sl)

#                 # SOFT-EXIT conditions  (volume die or EMA flip)
#                 vol_collapse = last['tick_volume'] < SOFT_VOL_DROP*vol_avg
#                 ema_cross    = last['close'] < last['ema'] if side=="BUY" else last['close'] > last['ema']
#                 if (vol_collapse or (SOFT_EMA_FLIP and ema_cross)) and gain_atr>0:
#                     logging.info("[SOFT] volume/EMA exit ticket=%s", pos.ticket)
#                     market_close(pos)
#                     live_ticket=None; opened_at=None
#                     continue

#                 # TIME-STOP
#                 if opened_at and now - opened_at >= timedelta(minutes=TIME_STOP_MIN):
#                     logging.info("[TIME] %dm timed exit ticket=%s", TIME_STOP_MIN, pos.ticket)
#                     market_close(pos)
#                     live_ticket=None; opened_at=None
#                     continue

#         # ===== update day P/L once per minute ================
#         if now.second==0:
#             day_trades = mt5.history_deals_get(
#                 datetime.combine(day0, dtime.min, tzinfo=timezone.utc),
#                 now, group=f"*{MAGIC}")
#             day_pl = sum(d.profit for d in day_trades)

#         time.sleep(1)

# # ─────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     try:
#         main()
#     except Exception:
#         logging.exception("Fatal loop error")
#     finally:
#         mt5.shutdown()
