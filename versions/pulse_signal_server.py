#!/usr/bin/env python3
# =============================================================
#   Pulse-Signal-Server – generates trade commands
#   (redis queue → MT5 executor)
# =============================================================
import os, time, json, logging
from datetime import datetime, timezone
import numpy  as np
import pandas as pd
import MetaTrader5 as mt5
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import BayesianRidge
import redis

# ───── account / infra ───────────────────────────────────────
LOGIN, PASSWORD, SERVER = 204215535, "Mgi@2005", "Exness-MT5Trial7"
SYMBOL                  = "XAUUSDm"
REDIS_HOST              = "localhost"
REDIS_Q                 = "PULSE:CMD"

# ───── model / signal params ─────────────────────────────────
TICK_WINDOW    = 3_000            # ticks kept in RAM
HMM_TRAIN_LEN  = 600              # M1 returns for training
PROB_TH        = 0.65             # posterior long / short threshold
IMB_Z_MULT     = 1.5              # order-flow imbalance σ trigger
ATR_LEN        = 14
# ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    handlers=[logging.FileHandler("logs/signal_server.log"),
              logging.StreamHandler()])

redis_cli = redis.Redis(host=REDIS_HOST)

# ───── convenience ──────────────────────────────────────────
def mt5_ready() -> bool:
    if mt5.account_info() and mt5.terminal_info():
        return True
    mt5.shutdown()
    ok = mt5.initialize(server=SERVER, login=LOGIN, password=PASSWORD)
    if ok:
        mt5.symbol_select(SYMBOL, True)
        logging.info("MT5 re-connected")
    else:
        logging.error("MT5 init error %s", mt5.last_error())
    return ok

def lee_ready(df: pd.DataFrame) -> pd.Series:
    """Return +/- volume imbalance using Lee–Ready rule."""
    mid = (df['bid'] + df['ask']) * .5
    sign = np.sign(mid.diff().fillna(0.0))
    zero = sign == 0
    sign[zero] = np.sign(mid - mid.shift())[zero]
    return sign * df['volume'].replace(0,1)

def atr(high, low, close, n=ATR_LEN):
    h,l,c = np.asarray(high), np.asarray(low), np.asarray(close)
    prev  = np.roll(c, 1); prev[0] = c[0]
    tr = np.maximum.reduce([h-l, np.abs(h-prev), np.abs(l-prev)])
    return pd.Series(tr).rolling(n).mean().iloc[-1]

hmm  = GaussianHMM(n_components=2, covariance_type="diag", n_iter=20)

# ───── main loop ────────────────────────────────────────────
def main():
    if not mt5_ready(): return
    logging.info("Signal-server up")

    while True:
        if not mt5_ready():
            time.sleep(5); continue

        now   = datetime.now(timezone.utc)
        ticks = mt5.copy_ticks_from(SYMBOL, now, TICK_WINDOW, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) < 1500:
            time.sleep(1); continue

        df  = pd.DataFrame(ticks)
        df['imb'] = lee_ready(df)
        df['mid'] = (df['bid']+df['ask'])*.5

        # ---- HMM regime probability ------------------------------------
        rets = np.log(df['mid']).diff().dropna().values.reshape(-1,1)
        hmm.fit(rets[-HMM_TRAIN_LEN:])
        post = hmm.predict_proba(rets[-1:])[0]
        up   = np.argmax(hmm.means_)
        prob_long = post[up]

        # ---- order-flow imbalance sigma test --------------------------
        im = df.groupby(df['time']//1000)['imb'].sum()
        imb_now = im.iloc[-1]
        imb_sig = im.rolling(60).std().iloc[-1]

        # ---- ATR for meta-data ----------------------------------------
        m1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 100)
        o  = pd.DataFrame(m1)
        cur_atr = atr(o['high'], o['low'], o['close'])

        # ---- trigger logic --------------------------------------------
        side = None
        if prob_long >= PROB_TH and imb_now >  IMB_Z_MULT*imb_sig: side="BUY"
        if (1-prob_long) >= PROB_TH and imb_now < -IMB_Z_MULT*imb_sig: side="SELL"

        logging.debug("prob=%.2f  imb=%.0f/%.0f  %s", prob_long, imb_now, imb_sig, side)
        if side:
            msg = json.dumps(dict(ts=int(time.time()), side=side, atr=cur_atr))
            redis_cli.rpush(REDIS_Q, msg)
            logging.info("queued %s", msg)

        time.sleep(1)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: pass
