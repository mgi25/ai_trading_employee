import pandas as pd

def analyze_log(file="ml_trading_log.csv"):
    df = pd.read_csv(file)
    if df.empty:
        print("Log is empty.")
        return

    total_trades = len(df)
    avg_pnl = df['pnl'].mean()
    win_rate = len(df[df['pnl'] > 0]) / total_trades * 100
    avg_spread = df['spread'].mean()
    avg_skew = df['skew'].mean()
    hedge_trades = df[df['hedge_mode'] == 1].shape[0]

    print(f"🧠 Log Summary:")
    print(f"Total Trades: {total_trades}")
    print(f"Average PnL: {avg_pnl:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Spread: {avg_spread:.5f}")
    print(f"Average Skew: {avg_skew:.5f}")
    print(f"Hedge Trades: {hedge_trades}")
    
    print("\n💡 Observations:")
    if win_rate < 50:
        print("- Low win rate. Might need better entry filtering.")
    if avg_spread > 0.2:
        print("- High spread. Could impact profit per trade.")
    if hedge_trades > total_trades * 0.3:
        print("- Too many hedge trades. Check trap/consolidation logic.")

if __name__ == "__main__":
    analyze_log()
