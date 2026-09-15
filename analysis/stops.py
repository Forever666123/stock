"""What a stop-loss costs on SOXL, and where leverage puts one for you.

    python3 -m analysis.stops [--entry -0.05] [--hold 20]

A stop triggers on the intraday low, the same mechanic as a broker liquidation.
The difference is only that you choose where a stop sits and you do not choose
where a liquidation sits.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis.data import load_merged

STOPS = [None, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30]
LIQ = [("10x", -0.095), ("5x", -0.195), ("3x", -0.328), ("2x", -0.495), ("现货", -1.0)]


def populations(m):
    c = m["close"].values
    cl = pd.Series(c)
    ret = cl.pct_change().values
    dd = c / cl.rolling(250, min_periods=60).max().values - 1
    gap = m["open"].values / np.r_[np.nan, c[:-1]] - 1
    return {"当日跌≥13%": ret <= -0.13,
            "深回撤+20日暴跌": (dd <= -0.55) & (cl.pct_change(20).values <= -0.25),
            "跳空≤−12%": gap <= -0.12}


def trade(c, low, n, i, entry, hold, stop, window=20):
    """-> (filled, return, stopped_out, max_adverse_excursion)"""
    px = c[i] * (1 + entry)
    fill = next((j for j in range(i + 1, min(i + 1 + window, n)) if low[j] <= px), None)
    if fill is None:
        return False, 0.0, False, 0.0
    end = fill + hold
    if end >= n:
        return True, np.nan, False, np.nan
    mae = low[fill:end + 1].min() / px - 1
    if stop is not None:
        for j in range(fill, end + 1):
            if low[j] <= px * (1 - stop):
                return True, -stop, True, mae
    return True, c[end] / px - 1, False, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=float, default=-0.05, help="limit offset from the signal close")
    ap.add_argument("--hold", type=int, default=20)
    a = ap.parse_args()
    m = load_merged()
    m = m[m.index >= "2010-03-12"].reset_index()
    c, low, n = m["close"].values, m["low"].values, len(m)
    pops = populations(m)
    idx = {k: [i for i in np.where(v)[0] if i + a.hold + 20 < n] for k, v in pops.items()}

    print(f"入场：挂 {a.entry*100:+.0f}% 成交后持有 {a.hold} 个交易日。止损按盘中最低触发。\n")
    print("| 止损 | " + " | ".join(f"{k}<br>EV / 触发率 / 触发后本可盈利" for k in pops) + " |")
    print("|---|" + "---|" * len(pops))
    for stop in STOPS:
        cells = []
        for k, ii in idx.items():
            rs, hit, regret, filled = [], 0, 0, 0
            for i in ii:
                f, r, st, _ = trade(c, low, n, i, a.entry, a.hold, stop)
                if np.isnan(r):
                    continue
                rs.append(r); filled += f
                if st:
                    hit += 1
                    _, r0, _, _ = trade(c, low, n, i, a.entry, a.hold, None)
                    regret += (not np.isnan(r0)) and r0 > 0
            cells.append(f"{np.mean(rs)*100:+.1f}% / {hit/max(filled,1)*100:.0f}% / {regret/max(hit,1)*100:.0f}%")
        print(f"| {'无止损' if stop is None else f'−{stop*100:.0f}%'} | " + " | ".join(cells) + " |")
    print("\n第三个数字是被止损的交易里，不设止损本来会赚钱的比例——那是止损的直接代价。\n")

    mae = {}
    print("| 参照组 | n | 5分位 | 中位 | 95分位 | 最差 | 最大浮亏中位 | 最大浮亏最深 |")
    print("|---|---|---|---|---|---|---|---|")
    for k, ii in idx.items():
        rs, ms = [], []
        for i in ii:
            f, r, _, mx = trade(c, low, n, i, a.entry, a.hold, None)
            if f and not np.isnan(r):
                rs.append(r); ms.append(mx)
        rs, ms = np.array(rs), np.array(ms); mae[k] = ms
        q = np.percentile(rs, [5, 50, 95])
        print(f"| {k} | {len(rs)} | {q[0]*100:+.1f}% | {q[1]*100:+.1f}% | {q[2]*100:+.1f}% "
              f"| {rs.min()*100:+.1f}% | {np.median(ms)*100:.1f}% | {ms.min()*100:.1f}% |")
    print()
    print("| 杠杆 | 强平跌幅 | " + " | ".join(f"{k} 被强平" for k in pops) + " |")
    print("|---|---|" + "---|" * len(pops))
    for name, lv in LIQ:
        print(f"| {name} | {lv*100:.1f}% | " + " | ".join(f"{(mae[k] <= lv).mean()*100:.0f}%" for k in pops) + " |")
    print("\n杠杆不是仓位工具，是一个你无法选址的止损。上面每一行都是同一个机制。\n")


if __name__ == "__main__":
    main()
