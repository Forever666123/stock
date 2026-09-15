"""Expected value of limit-order ladders versus buying at market.

    python3 -m analysis.ladder --close 101.13 --levels 99 96 93 91.5

Each tranche is an equal share of the position resting as a limit order.
A tranche that never fills inside the window contributes 0 (no position, not
a loss). Filled tranches are held a fixed number of trading days FROM THE FILL,
so a late fill is not penalised by a shortened holding period.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis.data import load_merged

HOLD = 20     # trading days held after a fill
WINDOW = 20   # trading days the order rests before being cancelled


def populations(m: pd.DataFrame) -> dict:
    c = m["close"].values
    prev = np.r_[np.nan, c[:-1]]
    cl = pd.Series(c)
    ret = cl.pct_change().values
    dd = c / cl.rolling(250, min_periods=60).max().values - 1
    return {
        "当日跌≥13%": ret <= -0.13,
        "深回撤+20日暴跌": (dd <= -0.55) & (cl.pct_change(20).values <= -0.25),
        "跳空≤−12%": (m["open"].values / prev - 1) <= -0.12,
    }


def fill_return(c, low, n, i, level):
    """Return of one tranche resting at close_i*(1+level); 0.0 if never filled."""
    px = c[i] * (1 + level)
    for j in range(i + 1, min(i + 1 + WINDOW, n)):
        if low[j] <= px:
            k = j + HOLD
            return np.nan if k >= n else c[k] / px - 1
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--close", type=float, required=True, help="reference close the levels are measured from")
    ap.add_argument("--levels", type=float, nargs="+", required=True, help="ladder prices, e.g. 99 96 93")
    ap.add_argument("--quote", type=float, default=None, help="current quote, for the noise check")
    a = ap.parse_args()

    m = load_merged()
    m = m[m.index >= "2010-03-12"].reset_index()
    c, low, n = m["close"].values, m["low"].values, len(m)
    pops = populations(m)
    rng = ((m["high"] - m["low"]) / pd.Series(c).shift(1)).tail(20).median()

    print(f"参考收盘 {a.close}。近 20 日振幅中位 {rng*100:.1f}%。\n")
    ref = a.quote or a.close
    print("| 价位 | 距参考 | 占日振幅 | 判断 |")
    print("|---|---|---|---|")
    for px in a.levels:
        d = px / ref - 1
        mult = abs(d) / rng
        verdict = "噪声内，等于市价" if mult < 0.6 else ("有效档" if mult < 2.2 else "太深，成交率低")
        print(f"| {px} | {d*100:+.1f}% | {mult:.2f}× | {verdict} |")
    print("\n档位之间至少要隔一个日振幅，否则两档在同一天里一起成交，分批就没意义了。\n")

    combos = [("全仓现价", [])] + [(f"单档 {p}", [p]) for p in a.levels]
    for i in range(len(a.levels)):
        for j in range(i + 1, len(a.levels)):
            combos.append((f"{a.levels[i]} + {a.levels[j]}", [a.levels[i], a.levels[j]]))
    combos.append((" + ".join(str(p) for p in a.levels), list(a.levels)))

    print("| 组合 | " + " | ".join(pops) + " |")
    print("|---|" + "---|" * len(pops))
    for name, levels in combos:
        cells = []
        for mask in pops.values():
            idx = [i for i in np.where(mask)[0] if i + HOLD + WINDOW < n]
            vals = []
            for i in idx:
                if not levels:
                    v = c[i + HOLD] / c[i] - 1
                else:
                    parts = [fill_return(c, low, n, i, p / a.close - 1) for p in levels]
                    v = np.nan if any(np.isnan(x) for x in parts) else float(np.mean(parts))
                if not np.isnan(v):
                    vals.append(v)
            cells.append(f"{np.mean(vals)*100:+.1f}% ({len(vals)})")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(f"\n持有 {HOLD} 日（从成交日起算），挂单有效期 {WINDOW} 日，未成交的那一份算 0。均值口径。")


if __name__ == "__main__":
    main()
