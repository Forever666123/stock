"""The one table for trading a single session: entry tiers, the extreme, and
two take-profit levels, as prices rather than percentages.

    python3 -m analysis.day_table --prev-close 101.13
    python3 -m analysis.day_table --prev-close 101.13 --open 99.4

Everything is measured from the OPEN, because the gap carries almost no
information about how much further price falls after it (correlation about
-0.2 on FOMC D-1, zero elsewhere) while |gap| does predict the day's range
(+0.42), so the open is the right anchor and the reference population is
chosen by yesterday's range, not by yesterday's direction.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis.data import load_merged

TIERS = [-0.015, -0.030, -0.050]
EXTREME_PCTL = 5      # a fill here happens about one session in twenty
TP_PCTLS = [50, 75]   # reached in about half and a quarter of sessions
TODAY = {"date": "2026-09-14", "open": 101.53, "high": 105.39, "low": 99.87,
         "close": 101.13, "qqq": 711.96}


def build():
    m = load_merged()
    m = m[m.index >= "2010-03-12"]
    d = pd.Timestamp(TODAY["date"])
    if d not in m.index:
        m = pd.concat([m, pd.Series({k: v for k, v in TODAY.items() if k != "date"},
                                    name=d).to_frame().T]).sort_index()
    for x in ("open", "high", "low", "close"):
        m[x] = m[x].astype(float)
    c = m["close"]
    m["prev"] = c.shift(1)
    m["gap"] = m["open"] / m["prev"] - 1
    m["rng"] = (m["high"] - m["low"]) / m["open"]
    m["prev_rng"] = m["rng"].shift(1)
    m["l_o"] = m["low"] / m["open"] - 1
    m["h_o"] = m["high"] / m["open"] - 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev-close", type=float, required=True)
    ap.add_argument("--open", type=float, default=None, dest="open_px")
    ap.add_argument("--prev-range", type=float, default=0.054,
                    help="yesterday's high-low over its open; picks the volatility regime")
    a = ap.parse_args()
    m = build()
    lo_cut = max(0.02, a.prev_range - 0.015)
    s = m[(m["prev_rng"] >= lo_cut) & (m["gap"].abs() <= 0.04)].dropna(subset=["l_o", "h_o"])
    o, low, high = s["open"].values, s["low"].values, s["high"].values
    ext = np.percentile(s["l_o"], EXTREME_PCTL)
    tps = [np.percentile(s["h_o"], p) for p in TP_PCTLS]

    print(f"母体 n={len(s)}（昨日振幅 ≥{lo_cut*100:.1f}%，今日跳空 ±4% 内）")
    print(f"当天振幅中位 {s['rng'].median()*100:.1f}%　开盘后最低中位 {s['l_o'].median()*100:+.1f}%"
          f"　最高中位 {s['h_o'].median()*100:+.1f}%\n")
    fills = [float(np.mean(low <= o * (1 + t))) for t in TIERS]
    print("触及概率：" + "　".join(f"{t*100:+.1f}% → {f*100:.0f}%" for t, f in zip(TIERS, fills))
          + f"　极限 {ext*100:+.1f}% → {EXTREME_PCTL}%")
    print("止盈：" + "　".join(f"{tp*100:+.1f}% → {np.mean(high >= o*(1+tp))*100:.0f}%" for tp in tps) + "\n")

    opens = [a.open_px] if a.open_px else [a.prev_close * (1 + g / 100) for g in range(4, -6, -1)]
    print("| 开盘 | 第一档 | 第二档 | 第三档 | 极限 | 止盈一 | 止盈二 |")
    print("|---|---|---|---|---|---|---|")
    for op in opens:
        cells = [f"{op*(1+t):.1f} ({t*100:+.1f}%)" for t in TIERS]
        cells.append(f"{op*(1+ext):.1f} ({ext*100:+.1f}%)")
        cells += [f"{op*(1+tp):.1f} (+{tp*100:.1f}%)" for tp in tps]
        print(f"| **{op:.0f}** | " + " | ".join(cells) + " |")

    print("\n成交之后当天还要跌多少（挂在哪一档几乎不影响这一列）：\n")
    print("| 档位 | 成交率 | 还跌 中位 | 最差 5% |")
    print("|---|---|---|---|")
    for t in TIERS + [ext]:
        f = low <= o * (1 + t)
        further = low[f] / (o[f] * (1 + t)) - 1
        print(f"| {t*100:+.1f}% | {f.mean()*100:.0f}% | {np.median(further)*100:+.1f}% | {np.percentile(further,5)*100:+.1f}% |")
    print()


if __name__ == "__main__":
    main()
