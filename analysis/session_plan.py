"""Price levels and leverage limits for one upcoming session.

    python3 -m analysis.session_plan --prev-close 101.13 --role dm1
    python3 -m analysis.session_plan --prev-close 101.13 --role d0 --prev-ret -0.17

Gives the open / high / low / close distribution as prices rather than percents,
split by how the position is held, because the binding constraint for a
leveraged trade is the intraday path, not the horizon return. Overnight risk,
not intraday risk, is what forces liquidation - the tables make that explicit.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis.data import load_merged
from analysis.fomc import fomc_table

QS = [1, 5, 25, 50, 75, 95, 99]
LIQ = [("10x", -0.095), ("5x", -0.195), ("3x", -0.328), ("2x", -0.495)]
TODAY = {"date": "2026-09-14", "open": 101.53, "high": 105.39, "low": 99.87,
         "close": 101.13, "qqq": 711.96}


def build():
    m = load_merged()
    m = m[m.index >= "2010-03-12"]
    d = pd.Timestamp(TODAY["date"])
    if d not in m.index:
        row = pd.Series({k: v for k, v in TODAY.items() if k != "date"}, name=d)
        m = pd.concat([m, row.to_frame().T]).sort_index()
    for x in ("open", "high", "low", "close", "qqq"):
        m[x] = m[x].astype(float)
    c = m["close"]
    m["prev"] = c.shift(1)
    m["ret"] = c / m["prev"] - 1
    m["gap"] = m["open"] / m["prev"] - 1
    m["dd"] = c / c.rolling(250, min_periods=60).max() - 1
    ft = fomc_table()
    ft = ft[~ft["emergency"]]
    idx = m.index
    roles = {"dm1": set(), "d0": set(), "dm2": set()}
    for dt, _ in ft.iterrows():
        p = idx.searchsorted(dt)
        if p < len(idx):
            roles["d0"].add(idx[p])
        if 1 <= p <= len(idx) - 1:
            roles["dm1"].add(idx[p - 1])
        if 2 <= p <= len(idx) - 1:
            roles["dm2"].add(idx[p - 2])
    return m, roles


def price_table(sub, prev, title):
    print(f"\n### {title}  n={len(sub)}\n")
    print("| | " + " | ".join(f"{q}%" for q in QS) + " |")
    print("|---|" + "---|" * len(QS))
    for lab, col in (("开盘", "o"), ("最高", "h"), ("最低", "l"), ("收盘", "cc")):
        q = [np.percentile(sub[col].dropna(), x) for x in QS]
        print(f"| {lab} | " + " | ".join(f"{prev*(1+v):.1f}" for v in q) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev-close", type=float, required=True)
    ap.add_argument("--role", choices=["dm2", "dm1", "d0", "none"], default="dm1")
    ap.add_argument("--prev-ret", type=float, default=None,
                    help="prior session return, to also cut on 'after a big drop'")
    ap.add_argument("--open", type=float, default=None, dest="open_px",
                    help="the actual open, once known; levels are then measured from it")
    a = ap.parse_args()
    m, roles = build()
    P = a.prev_close
    for k in ("o", "h", "l", "cc"):
        m[k] = m[{"o": "open", "h": "high", "l": "low", "cc": "close"}[k]] / m["prev"] - 1

    print(f"# 下一场交易日计划（前收 {P}，角色 {a.role}）")

    if a.role != "none":
        price_table(m[m.index.isin(roles[a.role])].dropna(subset=["cc"]), P,
                    f"全部 FOMC {a.role.upper().replace('DM','D−')} 交易日")
    if a.prev_ret is not None and a.prev_ret <= -0.10:
        big = m[(m["ret"].shift(1) <= -0.13) | (m["gap"].shift(1) <= -0.12)].dropna(subset=["cc"])
        price_table(big, P, "大跌/大跳空之后的次日（不限 FOMC）")

    print("\n## 杠杆：从开盘价入场，按持有天数算期间最深浮亏\n")
    op, lw = m["open"].values, m["low"].values
    n = len(m)
    groups = [("大跌次日", np.where(((m["ret"].shift(1) <= -0.13) | (m["gap"].shift(1) <= -0.12)).values)[0])]
    if a.role != "none":
        groups.append((f"FOMC {a.role.upper().replace('DM','D−')} 入场",
                       np.where(m.index.isin(roles[a.role]))[0]))
    for name, ii in groups:
        ii = [i for i in ii if i + 6 < n]
        if len(ii) < 5:
            continue
        print(f"**{name}** n={len(ii)}\n")
        print("| 持有 | 中位 | 5%分位 | 1%分位 | " + " | ".join(f"{k} 爆仓" for k, _ in LIQ) + " |")
        print("|---|---|---|---|" + "---|" * len(LIQ))
        for N in (1, 2, 3, 5):
            mae = np.array([lw[i:i + N].min() / op[i] - 1 for i in ii])
            cells = " | ".join(f"{(mae <= lv).mean()*100:.0f}%" for _, lv in LIQ)
            print(f"| {N}天 | {np.median(mae)*100:.1f}% | {np.percentile(mae,5)*100:.1f}% "
                  f"| {np.percentile(mae,1)*100:.1f}% | {cells} |")
        print()
    print("爆仓按盘中最低触发，和券商强平同一个机制。过夜是杠杆代价的来源：")
    print("同样的杠杆，当日了结和持有五天的爆仓概率差一个数量级。\n")

    # ---- levels measured from the open ---------------------------------
    # The gap carries almost no information about how much further price falls
    # after the open (correlation about -0.2 on FOMC D-1 and 0.0 elsewhere), so
    # the open is treated as a free anchor and everything is measured from it.
    m["l_o"] = m["low"] / m["open"] - 1
    m["h_o"] = m["high"] / m["open"] - 1
    pops = []
    if a.role != "none":
        pops.append((f"FOMC {a.role.upper().replace('DM','D−')}",
                     m[m.index.isin(roles[a.role])].dropna(subset=["l_o"])))
    if a.prev_ret is not None and a.prev_ret <= -0.10:
        pops.append(("大跌次日",
                     m[(m["ret"].shift(1) <= -0.13) | (m["gap"].shift(1) <= -0.12)].dropna(subset=["l_o"])))
    if not pops:
        return
    print("## 从开盘价往下量（跳空幅度对此几乎没有预测力，所以以开盘为锚）\n")
    print("| 母体 | n | 中位低 | 25%低 | 10%低 | 5%低 | 中位高 | 75%高 |")
    print("|---|---|---|---|---|---|---|---|")
    lows, highs = {}, {}
    for nm, s_ in pops:
        lo = {p: np.percentile(s_["l_o"], p) for p in (50, 25, 10, 5)}
        hi = {p: np.percentile(s_["h_o"], p) for p in (50, 75)}
        lows[nm], highs[nm] = lo, hi
        print(f"| {nm} | {len(s_)} | " + " | ".join(f"{lo[p]*100:+.1f}%" for p in (50, 25, 10, 5))
              + f" | {hi[50]*100:+.1f}% | {hi[75]*100:+.1f}% |")
    if len(pops) > 1:
        mix_l = {p: float(np.mean([lows[n][p] for n in lows])) for p in (50, 25, 10, 5)}
        mix_h = {p: float(np.mean([highs[n][p] for n in highs])) for p in (50, 75)}
        print(f"| **混合** | — | " + " | ".join(f"{mix_l[p]*100:+.1f}%" for p in (50, 25, 10, 5))
              + f" | {mix_h[50]*100:+.1f}% | {mix_h[75]*100:+.1f}% |")
    else:
        mix_l, mix_h = lows[pops[0][0]], highs[pops[0][0]]
    print()
    opens = [a.open_px] if a.open_px else [round(P * (1 + g / 100)) for g in range(3, -8, -1)]
    print("| 开盘价 | 一档(中位低) | 二档(25%) | 深档(10%) | 极限(5%) | 中位高 | 75%高 |")
    print("|---|---|---|---|---|---|---|")
    for o in opens:
        print(f"| {o:g} | " + " | ".join(f"{o*(1+mix_l[p]):.1f}" for p in (50, 25, 10, 5))
              + f" | {o*(1+mix_h[50]):.1f} | {o*(1+mix_h[75]):.1f} |")
    print("\n一档 = 一半的交易日会摸到；二档 = 四次里一次；深档 = 十次里一次；极限 = 二十次里一次。\n")


if __name__ == "__main__":
    main()
