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


if __name__ == "__main__":
    main()
