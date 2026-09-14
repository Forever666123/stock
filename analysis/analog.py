"""Nearest-neighbour analogues for a (possibly intraday) SOXL bar.

    python3 -m analysis.analog --date 2026-09-14 --open 101.74 --high 105.34 \
        --low 99.87 --close 104.41 --qqq 711.96 [--k 15]

The bar is appended to the history as a provisional row, features are computed
on the joined series, and the closest historical days (standardised Euclidean
distance) are listed with what happened next.  Also prints rule-based subsets
so the neighbour list can be sanity-checked against plain conditions.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis.data import load_merged

FEATS = ["gap", "low_ret", "ret", "recover_from_low", "qqq_ret", "prev1", "prev2", "dd60"]
WEIGHTS = {"gap": 1.5, "low_ret": 1.5, "ret": 1.5, "recover_from_low": 1.0, "qqq_ret": 1.5,
           "prev1": 0.7, "prev2": 0.7, "dd60": 0.7}


def pct(x, d=1):
    return "" if pd.isna(x) else f"{x*100:+.{d}f}%"


def build(args):
    m = load_merged()
    d = pd.Timestamp(args.date)
    if d in m.index:
        raise SystemExit(f"{args.date} already in data; drop the override")
    row = pd.Series({"open": args.open, "high": args.high, "low": args.low, "close": args.close,
                     "volume": np.nan, "qqq": args.qqq, "src": "manual"}, name=d)
    m = pd.concat([m, row.to_frame().T]).sort_index()
    for c in ["open", "high", "low", "close", "qqq"]:
        m[c] = m[c].astype(float)
    m["prev_close"] = m["close"].shift(1)
    m["ret"] = m["close"] / m["prev_close"] - 1
    m["gap"] = m["open"] / m["prev_close"] - 1
    m["low_ret"] = m["low"] / m["prev_close"] - 1
    m["recover_from_low"] = m["close"] / m["low"] - 1
    m["qqq_ret"] = m["qqq"] / m["qqq"].shift(1) - 1
    m["prev1"] = m["ret"].shift(1)
    m["prev2"] = m["ret"].shift(2)
    m["dd60"] = m["close"] / m["close"].rolling(60).max() - 1
    for h in (1, 2, 3, 5, 10):
        m[f"fwd{h}"] = m["close"].shift(-h) / m["close"] - 1
    m["next_gap"] = m["open"].shift(-1) / m["close"] - 1
    m["next_low"] = m["low"].shift(-1) / m["close"] - 1
    m["next_high"] = m["high"].shift(-1) / m["close"] - 1
    m["min5"] = m["low"][::-1].rolling(5, min_periods=5).min()[::-1].shift(-1) / m["close"] - 1
    return m, d


def summarize(df, name):
    if len(df) == 0:
        print(f"{name}: n=0\n"); return
    cols = {"次日": "fwd1", "次日最低": "next_low", "次日最高": "next_high", "2日(拿过决议)": "fwd2", "5日": "fwd5", "10日": "fwd10", "5日内最低": "min5"}
    parts = [f"{k} {pct(df[v].mean())}/{pct(df[v].median())}/{(df[v] > 0).mean()*100:.0f}%" if k not in ("次日最低", "5日内最低", "次日最高")
             else f"{k}中位 {pct(df[v].median())}" for k, v in cols.items()]
    print(f"**{name}** n={len(df)}（均值/中位/上涨率）：" + "；".join(parts) + "\n")


def main():
    ap = argparse.ArgumentParser()
    for k in ("open", "high", "low", "close", "qqq"):
        ap.add_argument(f"--{k}", type=float, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()
    m, d = build(args)
    today = m.loc[d]
    print(f"## 今日特征（{d.date()}，盘中快照）\n")
    print(", ".join(f"{f}={pct(today[f])}" for f in FEATS) + "\n")

    hist = m[(m.index < d) & m[FEATS].notna().all(axis=1)].copy()
    z = (hist[FEATS] - hist[FEATS].mean()) / hist[FEATS].std()
    zt = (today[FEATS].astype(float) - hist[FEATS].mean()) / hist[FEATS].std()
    w = np.array([WEIGHTS[f] for f in FEATS])
    hist["dist"] = np.sqrt((((z - zt) ** 2) * w).sum(axis=1))
    nn = hist.nsmallest(args.k, "dist")
    show = ["dist"] + FEATS + ["fwd1", "next_low", "fwd2", "fwd5", "min5"]
    t = nn[show].copy()
    for c in show[1:]:
        t[c] = t[c].map(pct)
    t["dist"] = t["dist"].round(2)
    print(f"## 最近邻 {args.k} 天（QQQ 数据 2018 起，所以只在 2018 后找）\n")
    print(t.to_markdown()); print()
    summarize(nn, f"最近邻 {args.k}")
    summarize(hist.nsmallest(8, "dist"), "最近邻 8")

    print("## 规则子集（不靠距离，只靠条件）\n")
    S = hist[hist["gap"] <= -0.10]
    summarize(S, "跳空 ≤ −10%")
    A = S[S["qqq_ret"] > -0.02]
    summarize(A, "跳空 ≤ −10% 且 QQQ > −2%（板块独跌型）")
    B = A[A["ret"] <= -0.12]
    summarize(B, "…且收跌 ≥ 12%")
    C = A[A["recover_from_low"] < 0.05]
    summarize(C, "…且收盘距最低收回 < 5%")
    D = S[S["qqq_ret"] <= -0.02]
    summarize(D, "对照：跳空 ≤ −10% 且 QQQ ≤ −2%（同步型）")
    full = m[(m.index < d)]
    E = full[(full["gap"] <= -0.10)]
    summarize(E, "对照：2010 起所有跳空 ≤ −10%（无 QQQ 条件）")
    print("## 明天挂单：以今天收盘为基准，次日最低分布（板块独跌型跳空日之后）\n")
    for lv in (-0.05, -0.08, -0.10, -0.15):
        hit = A[A["next_low"] <= lv]
        if len(hit) == 0:
            print(f"- 挂 {pct(lv,0)}：0/{len(A)} 触发"); continue
        fill_ret = hit["fwd1"] - lv  # next close vs fill, approx additive
        print(f"- 挂 {pct(lv,0)}：{len(hit)}/{len(A)} 触发；成交后当日收盘相对成交价 {pct(((1+hit['fwd1'])/(1+lv)-1).mean())}，上涨 {(((1+hit['fwd1'])/(1+lv)-1)>0).mean()*100:.0f}%")
    print()
    print("## 板块独跌型跳空日的逐日明细\n")
    tt = A[["gap", "low_ret", "ret", "recover_from_low", "qqq_ret", "fwd1", "next_low", "fwd2", "fwd5"]].copy()
    for c in tt.columns:
        tt[c] = tt[c].map(pct)
    print(tt.to_markdown())


if __name__ == "__main__":
    main()
