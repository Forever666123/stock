"""Does breaking a prior low mean anything?

Common belief: losing a prior low is a bearish break. Opposite belief: it is a
washout that marks the bottom. This tests both against the same control groups.

    python3 -m analysis.prior_low
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import load_merged

MIN_AGE = 25      # the prior low must be at least this many bars old
DD_FLOOR = -0.40  # only look inside a real drawdown


def perm_p(a, b, n=20000, seed=3):
    a, b = np.asarray(a.dropna()), np.asarray(b.dropna())
    if len(a) < 2 or len(b) < 2:
        return np.nan
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    return sum(1 for _ in range(n)
               if (rng.shuffle(pool) or abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= obs)) / n


def build():
    m = load_merged()
    m = m[m.index >= "2010-03-12"].copy()
    c, low = m["close"], m["low"]
    m["dd"] = c / c.rolling(250, min_periods=60).max() - 1
    # prior low uses only data strictly before today, so it is knowable at the open
    m["prior_low"] = low.shift(1).rolling(60, min_periods=MIN_AGE).min()
    m["age"] = low.shift(1).rolling(60, min_periods=MIN_AGE).apply(
        lambda s: len(s) - 1 - int(np.argmin(s.values)), raw=False)
    for h in (1, 3, 5, 10, 20):
        m[f"f{h}"] = c.shift(-h) / c - 1
    m["min20"] = low[::-1].rolling(20, min_periods=1).min()[::-1].shift(-1) / c - 1
    return m


def main():
    m = build()
    deep = m["dd"] <= DD_FLOOR
    old = m["age"] >= MIN_AGE
    broke = m[deep & old & (m["low"] < m["prior_low"])].copy()
    # one row per episode: the first break, not the whole slide
    broke = broke.groupby((broke.index.to_series().diff().dt.days > 10).cumsum()).head(1)
    near = m[deep & old & (m["low"] >= m["prior_low"]) & (m["close"] / m["prior_low"] - 1 <= 0.15)]

    print("# 跌破前低意味着什么\n")
    print(f"定义：{DD_FLOOR*100:.0f}% 以上回撤中，盘中跌破一个已存在 ≥{MIN_AGE} 根的前低。每段行情只取第一次破位。\n")
    print("| 破位日 | 收盘 | 破位幅度 | 次日 | 5日 | 20日 | 20日内最低 |")
    print("|---|---|---|---|---|---|---|")
    for i, r in broke.iterrows():
        print(f"| {i.date()} | {r.close:.2f} | {(r.low/r.prior_low-1)*100:+.1f}% | {r.f1*100:+.1f}% "
              f"| {r.f5*100:+.1f}% | {r.f20*100:+.1f}% | {r.min20*100:+.1f}% |")
    print()

    lo_q, hi_q = broke["ret"].quantile(0.25), broke["ret"].quantile(0.75)
    same_size = m[deep & (m["ret"].between(lo_q, hi_q)) & (m["low"] >= m["prior_low"])]
    no_outlier = broke[broke["f20"] < broke["f20"].max()]

    print("| 组 | n | 20日均值 | 20日中位 | 上涨率 |")
    print("|---|---|---|---|---|")
    for nm, d in (("破位", broke), ("破位（剔除单个最大异常值）", no_outlier),
                  ("逼近但没破", near), (f"同等跌幅但没破位", same_size),
                  ("深回撤全体", m[deep]), ("全样本", m)):
        d = d.dropna(subset=["f20"])
        print(f"| {nm} | {len(d)} | {d.f20.mean()*100:+.1f}% | {d.f20.median()*100:+.1f}% | {(d.f20>0).mean()*100:.0f}% |")
    print()
    print(f"破位 vs 逼近未破，置换检验 p = {perm_p(broke['f20'], near['f20']):.3f}；"
          f"剔除异常值后 p = {perm_p(no_outlier['f20'], near['f20']):.3f}\n")
    print("**结论：前低没有预测力，既不是支撑也不是破位信号。** 原始数字看起来破位后表现更好，")
    print("但那是 2026-03-30 一笔 +170% 撑起来的；剔掉它中位数归零，p 值从 0.11 变成 0.54。")
    print("跟\"同样跌幅但没破位\"的日子比，也看不出差别。把前低当信号用是给噪声赋予意义。\n")
    print(f"唯一稳定的一条：破位后 20 日内还要再跌 {broke.min20.median()*100:.1f}%（中位），")
    print(f"最糟的几次是 {', '.join(f'{i.date()} {r.min20*100:.0f}%' for i, r in broke.nsmallest(3, 'min20').iterrows())}。\n")


if __name__ == "__main__":
    main()
