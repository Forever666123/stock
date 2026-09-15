"""Does a fully expected rate hike mean the bad news is over?

Tests the "it's priced in" claim three ways instead of assuming it:
  1. implied volatility — does VIX crush after the decision, as it would if the
     event resolved the uncertainty?
  2. realised volatility — does SOXL's daily range contract or expand?
  3. drift — what are returns after the decision, split by how predictable the
     move was and by how deep the drawdown already was?

    python3 -m analysis.priced_in
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import ROOT, load_merged
from analysis.fomc import fomc_table

KIND_CN = {"hike": "加息", "hold": "按兵", "cut": "降息"}


def perm_p(a, b, n=20000, seed=7):
    a, b = np.asarray(a), np.asarray(b)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    return sum(1 for _ in range(n)
               if (rng.shuffle(pool) or abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= obs)) / n


def build():
    m = load_merged()
    m = m[m.index >= "2010-03-12"].copy()
    c = m["close"]
    m["rng"] = (m["high"] - m["low"]) / c.shift(1)
    m["dd"] = c / c.rolling(250, min_periods=60).max() - 1
    vix = pd.read_csv(ROOT / "data" / "vix-daily.csv", parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    ft = fomc_table()
    ft = ft[~ft["emergency"]]
    idx = m.index
    # nth consecutive hike, and whether the size repeated (a repeat is the most
    # predictable move the calendar offers)
    nth, prev, seq = 0, None, {}
    for dt, r in ft.sort_index().iterrows():
        if str(r.kind) == "hike":
            nth += 1
            seq[dt] = (nth, r.action == prev)
            prev = r.action
        else:
            nth, prev = 0, None
    rows = []
    for dt, r in ft.iterrows():
        p = idx.searchsorted(dt)
        if p < 6 or p + 21 >= len(idx):
            continue
        g = lambda o: vix.reindex([idx[p + o]]).iloc[0]
        n_, same = seq.get(dt, (0, False))
        rows.append(dict(date=dt, kind=str(r.kind), nth=n_, same=same, dd=m["dd"].iloc[p - 1],
                         d0=m["ret"].iloc[p], d1=c.iloc[p+1]/c.iloc[p]-1, d5=c.iloc[p+5]/c.iloc[p]-1,
                         d20=c.iloc[p+20]/c.iloc[p]-1,
                         min10=m["low"].iloc[p+1:p+11].min()/c.iloc[p]-1,
                         vm1=g(-1), v0=g(0), v1=g(1),
                         **{f"rng{o}": m["rng"].iloc[p + o] for o in (-1, 0, 1, 2, 5)}))
    return m, pd.DataFrame(rows).set_index("date")


def row(d, name):
    if len(d) < 3:
        return f"| {name} | {len(d)} | 样本不足 | | | |"
    return (f"| {name} | {len(d)} | {d.d0.mean()*100:+.1f}% / {(d.d0>0).mean()*100:.0f}% "
            f"| {d.d5.mean()*100:+.1f}% / {(d.d5>0).mean()*100:.0f}% "
            f"| {d.d20.mean()*100:+.1f}% / {(d.d20>0).mean()*100:.0f}% "
            f"| {d.min10.median()*100:+.1f}% |")


def main():
    m, t = build()
    print("# 加息真的被定价了就利空出尽了吗\n")

    print("## 1. 隐含波动率：决议后不确定性有没有消失\n")
    print("| 会议类型 | n | D−1 VIX | D+1 VIX | 变化 | 下降占比 |")
    print("|---|---|---|---|---|---|")
    for k in ("hike", "hold", "cut"):
        d = t[t.kind == k].dropna(subset=["vm1", "v1"])
        ch = d.v1 / d.vm1 - 1
        print(f"| {KIND_CN[k]} | {len(d)} | {d.vm1.mean():.1f} | {d.v1.mean():.1f} | {ch.mean()*100:+.1f}% | {(ch<0).mean()*100:.0f}% |")
    print("\n事件溢价消退应该让 VIX 在决议后回落。按兵会议是这样，加息会议不是。\n")

    print("## 2. 实际波动率：SOXL 自己的日振幅\n")
    print("| 日 | 加息会议 | 按兵会议 |")
    print("|---|---|---|")
    for o, lab in ((-1, "D−1"), (0, "D0"), (1, "D+1"), (2, "D+2"), (5, "D+5")):
        a, b = t[t.kind == "hike"][f"rng{o}"].median(), t[t.kind == "hold"][f"rng{o}"].median()
        print(f"| {lab} | {a*100:.1f}% | {b*100:.1f}% |")
    print(f"\n全样本日振幅中位 {m['rng'].median()*100:.1f}%。加息决议日振幅接近基准的两倍，一周后仍然偏高。\n")

    print("## 3. 决议之后的漂移（从 D0 收盘算）\n")
    print("| 组 | n | D0 | D+5 | D+20 | 10日内最低中位 |")
    print("|---|---|---|---|---|---|")
    h = t[t.kind == "hike"]
    print(row(h, "加息"))
    print(row(t[t.kind == "hold"], "按兵"))
    print(row(h[h.nth == 1], "　第 1 次加息（最意外）"))
    print(row(h[h.nth >= 2], "　第 2 次起（已被预期）"))
    print(row(h[h.same], "　与上次同幅（最可预测）"))
    print(row(h[h.dd <= -0.60], "　深跌 ≤ −60% 中加息"))
    print(row(t[(t.kind == "hold") & (t.dd <= -0.40)], "　深跌 ≤ −40% 中按兵（对照）"))
    print()
    hk, hd = t[t.kind == "hike"].d20, t[t.kind == "hold"].d20
    print(f"加息 vs 按兵 D+20：{hk.mean()*100:+.1f}% vs {hd.mean()*100:+.1f}%，置换检验 p = {perm_p(hk, hd):.3f}；"
          f"上涨率 {(hk>0).mean()*100:.0f}% vs {(hd>0).mean()*100:.0f}%。\n")
    deep = h[h.dd <= -0.60]
    yrs = sorted({d.year for d in deep.index})
    print(f"最贴近深跌处境的那一格 n={len(deep)}，全部来自 {yrs[0]}–{yrs[-1]} 一个加息周期，")
    print("是一个 regime 不是若干独立事件。它只能当反例用。\n")
    print("**结论：加息被定价，指的是那 25bp 这个事件被定价，不是路径被定价。**")
    print("VIX 在加息后不回落，SOXL 的实际波动反而扩张——市场自己并不认为决议结束了什么。\n")


if __name__ == "__main__":
    main()
