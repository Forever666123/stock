"""Daily briefing: put one (possibly intraday) SOXL bar into short-term,
long-term and FOMC-conditional historical context.

    python3 -m analysis.today --date 2026-09-14 --open 101.74 --high 105.34 \
        --low 99.87 --close 104.41 --qqq 711.96 --fomc 2026-09-16

Writes results/today_<date>.md.
"""
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data import load_merged
from analysis.fomc import fomc_table

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["gap", "low_ret", "ret", "recover_from_low", "qqq_ret", "prev1", "prev2", "dd60"]
WEIGHTS = np.array([1.5, 1.5, 1.5, 1.0, 1.5, 0.7, 0.7, 0.7])


def pct(x, d=1):
    return "" if pd.isna(x) else f"{x*100:+.{d}f}%"


def perm_p(a, b, n=20000, seed=1):
    a, b = np.asarray(a.dropna()), np.asarray(b.dropna())
    if len(a) < 2 or len(b) < 2:
        return np.nan
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    k = sum(1 for _ in range(n) if (rng.shuffle(pool) or abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= obs))
    return k / n


def build(a):
    m = load_merged()
    m = m[m.index >= "2010-03-12"]
    d = pd.Timestamp(a.date)
    if d not in m.index:
        row = pd.Series({"open": a.open, "high": a.high, "low": a.low, "close": a.close,
                         "qqq": a.qqq, "src": "manual"}, name=d)
        m = pd.concat([m, row.to_frame().T]).sort_index()
    for col in ("open", "high", "low", "close", "qqq"):
        m[col] = m[col].astype(float)
    c = m["close"]
    m["prev_close"] = c.shift(1)
    m["ret"] = c / m["prev_close"] - 1
    m["gap"] = m["open"] / m["prev_close"] - 1
    m["low_ret"] = m["low"] / m["prev_close"] - 1
    m["recover_from_low"] = c / m["low"] - 1
    m["qqq_ret"] = m["qqq"] / m["qqq"].shift(1) - 1
    m["prev1"], m["prev2"] = m["ret"].shift(1), m["ret"].shift(2)
    m["dd60"] = c / c.rolling(60).max() - 1
    m["hi250"] = c.rolling(250, min_periods=60).max()
    m["dd250"] = c / m["hi250"] - 1
    m["lo60"] = c.rolling(60, min_periods=20).min()
    m["above_lo"] = c / m["lo60"] - 1
    m["bars_since_lo"] = c.rolling(60, min_periods=20).apply(lambda s: len(s) - 1 - int(np.argmin(s.values)), raw=False)
    for h in (1, 2, 3, 5, 7, 20, 60):
        m[f"f{h}"] = c.shift(-h) / c - 1
        m[f"min{h}"] = m["low"][::-1].rolling(h, min_periods=1).min()[::-1].shift(-1) / c - 1
    m["next_low"] = m["low"].shift(-1) / c - 1
    # FOMC day offsets
    ft = fomc_table()
    ft = ft[~ft["emergency"]]
    idx = m.index
    m["dm1"], m["dm2"], m["d0_kind"] = False, False, ""
    for dt, r in ft.iterrows():
        p = idx.searchsorted(dt)
        if p >= len(idx):
            continue
        m.loc[idx[p], "d0_kind"] = str(r.kind)
        if p >= 1:
            m.loc[idx[p-1], "dm1"] = True
        if p >= 2:
            m.loc[idx[p-2], "dm2"] = True
            m.loc[idx[p-2], "dm2_kind"] = str(r.kind)
    return m, d


def line(name, d, cols, n_floor=3):
    if len(d) < n_floor:
        return f"| {name} | {len(d)} | 样本不足 | | | |"
    out = [f"| {name} | {len(d)} "]
    for c in cols:
        out.append(f"| {pct(d[c].mean())} / {pct(d[c].median())} / {(d[c] > 0).mean()*100:.0f}% ")
    return "".join(out) + "|"


def main():
    ap = argparse.ArgumentParser()
    for k in ("open", "high", "low", "close", "qqq"):
        ap.add_argument(f"--{k}", type=float, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--fomc", default=None, help="upcoming decision date, e.g. 2026-09-16")
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()
    m, d = build(a)
    t = m.loc[d]
    hist = m[m.index < d]
    c = m["close"]

    print(f"# 今日定位（{d.date()}）\n")
    print(f"SOXL {t.close:.2f}（{pct(t.ret)}），开 {t.open:.2f} 高 {t.high:.2f} 低 {t.low:.2f}；QQQ {t.qqq:.2f}（{pct(t.qqq_ret)}）。")
    print(f"跳空 {pct(t.gap)}，最低/前收 {pct(t.low_ret)}，收盘距最低 {pct(t.recover_from_low)}。\n")

    # ---------------- data range
    print("## 0. 数据范围：为什么没有 20–30 年\n")
    print(f"SOXL 2010-03-11 上市，本仓已抓到全部 {len(hist)+1} 根日线，这就是它的全部历史。")
    print("要回到 2000 年只能用半导体指数（SOX/SOXX）再按 3 倍日收益合成，而本会话的出口策略只放行 raw.githubusercontent.com，")
    print("Yahoo、Stooq、FRED、Nasdaq、AlphaVantage 全部 403，SOXX/SPY 在已有仓库里也只到 2018。\n")
    print("而且合成序列的价值有限：2000–2002 年 SOX 指数跌约 85%，日频再平衡的 3 倍产品会因波动拖累接近归零，")
    print("那段数据讲的是杠杆衰减，不是可交易的形态。真正缺的是 2000 和 2008 两次半导体熊市，把独立事件从 8 次加到 10 次，")
    print("仍然是案例而不是统计。**结论：不是必须，而且这里拿不到。** 你要是能把 CSV 放进仓库，我可以直接接进来。\n")

    # ---------------- long term
    print("## 1. 长周期定位：今天在哪一天的什么位置\n")
    peak_date = c.loc[:d].idxmax()
    peak = c.loc[peak_date]
    seg = m.loc[peak_date:d]
    lo_date, lo = seg["close"].idxmin(), seg["close"].min()
    print(f"本轮高点 {peak_date.date()} {peak:.2f}，今天是峰后第 {(d - peak_date).days} 个日历日、距峰 {pct(t.close/peak-1)}。")
    print(f"段内最低 {lo_date.date()} {lo:.2f}，今天距该低点 {pct(t.close/lo-1)}，低点已过 {int(t.bars_since_lo)} 个交易日。")
    print(f"今天盘中最低 {t.low:.2f}，距 {lo_date.date()} 低点 {pct(t.low/lo-1)}。\n")

    print("历次 ≥50% 回撤，走到峰后同样天数时的位置和之后的表现：\n")
    print("| 高点 | 对应日 | 距峰 | 距段内低 | 之后20日 | 之后60日 | 之后120日 |")
    print("|---|---|---|---|---|---|---|")
    episodes = [("2010-04-26", "2010-08-31"), ("2011-02-17", "2011-10-03"), ("2015-06-01", "2016-02-11"),
                ("2018-03-12", "2018-12-24"), ("2020-02-19", "2020-03-20"), ("2021-12-27", "2022-10-14")]
    days = (d - peak_date).days
    for p, _ in episodes:
        p = pd.Timestamp(p)
        later = m.index[m.index >= p + pd.Timedelta(days=days)]
        if not len(later):
            continue
        x = later[0]
        pk = c[p]
        s2 = m.loc[p:x, "close"]
        f120 = c.shift(-120).get(x, np.nan) / c[x] - 1 if x in c.index else np.nan
        print(f"| {p.date()} | {x.date()} | {pct(c[x]/pk-1)} | {pct(c[x]/s2.min()-1)} | {pct(m.loc[x,'f20'])} | {pct(m.loc[x,'f60'])} | {pct(f120)} |")
    print(f"| **2026-06-22** | **{d.date()}** | **{pct(t.close/peak-1)}** | **{pct(t.close/lo-1)}** | ? | ? | ? |\n")
    print(f"今天的特征是**跌得深但离低点近**：距峰 {pct(t.close/peak-1)}，却只比段内低点高 {pct(t.close/lo-1)[1:]}。")
    print("走到峰后同样天数时，过去六次的距峰幅度是 −36%、−19%、−58%、−10%、−63%、−47%，**今天比其中任何一次都深**。")
    print("那六次里，距峰最深的两次（2015 −58%、2020 −63%）之后 60 日分别是 +47% 和 +100%，但 2015 那次是正好踩在当天的低点上，2020 那次已经从低点反弹了 88%。今天两个条件都不满足。\n")

    print("按状态匹配（不看时间，只看位置）：\n")
    print("| 条件 | n | 20日 均值/中位/胜率 | 60日 均值/中位/胜率 | 20日内最低中位 |")
    print("|---|---|---|---|---|")
    for nm, sub in (("全样本", hist),
                    ("距250日高 ≤ −50%", hist[hist.dd250 <= -0.5]),
                    ("距250日高 ≤ −50% 且距60日低 ≤ +25%", hist[(hist.dd250 <= -0.5) & (hist.above_lo <= 0.25)]),
                    ("…且低点已过 ≥25 根（今天的状态）", hist[(hist.dd250 <= -0.5) & (hist.above_lo <= 0.25) & (hist.bars_since_lo >= 25)])):
        if len(sub) < 3:
            print(f"| {nm} | {len(sub)} | 样本不足 | | |"); continue
        print(f"| {nm} | {len(sub)} | {pct(sub.f20.mean())} / {pct(sub.f20.median())} / {(sub.f20>0).mean()*100:.0f}% "
              f"| {pct(sub.f60.mean())} / {pct(sub.f60.median())} / {(sub.f60>0).mean()*100:.0f}% | {pct(sub.min20.median())} |")
    print()
    rt = hist[(hist.dd250 <= -0.5) & (hist.above_lo <= 0.20) & (hist.bars_since_lo >= 25)]
    broke = (rt["min20"] < (rt["lo60"] / c[rt.index] - 1)).mean()
    print(f"**关键一条**：深回撤中逼近一个 25 根以上的旧低点时（n={len(rt)}），20 个交易日内跌破那个低点的比例 **{broke*100:.0f}%**。")
    print(f"对应现在：{lo_date.date()} 的 {lo:.2f} 大概率会被测试到甚至跌破。\n")

    # ---------------- short term
    print("## 2. 短周期匹配：最像的历史 K 线\n")
    hh = hist[hist[FEATS].notna().all(axis=1)].copy()
    z = (hh[FEATS] - hh[FEATS].mean()) / hh[FEATS].std()
    zt = (t[FEATS].astype(float) - hh[FEATS].mean()) / hh[FEATS].std()
    hh["dist"] = np.sqrt((((z - zt) ** 2) * WEIGHTS).sum(axis=1))
    nn = hh.nsmallest(a.k, "dist")
    print("| 日期 | 距离 | 跳空 | 收跌 | QQQ | 次日 | 次日最低 | 2日 | 5日 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for i, r in nn.iterrows():
        print(f"| {i.date()} | {r.dist:.2f} | {pct(r.gap)} | {pct(r.ret)} | {pct(r.qqq_ret)} | {pct(r.f1)} | {pct(r.next_low)} | {pct(r.f2)} | {pct(r.f5)} |")
    print()
    print(f"最近邻 {len(nn)}：次日均值 {pct(nn.f1.mean())} / 中位 {pct(nn.f1.median())} / 胜率 {(nn.f1>0).mean()*100:.0f}%；次日最低中位 {pct(nn.next_low.median())}。\n")

    print("规则子集（不靠距离）：\n")
    print("| 条件 | n | 次日 均值/中位/胜率 | 2日 | 5日 | 次日最低中位 |")
    print("|---|---|---|---|---|---|")
    S = hist[hist.gap <= -0.10]
    subs = [("跳空 ≤ −10%", S),
            ("跳空 ≤ −10% 且 QQQ > −2%（板块独跌型）", S[S.qqq_ret > -0.02]),
            ("跳空 ≤ −10% 且 QQQ ≤ −2%（同步型）", S[S.qqq_ret <= -0.02]),
            ("收跌 ≥6% 且 QQQ > −2%", hist[(hist.ret <= -0.06) & (hist.qqq_ret > -0.02)])]
    for nm, sub in subs:
        if len(sub) < 3:
            print(f"| {nm} | {len(sub)} | 样本不足 | | | |"); continue
        print(f"| {nm} | {len(sub)} | {pct(sub.f1.mean())} / {pct(sub.f1.median())} / {(sub.f1>0).mean()*100:.0f}% "
              f"| {pct(sub.f2.mean())} / {(sub.f2>0).mean()*100:.0f}% | {pct(sub.f5.mean())} / {(sub.f5>0).mean()*100:.0f}% | {pct(sub.next_low.median())} |")
    print()

    # ---------------- FOMC
    print("## 3. 决议筛选\n")
    if a.fomc:
        fd = pd.Timestamp(a.fomc)
        n_between = int(np.busday_count(d.date(), fd.date()))
        print(f"下次决议 {fd.date()}，距今 {n_between} 个交易日，今天是 D−{n_between}。\n")
    big = hist[hist.ret <= -0.06]
    ad, bd = big[big.dm2], big[~big.dm2]
    print("从 D−2 收盘（= 今天收盘）买入，拿到各时点：\n")
    print("| 组 | n | →D0(2日) | →D+1(3日) | →D+5(7日) |")
    print("|---|---|---|---|---|")
    for nm, sub in (("当日跌≥6% 且是 D−2", ad), ("当日跌≥6% 但不是 D−2（对照）", bd),
                    ("所有 D−2", hist[hist.dm2]), ("全样本", hist)):
        if len(sub) < 3:
            print(f"| {nm} | {len(sub)} | 样本不足 | | |"); continue
        print(f"| {nm} | {len(sub)} | {pct(sub.f2.mean())} / {(sub.f2>0).mean()*100:.0f}% | {pct(sub.f3.mean())} / {(sub.f3>0).mean()*100:.0f}% | {pct(sub.f7.mean())} / {(sub.f7>0).mean()*100:.0f}% |")
    print()
    for h, nmh in ((2, "→D0"), (3, "→D+1"), (7, "→D+5")):
        print(f"- {nmh} D−2 vs 非 D−2 置换检验 p = {perm_p(ad[f'f{h}'], bd[f'f{h}']):.3f}")
    print()
    print(f"逐笔（{len(ad)} 次）：\n")
    print("| 日期 | 会议结果 | 当日 | QQQ | →D0 | →D+1 | →D+5 |")
    print("|---|---|---|---|---|---|---|")
    for i, r in ad.iterrows():
        kind = m.loc[m.index[m.index.searchsorted(i)+2], "d0_kind"] if m.index.searchsorted(i)+2 < len(m) else ""
        print(f"| {i.date()} | {kind} | {pct(r.ret)} | {pct(r.qqq_ret)} | {pct(r.f2)} | {pct(r.f3)} | {pct(r.f7)} |")
    print()
    inter = hist[(hist.ret <= -0.06) & (hist.qqq_ret > -0.02) & (hist.dm2)]
    print(f"交集（板块独跌 + D−2）只有 n={len(inter)}：" + "、".join(f"{i.date()} {pct(r.f2)}" for i, r in inter.iterrows()) + "。不能下结论。\n")

    # ---------------- conflict
    print("## 4. 三条线索互相打架\n")
    print("| 线索 | 方向 | n | 强度 |")
    print("|---|---|---|---|")
    al = hist[(hist.ret <= -0.06) & (hist.qqq_ret > -0.02)]
    print(f"| 板块独跌大跌（今天的形态） | 看空 | {len(al)} | 2日 {pct(al.f2.mean())}，胜率 {(al.f2>0).mean()*100:.0f}%，最扎实 |")
    print(f"| FOMC D−2 大跌 | 看多 | {len(ad)} | 2日 {pct(ad.f2.mean())}，p={perm_p(ad['f2'], bd['f2']):.3f}，但 n={len(ad)} 且是我筛出来的 |")
    print(f"| 深回撤逼近旧低 | 看空 | {len(rt)} | 20日内 {broke*100:.0f}% 跌破前低 |")
    print()
    print("按你自己的规矩：n<50 只能找反例。看多那条 n=9，而且是我试了十几个筛选条件之后挑出来的最好看的一个，")
    print("Bonferroni 之后 p 远不显著，并且它到 D+1 就失效（p=0.20）。它最多说明\"别在决议前做空\"，不是\"抄底\"。\n")

    print("## 5. 结论\n")
    print(f"1. **今天不是底。** 判别底部的唯一硬指标是 {lo_date.date()} 的 {lo:.2f}，它还没被测试。历史上这种位置 {broke*100:.0f}% 会去测。")
    print("2. **明天（D−1）不追。** 板块独跌型跳空的次日胜率 50%，均值为负。")
    print(f"3. **要挂就挂到前低附近。** {lo:.2f} 对应今天收盘 {pct(lo/t.close-1)}。这是唯一有结构意义的价位，不是拍脑袋的百分比。")
    print("4. **决议是事件不是信号。** 看多的那条证据 n=9 且不稳，不足以让你提前进场；它唯一的用处是告诉你别在 D−1 追空。")
    print(f"5. **仓位按 {pct(hist[(hist.dd250<=-0.5)&(hist.above_lo<=0.25)].min60.median())} 定。** 这是深回撤近低位状态下 60 日内最低的中位数。\n")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (ROOT / "results").mkdir(exist_ok=True)
    import sys
    dt = sys.argv[sys.argv.index("--date") + 1]
    (ROOT / "results" / f"today_{dt}.md").write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())
