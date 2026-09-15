"""What daily OHLC can and cannot say about WHEN inside the session the low happened.

    python3 -m analysis.bt_intraday

MECHANISM, STATED BEFORE ANY NUMBER
-----------------------------------
A daily bar is four numbers. It identifies the *time* of the low in exactly two
degenerate cases and in no others:

  low == open   -> the low was printed at (or within ticks of) the opening print,
                   i.e. at the very start of the session.
  low == close  -> the session ended on its low, i.e. the low was at the very end.
  otherwise     -> the low happened strictly inside the session and the daily bar
                   contains ZERO information about when. A low at 09:35 and a low
                   at 15:45 produce byte-identical daily bars whenever open, high
                   and close agree. This is not a weak signal, it is an
                   unidentified parameter.

Second mechanism, which is the adversarial half of this study:
the class is very nearly a restatement of the sign and size of the day's own
return. If the day closes up, the open is a strong candidate for the low; if the
day closes down hard, the close is a strong candidate for the low. So
  "low at the open" ~= "today was an up day"
  "low at the close" ~= "today was a down day"
Any forward-return difference between the classes is therefore a short-horizon
reversal/momentum result wearing a costume, NOT an intraday-timing result. The
test below is built to expose that: every contrast is run twice, once raw and
once after removing the same-day return by stratifying on its decile. If the raw
difference dies under the control, the class carries nothing of its own.

Observability: low==open and low==close are ex-post. They classify history, they
can never trigger an entry. Nothing here is proposed as a signal; the point is to
bound what the data can support.

Tolerance: 0.2% of the previous close, matching analysis/data.py's low_at_open.

Multiple testing: every permutation test run is collected into one family and
Holm-corrected together. The count is printed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import load_merged
from analysis.fomc import fomc_table

TOL = 0.002          # 0.2% of previous close
NPERM = 20000
RNG = np.random.default_rng(20260915)
SMALL_N = 50         # below this: counter-example only, never a conclusion

# The most recent bar is not in the CSV.
LAST_BAR = dict(date="2026-09-14", open=101.53, high=105.39, low=99.87,
                close=101.13, qqq=711.96)


# ----------------------------------------------------------------- data

def frame() -> pd.DataFrame:
    m = load_merged()
    d = pd.Timestamp(LAST_BAR["date"])
    assert d not in m.index, "2026-09-14 already present -- do not double-append"
    m.loc[d, ["open", "high", "low", "close", "qqq"]] = [
        LAST_BAR["open"], LAST_BAR["high"], LAST_BAR["low"],
        LAST_BAR["close"], LAST_BAR["qqq"]]
    m = m.sort_index()
    pc = m["close"].shift(1)
    m["prev_close"] = pc
    m["ret"] = m["close"] / pc - 1
    m["gap"] = m["open"] / pc - 1
    m["low_ret"] = m["low"] / pc - 1
    m["range"] = (m["high"] - m["low"]) / pc
    for h in (1, 5):
        m[f"fwd{h}"] = m["close"].shift(-h) / m["close"] - 1
    # classification
    d_open = (m["open"] - m["low"]).abs() / pc
    d_close = (m["close"] - m["low"]).abs() / pc
    at_open, at_close = d_open < TOL, d_close < TOL
    m["cls"] = np.where(at_open & at_close, "both",
                np.where(at_open, "low@open",
                np.where(at_close, "low@close", "low@mid")))
    return m[m.index >= "2010-03-12"].copy()


def populations(m: pd.DataFrame, fomc_d1: pd.Index) -> dict:
    """Every mask is a property of the session being classified, or of sessions
    strictly before it. No forward information is used to select a population."""
    prev_ret = m["ret"].shift(1)
    return {
        "全样本 all days":                 pd.Series(True, index=m.index),
        "前一日跌≥13% after -13% day":     prev_ret <= -0.13,
        "低开 gap-down (gap<0)":           m["gap"] < 0,
        "大幅低开 gap<=-3%":               m["gap"] <= -0.03,
        "FOMC D-1":                        m.index.isin(fomc_d1),
    }


def fomc_dm1(m: pd.DataFrame) -> pd.Index:
    """The trading bar immediately before a scheduled statement day with a known
    action. 2026 rows with action=None stay excluded (fomc.py's rule)."""
    t = fomc_table(include_2026=True)
    t = t[~t["emergency"] & t["action"].notna()]
    pos = pd.Series(np.arange(len(m)), index=m.index)
    out = []
    for dt in t.index:
        if dt in pos.index and pos[dt] > 0:
            out.append(m.index[pos[dt] - 1])
    return pd.DatetimeIndex(out)


# ----------------------------------------------------------------- stats

TESTS: list[tuple[str, float]] = []


def perm_p(a, b, stat="median", n=NPERM) -> float:
    a = np.asarray(pd.Series(a).dropna(), float)
    b = np.asarray(pd.Series(b).dropna(), float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    f = np.median if stat == "median" else np.mean
    pool = np.concatenate([a, b])
    na, N = len(a), len(pool)
    obs = abs(f(a) - f(b))
    hits = 0
    for _ in range(n):
        idx = RNG.choice(N, na, replace=False)
        mask = np.zeros(N, bool); mask[idx] = True
        if abs(f(pool[mask]) - f(pool[~mask])) >= obs - 1e-15:
            hits += 1
    return (hits + 1) / (n + 1)


def record(label, a, b, stat="median") -> float:
    p = perm_p(a, b, stat)
    TESTS.append((label, p))
    return p


def holm(tests):
    k = len(tests)
    order = sorted(range(k), key=lambda i: tests[i][1])
    adj = [None] * k
    running = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (k - rank) * tests[i][1])
        running = max(running, v)
        adj[i] = running
    return adj


def desc(x) -> str:
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "n=0"
    return (f"n={len(x)} 均值{x.mean()*100:+.2f}% 中位{x.median()*100:+.2f}% "
            f"胜率{(x > 0).mean()*100:.0f}%")


# ----------------------------------------------------------------- report

ORDER = ["low@open", "low@mid", "low@close", "both"]


def main():
    m = frame()
    d1 = fomc_dm1(m)
    pops = populations(m, d1)

    print("# 盘中低点出现在什么时候：日线能回答的和不能回答的\n")
    print("**先说机制。** 一根日K只有四个数。它能定位低点时刻的情形只有两种：")
    print("low==open（低点在开盘那一刻）和 low==close（收在最低，低点在尾盘）。")
    print("其余所有日子，低点落在盘中某处，**日线对具体时刻的信息量恰好为零**——")
    print("09:35 见低和 15:45 见低，只要 open/high/close 相同，日K一模一样。")
    print("这不是弱信号，这是一个不可识别的参数。\n")
    print(f"分类容差：前收的 {TOL*100:.1f}%。'both' = 当天振幅小于容差，开=收=低，无法区分。\n")

    # ---- 1. frequencies
    print("## 1. 三类的出现频率\n")
    print("| 参照组 | n | low@open | low@mid | low@close | both |")
    print("|---|---|---|---|---|---|")
    counts = {}
    for name, mask in pops.items():
        d = m[mask & m["cls"].notna() & m["prev_close"].notna()]
        counts[name] = d
        vc = d["cls"].value_counts()
        cells = " | ".join(f"{vc.get(c,0)/max(len(d),1)*100:.1f}% ({vc.get(c,0)})" for c in ORDER)
        flag = "  ⚠n<50" if len(d) < SMALL_N else ""
        print(f"| {name}{flag} | {len(d)} | {cells} |")
    print()
    for name, d in counts.items():
        if len(d) < SMALL_N:
            print(f"> ⚠ **{name} n={len(d)} < {SMALL_N}：只能作为反例，不能作为结论。**")
    print()

    # ---- 1b. the confound, shown before any forward test
    print("## 2. 先把混淆变量摆出来：类别几乎就是当天涨跌\n")
    print("| 类别 | n | 当日收益 中位 | 当日收益 均值 | 上涨占比 | 当日振幅 中位 |")
    print("|---|---|---|---|---|---|")
    allm = counts["全样本 all days"]
    for c in ORDER:
        d = allm[allm["cls"] == c]
        if not len(d):
            continue
        print(f"| {c} | {len(d)} | {d['ret'].median()*100:+.2f}% | {d['ret'].mean()*100:+.2f}% "
              f"| {(d['ret'] > 0).mean()*100:.0f}% | {d['range'].median()*100:.1f}% |")
    print("\n这就是全部问题所在：low@open 基本等于'今天收阳'，low@close 基本等于'今天收阴'。")
    print("任何类别间的前瞻差异，先验上应该被解读成短周期反转/动量，而不是盘中时点。\n")

    # ---- 1c. regime stability
    print("**频率随时间是否稳定**（全样本，按年份段）：\n")
    print("| 区间 | n | low@open | low@mid | low@close |")
    print("|---|---|---|---|---|")
    for a, b in [("2010", "2014"), ("2015", "2019"), ("2020", "2022"), ("2023", "2026")]:
        d = allm[(allm.index.year >= int(a)) & (allm.index.year <= int(b))]
        vc = d["cls"].value_counts()
        print(f"| {a}-{b} | {len(d)} | " + " | ".join(
            f"{vc.get(c,0)/len(d)*100:.1f}%" for c in ORDER[:3]) + " |")
    print()

    # ---- 2. forward predictive value
    print("## 3. 类别有没有前瞻预测力（1日 / 5日）\n")
    print("对照组一律取 low@mid（低点在盘中、时点不可知的那一类）。\n")
    for name in ["全样本 all days", "低开 gap-down (gap<0)", "大幅低开 gap<=-3%",
                 "前一日跌≥13% after -13% day", "FOMC D-1"]:
        d = counts[name]
        small = len(d) < SMALL_N
        print(f"### {name}  (n={len(d)}){'  ⚠n<50 只能当反例' if small else ''}\n")
        print("| 类别 | H | n | 均值 | 中位 | 胜率 | vs low@mid 中位p | vs low@mid 均值p |")
        print("|---|---|---|---|---|---|---|---|")
        for h in (1, 5):
            ref = d[d["cls"] == "low@mid"][f"fwd{h}"]
            for c in ORDER:
                x = d[d["cls"] == c][f"fwd{h}"].dropna()
                if len(x) < 2:
                    continue
                if c == "low@mid":
                    pm = pa = float("nan")
                else:
                    pm = record(f"{name}|{c}|fwd{h}|median", x, ref, "median")
                    pa = record(f"{name}|{c}|fwd{h}|mean", x, ref, "mean")
                print(f"| {c} | {h}d | {len(x)} | {x.mean()*100:+.2f}% | {x.median()*100:+.2f}% "
                      f"| {(x>0).mean()*100:.0f}% | " +
                      ("—" if np.isnan(pm) else f"{pm:.3f}") + " | " +
                      ("—" if np.isnan(pa) else f"{pa:.3f}") + " |")
        print()

    # ---- 3. the control: strip out same-day return
    print("## 4. 对照检验：扣掉当天涨跌之后还剩什么\n")
    print("做法：把全样本按当日收益分成十档，在每档内减去该档 fwd 的中位数，")
    print("得到的残差再做类别对比。这样类别里'今天涨了/跌了'的成分被抽掉，")
    print("剩下的才是类别本身（低点位置）可能携带的信息。\n")
    print("| H | 类别 | n | 残差均值 | 残差中位 | vs low@mid 中位p |")
    print("|---|---|---|---|---|---|")
    dec = pd.qcut(allm["ret"], 10, labels=False, duplicates="drop")
    for h in (1, 5):
        r = allm[f"fwd{h}"] - allm.groupby(dec)[f"fwd{h}"].transform("median")
        ref = r[allm["cls"] == "low@mid"]
        for c in ORDER[:3]:
            x = r[allm["cls"] == c].dropna()
            p = float("nan") if c == "low@mid" else record(
                f"控制当日收益|{c}|fwd{h}|median", x, ref, "median")
            print(f"| {h}d | {c} | {len(x)} | {x.mean()*100:+.2f}% | {x.median()*100:+.2f}% | " +
                  ("—" if np.isnan(p) else f"{p:.3f}") + " |")
    print()

    # ---- 4. FOMC D-1 fill question
    print("## 5. FOMC D-1：挂在前收之下的单子到底能不能成交\n")
    dd = counts["FOMC D-1"]
    base = allm
    print("| 组 | n | 低点<前收 | 低点≥前收（全天没跌破） | low_ret 中位 | low_ret 5分位 | "
          "跌破−5% | 跌破−9% |")
    print("|---|---|---|---|---|---|---|---|")
    for nm, d in (("FOMC D-1", dd), ("全样本", base),
                  ("FOMC D-1 且平开 |gap|<1%", dd[dd["gap"].abs() < 0.01]),
                  ("全样本 且平开 |gap|<1%", base[base["gap"].abs() < 0.01])):
        lr = d["low_ret"].dropna()
        if not len(lr):
            continue
        flag = " ⚠n<50" if len(lr) < SMALL_N else ""
        print(f"| {nm}{flag} | {len(lr)} | {(lr<0).mean()*100:.1f}% | {(lr>=0).mean()*100:.1f}% "
              f"| {lr.median()*100:+.2f}% | {np.percentile(lr,5)*100:+.1f}% "
              f"| {(lr<=-0.05).mean()*100:.1f}% | {(lr<=-0.09).mean()*100:.1f}% |")
    print()
    p_fill = record("FOMC D-1 vs 全样本|low_ret|median",
                    dd["low_ret"], base["low_ret"], "median")
    p_fill_m = record("FOMC D-1 vs 全样本|low_ret|mean",
                      dd["low_ret"], base["low_ret"], "mean")
    print(f"FOMC D-1 的 low_ret 与全样本比：中位 p={p_fill:.3f}，均值 p={p_fill_m:.3f}。\n")
    print("D-1 当天低点在前收之上（挂单完全不成交）的那些日子：")
    nofill = dd[dd["low_ret"] >= 0]
    print(f"共 {len(nofill)} 天，占 {len(nofill)/len(dd.dropna(subset=['low_ret']))*100:.1f}%；"
          f"其中最近几次 {', '.join(str(i.date()) for i in nofill.index[-5:])}\n")

    # ---- 5. multiple testing
    adj = holm(TESTS)
    sig = [(l, p, a) for (l, p), a in zip(TESTS, adj) if a < 0.05]
    print("## 6. 多重检验\n")
    print(f"本文共跑了 **{len(TESTS)} 个置换检验**（每个 {NPERM} 次重排）。")
    print(f"Bonferroni 阈值 = {0.05/len(TESTS):.5f}。Holm 校正后 p<0.05 的检验：\n")
    if not sig:
        print("**一个都没有。** 未校正时看起来显著的几条，全部死在多重检验校正上。\n")
    else:
        print("| 检验 | 原始p | Holm校正p |")
        print("|---|---|---|")
        for l, p, a in sorted(sig, key=lambda t: t[2]):
            print(f"| {l} | {p:.4f} | {a:.4f} |")
        print()
    raw_sig = [(l, p) for l, p in TESTS if p < 0.05]
    print(f"未校正 p<0.05 的有 {len(raw_sig)} 条，随机预期 {0.05*len(TESTS):.1f} 条。\n")

    # ---- 6. today
    print("## 7. 今天\n")
    t = m.loc[pd.Timestamp(LAST_BAR['date'])]
    print(f"2026-09-14：open {t.open} high {t.high} low {t.low} close {t.close} → "
          f"分类 **{t.cls}**（低点距开盘 {(t.open-t.low)/t.prev_close*100:.1f}%，"
          f"距收盘 {(t.close-t.low)/t.prev_close*100:.1f}%）。")
    print("所以 9/14 的低点落在盘中，日线无法告诉我们是几点。\n")
    print("**不能说的话：** '低点通常出现在十点'、'尾盘才是买点'、'开盘半小时最凶'——")
    print("这些陈述在本数据集上全部不可检验。要回答它们需要分钟级数据，此处没有。\n")


if __name__ == "__main__":
    main()
