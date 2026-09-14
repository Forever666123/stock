"""Re-run every condition from the SOXL / FOMC historical study and write
results/report.md.  Usage:  python3 -m analysis.run
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data import load_merged
from analysis.fomc import fomc_table

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

GAP_TH = -0.10          # "gap-down scenario": open <= 90% of prior close
LEVELS = {"-15.5%": -0.155, "-22%": -0.22, "-24.9%": -0.249}
LIQ = {"10x": -0.095, "5x": -0.195, "3x": -0.328}   # per-user liquidation drops
SECTOR_TH, QQQ_TH = -0.08, -0.02


def pct(x, d=2):
    return "nan" if pd.isna(x) else f"{x*100:+.{d}f}%"


def rate(s):
    s = s.dropna()
    return f"{(s > 0).mean()*100:.0f}% ({int((s > 0).sum())}/{len(s)})" if len(s) else "n/a"


def binom_p(k, n, p0):
    """Two-sided-ish: P(X >= k) under Binomial(n, p0) (one-sided, the direction claimed)."""
    from math import comb
    return sum(comb(n, i) * p0**i * (1-p0)**(n-i) for i in range(k, n+1))


def perm_p(a, b, n=20000, seed=0):
    """Permutation p-value for mean(a) - mean(b) (two-sided)."""
    a, b = np.asarray(a.dropna()), np.asarray(b.dropna())
    if len(a) < 2 or len(b) < 2:
        return np.nan
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= abs(obs):
            cnt += 1
    return cnt / n


def md_table(df: pd.DataFrame, floatfmt=None) -> str:
    df = df.copy()
    if floatfmt:
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt(v))
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i.date() if hasattr(i, "date") else i)] + [str(v) for v in r]) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def fomc_flags(m: pd.DataFrame):
    f = fomc_table()
    idx = m.index
    m["fomc_kind"] = ""; m["fomc_dm1"] = ""; m["fomc_emerg"] = False; m["presser"] = False
    rows = []
    for d, r in f.iterrows():
        p = idx.searchsorted(d)
        if p >= len(idx):
            continue
        d0, dm1, dm4 = idx[p], idx[p-1], idx[p-4]
        m.loc[d0, "fomc_kind"] = str(r.kind); m.loc[d0, "fomc_emerg"] = bool(r.emergency)
        m.loc[dm1, "fomc_dm1"] = str(r.kind)
        # press conferences: every meeting from 2019; before that Mar/Jun/Sep/Dec only
        presser = d.year >= 2019 or d.month in (3, 6, 9, 12)
        m.loc[d0, "presser"] = presser
        rows.append(dict(date=d, kind=str(r.kind), action=r.action, emergency=bool(r.emergency),
                         hike_option=bool(r.hike_option), presser=presser,
                         soxl_dm1=m.loc[dm1, "ret"], soxl_d0=m.loc[d0, "ret"], soxl_d1=m.loc[d0, "fwd1"],
                         soxl_pre3=m.loc[dm1, "close"]/m.loc[dm4, "close"]-1,
                         qqq_pre3=m.loc[dm1, "qqq"]/m.loc[dm4, "qqq"]-1,
                         qqq_d0=m.loc[d0, "qqq_ret"], dm1_date=dm1))
    return pd.DataFrame(rows).set_index("date")


def main():
    m = load_merged()
    m = m[m.index >= "2010-03-12"].copy()
    ft = fomc_flags(m)
    last = m.index[-1].date()
    q = m[m["qqq"].notna()]
    base_up = (m["fwd1"] > 0).mean()

    print(f"# SOXL 历史条件复核报告\n")
    print(f"数据：SOXL 日线 2010-03-12 → {last}（{len(m)} 根）；QQQ 收盘 2018-01-02 → {last}（{len(q)} 根，两源在 2026-05-26 拼接，收益率不跨接缝）。")
    print(f"FOMC：2015–2025 共 {len(ft)} 次（含 2 次 2020 紧急会议），2026 决议未验证、不入统计。\n")
    print(f"全样本次日上涨率基准：{base_up*100:.1f}%（n={m['fwd1'].notna().sum()}）。\n")

    # ------------------------------------------------------------------ S
    S = m[m["gap"] <= GAP_TH].copy()
    print("## 0. 场景样本：跳空低开 ≤ −10%\n")
    print(f"n = {len(S)}。下面条件 1、8、9 都以这个子集为前提（模拟决议日/财报日大幅低开）。\n")
    t = S[["gap", "low_ret", "ret", "recover_from_low", "qqq_ret", "fwd1"]].copy()
    t.columns = ["跳空", "最低/前收", "收盘/前收", "收盘/最低", "QQQ当日", "次日"]
    print(md_table(t, pct)); print()

    # ------------------------------------------------------------------ 1
    print("## 1. 杠杆决定你有没有资格等到结果（结构性）\n")
    lvl = LEVELS["-22%"]
    S1 = S[S["low_ret"] <= lvl].copy()
    S1["fill"] = S1["prev_close"] * (1 + lvl)
    S1["same_day_worst"] = S1["low"] / S1["fill"] - 1
    for h in (5, 20):
        S1[f"worst{h}d"] = np.minimum(S1["same_day_worst"],
                                      (m["low"][::-1].rolling(h, min_periods=1).min()[::-1].shift(-1).reindex(S1.index)) / S1["fill"] - 1)
    S1["next_close"] = m["close"].shift(-1).reindex(S1.index) / S1["fill"] - 1
    print(f"跳空 ≤ −10% 且当日最低已到 −22%（对应 121.82 → 95 入场）：n = {len(S1)}。以 −22% 价成交后：\n")
    t = S1[["gap", "low_ret", "same_day_worst", "worst5d", "worst20d", "next_close"]].copy()
    t.columns = ["跳空", "最低/前收", "成交后当日再跌", "5日内最深", "20日内最深", "次日收盘/成交价"]
    print(md_table(t, pct)); print()
    print(f"成交后当日继续下跌中位数：{pct(S1['same_day_worst'].median())}；20 日内最深中位数：{pct(S1['worst20d'].median())}\n")
    rows = []
    for name, th in LIQ.items():
        rows.append(dict(杠杆=name, 强平跌幅=pct(th),
                         当日穿仓=f"{int((S1['same_day_worst'] <= th).sum())}/{len(S1)}",
                         五日内穿仓=f"{int((S1['worst5d'] <= th).sum())}/{len(S1)}",
                         二十日内穿仓=f"{int((S1['worst20d'] <= th).sum())}/{len(S1)}"))
    print(md_table(pd.DataFrame(rows).set_index("杠杆"))); print()
    print("强平是盘中按最低价触发的，\"当天了结\"救不了盘中被打穿的仓位。\n")

    # ------------------------------------------------------------------ 2
    print("## 2. 高杠杆加仓压不住强平价（算术）\n")
    mm = 0.05  # liquidation when equity falls to 5% of posted margin

    def liq_price(tranches, L):
        """tranches: list of (margin, price). Liquidation P where equity = mm * total margin."""
        M = sum(t[0] for t in tranches)
        n = sum(t[0] * L / t[1] for t in tranches)
        return M * (L + mm - 1) / n

    print(f"假设：强平发生在权益跌到保证金的 {mm*100:.0f}%（10x 对应 −9.5%）。总保证金 1,850，首笔 95。\n")
    rows = []
    for L in (10, 5, 3):
        one = liq_price([(1850, 95)], L)
        two = liq_price([(925, 95), (925, 90)], L)
        four = liq_price([(462.5, 95), (462.5, 92), (462.5, 89), (462.5, 86)], L)
        rows.append({"杠杆": f"{L}x", "一笔 1850@95 强平价": f"{one:.1f}", "两笔 925@95/90 强平价": f"{two:.1f}",
                     "四笔 462.5@95/92/89/86 强平价": f"{four:.1f}",
                     "四笔加满后距最后一笔": pct(four/86-1)})
    print(md_table(pd.DataFrame(rows).set_index("杠杆"))); print()
    print("等额加仓时均价只走价格的一半，但强平价跟着均价走，离现价越来越近。要分批就减股数，不要加杠杆。\n")

    # ------------------------------------------------------------------ 3
    print("## 3. 挂单保证你在最坏的日子成交\n")
    l1, l2 = LEVELS["-15.5%"], LEVELS["-22%"]
    T = m[m["low_ret"] <= l1].copy()
    T["fill1"] = T["prev_close"] * (1 + l1)
    T["beyond"] = T["low"] / T["fill1"] - 1
    T["second"] = T["low_ret"] <= l2
    T["next_close"] = m["close"].shift(-1).reindex(T.index)
    T["win"] = T["next_close"] > T["fill1"]
    at_low = (T["beyond"] > -0.001).sum()
    print(f"全历史挂 −15.5% 限价：触发 {len(T)} 次，成交价恰为当日最低（±0.1%）{at_low} 次；成交后当日继续下跌中位数 {pct(T['beyond'].median())}，最深 {pct(T['beyond'].min())}。\n")
    a, b = T[~T["second"]], T[T["second"]]
    print(f"- 只成交第一单（最低未到 −22%）：n={len(a)}，次日收盘高于成交价 {rate(a['next_close']/a['fill1']-1)}")
    print(f"- 两单都成交（最低 ≤ −22%）：n={len(b)}，次日收盘高于第一单成交价 {rate(b['next_close']/b['fill1']-1)}\n")
    print("第二单能成交本身就是坏消息。\n")

    # ------------------------------------------------------------------ 4
    print("## 4. 美联储不会因会前大跌改主意\n")
    ho = ft[ft["hike_option"] & ~ft["emergency"]]
    sel = ho[ho["qqq_pre3"] <= -0.02]
    print(f"加息周期内（2015-12→2018-12、2022-03→2023-07）的 {len(ho)} 次会议里，会前 3 个交易日 QQQ 跌超 2% 的（QQQ 数据 2018 起）：\n")
    t = sel[["kind", "action", "qqq_pre3", "soxl_pre3", "qqq_d0", "soxl_d0"]].copy()
    t.columns = ["结果", "bp", "QQQ会前3日", "SOXL会前3日", "QQQ决议日", "SOXL决议日"]
    print(md_table(t, pct)); print()
    print(f"{len(sel)} 次全部加息，其中 2022-06-15 会前 QQQ {pct(sel.loc['2022-06-15','qqq_pre3'])} 换来 75bp。会前下跌没有换来任何一次推迟。\n")
    sel2 = ho[ho["soxl_pre3"] <= -0.06]
    t = sel2[["kind", "action", "soxl_pre3", "soxl_d0"]].copy(); t.columns = ["结果", "bp", "SOXL会前3日", "SOXL决议日"]
    print(f"用 SOXL 会前 3 日 ≤ −6% 覆盖 2015–2017：\n"); print(md_table(t, pct)); print()

    # ------------------------------------------------------------------ 5
    print("## 5. 决议前一日（D-1）偏强\n")
    w = m.loc["2015-01-01":].copy()
    sched_dm1 = (w["fomc_dm1"] != "") & ~w["fomc_emerg"].shift(-1, fill_value=False)
    dm1, rest = w.loc[sched_dm1, "ret"], w.loc[~sched_dm1, "ret"]
    print(f"2015 起，仅计划内会议。SOXL D-1 日均 {pct(dm1.mean(), 3)}（n={len(dm1)}，上涨率 {rate(dm1)}） vs 其余 {pct(rest.mean(), 3)}（n={len(rest)}，上涨率 {rate(rest)}）；置换检验 p = {perm_p(dm1, rest):.3f}。\n")
    g = ft[~ft["emergency"]].groupby("kind")["soxl_dm1"]
    t = pd.DataFrame({"n": g.count(), "D-1均值": g.mean().map(pct), "D-1上涨率": g.apply(rate)})
    print(md_table(t)); print()
    hk = ft[(ft["kind"] == "hike")]["soxl_dm1"]
    print(f"加息会议 D-1 上涨 {rate(hk)}；在基准 {base_up*100:.0f}% 下 ≥{int((hk>0).sum())}/{len(hk)} 的概率 p = {binom_p(int((hk>0).sum()), len(hk), base_up):.3f}。\n")
    pr = ft[~ft["emergency"]].groupby("presser")["soxl_dm1"]
    t = pd.DataFrame({"n": pr.count(), "D-1均值": pr.mean().map(pct), "D-1上涨率": pr.apply(rate)}); t.index = t.index.map({True: "有发布会", False: "无发布会"})
    print("按有无发布会（2019 起每次都有；之前只有 3/6/9/12 月）：\n"); print(md_table(t)); print()
    print(f"决议日 D0 本身：均值 {pct(ft[~ft.emergency]['soxl_d0'].mean())}，上涨率 {rate(ft[~ft.emergency]['soxl_d0'])}；D+1：{pct(ft[~ft.emergency]['soxl_d1'].mean())}，{rate(ft[~ft.emergency]['soxl_d1'])}。\n")
    print("CPI/PPI → FOMC 那 15 次的周一统计需要 BLS 发布日历，本环境访问不到 bls.gov，未复核。\n")

    # ------------------------------------------------------------------ 6
    print("## 6. 板块独跌的中期跑输\n")
    alone = q[(q["ret"] <= SECTOR_TH) & (q["qqq_ret"] > QQQ_TH)]
    joint = q[(q["ret"] <= SECTOR_TH) & (q["qqq_ret"] <= QQQ_TH)]
    rows = []
    groups = (("板块独跌 (SOXL≤−8%, QQQ>−2%)", alone), ("同步下跌 (SOXL≤−8%, QQQ≤−2%)", joint),
              ("板块独跌，剔除 2026", alone[alone.index < "2026-01-01"]), ("同步下跌，剔除 2026", joint[joint.index < "2026-01-01"]),
              ("全样本 2018+", q))
    for name, d in groups:
        r = {"组": name, "n": len(d)}
        for h in (1, 3, 10):
            r[f"{h}日均值"] = pct(d[f"fwd{h}"].mean()); r[f"{h}日中位"] = pct(d[f"fwd{h}"].median()); r[f"{h}日上涨率"] = rate(d[f"fwd{h}"])
        rows.append(r)
    print(md_table(pd.DataFrame(rows).set_index("组"))); print()
    print(f"独跌 vs 同步：10 日均值差置换检验 p = {perm_p(alone['fwd10'], joint['fwd10']):.3f}，3 日 p = {perm_p(alone['fwd3'], joint['fwd3']):.3f}（窗口重叠，p 值偏乐观）。")
    n26 = int((alone.index >= "2026-01-01").sum())
    print(f"注意：独跌组 {n26}/{len(alone)} 个样本来自 2026 年，这一年独跌后的 10 日均值 {pct(alone[alone.index >= '2026-01-01']['fwd10'].mean())}，把整体均值拉高了。上涨率和中位数上\"独跌 3 日偏弱\"仍成立，但均值上不成立，也不显著。\n")
    ratio = q["ret"] / q["qqq_ret"]
    dec = q[(q["ret"] < 0) & (q["qqq_ret"] < 0) & (ratio > 7)]
    print(f"脱钩比值 >7（两者同跌、SOXL/QQQ 跌幅比 >7）：n={len(dec)}，3 日 {pct(dec['fwd3'].mean())} / {rate(dec['fwd3'])}，10 日 {pct(dec['fwd10'].mean())} / {rate(dec['fwd10'])}。\n")
    print("机制：板块自身的问题不会被大盘情绪修复。\n")

    # ------------------------------------------------------------------ 7
    print("## 7. SOXL 没有超跌反弹？\n")
    rows = []
    for th in (-0.05, -0.08, -0.10, -0.12, -0.15):
        d = m[m["ret"] <= th]
        k, n = int((d["fwd1"] > 0).sum()), int(d["fwd1"].notna().sum())
        rows.append({"当日跌幅≤": pct(th, 0), "n": n, "次日上涨率": rate(d["fwd1"]), "次日均值": pct(d["fwd1"].mean()),
                     "次日中位": pct(d["fwd1"].median()), "5日均值": pct(d["fwd5"].mean()), "5日上涨率": rate(d["fwd5"]),
                     "p(≥k|基准)": f"{binom_p(k, n, base_up):.2f}"})
    print(md_table(pd.DataFrame(rows).set_index("当日跌幅≤"))); print()
    print(f"基准 {base_up*100:.1f}%。跌幅越大次日上涨率略升，但 −8% 档 p≈0.1、−10% 档 p≈0.09，都不显著，且均值被少数大反弹拉起来（看中位数）。\"跌多了会弹\"在 SOXL 上没有可交易的边际。\n")

    # ------------------------------------------------------------------ 8
    print("## 8. QQQ 是否决指标，不是进场指标\n")
    Sq = S[S["qqq_ret"].notna()].copy()
    Sq["good"] = Sq["recover_from_low"] >= 0.05
    g, b = Sq[Sq["good"]], Sq[~Sq["good"]]
    print(f"跳空 ≤ −10% 且有 QQQ 数据：n={len(Sq)}。按收盘距最低收回 ≥5% 分好/坏日（事后指标，仅用于分类）：\n")
    rows = [{"组": "好日子（收回≥5%）", "n": len(g), "QQQ当日中位": pct(g["qqq_ret"].median()), "QQQ收涨": rate(g["qqq_ret"]), "QQQ≤−3%": int((g["qqq_ret"] <= -0.03).sum())},
            {"组": "坏日子（收回<5%）", "n": len(b), "QQQ当日中位": pct(b["qqq_ret"].median()), "QQQ收涨": rate(b["qqq_ret"]), "QQQ≤−3%": int((b["qqq_ret"] <= -0.03).sum())}]
    print(md_table(pd.DataFrame(rows).set_index("组"))); print()
    big = Sq[Sq["qqq_ret"] <= -0.03]; small = Sq[Sq["qqq_ret"] > -0.03]
    print(f"反过来看：QQQ 当日 ≤ −3% 的 {len(big)} 天里 SOXL 收回 ≥5% 的有 {int(big['good'].sum())} 天；QQQ > −3% 的 {len(small)} 天里有 {int(small['good'].sum())} 天收回、{int((~small['good']).sum())} 天没收回。")
    print("QQQ 大跌 → SOXL 基本不弹；QQQ 撑住 → 不保证弹。且这只在同步下跌时有信息量，板块独跌时 QQQ 本来就没跌。\n")

    # ------------------------------------------------------------------ 9
    print("## 9. 深挂单 EV 才转正\n")
    rows = []
    for lv in (-0.12, -0.155, -0.18, -0.20, -0.22, -0.249, -0.28):
        hit = S[S["low_ret"] <= lv].copy()
        fill = hit["prev_close"] * (1 + lv)
        nxt = m["close"].shift(-1).reindex(hit.index) / fill - 1
        same = hit["close"] / fill - 1
        worst = np.minimum(hit["low"] / fill - 1, m["low"][::-1].rolling(5, min_periods=1).min()[::-1].shift(-1).reindex(hit.index) / fill - 1)
        sd = hit["low"] / fill - 1
        rows.append({"挂单": pct(lv, 1), "触发": f"{len(hit)}/{len(S)} ({len(hit)/len(S)*100:.0f}%)", "当日收盘EV": pct(same.mean()),
                     "次日收盘EV": pct(nxt.mean()), "次日上涨": rate(nxt), "当日再跌中位": pct(sd.median()), "当日再跌最深": pct(sd.min()),
                     "5日内最糟中位": pct(worst.median()), "5日内最糟最深": pct(worst.min())})
    print(md_table(pd.DataFrame(rows).set_index("挂单"))); print()
    print("触发率高的浅档 EV 为负，−22% 附近才转正；更深的档次触发少、成交后当日再被砸的空间也小（−15.5% 最深 −28%，−24.9% 最深 −19%）。但拉长到 5 日，每一档的最糟都是 −50% 上下（2020-03、2025-04），深挂单只减少当日的痛，不减少持仓风险。n 到个位数时看\"最糟\"那列，别看胜率。\n")

    # ------------------------------------------------------------------ dead
    print("## 三、已推翻的说法（回归复核）\n")
    rows = []
    nr = m[m["range"] < 0.04]
    rows.append({"说法": "窄幅磨（日振幅<4%）= 坏", "n": len(nr), "结果": f"次日上涨率 {rate(nr['fwd1'])} vs 基准 {base_up*100:.0f}%", "状态": "死"})
    cs = []
    for nm, d in (("跳空≤−10%", S), ("收跌≤−10%", m[m["ret"] <= -0.10]), ("收跌≤−8%", m[m["ret"] <= -0.08])):
        d = d.copy(); d["after"] = m["close"].shift(-6).reindex(d.index) / m["close"].shift(-1).reindex(d.index) - 1
        cs.append(f"{nm} n={int(d['after'].notna().sum())} r={d[['fwd1', 'after']].corr().iloc[0, 1]:+.2f}")
    rows.append({"说法": "反弹确认再买", "n": "", "结果": "次日反弹幅度 vs 之后 5 日收益相关性：" + "；".join(cs), "状态": "死（反向，越弹越差）"})
    lo = m[(m["ret"] < 0) & ((m["open"] - m["low"]) / m["prev_close"] < 0.005)]
    rows.append({"说法": "下跌日最低在开盘附近 = 好", "n": len(lo), "结果": f"次日上涨率 {rate(lo['fwd1'])} vs 基准 {base_up*100:.0f}%", "状态": "死"})
    c2 = ft[~ft.emergency][["soxl_pre3", "soxl_d0"]].corr().iloc[0, 1]
    rows.append({"说法": "会前涨越多决议日跌越狠", "n": int((~ft.emergency).sum()), "结果": f"会前3日收益 vs 决议日收益相关性 {c2:+.3f}", "状态": "死"})
    up_dm1 = ft[(~ft.emergency) & (ft["soxl_dm1"] > 0)]
    hold = m["close"].shift(-1).reindex(up_dm1.index) / m["close"].reindex(up_dm1["dm1_date"]).values - 1
    rows.append({"说法": "反弹只活一天，别拿过决议", "n": len(up_dm1), "结果": f"D-1 上涨后：决议日当天上涨 {rate(up_dm1['soxl_d0'])}；从 D-1 收盘拿到 D+1 收盘 {rate(hold)}", "状态": "死"})
    rows.append({"说法": "样本里没有加息会议", "n": int((ft.kind == 'hike').sum()), "结果": "2015–2023 共 20 次加息会议，2022-09-21、2023-03-22 均在内", "状态": "错"})
    c3 = S[["dd60", "recover_from_low"]].corr().iloc[0, 1]
    rows.append({"说法": "顶部第一砸比深跌好抄", "n": len(S), "结果": f"跳空日前收距 60 日高 vs 当日收回幅度相关性 {c3:+.3f}（越靠近顶部收回越少）", "状态": "死（反向）"})
    rows.append({"说法": "最相似的 2 天（2016-06-24、2020-02-24）", "n": 2, "结果": "n=2", "状态": "不算证据"})
    print(md_table(pd.DataFrame(rows).set_index("说法"))); print()

    # ------------------------------------------------------------------ now
    print("## 四、把当前位置放进这些条件里\n")
    r = m.iloc[-1]; r2 = m.iloc[-2]
    print(f"最后一根：{last}，SOXL 收 {r['close']:.2f}（{pct(r['ret'])}），QQQ {r['qqq']:.1f}（{pct(r['qqq_ret'])}）。")
    print(f"前一日 {m.index[-2].date()}：SOXL {pct(r2['ret'])}，QQQ {pct(r2['qqq_ret'])} → {'满足' if (r2['ret'] <= SECTOR_TH and r2['qqq_ret'] > QQQ_TH) else '不满足'}\"板块独跌\"定义（条件 6），历史上这组 3 日均值 {pct(alone['fwd3'].mean())}、10 日 {pct(alone['fwd10'].mean())}。")
    print(f"随后一天反弹 {pct(r['ret'])}：条件\"反弹确认\"相关性为负，反弹本身不是买入理由。")
    lo91 = 91.5
    print(f"距 7/29 前低 {lo91}：{pct(lo91/r['close']-1)}；跳空 ≤ −10% 后挂到该位（≈−24.9%）历史触发 {len(S[S['low_ret']<=-0.249])}/{len(S)}。")
    print(f"距 60 日高 {r['hi60']:.2f}：{pct(r['dd60'])}。")
    print(f"2026-09-16 为 FOMC 决议日（未入统计），9/15 为 D-1：条件 5（D-1 偏强）与条件 6（板块独跌后 3 日偏弱）方向相反，两条都不足以单独进场。\n")

    print("## 五、方法论规矩（本次复核沿用）\n")
    print("1. n<50 只能找反例，不能下结论。\n2. 先有机制再看数字。\n3. 指标必须盘中可得：\"收盘距最低收回\"只用于事后分类，进场基准只用前收盘/开盘。\n4. 一次只用一个进场条件。\n")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    OUT.mkdir(exist_ok=True)
    (OUT / "report.md").write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())
