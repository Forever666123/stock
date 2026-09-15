"""Load the 15-minute SOXL/QQQ bars and answer the timing questions daily bars cannot.

    python3 -m analysis.intraday

The file is a yfinance two-level export (Ticker / Price). Times are US Eastern
and cover the regular session only, so bar 0 is 09:30 and the last is 15:45.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import ROOT

PATH = ROOT / "data" / "intraday" / "SOXL_QQQ_15m.csv"


def load() -> pd.DataFrame:
    d = pd.read_csv(PATH, header=[0, 1], index_col=0, parse_dates=True)
    d.index = pd.to_datetime(d.index, utc=True).tz_convert("America/New_York")
    d.index.name = "ts"
    out = pd.DataFrame(index=d.index)
    for tk in ("SOXL", "QQQ"):
        for px in ("Open", "High", "Low", "Close", "Volume"):
            out[f"{tk.lower()}_{px.lower()}"] = d[(tk, px)]
    out["date"] = out.index.date
    out["bar"] = out.groupby("date").cumcount()          # 0 = 09:30
    out["time"] = out.index.strftime("%H:%M")
    return out.dropna(subset=["soxl_close"])


def sessions(d: pd.DataFrame) -> pd.DataFrame:
    """One row per session, with the bar index of the low and of the high."""
    g = d.groupby("date")
    s = pd.DataFrame({
        "open": g["soxl_open"].first(),
        "high": g["soxl_high"].max(),
        "low": g["soxl_low"].min(),
        "close": g["soxl_close"].last(),
        "bars": g.size(),
        "qqq_open": g["qqq_open"].first(),
        "qqq_close": g["qqq_close"].last(),
    })
    s["low_bar"] = g.apply(lambda x: int(x["soxl_low"].values.argmin()), include_groups=False)
    s["high_bar"] = g.apply(lambda x: int(x["soxl_high"].values.argmax()), include_groups=False)
    s["low_time"] = g.apply(lambda x: x["time"].values[int(x["soxl_low"].values.argmin())], include_groups=False)
    s["high_time"] = g.apply(lambda x: x["time"].values[int(x["soxl_high"].values.argmax())], include_groups=False)
    s["ret_oc"] = s["close"] / s["open"] - 1
    s["low_o"] = s["low"] / s["open"] - 1
    s["high_o"] = s["high"] / s["open"] - 1
    s["rng"] = (s["high"] - s["low"]) / s["open"]
    s["qqq_oc"] = s["qqq_close"] / s["qqq_open"] - 1
    s["prev_close"] = s["close"].shift(1)
    s["gap"] = s["open"] / s["prev_close"] - 1
    s["ret_cc"] = s["close"] / s["prev_close"] - 1
    return s


def main():
    d = load()
    s = sessions(d)
    full = s[s["bars"] >= 24]     # drop half days and today's partial session
    print(f"# 15 分钟数据：{s.index.min()} → {s.index.max()}，{len(s)} 个交易日，{len(d)} 根 K 线")
    print(f"完整交易日 {len(full)} 天（≥24 根）\n")

    print("## 1. 当日最低点出现在第几根\n")
    bins = [(0, 1, "09:30–09:45 开盘两根"), (2, 3, "10:00–10:15"), (4, 7, "10:30–11:15"),
            (8, 15, "11:30–13:15"), (16, 21, "13:30–15:00"), (22, 26, "15:15–收盘")]
    print("| 时段 | 天数 | 占比 | 该组当日收盘距最低 中位 |")
    print("|---|---|---|---|")
    for a, b, lab in bins:
        sub = full[(full["low_bar"] >= a) & (full["low_bar"] <= b)]
        rec = (sub["close"] / sub["low"] - 1).median() * 100 if len(sub) else np.nan
        print(f"| {lab} | {len(sub)} | {len(sub)/len(full)*100:.0f}% | {rec:+.1f}% |")
    print(f"\n中位低点在第 {int(full['low_bar'].median())} 根（{full['low_time'].mode().iloc[0] if len(full) else ''} 附近），"
          f"最高点中位在第 {int(full['high_bar'].median())} 根\n")

    big = full[full["ret_cc"] <= -0.06]
    if len(big):
        print(f"其中「当日跌 ≥6%」的 {len(big)} 天：")
        for a, b, lab in bins:
            sub = big[(big["low_bar"] >= a) & (big["low_bar"] <= b)]
            if len(sub):
                print(f"  {lab}: {len(sub)}/{len(big)} —— " + "、".join(str(x) for x in sub.index))
        print()

    print("## 2. 开盘 30 分钟能不能预测当日低点\n")
    f = full.copy()
    first2 = d[d["bar"] <= 1].groupby("date").agg(lo=("soxl_low", "min"), hi=("soxl_high", "max"),
                                                  cl=("soxl_close", "last"))
    f = f.join(first2)
    f["f2_low"] = f["lo"] / f["open"] - 1          # 前两根的最低，相对开盘
    f["f2_ret"] = f["cl"] / f["open"] - 1          # 前两根走完的涨跌
    f["rest_low"] = np.where(f["low_bar"] >= 2, f["low"] / f["open"] - 1, np.nan)
    later = f[f["low_bar"] >= 2]
    print(f"低点出现在第 3 根之后的有 {len(later)}/{len(full)} 天（{len(later)/len(full)*100:.0f}%）。")
    print(f"前两根跌幅 vs 全日低点，相关性 {f['f2_low'].corr(f['low_o']):+.3f}；"
          f"前两根收益 vs 全日低点 {f['f2_ret'].corr(f['low_o']):+.3f}")
    print(f"前两根跌幅 vs 当日振幅 {f['f2_low'].abs().corr(f['rng']):+.3f}\n")

    print("## 3. 2026-07-29 议息日的分钟走势（明天的对照）\n")
    fomc = d[d["date"] == pd.Timestamp("2026-07-29").date()]
    if len(fomc):
        print("| 时间 | 开 | 高 | 低 | 收 | 成交量 |")
        print("|---|---|---|---|---|---|")
        for ts, r in fomc.iterrows():
            mark = " ← 决议" if r["time"] == "14:00" else (" ← 发布会" if r["time"] == "14:30" else "")
            print(f"| {r['time']}{mark} | {r['soxl_open']:.2f} | {r['soxl_high']:.2f} | {r['soxl_low']:.2f} "
                  f"| {r['soxl_close']:.2f} | {r['soxl_volume']/1e6:.1f}M |")
    print()


if __name__ == "__main__":
    main()
