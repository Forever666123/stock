"""Read CME FedWatch meeting CSVs: one file per meeting, one row per date,
one column per target-rate bucket, values are the market-implied probabilities.

    python3 -m analysis.fedwatch

A past meeting's final row collapses onto the bucket the Fed actually chose, so
these files also recover what was decided. A future meeting's final row is the
path the market is pricing right now.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data import ROOT

DIR = ROOT / "data" / "fedwatch"


def bucket_mid(col: str) -> float:
    a, b = re.match(r"\((\d+)-(\d+)\)", col).groups()
    return (int(a) + int(b)) / 2


def load() -> dict[pd.Timestamp, pd.DataFrame]:
    out = {}
    for f in sorted(glob.glob(str(DIR / "FedMeeting_*.csv"))):
        dt = pd.Timestamp(re.search(r"(\d{8})", Path(f).name).group(1))
        d = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()
        d = d.loc[:, (d.fillna(0) > 0).any()]
        out[dt] = d
    return out


def expected_rate(d: pd.DataFrame) -> pd.Series:
    mids = np.array([bucket_mid(c) for c in d.columns])
    return pd.Series(np.nan_to_num(d.values) @ mids, index=d.index)


def main():
    meets = load()
    if not meets:
        raise SystemExit(f"no FedMeeting_*.csv under {DIR}")
    print("# FedWatch 定价路径\n")
    print(f"共 {len(meets)} 场会议，数据起自 {min(d.index.min() for d in meets.values()).date()}。\n")

    print("## 1. 已开过的会：反推实际决议\n")
    print("| 会议 | 最后一行的分布 | 判定 |")
    print("|---|---|---|")
    settled = {}
    today = max(d.index.max() for d in meets.values())
    for dt, d in meets.items():
        last = d.iloc[-1]
        nz = last[last > 0.02].sort_values(ascending=False)
        top = nz.index[0]
        # a meeting whose file runs to its own date has settled
        done = d.index.max() >= dt
        if done:
            settled[dt] = bucket_mid(top)
            print(f"| {dt.date()} | " + "，".join(f"{k} {v*100:.0f}%" for k, v in nz.items()) + f" | 定在 {top} |")
    print()
    if len(settled) >= 2:
        ks = sorted(settled)
        moves = [(b, settled[b] - settled[a]) for a, b in zip(ks, ks[1:])]
        print("已结束会议之间的利率变化：" + "，".join(
            f"{b.date()} {'+' if m > 0 else ''}{m:.0f}bp" if m else f"{b.date()} 按兵不动" for b, m in moves))
        print(f"最近一次已知利率水平：{settled[ks[-1]]:.0f}bp 中值（{ks[-1].date()}）\n")

    print("## 2. 还没开的会：市场定价的路径\n")
    base = max(settled.values()) if settled else np.nan
    print("| 会议 | 当前分布 | 隐含期望利率 | 相对现行 |")
    print("|---|---|---|---|")
    for dt in sorted(meets):
        if dt in settled:
            continue
        d = meets[dt]
        last = d.iloc[-1]
        nz = last[last > 0.02].sort_values(ascending=False)
        er = expected_rate(d).iloc[-1]
        print(f"| {dt.date()} | " + "，".join(f"{k} {v*100:.0f}%" for k, v in nz.items())
              + f" | {er:.0f}bp | {er-base:+.0f}bp |")
    print()

    print("## 3. 定价是什么时候变的\n")
    for dt in sorted(meets):
        if dt in settled:
            continue
        er = expected_rate(meets[dt])
        marks = [er.index[-1] - pd.Timedelta(days=k) for k in (180, 90, 60, 30, 14, 7, 1)]
        vals = [(m, er.asof(m)) for m in marks if m >= er.index[0]]
        print(f"**{dt.date()} 会议的隐含期望利率**")
        print("| " + " | ".join(str(m.date()) for m, _ in vals) + f" | {er.index[-1].date()} |")
        print("|" + "---|" * (len(vals) + 1))
        print("| " + " | ".join(f"{v:.0f}bp" for _, v in vals) + f" | {er.iloc[-1]:.0f}bp |")
        print()


if __name__ == "__main__":
    main()
