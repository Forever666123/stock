"""What the curve prices for the whole hiking cycle, not just the next meeting.

FedWatch's aggregated view gives the probability the target range IS a given
level at each future meeting, so it reads off the cumulative path. Fed Funds
futures give the same path as a continuous rate. Both are snapshots; re-export
them from FedWatch and drop the files in data/fedwatch/ to refresh.

    python3 -m analysis.rate_path
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

from analysis.data import ROOT

DIR = ROOT / "data" / "fedwatch"
CURRENT_MID_BP = 362.5   # 350-375, confirmed by the settled meeting files


def latest(pattern: str) -> Path:
    files = sorted(glob.glob(str(DIR / pattern)))
    if not files:
        raise SystemExit(f"no {pattern} under {DIR}")
    return Path(files[-1])


def main():
    agg_file = latest("aggregated_*.csv")
    a = pd.read_csv(agg_file, comment="#")
    snap = re.search(r"(\d{8})", agg_file.name).group(1)
    mids = {c: (int(c.split("-")[0]) + int(c.split("-")[1])) / 2 for c in a.columns[1:]}
    a["er"] = sum(a[c] * m for c, m in mids.items())

    print(f"# 市场给整条加息路径的定价（快照 {snap[:4]}-{snap[4:6]}-{snap[6:]}）\n")
    print(f"基准：现行目标区间 350-375，中值 {CURRENT_MID_BP:.1f}bp。\n")
    print("| 会议 | 主导区间 | 概率 | 期望利率 | 累计加息 | 折合次数 |")
    print("|---|---|---|---|---|---|")
    for _, r in a.iterrows():
        probs = {c: r[c] for c in mids}
        top = max(probs, key=probs.get)
        cum = r["er"] - CURRENT_MID_BP
        print(f"| {r.meeting} | {top} | {probs[top]*100:.0f}% | {r['er']:.0f}bp | +{cum:.0f}bp | {cum/25:.1f} |")
    peak = a["er"].max()
    peak_row = a.loc[a["er"].idxmax(), "meeting"]
    print(f"\n终端定价 **{peak:.0f}bp（{peak/100:.2f}%）**，出现在 {peak_row}，"
          f"相对现行 **+{peak-CURRENT_MID_BP:.0f}bp，约 {(peak-CURRENT_MID_BP)/25:.1f} 次加息**。\n")

    try:
        f = pd.read_csv(latest("ff_futures_*.csv"), comment="#")
    except SystemExit:
        f = None
    if f is not None:
        f["rate"] = (100 - f["price"]) * 100
        print("## Fed Funds 期货隐含路径（独立口径，用来交叉验证）\n")
        print("| 合约 | 月份 | 价格 | 隐含利率 |")
        print("|---|---|---|---|")
        for _, r in f.iterrows():
            print(f"| {r.contract} | {r.month} | {r.price:.4f} | {r['rate']:.0f}bp |")
        print(f"\n期货终端 {f['rate'].max():.0f}bp，聚合表终端 {peak:.0f}bp，两个口径一致。\n")

    print("## 周三点阵图的判别标准\n")
    print(f"曲线已经把终端定在 {peak/100:.2f}%。点阵图只有偏离这个数才构成信息：\n")
    print("| 点阵图隐含的终端 | 相对定价 | 方向 |")
    print("|---|---|---|")
    print(f"| 高于 4.75% | 比曲线更鹰 | 后端会议重定价，杀 |")
    print(f"| 4.50–4.75% | 与曲线一致 | 没有新信息，利率这条腿结束 |")
    print(f"| 低于 4.50% | 比曲线更鸽 | 宽松 |")
    print()
    print("注意这是对利率资产说的。SOXL 的日收益和加息概率变动相关性只有 −0.09，")
    print("这套框架对 QQQ / SPY 有意义，对半导体的解释力接近零。\n")
    print("**没法回测的地方**：历史 Fed Funds 期货数据本会话拿不到，所以")
    print("\"整条周期已被定价时加息落地会怎样\" 无法用历史检验。上面的历史分组")
    print("（第 1 次加息 D+20 −10.0%）来自那些定价程度未知的会议，不能直接套用。\n")


if __name__ == "__main__":
    main()
