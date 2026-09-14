"""Load and stitch the raw price data.

SOXL: daily OHLC 2010-03-11 -> last trading day (Yahoo, split/dividend adjusted).
QQQ : close 2018-01-02 -> 2026-05-22 (parquet) + last 252 days (holdings JSON).
      The two QQQ sources differ by a constant ~0.1% (dividend treatment), so the
      stitched series is only ever used for *within-source* return calculations:
      returns are computed per source and then concatenated, never across the seam.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_soxl() -> pd.DataFrame:
    df = pd.read_csv(DATA / "SOXL_OHLC.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    df = df.rename(columns=str.lower)
    df["prev_close"] = df["close"].shift(1)
    df["ret"] = df["close"] / df["prev_close"] - 1            # close-to-close
    df["gap"] = df["open"] / df["prev_close"] - 1             # open vs prior close
    df["low_ret"] = df["low"] / df["prev_close"] - 1          # intraday low vs prior close
    df["high_ret"] = df["high"] / df["prev_close"] - 1
    df["recover_from_low"] = df["close"] / df["low"] - 1      # ex-post, only for classification
    df["range"] = (df["high"] - df["low"]) / df["prev_close"]
    df["low_at_open"] = (df["open"] - df["low"]).abs() / df["prev_close"] < 0.002
    for h in (1, 3, 5, 10, 20):
        df[f"fwd{h}"] = df["close"].shift(-h) / df["close"] - 1
        df[f"fwd_min{h}"] = df["low"][::-1].rolling(h, min_periods=1).min()[::-1].shift(-1) / df["close"] - 1
    df["hi60"] = df["close"].rolling(60).max()
    df["dd60"] = df["close"] / df["hi60"] - 1
    return df


def load_qqq() -> pd.DataFrame:
    """Stitched QQQ close with per-source daily returns (no cross-seam returns)."""
    a = pd.read_parquet(DATA / "phase5_close_qqq.parquet")
    a.index = pd.to_datetime(a.index)
    a = a.rename(columns={"QQQ": "qqq"})
    a["src"] = "parquet"

    j = json.load(open(DATA / "holdings_prices_1y.json"))
    q = j["prices"]["QQQ"]
    b = pd.DataFrame({"qqq": q["prices"]}, index=pd.to_datetime(q["dates"]))
    b["src"] = "json"

    # Sanity: overlap should be a near-constant ratio (dividend treatment).
    ov = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    ratio = (ov["qqq_b"] / ov["qqq_a"])
    assert ratio.std() < 0.002, f"QQQ sources diverge on overlap: std={ratio.std():.4f}"

    a["qqq_ret"] = a["qqq"].pct_change()
    b["qqq_ret"] = b["qqq"].pct_change()
    for h in (1, 3, 5, 10, 20):
        a[f"qqq_fwd{h}"] = a["qqq"].shift(-h) / a["qqq"] - 1
        b[f"qqq_fwd{h}"] = b["qqq"].shift(-h) / b["qqq"] - 1
    # Use parquet up to its end, json after. Drop json rows inside the overlap.
    out = pd.concat([a, b[b.index > a.index.max()]]).sort_index()
    # First json-row return would straddle the seam: recompute it from json only.
    first_json = b.index[b.index > a.index.max()][0]
    out.loc[first_json, "qqq_ret"] = b.loc[first_json, "qqq_ret"]
    # Forward returns near the seam are fine: they are computed inside each source,
    # but the last 20 parquet rows lack forward values (NaN) rather than crossing.
    return out


def load_merged() -> pd.DataFrame:
    s = load_soxl()
    q = load_qqq()
    m = s.join(q, how="left")
    m["seam"] = m["src"].ne(m["src"].shift(1)) & m["src"].notna() & m["src"].shift(1).notna()
    return m


if __name__ == "__main__":
    m = load_merged()
    print(m.tail(3).T)
    print("SOXL rows", len(m), "QQQ rows", m["qqq"].notna().sum())
    ov = load_qqq()
    print("seam at", ov.index[ov["src"].ne(ov["src"].shift(1))][1:])
