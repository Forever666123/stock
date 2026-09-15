"""Before vs after the FOMC decision: D-2 / D-1 / D0 / D+1 entries, held a fixed
number of trading days FROM EACH ENTRY.

    python3 -m analysis.bt_entry_timing

Mechanism stated first, so the numbers can only confirm or refuse it
-------------------------------------------------------------------
1. An FOMC statement is a scheduled release of information. Option markets price
   a wider distribution into the decision day for that reason. Anyone holding
   into the close of D0 is paid (or charged) that extra dispersion; anyone who
   buys at the D0 close has skipped it. So the PRIOR is: D-1 and D0 should have
   similar *central* outcomes (the drift of a 3x semi ETF over 5-20 days is
   dominated by everything except one afternoon), and D-1 should have the wider
   *spread*. Waiting is not "gambling on the news" -- holding into it is.
2. If that prior is right, the honest answer to "which is better" is a variance
   statement, not a mean statement, and any mean difference on n<=90 meetings
   is noise unless it is very large.
3. Everything used to *classify* a meeting here is observable before the entry
   it gates: the 250-day drawdown measured at the D-2 close, and the D-2 return
   itself. Nothing is computed from a close later than the entry close.

Multiple testing: 12 permutation tests are run on the D-1 vs D0 difference
(4 cuts x 3 horizons). Bonferroni at 0.05 is 0.00417; Holm is reported too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import load_merged
from analysis.fomc import fomc_table

RNG = np.random.default_rng(20260915)
NPERM = 20000
HORIZONS = (5, 10, 20)
ENTRIES = {"D-2": -2, "D-1": -1, "D0": 0, "D+1": +1}

# The most recent bar is not in the CSV.
LAST_BAR = dict(date="2026-09-14", open=101.53, high=105.39, low=99.87,
                close=101.13, qqq=711.96)


def frame() -> pd.DataFrame:
    m = load_merged()
    d = pd.Timestamp(LAST_BAR["date"])
    assert d not in m.index, "2026-09-14 already present -- do not double-append"
    m.loc[d, ["open", "high", "low", "close", "qqq"]] = [
        LAST_BAR["open"], LAST_BAR["high"], LAST_BAR["low"],
        LAST_BAR["close"], LAST_BAR["qqq"]]
    m = m.sort_index()
    m["ret"] = m["close"] / m["close"].shift(1) - 1
    return m


def meetings(m: pd.DataFrame) -> pd.DataFrame:
    """Scheduled FOMC statement days with a known action, mapped to a bar index."""
    t = fomc_table(include_2026=True)
    t = t[~t["emergency"] & t["action"].notna()]
    pos = pd.Series(np.arange(len(m)), index=m.index)
    rows = []
    for dt, r in t.iterrows():
        if dt not in pos.index:          # statement day must be a trading day we have
            continue
        rows.append(dict(date=dt, i=int(pos[dt]), action=float(r["action"]),
                         kind=str(r["kind"])))
    return pd.DataFrame(rows).set_index("date")


def trades(m: pd.DataFrame, mt: pd.DataFrame, hold: int) -> pd.DataFrame:
    """One row per meeting; columns are the 4 entries. Common sample: a meeting is
    kept only if ALL FOUR entries have a full `hold`-day window, so no entry is
    advantaged by a different set of meetings."""
    c = m["close"].values
    lo = m["low"].values
    n = len(c)
    out = {}
    for name, off in ENTRIES.items():
        ret, mae, ok = [], [], []
        for i in mt["i"].values:
            j = i + off
            e = j + hold
            good = (j - 2 >= 0) and (e < n) and (i + 1 + hold < n)
            ok.append(good)
            if not good:
                ret.append(np.nan); mae.append(np.nan); continue
            ret.append(c[e] / c[j] - 1)
            mae.append(lo[j + 1:e + 1].min() / c[j] - 1)
        out[f"{name}_ret"] = ret
        out[f"{name}_mae"] = mae
        out[f"{name}_ok"] = ok
    df = pd.DataFrame(out, index=mt.index)
    keep = np.all([df[f"{k}_ok"] for k in ENTRIES], axis=0)
    return df[keep].join(mt[keep])


def desc(x: np.ndarray, mae: np.ndarray) -> dict:
    x = np.asarray(x, float); mae = np.asarray(mae, float)
    return dict(n=len(x), mean=x.mean(), median=np.median(x),
                hit=(x > 0).mean(), sd=x.std(ddof=1),
                p5=np.percentile(x, 5), p95=np.percentile(x, 95),
                mae_mean=mae.mean(), mae_med=np.median(mae), mae_worst=mae.min())


def paired_perm(a: np.ndarray, b: np.ndarray, nperm=NPERM) -> tuple:
    """Sign-flip permutation on the paired difference a-b (same meetings)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    obs = d.mean()
    s = RNG.choice([-1.0, 1.0], size=(nperm, d.size))
    null = (s * d).mean(axis=1)
    p = (1 + (np.abs(null) >= abs(obs)).sum()) / (nperm + 1)
    return obs, np.median(d), p, d.size


def holm(ps: list) -> dict:
    order = np.argsort([p for _, p in ps])
    k = len(ps)
    adj, prev = {}, 0.0
    for rank, idx in enumerate(order):
        name, p = ps[idx]
        v = min(1.0, max(prev, (k - rank) * p))
        adj[name] = v
        prev = v
    return adj


def table(title: str, tr: pd.DataFrame, hold: int):
    print(f"\n  {title}   hold={hold}d   n={len(tr)}")
    print(f"    {'entry':<6}{'mean':>9}{'median':>9}{'hit':>8}{'sd':>9}"
          f"{'p5':>9}{'p95':>9}{'MAEmean':>10}{'MAEmed':>9}{'MAEworst':>10}")
    for name in ENTRIES:
        d = desc(tr[f"{name}_ret"].values, tr[f"{name}_mae"].values)
        print(f"    {name:<6}{d['mean']:>8.2%}{d['median']:>9.2%}{d['hit']:>8.0%}"
              f"{d['sd']:>9.2%}{d['p5']:>9.2%}{d['p95']:>9.2%}"
              f"{d['mae_mean']:>10.2%}{d['mae_med']:>9.2%}{d['mae_worst']:>10.2%}")


def main():
    m = frame()
    mt = meetings(m)
    print("=" * 104)
    print("ENTRY TIMING AROUND THE FOMC DECISION -- SOXL, buy at the close, hold N days from that close")
    print("=" * 104)
    print(f"bars {m.index[0].date()}..{m.index[-1].date()} (2026-09-14 appended manually)")
    print(f"scheduled meetings with a known action: {len(mt)}  "
          f"({mt['kind'].value_counts().to_dict()})")

    # ---- observable classifiers, measured at the D-2 close -------------------
    c = m["close"].values
    hi250 = pd.Series(c).rolling(250, min_periods=60).max().values
    dd250 = c / hi250 - 1
    ret = m["ret"].values
    mt["dd250_D2"] = dd250[mt["i"].values - 2]
    mt["ret_D2"] = ret[mt["i"].values - 2]
    mt["ret_D0"] = ret[mt["i"].values]
    mt["sd20_preD0"] = pd.Series(ret).rolling(20).std().values[mt["i"].values - 1]

    cuts = {
        "ALL scheduled": np.ones(len(mt), bool),
        "HIKE meetings": (mt["action"] > 0).values,
        "dd250 at D-2 < -40%": (mt["dd250_D2"] <= -0.40).values,
        "D-2 fell >= 6%": (mt["ret_D2"] <= -0.06).values,
    }
    print("\ncut sizes:", {k: int(v.sum()) for k, v in cuts.items()})
    for k, v in cuts.items():
        if k == "ALL scheduled":
            continue
        print(f"  {k}: {[str(d.date()) for d in mt.index[v]]}")

    # ---- where today sits ----------------------------------------------------
    print(f"\ntoday's analogue (2026-09-16 meeting): D-2 = 2026-09-14, "
          f"ret_D2 = {ret[-1]:+.2%}, dd250 at D-2 = {dd250[-1]:+.1%}")
    print(f"  -> qualifies for 'D-2 fell >= 6%': {ret[-1] <= -0.06}; "
          f"'dd250 < -40%': {dd250[-1] <= -0.40}; a hike is 92.3% priced.")

    # ---- main tables ---------------------------------------------------------
    pvals, store = [], {}
    for hold in HORIZONS:
        tr_all = trades(m, mt, hold)
        print("\n" + "-" * 104)
        for cname, mask in cuts.items():
            sub = tr_all[mask[np.isin(mt.index, tr_all.index)]] if len(tr_all) != len(mt) \
                else tr_all[mask]
            if len(sub) == 0:
                print(f"\n  {cname}  hold={hold}d  n=0 -- skipped"); continue
            table(cname, sub, hold)
            obs, med, p, nn = paired_perm(sub["D-1_ret"].values, sub["D0_ret"].values)
            key = f"{cname} @{hold}d"
            pvals.append((key, p))
            store[key] = (obs, med, p, nn)
            flag = "  [n<50: COUNTER-EXAMPLE ONLY, NOT A CONCLUSION]" if nn < 50 else ""
            print(f"    paired D-1 minus D0: mean {obs:+.2%}  median {med:+.2%}  "
                  f"perm p={p:.3f}  n={nn}{flag}")

    # ---- multiple-testing ----------------------------------------------------
    adj = holm(pvals)
    print("\n" + "=" * 104)
    print("MULTIPLE TESTING on the 12 D-1 vs D0 tests (Bonferroni 0.05/12 = 0.00417)")
    print(f"  {'test':<34}{'raw p':>9}{'Holm p':>10}{'survives Holm 0.05':>22}")
    for k, p in pvals:
        print(f"  {k:<34}{p:>9.3f}{adj[k]:>10.3f}{str(adj[k] < 0.05):>22}")

    # ---- the thing actually feared: decision-day dispersion -------------------
    print("\n" + "=" * 104)
    print("DISPERSION OF THE DECISION-DAY MOVE (D0 close-to-close return)")
    fomc_idx = set(mt["i"].values)
    near = set()
    for i in fomc_idx:
        near.update({i - 1, i, i + 1})
    era = m.index >= "2015-01-01"
    allpos = np.arange(len(m))
    rand_mask = era & ~np.isin(allpos, list(near)) & ~np.isnan(ret)
    rand = ret[rand_mask]

    groups = {
        "FOMC D0 -- hikes": mt.loc[mt["action"] > 0, "ret_D0"].values,
        "FOMC D0 -- holds": mt.loc[mt["action"] == 0, "ret_D0"].values,
        "FOMC D0 -- cuts": mt.loc[mt["action"] < 0, "ret_D0"].values,
        "FOMC D0 -- all": mt["ret_D0"].values,
        "non-FOMC day (2015+, excl D-1/D0/D+1)": rand,
    }
    print(f"  {'group':<40}{'n':>5}{'sd':>9}{'mean':>9}{'median':>9}{'p5':>9}{'p95':>9}{'|ret| med':>11}")
    for k, v in groups.items():
        v = np.asarray(v, float); v = v[~np.isnan(v)]
        print(f"  {k:<40}{len(v):>5}{v.std(ddof=1):>9.2%}{v.mean():>9.2%}"
              f"{np.median(v):>9.2%}{np.percentile(v,5):>9.2%}"
              f"{np.percentile(v,95):>9.2%}{np.median(np.abs(v)):>11.2%}")

    base = rand.std(ddof=1)
    for k in ("FOMC D0 -- hikes", "FOMC D0 -- holds", "FOMC D0 -- all"):
        v = np.asarray(groups[k], float); v = v[~np.isnan(v)]
        print(f"  width ratio {k} / non-FOMC day: sd {v.std(ddof=1)/base:.2f}x, "
              f"5-95 span {(np.percentile(v,95)-np.percentile(v,5))/(np.percentile(rand,95)-np.percentile(rand,5)):.2f}x")

    # variance permutation: is D0 really wider, or is FOMC in high-vol regimes?
    def var_perm(a, b, nperm=NPERM):
        a = np.asarray(a, float); a = a[~np.isnan(a)]
        b = np.asarray(b, float); b = b[~np.isnan(b)]
        obs = a.std(ddof=1) / b.std(ddof=1)
        pool = np.r_[a, b]; na = a.size
        cnt = 0
        for _ in range(nperm):
            p = RNG.permutation(pool)
            r = p[:na].std(ddof=1) / p[na:].std(ddof=1)
            cnt += (r >= obs) or (1 / r >= obs)
        return obs, (1 + cnt) / (nperm + 1)

    r, p = var_perm(groups["FOMC D0 -- all"], rand, 4000)
    print(f"\n  raw sd ratio all-D0 / non-FOMC = {r:.2f}x, two-sided perm p={p:.3f}  (n_D0={len(mt)})")

    # regime-controlled: standardise every day by its own trailing 20d vol
    sd20 = pd.Series(ret).rolling(20).std().shift(1).values
    z = ret / sd20
    z_d0 = z[mt["i"].values]
    z_rand = z[rand_mask & ~np.isnan(z)]
    z_d0 = z_d0[~np.isnan(z_d0)]
    rz, pz = var_perm(z_d0, z_rand, 4000)
    print(f"  VOL-CONTROLLED (each day / its own trailing-20d sd, sd measured through the")
    print(f"  prior close so it is observable before the decision):")
    print(f"    sd(z) FOMC D0 = {z_d0.std(ddof=1):.3f} vs non-FOMC {z_rand.std(ddof=1):.3f}"
          f"  ratio {rz:.2f}x  perm p={pz:.3f}")

    # ---- dispersion of the actual TRADE, D-1 vs D0 ---------------------------
    print("\n" + "=" * 104)
    print("DISPERSION OF THE HELD TRADE (what the extra day of exposure costs in width)")
    print(f"  {'hold':<6}{'cut':<22}{'sd D-1':>9}{'sd D0':>9}{'ratio':>8}"
          f"{'5-95 D-1':>11}{'5-95 D0':>11}{'wider by':>10}")
    for hold in HORIZONS:
        tr_all = trades(m, mt, hold)
        for cname, mask in (("ALL scheduled", cuts["ALL scheduled"]),
                            ("HIKE meetings", cuts["HIKE meetings"])):
            sub = tr_all[mask[np.isin(mt.index, tr_all.index)]] if len(tr_all) != len(mt) \
                else tr_all[mask]
            a = sub["D-1_ret"].values; b = sub["D0_ret"].values
            sa, sb = a.std(ddof=1), b.std(ddof=1)
            wa = np.percentile(a, 95) - np.percentile(a, 5)
            wb = np.percentile(b, 95) - np.percentile(b, 5)
            print(f"  {hold:<6}{cname:<22}{sa:>9.2%}{sb:>9.2%}{sa/sb:>8.2f}"
                  f"{wa:>11.1%}{wb:>11.1%}{wa/wb:>10.2f}x")

    # ---- is ANY of this different from a random day? -------------------------
    print("\n" + "=" * 104)
    print("UNCONDITIONAL BASELINE (2015+, any day, buy the close, same horizons)")
    print("  If the FOMC entries are indistinguishable from this, the whole question")
    print("  of *when* around the meeting is second-order to *whether* you buy at all.")
    c = m["close"].values; lo = m["low"].values; n = len(c)
    start = int(np.argmax(m.index >= "2015-01-01"))
    print(f"  {'hold':<6}{'n':>6}{'mean':>9}{'median':>9}{'hit':>7}{'sd':>9}"
          f"{'p5':>9}{'p95':>9}{'MAEmed':>9}")
    for hold in HORIZONS:
        idx = np.arange(start, n - hold)
        r = c[idx + hold] / c[idx] - 1
        mae = np.array([lo[i + 1:i + hold + 1].min() / c[i] - 1 for i in idx])
        print(f"  {hold:<6}{len(r):>6}{r.mean():>9.2%}{np.median(r):>9.2%}"
              f"{(r>0).mean():>7.0%}{r.std(ddof=1):>9.2%}{np.percentile(r,5):>9.2%}"
              f"{np.percentile(r,95):>9.2%}{np.median(mae):>9.2%}")

    print("\n" + "=" * 104)
    print("NOTE: every cut other than 'ALL scheduled' has n well under 50. Those rows are")
    print("counter-examples and descriptions of what happened, never conclusions.")


if __name__ == "__main__":
    main()
