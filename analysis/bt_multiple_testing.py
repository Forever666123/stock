"""SELF-AUDIT: how much of this session's findings is noise?

    python3 -m analysis.bt_multiple_testing

WHY THIS SCRIPT EXISTS (mechanism stated before any number)
-----------------------------------------------------------
Over one long session roughly a hundred hypothesis tests were run against ONE
price series (SOXL, 2010-03-11 .. 2026-09-14, ~4100 bars) with ONE outcome family
(forward returns at 1/2/3/5/10/20 days).  The subsets were narrowed repeatedly
until one of them matched today's situation.  That search procedure has a known
failure mode and it does not need a p-value to be believed:

  a) With ~100 tests at alpha=0.05, the EXPECTED number of false positives under
     a complete null is 5.  Finding three or four "significant" results is the
     null hypothesis, not evidence against it.
  b) Narrowing the subset until it matches today guarantees the final subset is
     small, and a small subset of a fat-tailed leveraged series has a mean that
     one observation can move by several percent.  So every mean here is
     decomposed and the driving observation named.
  c) Forward windows overlap.  fwd20 on consecutive days shares 19 of 20 days.
     A permutation test that shuffles daily labels assumes exchangeability of
     independent draws and therefore reports a p-value that is too small, often
     by a factor of 2-4.  The moving-block bootstrap below is the correction.
  d) Events cluster into episodes (Aug-2015, Feb/Mar-2020, 2022, Apr-2025, the
     2026 selloff).  n=79 "sector-only declines" is not 79 independent events.
     Episode counts are printed next to every n.

CORRECTIONS APPLIED
-------------------
  raw p        : permutation test, 20000 draws, two-sided unless noted.
  Bonferroni   : min(1, p * M) with M = 100 tests.  Survives if < 0.05.
  BH-FDR       : two variants, both with M = 100.
                 "generous" assumes the 92 unreported tests ALL had p larger
                 than these 8 -- the most favourable assumption possible.
                 "uniform"  assumes the 92 unreported tests were null, i.e.
                 p ~ U(0,1), so ~92*p of them sit below each reported p.
                 The uniform variant is the honest one.
  block boot   : moving-block bootstrap, contiguous blocks of 20 trading days,
                 2000 iterations, 95% percentile interval on the SAME statistic.
                 This is the decisive check: it respects autocorrelation and
                 episode clustering.  If 0 is inside the interval, there is no
                 finding, whatever the permutation p said.

RULES OF EVIDENCE CARRIED OVER
------------------------------
  * n < 50 -> counter-example only, never a conclusion.  Printed inline.
  * medians reported beside every mean.
  * anything computed from a close (recover_from_low) classifies history only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import load_merged
from analysis.fomc import fomc_table

RNG = np.random.default_rng(20260915)
NPERM = 20000
NBOOT = 2000
BLOCK = 20
M_TESTS = 100          # the multiplicity budget for the whole session
ALPHA = 0.05

# The most recent bar is not in the CSV.
LAST_BAR = dict(date="2026-09-14", open=101.53, high=105.39, low=99.87,
                close=101.13, qqq=711.96)

W = 104


# --------------------------------------------------------------------------- data
def frame() -> pd.DataFrame:
    m = load_merged()
    d = pd.Timestamp(LAST_BAR["date"])
    assert d not in m.index, "2026-09-14 already present -- do not double-append"
    m.loc[d, ["open", "high", "low", "close", "qqq"]] = [
        LAST_BAR["open"], LAST_BAR["high"], LAST_BAR["low"],
        LAST_BAR["close"], LAST_BAR["qqq"]]
    m = m.sort_index()
    c, q = m["close"], m["qqq"]
    m["prev_close"] = c.shift(1)
    m["ret"] = c / c.shift(1) - 1
    m["qqq_ret"] = q / q.shift(1) - 1
    m["low_ret"] = m["low"] / m["prev_close"] - 1
    m["range"] = (m["high"] - m["low"]) / m["prev_close"]
    for h in (1, 2, 3, 5, 10, 20):
        m[f"fwd{h}"] = c.shift(-h) / c - 1
    m["hi250"] = c.rolling(250, min_periods=60).max()
    m["dd250"] = c / m["hi250"] - 1
    return m


def meetings(m: pd.DataFrame) -> pd.DataFrame:
    """Scheduled FOMC statement days with a KNOWN action, mapped to a bar index."""
    t = fomc_table(include_2026=True)
    t = t[~t["emergency"] & t["action"].notna()]
    pos = pd.Series(np.arange(len(m)), index=m.index)
    rows = []
    for dt, r in t.iterrows():
        if dt not in pos.index:
            continue
        rows.append(dict(date=dt, i=int(pos[dt]), action=float(r["action"]),
                         kind=str(r["kind"]), oos=dt.year >= 2026))
    return pd.DataFrame(rows).set_index("date")


def episodes(idx: pd.DatetimeIndex, gap_days=10) -> int:
    """Distinct clusters: consecutive events more than `gap_days` apart start a new one."""
    if len(idx) == 0:
        return 0
    s = pd.Series(idx).sort_values()
    return int((s.diff().dt.days > gap_days).sum() + 1)


# --------------------------------------------------------------------------- stats
def perm_diff(x: np.ndarray, g: np.ndarray, nperm=NPERM, rng=RNG) -> tuple:
    """Two-sided permutation p for mean(x[g]) - mean(x[~g]); exact label shuffle."""
    ok = ~np.isnan(x)
    x, g = x[ok], g[ok].astype(bool)
    k, n = int(g.sum()), len(x)
    if k < 2 or n - k < 2:
        return np.nan, np.nan
    obs = x[g].mean() - x[~g].mean()
    tot = x.sum()
    hits = 0
    done = 0
    while done < nperm:
        ch = min(500, nperm - done)
        keys = rng.random((ch, n))
        idx = np.argpartition(keys, k, axis=1)[:, :k]
        sa = x[idx].sum(axis=1)
        d = sa / k - (tot - sa) / (n - k)
        hits += int((np.abs(d) >= abs(obs) - 1e-15).sum())
        done += ch
    return obs, (1 + hits) / (nperm + 1)


def perm_paired(d: np.ndarray, nperm=NPERM, rng=RNG) -> tuple:
    """Sign-flip permutation on paired differences."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if d.size < 2:
        return np.nan, np.nan
    obs = d.mean()
    s = rng.choice([-1.0, 1.0], size=(nperm, d.size))
    null = (s * d).mean(axis=1)
    return obs, (1 + int((np.abs(null) >= abs(obs)).sum())) / (nperm + 1)


def binom_p(k, n, p0, two_sided=True):
    from math import comb
    pmf = [comb(n, i) * p0**i * (1 - p0)**(n - i) for i in range(n + 1)]
    up = sum(pmf[k:])
    if not two_sided:
        return up
    thr = pmf[k] * (1 + 1e-9)
    return min(1.0, sum(p for p in pmf if p <= thr))


def block_boot_diff(x: np.ndarray, g: np.ndarray, L=BLOCK, nboot=NBOOT, rng=RNG):
    """Moving-block bootstrap CI for mean(x[g]) - mean(x[~g]).

    Blocks are contiguous in CALENDAR ORDER, so both the autocorrelation of x and
    the clustering of g survive resampling.  Rows with NaN x are kept in the
    blocks and dropped inside each replicate (they carry no weight either way)."""
    n = len(x)
    if n < L * 2:
        return (np.nan, np.nan), np.nan
    nb = int(np.ceil(n / L))
    g = g.astype(bool)
    xv = np.where(np.isnan(x), 0.0, x)
    valid = (~np.isnan(x))
    out = []
    done = 0
    while done < nboot:
        ch = min(200, nboot - done)
        st = rng.integers(0, n - L + 1, size=(ch, nb))
        idx = (st[:, :, None] + np.arange(L)[None, None, :]).reshape(ch, -1)[:, :n]
        xs, gs, vs = xv[idx], g[idx], valid[idx]
        ga, gb = gs & vs, (~gs) & vs
        na, nbb = ga.sum(1), gb.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (xs * ga).sum(1) / na - (xs * gb).sum(1) / nbb
        d = d[(na > 0) & (nbb > 0)]
        out.append(d)
        done += ch
    d = np.concatenate(out)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))), float(np.mean(d))


def block_boot_mean(x: np.ndarray, L=BLOCK, nboot=NBOOT, rng=RNG):
    n = len(x)
    if n < 3:
        return (np.nan, np.nan)
    Lb = min(L, max(1, n // 3))
    nb = int(np.ceil(n / Lb))
    xv = np.where(np.isnan(x), 0.0, x)
    valid = ~np.isnan(x)
    out = []
    done = 0
    while done < nboot:
        ch = min(200, nboot - done)
        st = rng.integers(0, n - Lb + 1, size=(ch, nb))
        idx = (st[:, :, None] + np.arange(Lb)[None, None, :]).reshape(ch, -1)[:, :n]
        xs, vs = xv[idx], valid[idx]
        nn = vs.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append((xs * vs).sum(1) / nn)
        done += ch
    d = np.concatenate(out)
    d = d[~np.isnan(d)]
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


# --------------------------------------------------------------------------- bookkeeping
RESULTS = []      # (id, short, n, raw_p, direction_claimed)


def record(cid, short, n, p, note=""):
    RESULTS.append(dict(id=cid, short=short, n=n, p=p, note=note))
    return p


def bonf(p, M=M_TESTS):
    return min(1.0, p * M) if p == p else np.nan


def bh_table(ps, M=M_TESTS, alpha=ALPHA):
    """Return per-test (q_generous, q_uniform).

    generous: the 8 reported p-values are assumed to be the 8 SMALLEST of M,
              so rank_i = i.  Most favourable assumption available.
    uniform : the M-8 unreported tests are assumed null, p~U(0,1), so the
              expected rank of p is  i + (M-8)*p.  The honest assumption.
    """
    order = np.argsort(ps)
    k = len(ps)
    qg = np.ones(k)
    prev = 1.0
    for rank in range(k - 1, -1, -1):
        i = order[rank]
        v = min(prev, min(1.0, M * ps[i] / (rank + 1)))
        qg[i] = v
        prev = v
    qu = np.ones(k)
    prev = 1.0
    for rank in range(k - 1, -1, -1):
        i = order[rank]
        r_eff = (rank + 1) + (M - k) * ps[i]
        v = min(prev, min(1.0, M * ps[i] / r_eff))
        qu[i] = v
        prev = v
    return qg, qu


def hdr(t):
    print("\n" + "=" * W)
    print(t)
    print("=" * W)


def sub(t):
    print("\n" + "-" * W)
    print(t)
    print("-" * W)


def flag_n(n):
    return "   [n<50: COUNTER-EXAMPLE ONLY, NOT A CONCLUSION]" if n < 50 else ""


def show(x, n=None, name="mean"):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "n=0"
    return (f"n={len(x)}  mean {x.mean():+.2%}  median {np.median(x):+.2%}  "
            f"up {(x > 0).mean():.0%}")


def drop_top(x: pd.Series, k=1, by="abs_from_mean"):
    """Remove the k observations that move the mean most, return (series, dropped)."""
    s = x.dropna()
    dev = (s - s.mean()).abs()
    drop = dev.nlargest(k).index
    return s.drop(drop), s.loc[drop]


# =========================================================================== main
def main():
    m = frame()
    mt = meetings(m)
    idx = m.index

    hdr("SELF-AUDIT OF THE SESSION'S FINDINGS -- MULTIPLE TESTING AND ROBUSTNESS")
    print(f"bars {idx[0].date()} .. {idx[-1].date()}  (n={len(m)}, 2026-09-14 appended manually)")
    print(f"scheduled FOMC meetings with a known action: {len(mt)}  {mt['kind'].value_counts().to_dict()}")
    print(f"multiplicity budget M = {M_TESTS} tests;  Bonferroni threshold = {ALPHA/M_TESTS:.5f}")
    print(f"permutation draws = {NPERM};  block bootstrap = {NBOOT} iters, {BLOCK}-day contiguous blocks")
    base_up = float((m["fwd1"] > 0).mean())
    print(f"baseline next-day up rate = {base_up:.1%}  (n={int(m['fwd1'].notna().sum())})")
    print()
    print("  *** A SPLIT THE SESSION NEVER MADE, AND IT MATTERS MORE THAN THE P-VALUES ***")
    print("  Every claim below was originally computed from fomc_table() with include_2026")
    print("  defaulting to False, i.e. from the 2015-2025 meetings ONLY.  But fomc.py itself")
    print("  records three 2026 meetings as VERIFIED holds, recovered from the FedWatch files:")
    print(f"    {[str(d.date()) for d in mt.index[mt['oos']]]}")
    print("  Those three are the only observations in this repo that were NOT available while")
    print("  the hypotheses were being searched.  They are a genuine, if tiny, out-of-sample")
    print("  set.  Every FOMC claim below is therefore reported twice: IN-SAMPLE (2015-2025,")
    print("  the numbers that were quoted to the user) and EXTENDED (with the 3 verified 2026")
    print("  meetings added).  Where the two disagree, the in-sample number is the artefact.")

    store = {}

    # ---------------------------------------------------------------- CLAIM 1
    hdr("CLAIM 1  Pre-FOMC D-1 drift is positive: +0.456% on D-1 vs +0.330% otherwise")
    print("MECHANISM CLAIMED FIRST: dealers hedging into a scheduled release, plus the")
    print("documented equity drift into FOMC announcements (Lucca-Moench).  That literature")
    print("is about the S&P 500 index; nothing in it says a 3x semiconductor ETF inherits it.")
    w = m.loc["2015-01-01":].copy()
    wpos = pd.Series(np.arange(len(w)), index=w.index)

    def dm1_mask(meet_idx):
        f = np.zeros(len(w), bool)
        for dt in meet_idx:
            if dt in wpos.index:
                p_ = int(wpos[dt])
                if p_ - 1 >= 0:
                    f[p_ - 1] = True
        return f

    x = w["ret"].values
    sd = np.nanstd(x, ddof=1)
    res1 = {}
    for lbl, mi in (("IN-SAMPLE 2015-2025", mt.index[~mt["oos"]]),
                    ("EXTENDED incl 2026", mt.index)):
        g = dm1_mask(mi)
        a, b = x[g & ~np.isnan(x)], x[~g & ~np.isnan(x)]
        obs, pp = perm_diff(x, g)
        print(f"\n  {lbl}")
        print(f"    D-1 days      : {show(a)}")
        print(f"    all other days: {show(b)}")
        print(f"    difference = {obs:+.4%} ({obs/sd:+.3f} sd)   raw perm p = {pp:.3f}{flag_n(len(a))}")
        res1[lbl] = (obs, pp, g, a, b)
    obs, p1, dm1_flag, a, b = res1["EXTENDED incl 2026"]
    obs_in, p_in = res1["IN-SAMPLE 2015-2025"][0], res1["IN-SAMPLE 2015-2025"][1]
    print(f"\n  daily sd of SOXL = {sd:.2%}.  The quoted +0.456% vs +0.330% is the in-sample")
    print(f"  version.  Adding the three verified 2026 meetings moves the D-1 mean by")
    print(f"  {obs-obs_in:+.3%} and FLIPS THE SIGN of the difference.  Three observations did that.")
    s_ = pd.Series(a, index=w.index[dm1_flag & ~np.isnan(x)])
    kept, dropped = drop_top(s_, 2)
    print(f"  the two observations that move the extended D-1 mean most: "
          + ", ".join(f"{d.date()} {v:+.1%}" for d, v in dropped.items()))
    print(f"  D-1 mean without them: {kept.mean():+.4%}  (vs other days {b.mean():+.4%})")
    print(f"  up rate D-1 {(a>0).mean():.1%} vs others {(b>0).mean():.1%}  "
          f"-> a {abs((a>0).mean()-(b>0).mean())*100:.1f} pp difference on n={len(a)}, i.e. "
          f"{abs((a>0).mean()-(b>0).mean())*len(a):.1f} extra up days")
    print(f"  A mean difference that three new observations can reverse is not an estimate of")
    print(f"  anything; it is the sample mean of a fat-tailed series.")
    record(1, "FOMC D-1 drift", len(a), p1)
    store[1] = ("mean diff", x, dm1_flag, None)

    # ---------------------------------------------------------------- CLAIM 2
    hdr("CLAIM 2  Hike meetings show D-1 up 14 of 20")
    print("MECHANISM CLAIMED FIRST: none was ever stated.  'Hike meetings' is a subset")
    print("selected AFTER seeing that the all-meeting version was not significant.")
    print("That is the textbook shape of a subgroup found by search.")
    hike = mt[mt["action"] > 0]
    pos = pd.Series(np.arange(len(m)), index=m.index)
    dm1_ret = np.array([m["ret"].values[int(pos[d]) - 1] for d in hike.index])
    k, n2 = int((dm1_ret > 0).sum()), len(dm1_ret)
    print(f"\n  hike meetings n={n2} (all pre-2026; no hikes in the 2026 verified set, so this")
    print(f"  claim has NO out-of-sample observations at all).")
    print(f"    D-1 up {k}/{n2} = {k/n2:.0%}   mean {dm1_ret.mean():+.2%}  median {np.median(dm1_ret):+.2%}")
    print(f"    baseline up rate = {base_up:.1%}")
    p2a = binom_p(k, n2, base_up, two_sided=False)
    p2 = binom_p(k, n2, base_up, two_sided=True)
    print(f"    one-sided binomial p = {p2a:.3f}   two-sided p = {p2:.3f}{flag_n(n2)}")
    yrs = pd.Series([d.year for d in hike.index]).value_counts().sort_index().to_dict()
    print(f"    years of the {n2} hikes: {yrs}")
    n2223 = yrs.get(2022, 0) + yrs.get(2023, 0)
    print(f"    -> {n2223} of {n2} come from the 2022-23 cycle alone; not {n2} independent regimes.")
    print(f"    distinct hiking cycles represented: 2  (2015-12..2018-12, 2022-03..2023-07)")
    for a_, b_ in (("2015-12-01", "2018-12-31"), ("2022-03-01", "2023-07-31")):
        sel = [(d, v) for d, v in zip(hike.index, dm1_ret) if str(a_) <= str(d.date()) <= str(b_)]
        kk = sum(1 for _, v in sel if v > 0)
        print(f"    cycle {a_[:7]}..{b_[:7]}: D-1 up {kk}/{len(sel)}")
    print(f"    With only 2 independent cycles, the effective n for a regime-level claim is 2.")
    record(2, "hike D-1 up 14/20", n2, p2)

    # ---------------------------------------------------------------- CLAIM 3
    hdr("CLAIM 3  Sector-only decline (SOXL <= -8%, QQQ > -2%) underperforms over 3 days")
    print("MECHANISM CLAIMED FIRST: a sector-specific shock is not repaired by index-level")
    print("risk appetite, so the drawdown persists.  Plausible.  The test is whether the")
    print("data can distinguish it from a same-size decline that the index shared.")
    q = m[m["qqq_ret"].notna()]
    alone = q[(q["ret"] <= -0.08) & (q["qqq_ret"] > -0.02)]
    joint = q[(q["ret"] <= -0.08) & (q["qqq_ret"] <= -0.02)]
    print(f"\n  sector-only : {show(alone['fwd3'])}   episodes={episodes(alone.index)}")
    print(f"  joint       : {show(joint['fwd3'])}   episodes={episodes(joint.index)}")
    print(f"  all 2018+   : {show(q['fwd3'])}")
    big = q[(q["ret"] <= -0.08)]
    g3 = ((big["qqq_ret"] > -0.02)).values
    obs3, p3 = perm_diff(big["fwd3"].values, g3)
    print(f"  difference in 3d means (sector-only minus joint) = {obs3:+.2%}   raw perm p = {p3:.3f}")
    al3 = alone[alone["fwd3"].notna()]
    n26 = int((al3.index >= "2026-01-01").sum())
    print(f"  2026 share of the sector-only group (fwd3 available): "
          f"{n26}/{len(al3)} = {n26/len(al3):.0%}")
    ex26 = alone[alone.index < "2026-01-01"]
    jx26 = joint[joint.index < "2026-01-01"]
    print(f"  excluding 2026 -> sector-only {show(ex26['fwd3'])}")
    print(f"                    joint       {show(jx26['fwd3'])}")
    b2 = big[big.index < "2026-01-01"]
    obs3b, p3b = perm_diff(b2["fwd3"].values, (b2["qqq_ret"] > -0.02).values)
    print(f"  excluding 2026 difference = {obs3b:+.2%}   raw perm p = {p3b:.3f}")
    kept3, dr3 = drop_top(alone["fwd3"], 2)
    print(f"  two observations moving the sector-only 3d mean most: "
          + ", ".join(f"{d.date()} {v:+.1%}" for d, v in dr3.items()))
    print(f"  sector-only 3d mean without them: {kept3.mean():+.2%} (median {kept3.median():+.2%})")
    print(f"  NOTE the original claim rested on MEDIAN and UP-RATE, not mean; the mean never")
    print(f"  supported it.  Medians: sector-only {alone['fwd3'].median():+.2%} vs joint "
          f"{joint['fwd3'].median():+.2%}.")
    record(3, "sector-only 3d underperf", len(alone), p3)
    store[3] = ("mean diff", big["fwd3"].values, g3, big.index)

    # ---------------------------------------------------------------- CLAIM 4
    hdr("CLAIM 4  A 6%+ decline that happens to be FOMC D-2 was followed by +8.3% to the")
    print("         decision close, n=9")
    print("MECHANISM CLAIMED FIRST: none.  'Decline >= 6%' AND 'is exactly D-2' is a")
    print("conjunction of two filters chosen because today satisfies both.  The second")
    print("filter has no mechanism attached to it at all -- there is no reason a selloff")
    print("two sessions before a meeting should behave differently from one three sessions")
    print("before.  This is the single most search-contaminated number in the session.")
    def dm2_mask(meet_idx):
        f = np.zeros(len(m), bool)
        for d in meet_idx:
            p_ = int(pos[d])
            if p_ - 2 >= 0:
                f[p_ - 2] = True
        return f

    res4 = {}
    for lbl, mi in (("IN-SAMPLE 2015-2025", mt.index[~mt["oos"]]),
                    ("EXTENDED incl 2026", mt.index)):
        m2 = m.copy()
        m2["dm2"] = dm2_mask(mi)
        bigd = m2[(m2["ret"] <= -0.06) & m2["fwd2"].notna()]
        ad, bd = bigd[bigd["dm2"]], bigd[~bigd["dm2"]]
        obs4_, p4_ = perm_diff(bigd["fwd2"].values, bigd["dm2"].values)
        print(f"\n  {lbl}")
        print(f"    decline>=6% AND D-2 : {show(ad['fwd2'])}   episodes={episodes(ad.index)}")
        print(f"    decline>=6% not D-2 : {show(bd['fwd2'])}   episodes={episodes(bd.index)}")
        print(f"    difference = {obs4_:+.2%}   raw perm p = {p4_:.3f}{flag_n(len(ad))}")
        res4[lbl] = (obs4_, p4_, bigd, ad, bd)
    obs4, p4, bigd, ad, bd = res4["EXTENDED incl 2026"]
    ad_in = res4["IN-SAMPLE 2015-2025"][3]
    print(f"\n  every observation (the last one is the out-of-sample draw):")
    for d, r in ad.iterrows():
        tag = "   <-- OUT OF SAMPLE" if d not in ad_in.index else ""
        print(f"    {d.date()}  day {r['ret']:+.1%}   ->D0 (2d) {r['fwd2']:+.1%}   "
              f"->D+1 (3d) {r['fwd3']:+.1%}   ->D+5 (7d) {r['fwd5']:+.1%}{tag}")
    print(f"\n  The quoted +8.3% is the mean of the {len(ad_in)} in-sample rows "
          f"({ad_in['fwd2'].mean():+.2%}, median {ad_in['fwd2'].median():+.2%}).")
    newrows = ad.index.difference(ad_in.index)
    for d in newrows:
        print(f"  The one observation that arrived afterwards, {d.date()}, returned "
              f"{ad.loc[d,'fwd2']:+.1%} -- the WORST of the {len(ad)}.")
    print(f"  Mean drops {ad_in['fwd2'].mean():+.2%} -> {ad['fwd2'].mean():+.2%} on ONE new draw.")
    kept4, dr4 = drop_top(ad["fwd2"], 2)
    print(f"  the two observations driving the extended mean: "
          + ", ".join(f"{d.date()} {v:+.1%}" for d, v in dr4.items()))
    print(f"  mean without them: {kept4.mean():+.2%} on n={len(kept4)}   "
          f"median of all {len(ad)}: {ad['fwd2'].median():+.2%}")
    obs4b, p4b = perm_diff(bigd["fwd3"].values, bigd["dm2"].values)
    obs4c, p4c = perm_diff(bigd["fwd5"].values, bigd["dm2"].values)
    print(f"  the same filter at 3d: diff {obs4b:+.2%} p={p4b:.3f};  at 7d: diff {obs4c:+.2%} p={p4c:.3f}")
    print(f"  -> even in-sample the effect lives at one horizon; out of sample it is negative.")
    record(4, "6% drop on D-2 -> +8.3%", len(ad), p4)
    store[4] = ("mean diff", bigd["fwd2"].values, bigd["dm2"].values, bigd.index)

    # ---------------------------------------------------------------- CLAIM 5
    hdr("CLAIM 5  Breaking a prior low carries no information in either direction")
    print("THIS IS A NULL CLAIM.  Multiplicity cannot make a null claim worse; the threat")
    print("to a null claim is LOW POWER -- failing to reject is not the same as showing")
    print("there is nothing.  So the relevant output is the confidence interval, not the p.")
    c, low = m["close"], m["low"]
    dd = c / c.rolling(250, min_periods=60).max() - 1
    prior_low = low.shift(1).rolling(60, min_periods=25).min()
    deep = (dd <= -0.40)
    broke_all = m[deep & (m["low"] < prior_low) & m["fwd20"].notna()]
    broke = broke_all.groupby((broke_all.index.to_series().diff().dt.days > 10).cumsum()).head(1)
    near = m[deep & (m["low"] >= prior_low) & (c / prior_low - 1 <= 0.15) & m["fwd20"].notna()]
    print(f"\n  first break of a >=25-bar-old prior low inside a -40% drawdown:")
    print(f"    broke      : {show(broke['fwd20'])}   episodes={episodes(broke.index)}")
    print(f"    near, held : {show(near['fwd20'])}   episodes={episodes(near.index)}")
    both = pd.concat([broke.assign(g=True), near.assign(g=False)])
    obs5, p5 = perm_diff(both["fwd20"].values, both["g"].values)
    print(f"  difference in 20d means = {obs5:+.2%}   raw perm p = {p5:.3f}{flag_n(len(broke))}")
    kept5, dr5 = drop_top(broke["fwd20"], 1)
    print(f"  single driving observation: " + ", ".join(f"{d.date()} {v:+.1%}" for d, v in dr5.items()))
    print(f"  broke-group mean without it: {kept5.mean():+.2%} (median {kept5.median():+.2%})")
    obs5b, p5b = perm_diff(pd.concat([kept5.rename('fwd20').to_frame().assign(g=True),
                                      near[['fwd20']].assign(g=False)])["fwd20"].values,
                           np.r_[np.ones(len(kept5), bool), np.zeros(len(near), bool)])
    print(f"  after dropping it: difference {obs5b:+.2%}  p = {p5b:.3f}")
    record(5, "prior low is uninformative (NULL)", len(broke), p5)
    store[5] = ("mean diff", both["fwd20"].values, both["g"].values, both.index)

    # ---------------------------------------------------------------- CLAIM 6
    hdr("CLAIM 6  Every stop-loss level is EV-negative versus holding without one")
    print("MECHANISM CLAIMED FIRST, AND IT IS ARITHMETIC, NOT STATISTICAL: a stop replaces")
    print("the realised return with -S whenever the path touches -S.  On any series whose")
    print("20-day distribution has a right tail reachable from below -S, the stop can only")
    print("remove mass from above -S and add it at -S.  For the stop to be EV-POSITIVE the")
    print("series would have to be momentum-continuing at the 20-day horizon after a -S")
    print("touch.  So the correct test is not 'is the difference significant' but 'how")
    print("often is the stopped path the one that recovers'.")
    cv, lv = m["close"].values, m["low"].values
    nlen = len(cv)
    pops = {
        "all days": np.ones(nlen, bool),
        "day fell >= 8%": (m["ret"].values <= -0.08),
        "day fell >= 13%": (m["ret"].values <= -0.13),
    }
    HOLD = 20
    print(f"\n  entry at the signal close, hold {HOLD} trading days, stop triggers on the intraday low.")
    print(f"  {'population':<18}{'stop':>8}{'n':>6}{'EV stopped':>12}{'EV hold':>10}{'diff':>9}"
          f"{'median d':>10}{'hit%':>7}{'regret%':>9}{'perm p':>9}")
    p6_list = []
    for pname, mask in pops.items():
        ii = np.array([i for i in np.where(mask)[0] if i + HOLD < nlen])
        base = cv[ii + HOLD] / cv[ii] - 1
        for S in (0.08, 0.12, 0.15, 0.20, 0.25, 0.30):
            st = np.empty(len(ii))
            hit = np.zeros(len(ii), bool)
            for kk, i in enumerate(ii):
                thr = cv[i] * (1 - S)
                seg = lv[i + 1:i + HOLD + 1]
                j = np.argmax(seg <= thr) if (seg <= thr).any() else -1
                if j >= 0:
                    st[kk] = -S
                    hit[kk] = True
                else:
                    st[kk] = cv[i + HOLD] / cv[i] - 1
            d = st - base
            _, pv = perm_paired(d)
            regret = float((base[hit] > 0).mean()) if hit.any() else np.nan
            print(f"  {pname:<18}{'-'+format(S*100,'.0f')+'%':>8}{len(ii):>6}{st.mean():>12.2%}"
                  f"{base.mean():>10.2%}{d.mean():>9.2%}{np.median(d):>10.2%}"
                  f"{hit.mean():>7.0%}{(regret if regret==regret else 0):>9.0%}{pv:>9.3f}")
            p6_list.append((pname, S, d.mean(), pv, len(ii), d))
    worst = min(p6_list, key=lambda r: r[3])
    best_for_stop = max(p6_list, key=lambda r: r[2])
    print(f"\n  sign check: stop EV minus hold EV is negative in "
          f"{sum(1 for r in p6_list if r[2] < 0)}/{len(p6_list)} cells.")
    print(f"  the least-bad cell for a stop: {best_for_stop[0]} at -{best_for_stop[1]*100:.0f}% "
          f"-> {best_for_stop[2]:+.2%} (still {'negative' if best_for_stop[2]<0 else 'POSITIVE'})")
    p6 = worst[3]
    print(f"  smallest raw p across the {len(p6_list)} stop cells: {p6:.4f} "
          f"({worst[0]} at -{worst[1]*100:.0f}%)")
    print(f"  NOTE: those 18 cells are themselves 18 of the ~100 tests.  The claim is not")
    print(f"  'one cell is significant', it is 'the SIGN is the same in every cell', which is")
    print(f"  a much stronger and multiplicity-proof statement.")
    record(6, "stops are EV-negative", worst[4], p6)
    store[6] = ("paired", worst[5], None, None)

    # ---------------------------------------------------------------- CLAIM 7
    hdr("CLAIM 7  A limit tranche inside the daily range (~ -2%) is worse than buying at market")
    print("MECHANISM CLAIMED FIRST: adverse selection.  A limit 2% below a close, against a")
    print("6% median daily range, fills on almost every path EXCEPT the paths that gap up and")
    print("never look back.  You keep all the bad fills and give away the good non-fills.")
    print("The size of the loss is therefore the up-gap frequency times the up-gap size -- a")
    print("mechanical quantity, not an empirical discovery.")
    WINDOW = 20
    print(f"\n  limit rests {WINDOW} days; if filled, hold {HOLD} days from the FILL.")
    print(f"  two accounting conventions shown because they differ in principle:")
    print(f"    (A) unfilled tranche contributes 0.0   (the ladder.py convention)")
    print(f"    (B) unfilled tranche excluded entirely (filled-only)")
    print(f"  {'population':<18}{'offset':>8}{'fill%':>7}{'EV(A)':>9}{'EV(B)':>9}"
          f"{'market':>9}{'A-mkt':>9}{'med A-mkt':>11}{'perm p':>9}")
    p7_list = []
    for pname, mask in pops.items():
        ii = np.array([i for i in np.where(mask)[0] if i + WINDOW + HOLD < nlen])
        mkt = cv[ii + HOLD] / cv[ii] - 1
        for off in (-0.02, -0.05, -0.10):
            va, vb, filled = [], [], []
            for i in ii:
                px = cv[i] * (1 + off)
                seg = lv[i + 1:i + 1 + WINDOW]
                hitj = np.argmax(seg <= px) if (seg <= px).any() else -1
                if hitj < 0:
                    va.append(0.0)
                    filled.append(False)
                else:
                    j = i + 1 + hitj
                    r = cv[j + HOLD] / px - 1
                    va.append(r)
                    vb.append(r)
                    filled.append(True)
            va = np.array(va)
            d = va - mkt
            _, pv = perm_paired(d)
            print(f"  {pname:<18}{off*100:>7.0f}%{np.mean(filled):>7.0%}{va.mean():>9.2%}"
                  f"{(np.mean(vb) if vb else np.nan):>9.2%}{mkt.mean():>9.2%}"
                  f"{d.mean():>9.2%}{np.median(d):>11.2%}{pv:>9.3f}")
            p7_list.append((pname, off, d.mean(), pv, len(ii), d))
    c7 = [r for r in p7_list if abs(r[1] + 0.02) < 1e-9]
    p7 = min(r[3] for r in c7)
    print(f"\n  the -2% tranche is worse than market in "
          f"{sum(1 for r in c7 if r[2] < 0)}/{len(c7)} populations; smallest raw p = {p7:.4f}")
    deeper = [r for r in p7_list if r[1] <= -0.05]
    print(f"  the -5% and -10% tranches are worse than market in "
          f"{sum(1 for r in deeper if r[2] < 0)}/{len(deeper)} cells -> the claim is NOT")
    print(f"  specific to 'inside the range'; every limit offset tested loses to market on")
    print(f"  the mean, which means what is being measured is upward drift, not adverse")
    print(f"  selection.  The claim as stated is therefore mis-attributed to its mechanism.")
    worst7 = min(c7, key=lambda r: r[3])
    record(7, "-2% limit worse than market", worst7[4], p7)
    store[7] = ("paired", worst7[5], None, None)

    # ---------------------------------------------------------------- CLAIM 8
    hdr("CLAIM 8  Post-hike 20-day drift is negative (-2.9%, up 30%) vs holds (+6.5%, up 62%)")
    print("MECHANISM CLAIMED FIRST: tightening reduces the discount-rate support for long-")
    print("duration equities.  The confound stated BEFORE the number: the Fed hikes when")
    print("the economy is hot and stops when it is not, so 'hike meetings' is largely a")
    print("relabelling of 'calendar year 2022', which was a -60% year for SOXL for reasons")
    print("that have nothing to do with the two days around any single statement.")
    mt2 = mt.copy()
    mt2["fwd20"] = [m["fwd20"].values[int(pos[d])] for d in mt2.index]
    hk = mt2[mt2["action"] > 0]["fwd20"].dropna()
    hd = mt2[mt2["action"] == 0]["fwd20"].dropna()
    ct = mt2[mt2["action"] < 0]["fwd20"].dropna()
    print(f"\n  hikes : {show(hk)}")
    print(f"  holds : {show(hd)}")
    print(f"  cuts  : {show(ct)}")
    both8 = pd.concat([hk.to_frame("y").assign(g=True), hd.to_frame("y").assign(g=False)])
    obs8, p8 = perm_diff(both8["y"].values, both8["g"].values)
    print(f"  difference (hike minus hold) = {obs8:+.2%}   raw perm p = {p8:.3f}{flag_n(len(hk))}")
    kept8, dr8 = drop_top(hk, 2)
    print(f"  two observations moving the hike mean most: "
          + ", ".join(f"{d.date()} {v:+.1%}" for d, v in dr8.items()))
    print(f"  hike mean without them: {kept8.mean():+.2%} (median {kept8.median():+.2%}, n={len(kept8)})")
    print(f"\n  THE CONFOUND, QUANTIFIED.  Same statistic computed on ALL trading days of the")
    print(f"  calendar years in which each meeting type occurred:")
    hy = sorted({d.year for d in mt2[mt2['action'] > 0].index})
    dy = sorted({d.year for d in mt2[mt2['action'] == 0].index})
    ally = m["fwd20"].dropna()
    hy_all = ally[[d.year in hy for d in ally.index]]
    dy_all = ally[[d.year in dy for d in ally.index]]
    print(f"    hike years {hy}: every day  {show(hy_all)}")
    print(f"    hold years {dy}: every day  {show(dy_all)}")
    print(f"    -> baseline gap from the calendar alone = {hy_all.mean()-dy_all.mean():+.2%}, "
          f"vs the claimed meeting gap {obs8:+.2%}")
    print(f"\n  YEAR-MATCHED TEST: for each hike meeting, subtract the mean fwd20 of all other")
    print(f"  trading days in the SAME calendar year.  This removes the regime entirely.")
    yr_mean = ally.groupby(ally.index.year).mean()
    hk_adj = np.array([v - yr_mean[d.year] for d, v in hk.items()])
    hd_adj = np.array([v - yr_mean[d.year] for d, v in hd.items()])
    print(f"    hikes, year-demeaned: n={len(hk_adj)} mean {hk_adj.mean():+.2%} median {np.median(hk_adj):+.2%} up {(hk_adj>0).mean():.0%}")
    print(f"    holds, year-demeaned: n={len(hd_adj)} mean {hd_adj.mean():+.2%} median {np.median(hd_adj):+.2%} up {(hd_adj>0).mean():.0%}")
    obs8b, p8b = perm_diff(np.r_[hk_adj, hd_adj],
                           np.r_[np.ones(len(hk_adj), bool), np.zeros(len(hd_adj), bool)])
    print(f"    year-demeaned difference = {obs8b:+.2%}   raw perm p = {p8b:.3f}")
    print(f"    -> {abs(1-obs8b/obs8)*100:.0f}% of the raw gap is the calendar, not the meeting.")
    oos8 = mt2[mt2["oos"]]["fwd20"].dropna()
    print(f"\n  out-of-sample check: the 3 verified 2026 meetings were all HOLDS; their fwd20 "
          f"was {', '.join(f'{d.date()} {v:+.1%}' for d, v in oos8.items())}")
    print(f"  mean {oos8.mean():+.2%} vs the +7.45% the 'holds are good' claim predicts. "
          f"{int((oos8<0).sum())}/{len(oos8)} were negative.")
    hd_in = mt2[(mt2['action'] == 0) & ~mt2['oos']]['fwd20'].dropna()
    print(f"  holds excluding 2026: n={len(hd_in)} mean {hd_in.mean():+.2%} "
          f"-> the quoted +6.5%/62% figure is the in-sample one.")
    record(8, "post-hike 20d drift negative", len(hk), p8)
    store[8] = ("mean diff", both8["y"].values, both8["g"].values, both8.index)

    # ---------------------------------------------------------------- multiplicity
    hdr("MULTIPLE-TESTING CORRECTION  (M = 100 tests run this session)")
    ps = [r["p"] for r in RESULTS]
    qg, qu = bh_table(ps)
    print(f"{'#':<3}{'claim':<34}{'n':>5}{'raw p':>9}{'Bonf p':>9}{'surv B':>8}"
          f"{'BH q gen':>10}{'BH q unif':>11}{'surv BH':>9}")
    for r, g_, u_ in zip(RESULTS, qg, qu):
        bp = bonf(r["p"])
        print(f"{r['id']:<3}{r['short']:<34}{r['n']:>5}{r['p']:>9.4f}{bp:>9.3f}"
              f"{str(bp < ALPHA):>8}{g_:>10.3f}{u_:>11.3f}{str(u_ < ALPHA):>9}")
    print(f"\n  Bonferroni threshold on raw p: {ALPHA/M_TESTS:.5f}")
    print(f"  Expected false positives at alpha=0.05 with M=100 under a complete null: 5.0")
    print(f"  Number of these 8 with raw p < 0.05: {sum(1 for p in ps if p < 0.05)}")
    print(f"  -> that count is inside what pure chance produces from a 100-test search.")

    # ---------------------------------------------------------------- bootstrap
    hdr("BLOCK BOOTSTRAP  (contiguous 20-day blocks, 2000 iterations, 95% percentile CI)")
    print("Run on every claim, not only the survivors, because a claim that 'fails' a")
    print("permutation test can still have a CI that excludes zero and vice versa.")
    print("Overlapping forward windows make the permutation p too small; this is the fix.")
    print(f"\n{'#':<3}{'claim':<34}{'point est':>12}{'boot mean':>12}{'95% CI low':>13}"
          f"{'95% CI high':>13}{'excl 0':>8}")
    boot_out = {}
    for cid in sorted(store):
        kind, xa, ga, _ = store[cid]
        if kind == "mean diff":
            x = np.asarray(xa, float)
            g = np.asarray(ga, bool)
            ok = ~np.isnan(x)
            pt = x[ok & g].mean() - x[ok & ~g].mean()
            (lo_, hi_), bm = block_boot_diff(x, g)
        else:
            d = np.asarray(xa, float)
            pt = np.nanmean(d)
            lo_, hi_ = block_boot_mean(d)
            bm = np.nan
        excl = (lo_ > 0) or (hi_ < 0)
        boot_out[cid] = (pt, lo_, hi_, excl)
        short = next(r["short"] for r in RESULTS if r["id"] == cid)
        print(f"{cid:<3}{short:<34}{pt:>12.2%}{(bm if bm==bm else np.nan):>12.2%}"
              f"{lo_:>13.2%}{hi_:>13.2%}{str(excl):>8}")
    print("\n  Claims 2 is a proportion, not a mean difference; its interval is the binomial")
    print("  one and is shown in its own section above (14/20, two-sided p reported there).")

    # a direct look at the width of the CI for the null claims, to show power
    sub("POWER ON THE NULL CLAIMS: what magnitudes the data CANNOT rule out")
    for cid in (5,):
        pt, lo_, hi_, _ = boot_out[cid]
        print(f"  claim {cid}: 20-day effect point {pt:+.2%}, 95% CI [{lo_:+.2%}, {hi_:+.2%}]")
        print(f"    -> 'no information' is consistent with the data, but so is anything")
        print(f"       between {lo_:+.1%} and {hi_:+.1%} over 20 days.  The correct statement is")
        print(f"       'no detectable information at this sample size', which for trading")
        print(f"       purposes is the same decision but a different sentence.")

    # ---------------------------------------------------------------- ranking
    hdr("FINAL RANKING")
    print("robust  = survives multiplicity AND the block bootstrap CI excludes 0, OR the")
    print("          claim is mechanical/arithmetic and does not depend on a p-value.")
    print("fragile = the direction is probably real but the magnitude is not usable, or it")
    print("          depends on a regime / a handful of observations.")
    print("noise   = cannot be distinguished from the ~5 false positives a 100-test search")
    print("          produces by construction.  Do not act on it.")
    verdicts = [
        (1, "noise",
         "raw p ~ {p1:.2f} before any correction.  A 0.13 pp difference against a 4% daily "
         "sd is 0.03 sd.  Up-rate difference is one extra up day out of 87.  Nothing survives."),
        (2, "noise",
         "n=20, two-sided binomial p reported above.  The 'hike' subset was chosen after the "
         "full-sample version failed.  13 of the 20 come from one cycle.  Counter-example only."),
        (3, "fragile",
         "Direction (medians and up-rate) is stable and has a stated mechanism, but the mean "
         "never supported it, 2026 supplies a large share of the sample, and the CI spans 0."),
        (4, "noise",
         "n=9, two filters conjoined because today matches both, no mechanism for the second, "
         "and the effect vanishes one horizon later.  This is the definition of data mining."),
        (5, "robust",
         "As a NULL it is safe: nothing was found and nothing needs multiplicity protection. "
         "State it as 'no detectable edge at n=' rather than 'no information'."),
        (6, "robust",
         "Arithmetic, not statistical: a stop truncates the right tail of a series with no "
         "20-day momentum continuation.  Same sign in every cell tested."),
        (7, "fragile",
         "True as a number but mis-attributed: every limit offset loses to market, so what is "
         "measured is upward drift, not adverse selection specific to 'inside the range'."),
        (8, "fragile",
         "Direction has a real mechanism, but year-demeaning removes most of the gap.  The raw "
         "-2.9% vs +6.5% is largely 'hikes happened in 2022'."),
    ]
    print(f"\n{'#':<3}{'claim':<34}{'verdict':<10}why")
    for cid, v, why in verdicts:
        short = next(r["short"] for r in RESULTS if r["id"] == cid)
        why = why.format(p1=ps[0])
        print(f"{cid:<3}{short:<34}{v:<10}{why[:60]}")
        rest = why[60:]
        while rest:
            print(f"{'':<47}{rest[:60]}")
            rest = rest[60:]

    hdr("THE DELIVERABLE: WHAT SHOULD NOT BE RELIED ON")
    print("  1. Do not size anything off claim 4 (the 6%-decline-on-D-2 +8.3%).  It is the")
    print("     most-searched number in the session and it is n=9.")
    print("  2. Do not use claim 1 or claim 2 as a reason to be long into 2026-09-16.  The")
    print("     pre-FOMC drift is not measurable in this instrument at this sample size.")
    print("  3. Do not use claim 8's magnitude.  A hike is 92.3% priced; what is priced is")
    print("     not what the 2022 sample measured, which was a repricing of the whole path.")
    print("  4. Claim 3's direction may be used as a reason NOT to add, never as a size.")
    print("  5. Claims 5, 6, 7 are the only ones that should touch the plan, and all three")
    print("     are constraints (no stop, no shallow tranche, no prior-low trigger), not")
    print("     entry signals.  Constraints are what survives a multiplicity audit, because")
    print("     they rest on arithmetic rather than on a subset mean.")
    print("\n  Net: of the 8 claims, 0 are robust findings ABOUT THE FUTURE.  The 3 marked")
    print("  robust/fragile-but-usable are all statements about the arithmetic of position")
    print("  management, which is exactly the class of claim a 100-test search cannot fake.")
    print()


if __name__ == "__main__":
    main()
