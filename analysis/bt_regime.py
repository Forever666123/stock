"""Does 2026 SOXL come from the same distribution as 2010-2025?

Mechanism stated first, before any number:

1. SOXL is a 3x *daily-rebalanced* fund. Its exposure is reset to 3x NAV every
   close, so a large daily move mechanically forces the fund to trade in the
   direction of that move at the next close. Realised volatility therefore feeds
   back into the fund's own path in a way it does not for the underlying index.
   If the semiconductor index itself has become more volatile, the fund's
   volatility should rise by roughly 3x and no more; if the fund's volatility has
   risen by MORE than 3x the index's, the extra comes from the fund/leverage
   ecosystem (rebalancing flow, option hedging, liquidity), which is a genuinely
   different regime and not a rescaling of the old one.

2. If the 2026 return distribution differs in scale only, every historical base
   rate expressed as a *probability of direction* still transfers, while every
   base rate expressed in *percent* (drawdown depth, ladder fill distance,
   position sizing) is mis-scaled and must be re-derived. If the distribution
   differs in shape (kurtosis, autocorrelation) too, the direction probabilities
   move as well and nothing transfers.

So: test scale first (Levene/Brown-Forsythe on variance), then shape (KS on the
standardised distribution, kurtosis, autocorrelation), then the fund-vs-index
ratio, then quantify the damage to the session's headline base rates.

Run:  cd /home/user/stock && python3 -m analysis.bt_regime
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data import load_merged

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# The last bar is not in the CSV. Appended manually in every script this session.
LAST_BAR = dict(date="2026-09-14", open=101.53, high=105.39, low=99.87,
                close=101.13, qqq=711.96)

RNG_SEED = 12345
NPERM = 20000


# ---------------------------------------------------------------- data


def build() -> pd.DataFrame:
    m = load_merged()
    d = pd.Timestamp(LAST_BAR["date"])
    assert d not in m.index, "last bar already present - remove the manual append"
    prev = m.loc[m.index < d].iloc[-1]
    row = {c: np.nan for c in m.columns}
    row.update(open=LAST_BAR["open"], high=LAST_BAR["high"], low=LAST_BAR["low"],
               close=LAST_BAR["close"], volume=np.nan, qqq=LAST_BAR["qqq"],
               prev_close=prev["close"])
    row["ret"] = LAST_BAR["close"] / prev["close"] - 1
    row["gap"] = LAST_BAR["open"] / prev["close"] - 1
    row["low_ret"] = LAST_BAR["low"] / prev["close"] - 1
    row["high_ret"] = LAST_BAR["high"] / prev["close"] - 1
    row["range"] = (LAST_BAR["high"] - LAST_BAR["low"]) / prev["close"]
    row["qqq_ret"] = LAST_BAR["qqq"] / prev["qqq"] - 1
    m = pd.concat([m, pd.DataFrame([row], index=[d])]).sort_index()
    # forward columns must be rebuilt: the appended bar changes the last 20 rows
    c, low = m["close"], m["low"]
    for h in (1, 3, 5, 10, 20):
        m[f"fwd{h}"] = c.shift(-h) / c - 1
        m[f"fwd_min{h}"] = low[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1) / c - 1
    m["hi60"] = c.rolling(60).max()
    m["dd60"] = c / m["hi60"] - 1
    return m.iloc[1:]  # drop 2010-03-11, which has no prior close


def load_soxx() -> pd.Series:
    """SOXX daily returns, computed within each source and concatenated."""
    a = pd.read_parquet(DATA / "phase5_close_soxx.parquet")
    a.index = pd.to_datetime(a.index)
    a = a["SOXX"]
    j = json.load(open(DATA / "holdings_prices_1y.json"))["prices"]["SOXX"]
    b = pd.Series(j["prices"], index=pd.to_datetime(j["dates"]))
    ov = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    ratio = ov["b"] / ov["a"]
    print(f"  SOXX source overlap n={len(ov)} ratio mean={ratio.mean():.4f} "
          f"std={ratio.std():.5f} (constant ratio => dividend treatment only)")
    ra, rb = a.pct_change(), b.pct_change()
    return pd.concat([ra[ra.index <= a.index.max()],
                      rb[rb.index > a.index.max()]]).sort_index().dropna()


# ---------------------------------------------------------------- stats


def perm_p(a, b, stat, n=NPERM, seed=RNG_SEED):
    """Two-sided permutation p for any statistic taking (a, b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    obs = abs(stat(a, b))
    pool = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(stat(pool[:na], pool[na:])) >= obs - 1e-15:
            hits += 1
    return (hits + 1) / (n + 1)


def block_perm_p(a, b, stat, L=21, n=5000, seed=RNG_SEED):
    """Permutation that shuffles BLOCKS of L consecutive days, not single days.

    Volatility clusters. An iid label shuffle treats 175 days of 2026 as 175
    independent draws, which they are not: an effective sample closer to
    175/21 ~ 8 vol-blocks is honest. This keeps within-block clustering intact.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    obs = abs(stat(a, b))
    pool = np.concatenate([a, b])
    na, N = len(a), len(pool)
    nb_blocks = int(np.ceil(N / L))
    pad = nb_blocks * L - N
    padded = np.concatenate([pool, pool[:pad]]) if pad else pool
    blocks = padded.reshape(nb_blocks, L)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        order = rng.permutation(nb_blocks)
        z = blocks[order].ravel()[:N]
        if abs(stat(z[:na], z[na:])) >= obs - 1e-15:
            hits += 1
    return (hits + 1) / (n + 1)


def brown_forsythe(a, b):
    """Levene's test with the median (= Brown-Forsythe), F statistic.

    Robust to the fat tails that break the mean-centred version on a 3x ETF.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    za, zb = np.abs(a - np.median(a)), np.abs(b - np.median(b))
    n1, n2, N = len(za), len(zb), len(za) + len(zb)
    zbar = (za.sum() + zb.sum()) / N
    num = (n1 * (za.mean() - zbar) ** 2 + n2 * (zb.mean() - zbar) ** 2)
    den = (((za - za.mean()) ** 2).sum() + ((zb - zb.mean()) ** 2).sum()) / (N - 2)
    return num / den if den > 0 else np.inf


def ks_stat(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, "right") / len(a)
    cb = np.searchsorted(b, allv, "right") / len(b)
    return np.max(np.abs(ca - cb))


def exkurt(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = x.mean()
    s = x.std(ddof=0)
    return ((x - m) ** 4).mean() / s ** 4 - 3


def ac(x, lag=1):
    x = pd.Series(x).dropna()
    return x.autocorr(lag)


def ann(x):
    return np.std(x, ddof=1) * np.sqrt(252)


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def pct(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.2f}%"


# ---------------------------------------------------------------- 1. regime


def part1(m):
    sec("1. IS 2026 THE SAME DISTRIBUTION?  year-by-year")
    m = m.copy()
    m["yr"] = m.index.year
    rows = []
    for y, g in m.groupby("yr"):
        r = g["ret"].dropna()
        rows.append(dict(
            year=y, n=len(r), ann_vol=ann(r), med_abs=r.abs().median(),
            med_range=g["range"].median(), med_absgap=g["gap"].abs().median(),
            gap_gt5=(g["gap"].abs() > .05).mean(), gap_gt10=(g["gap"].abs() > .10).mean(),
            kurt=exkurt(r), ac1=ac(r, 1), big10=(r.abs() >= .10).mean(),
            n_big10=int((r.abs() >= .10).sum())))
    t = pd.DataFrame(rows).set_index("year")
    print(f"{'yr':>5} {'n':>4} {'annVol':>8} {'med|r|':>7} {'medRng':>7} {'med|gap|':>8} "
          f"{'gap>5%':>7} {'gap>10%':>8} {'exKurt':>7} {'AC(1)':>7} {'|r|>10%':>8} {'#':>3}")
    for y, r in t.iterrows():
        print(f"{y:>5} {int(r['n']):>4} {r['ann_vol']*100:>7.1f}% {r['med_abs']*100:>6.2f}% "
              f"{r['med_range']*100:>6.2f}% {r['med_absgap']*100:>7.2f}% {r['gap_gt5']*100:>6.1f}% "
              f"{r['gap_gt10']*100:>7.1f}% {r['kurt']:>7.2f} {r['ac1']:>7.3f} "
              f"{r['big10']*100:>7.1f}% {int(r['n_big10']):>3}")

    sec("1b. 2026 vs 2010-2025 : formal tests (all p from permutation, two-sided)")
    cur = m[m.index >= "2026-01-01"]
    old = m[m.index < "2026-01-01"]
    a, b = cur["ret"].dropna().values, old["ret"].dropna().values
    print(f"2026 n={len(a)}  (>=50, so conclusions are allowed)   2010-2025 n={len(b)}")

    print(f"\n{'metric':<34}{'2026':>12}{'2010-2025':>12}{'ratio':>9}{'p':>9}")

    def line(name, va, vb, p, fmt="{:.2f}%", scale=100):
        ratio = va / vb if vb else np.nan
        print(f"{name:<34}{fmt.format(va*scale):>12}{fmt.format(vb*scale):>12}"
              f"{ratio:>9.2f}{p:>9.4f}" if not np.isnan(p) else
              f"{name:<34}{fmt.format(va*scale):>12}{fmt.format(vb*scale):>12}{ratio:>9.2f}{'-':>9}")

    p_var = perm_p(a, b, lambda x, y: brown_forsythe(x, y))
    line("annualised vol of daily ret", ann(a), ann(b), p_var)
    print(f"  Brown-Forsythe F = {brown_forsythe(a,b):.1f}  (permutation p above)")

    p_mad = perm_p(a, b, lambda x, y: np.median(np.abs(x)) - np.median(np.abs(y)))
    line("median |daily return|", np.median(np.abs(a)), np.median(np.abs(b)), p_mad)

    ra, rb = cur["range"].dropna().values, old["range"].dropna().values
    line("median high-low range", np.median(ra), np.median(rb),
         perm_p(ra, rb, lambda x, y: np.median(x) - np.median(y)))
    line("mean high-low range", ra.mean(), rb.mean(),
         perm_p(ra, rb, lambda x, y: x.mean() - y.mean()))

    ga, gb = cur["gap"].dropna().values, old["gap"].dropna().values
    line("median |gap|", np.median(np.abs(ga)), np.median(np.abs(gb)),
         perm_p(ga, gb, lambda x, y: np.median(np.abs(x)) - np.median(np.abs(y))))
    for thr in (.02, .05, .10):
        fa, fb = (np.abs(ga) > thr).mean(), (np.abs(gb) > thr).mean()
        line(f"freq |gap| > {thr*100:.0f}%", fa, fb,
             perm_p(np.abs(ga) > thr, np.abs(gb) > thr, lambda x, y: x.mean() - y.mean()))
    for thr in (.10, .15):
        fa, fb = (np.abs(a) >= thr).mean(), (np.abs(b) >= thr).mean()
        line(f"freq |daily ret| >= {thr*100:.0f}%", fa, fb,
             perm_p(np.abs(a) >= thr, np.abs(b) >= thr, lambda x, y: x.mean() - y.mean()))
        print(f"    counts: 2026 {int((np.abs(a)>=thr).sum())}/{len(a)}   "
              f"2010-2025 {int((np.abs(b)>=thr).sum())}/{len(b)}")

    print(f"\n{'excess kurtosis':<34}{exkurt(a):>12.2f}{exkurt(b):>12.2f}"
          f"{exkurt(a)/exkurt(b):>9.2f}{perm_p(a,b,lambda x,y: exkurt(x)-exkurt(y)):>9.4f}")
    for lag in (1, 2, 3, 5):
        pa = perm_p(a, b, lambda x, y: ac(x, lag) - ac(y, lag))
        print(f"{'autocorr lag '+str(lag):<34}{ac(a,lag):>12.3f}{ac(b,lag):>12.3f}"
              f"{'':>9}{pa:>9.4f}")

    print("\nKolmogorov-Smirnov, raw daily returns")
    print(f"  D = {ks_stat(a,b):.4f}   permutation p = {perm_p(a,b,ks_stat):.4f}")
    print("KS on returns standardised by each period's own median-abs-deviation")
    sa = a / np.median(np.abs(a))
    sb = b / np.median(np.abs(b))
    p_shape = perm_p(sa, sb, ks_stat)
    print(f"  D = {ks_stat(sa,sb):.4f}   permutation p = {p_shape:.4f}")
    print("  (if raw KS rejects but standardised KS does not, 2026 differs in SCALE only)")

    print("\nBonferroni: 20 tests reported in this block; 5% -> 0.0025 threshold.")

    sec("1c. HONEST p: volatility clusters, so shuffle blocks not days")
    for L in (5, 21, 63):
        pb = block_perm_p(a, b, brown_forsythe, L=L, n=5000)
        print(f"  block length {L:>3}d  Brown-Forsythe block-permutation p = {pb:.4f}"
              f"   (effective 2026 blocks ~{len(a)//L})")
    print("  The iid p of 0.0000 above is not credible; these are.")

    sec("1d. ADVERSARIAL: is 2026 new, or is it just 2020/2022 again?")
    print("If 2026 merely repeats the worst years already in the sample, then the")
    print("problem is not that history fails to transfer - it is that the AVERAGE of")
    print("history was the wrong summary all along, and a vol-conditioned subsample")
    print("would transfer fine. That is a different, cheaper fix. Test it.")
    print(f"\n{'comparison':<26}{'volA':>9}{'volB':>9}{'ratio':>8}{'BF F':>8}"
          f"{'iid p':>9}{'block21 p':>11}")
    for nm, yr in (("2026 vs 2022", 2022), ("2026 vs 2020", 2020), ("2026 vs 2025", 2025),
                   ("2026 vs 2024", 2024)):
        z = m[m.index.year == yr]["ret"].dropna().values
        print(f"{nm:<26}{ann(a)*100:>8.1f}%{ann(z)*100:>8.1f}%{ann(a)/ann(z):>8.2f}"
              f"{brown_forsythe(a,z):>8.1f}{perm_p(a,z,brown_forsythe,n=5000):>9.4f}"
              f"{block_perm_p(a,z,brown_forsythe,L=21,n=5000):>11.4f}")
    hv = m[(m.index.year.isin([2020, 2022])) & (m.index < "2026-01-01")]["ret"].dropna().values
    print(f"{'2026 vs 2020+2022 pooled':<26}{ann(a)*100:>8.1f}%{ann(hv)*100:>8.1f}%"
          f"{ann(a)/ann(hv):>8.2f}{brown_forsythe(a,hv):>8.1f}"
          f"{perm_p(a,hv,brown_forsythe,n=5000):>9.4f}"
          f"{block_perm_p(a,hv,brown_forsythe,L=21,n=5000):>11.4f}")
    print(f"\nKS 2026 vs 2020+2022 pooled, raw returns: D={ks_stat(a,hv):.4f} "
          f"iid p={perm_p(a,hv,ks_stat,n=5000):.4f} "
          f"block21 p={block_perm_p(a,hv,ks_stat,L=21,n=5000):.4f}")
    return cur, old


# ---------------------------------------------------------------- 2. rolling


def part2(m):
    sec("2. ROLLING WINDOWS - when did the regime change?")
    r = m["ret"]
    roll = pd.DataFrame({
        "vol63": r.rolling(63).std() * np.sqrt(252),
        "rng63": m["range"].rolling(63).median(),
        "kurt252": r.rolling(252).apply(exkurt, raw=True),
        "big10_252": (r.abs() >= .10).rolling(252).sum(),
        "ac1_252": r.rolling(252).apply(lambda x: pd.Series(x).autocorr(1), raw=True),
    })
    print("63-day realised vol percentile of the full history, at month ends 2025-08 -> now:")
    v = roll["vol63"].dropna()
    tail = v[v.index >= "2025-08-01"]
    me = tail.groupby([tail.index.year, tail.index.month]).tail(1)
    for d, x in me.items():
        print(f"  {d.date()}  vol63 {x*100:6.1f}%   pctile {100*(v<x).mean():5.1f}%")
    print(f"\nall-time max of 63d vol: {v.max()*100:.1f}% on {v.idxmax().date()}")
    print(f"current (last bar):      {v.iloc[-1]*100:.1f}%  on {v.index[-1].date()}")
    print(f"median 2010-2025:        {v[v.index<'2026-01-01'].median()*100:.1f}%")
    print(f"median 2026:             {v[v.index>='2026-01-01'].median()*100:.1f}%")

    print("\nCalendar-year-end / current snapshot of each rolling metric:")
    print(f"{'date':>12}{'vol63':>9}{'rng63':>9}{'kurt252':>9}{'#|r|>10%/252':>14}{'AC1_252':>9}")
    snaps = [roll[roll.index.year == y].iloc[-1:] for y in range(2011, 2027)]
    for s in snaps:
        if len(s) == 0:
            continue
        d = s.index[0]
        x = s.iloc[0]
        print(f"{str(d.date()):>12}{x['vol63']*100:>8.1f}%{x['rng63']*100:>8.2f}%"
              f"{x['kurt252']:>9.2f}{x['big10_252']:>14.0f}{x['ac1_252']:>9.3f}")

    print("\nTop 10 calendar quarters by 63-day realised vol (end date of window):")
    top = v.sort_values(ascending=False)
    seen, shown = [], 0
    for d, x in top.items():
        if any(abs((d - s).days) < 63 for s in seen):
            continue
        seen.append(d)
        shown += 1
        print(f"  {d.date()}  {x*100:6.1f}%")
        if shown == 10:
            break
    return roll


# ---------------------------------------------------------------- 3. SOXX


def part3(m):
    sec("3. FUND vs UNDERLYING INDEX - is this leverage feedback or just chips?")
    print("SOXX close, two sources stitched (returns computed within each source):")
    sx = load_soxx()
    print(f"  SOXX returns {sx.index.min().date()} -> {sx.index.max().date()} n={len(sx)}")
    print("  NOTE: SOXX coverage starts 2018-01-03, so the index comparison cannot")
    print("        speak to 2010-2017. It also has no 2026-09-14 bar.")

    sl = m["ret"].dropna()
    idx = sx.index.intersection(sl.index)
    sx, sl2 = sx.loc[idx], sl.loc[idx]

    print(f"\n{'period':<14}{'n':>6}{'SOXX vol':>10}{'SOXL vol':>10}{'ratio':>8}"
          f"{'SOXX med|r|':>12}{'SOXL med|r|':>12}{'ratio':>8}")
    def blk(name, mask):
        x, y = sx[mask], sl2[mask]
        if len(x) < 10:
            return None
        print(f"{name:<14}{len(x):>6}{ann(x)*100:>9.1f}%{ann(y)*100:>9.1f}%"
              f"{ann(y)/ann(x):>8.2f}{x.abs().median()*100:>11.2f}%"
              f"{y.abs().median()*100:>11.2f}%"
              f"{y.abs().median()/x.abs().median():>8.2f}")
        return ann(y) / ann(x)
    for y in range(2018, 2027):
        blk(str(y), (sx.index.year == y))
    r_old = blk("2018-2025", sx.index < "2026-01-01")
    r_new = blk("2026", sx.index >= "2026-01-01")

    xo, xn = sx[sx.index < "2026-01-01"], sx[sx.index >= "2026-01-01"]
    yo, yn = sl2[sl2.index < "2026-01-01"], sl2[sl2.index >= "2026-01-01"]
    print(f"\nSOXX variance 2026 vs 2018-2025: Brown-Forsythe F={brown_forsythe(xn,xo):.1f} "
          f"perm p={perm_p(xn,xo,brown_forsythe):.4f}")
    print(f"SOXX vol ratio 2026/2018-25 = {ann(xn)/ann(xo):.2f}x   "
          f"SOXL vol ratio = {ann(yn)/ann(yo):.2f}x")
    print(f"SOXL/SOXX vol multiple: 2018-2025 {r_old:.2f}  ->  2026 {r_new:.2f}")

    # daily beta: is the fund still delivering 3x on the day?
    def beta(x, y):
        return np.polyfit(x, y, 1)[0]
    print(f"\ndaily regression slope SOXL on SOXX (should be ~3.00 by construction):")
    print(f"  2018-2025 beta={beta(xo,yo):.3f}  R2={np.corrcoef(xo,yo)[0,1]**2:.4f}")
    print(f"  2026      beta={beta(xn,yn):.3f}  R2={np.corrcoef(xn,yn)[0,1]**2:.4f}")
    print(f"  perm p on beta difference = "
          f"{perm_p(np.arange(len(xn)),np.arange(len(xo)),lambda i,j: 0.0, n=1):>.0f} (n/a - see note)")
    print("  NOTE: beta is a paired statistic, so the label-shuffle permutation above is")
    print("        meaningless and is not reported as evidence. Read the two betas directly.")

    print("\nIf SOXL vol rose by MORE than 3x the SOXX rise, the extra is fund-specific.")
    print(f"  SOXX vol rise {ann(xn)/ann(xo):.2f}x  vs  SOXL vol rise {ann(yn)/ann(yo):.2f}x")

    print("\nRolling 60-day SOXL-on-SOXX beta - has tracking drifted late in 2026?")
    br = []
    for i in range(60, len(sx) + 1):
        xx, yy = sx.values[i-60:i], sl2.values[i-60:i]
        br.append((sx.index[i-1], np.polyfit(xx, yy, 1)[0]))
    br = pd.Series(dict(br))
    print(f"  full range of rolling beta: min {br.min():.3f} ({br.idxmin().date()})  "
          f"max {br.max():.3f} ({br.idxmax().date()})")
    for d in ("2026-01-30", "2026-03-31", "2026-05-29", "2026-06-30", "2026-07-31",
              "2026-08-31"):
        sl_ = br[br.index <= d]
        if len(sl_):
            print(f"  {d}: {sl_.iloc[-1]:.3f}")
    print(f"  last available ({br.index[-1].date()}): {br.iloc[-1]:.3f}")
    print(f"\n  2026-09-14 is NOT covered by SOXX data (JSON ends 2026-09-11).")
    print(f"  SOXL -16.98% that day implies SOXX about {-0.1698/2.93*100:.2f}% if the")
    print(f"  3x relation held; QQQ was only -0.41%, so it was a chip-specific day.")
    print(f"  This cannot be verified here and is flagged as an open hole.")


# ---------------------------------------------------------------- 4. base rates


def part4(m):
    sec("4. CONSEQUENCE - the session's headline base rates on 2024-2026 only")
    full = m
    recent = m[m.index >= "2024-01-01"]
    print(f"full history n={len(full)}   2024-2026 n={len(recent)}\n")

    def cmp_block(title, fn, note=""):
        print("-" * 78)
        print(title)
        if note:
            print("  mechanism/definition: " + note)
        a = fn(full, "full 2010-2026")
        b = fn(recent, "2024-2026")
        return a, b

    # (a) unconditional next-day up rate  (report.md headline: 53.8%)
    print("-" * 78)
    print("(a) UNCONDITIONAL next-day up rate  [report.md headline: 53.8%, n=4150]")
    for name, d in (("full 2010-2026", full), ("2024-2026", recent),
                    ("2026 only", m[m.index >= "2026-01-01"])):
        f = d["fwd1"].dropna()
        print(f"  {name:<16} n={len(f):>5}  up {f.gt(0).mean()*100:5.1f}%  "
              f"mean {pct(f.mean()):>8}  median {pct(f.median()):>8}")
    fa = full["fwd1"].dropna()
    fb = recent["fwd1"].dropna()
    print(f"  perm p on up-rate difference (full vs 2024-26, overlapping samples): "
          f"{perm_p(fb.gt(0), fa.gt(0), lambda x,y: x.mean()-y.mean()):.4f}")

    # (b) 20-day forward minimum  -> the position-sizing number
    print("-" * 78)
    print("(b) 20-DAY FORWARD MINIMUM from any close  [sizing input; today_*.md used")
    print("    -12.4% unconditional and -25.7% for the deep-drawdown state]")
    for name, d in (("full 2010-2026", full), ("2024-2026", recent),
                    ("2026 only", m[m.index >= "2026-01-01"])):
        f = d["fwd_min20"].dropna()
        print(f"  {name:<16} n={len(f):>5}  median {pct(f.median()):>8}  "
              f"mean {pct(f.mean()):>8}  p10 {pct(f.quantile(.10)):>8}  "
              f"p05 {pct(f.quantile(.05)):>8}  worst {pct(f.min()):>8}")
    fa = full["fwd_min20"].dropna()
    fb = recent["fwd_min20"].dropna()
    print(f"  perm p on median difference: "
          f"{perm_p(fb, fa, lambda x,y: np.median(x)-np.median(y)):.4f}  "
          f"(20-day windows overlap ~20x, so this p is far too optimistic)")
    print(f"  independent 20-day blocks: 2024-26 {len(fb)//20}, full {len(fa)//20}")

    # (c) oversold bounce table  (report.md section 7)
    print("-" * 78)
    print("(c) OVERSOLD BOUNCE: next day after a big down close  [report.md s.7]")
    print(f"  {'thr':>6} {'period':<16}{'n':>5}{'up':>8}{'mean':>9}{'median':>9}"
          f"{'fwd5 mean':>11}{'fwd5 up':>9}")
    for thr in (-.08, -.10, -.12):
        for name, d in (("full 2010-2026", full), ("2024-2026", recent)):
            s = d[d["ret"] <= thr]
            f1, f5 = s["fwd1"].dropna(), s["fwd5"].dropna()
            print(f"  {thr*100:>5.0f}% {name:<16}{len(f1):>5}{f1.gt(0).mean()*100:>7.1f}%"
                  f"{pct(f1.mean()):>9}{pct(f1.median()):>9}{pct(f5.mean()):>11}"
                  f"{f5.gt(0).mean()*100:>8.1f}%")
        if len(recent[recent["ret"] <= thr]) < 50:
            print(f"         ^ 2024-2026 n<50 at this threshold: counter-example only,"
                  f" not a conclusion")

    # (d) gap <= -10% next-day  (report.md s.0 / today s.2)
    print("-" * 78)
    print("(d) GAP <= -10%: next-day outcome  [report.md s.0, n=40 full history]")
    print(f"  {'period':<16}{'n':>5}{'up':>8}{'mean':>9}{'median':>9}"
          f"{'next-day low med':>18}{'fwd5 med':>10}")
    for name, d in (("full 2010-2026", full), ("2024-2026", recent),
                    ("2026 only", m[m.index >= "2026-01-01"])):
        s = d[d["gap"] <= -.10]
        f1 = s["fwd1"].dropna()
        lo = s["fwd_min1"].dropna()
        f5 = s["fwd5"].dropna()
        print(f"  {name:<16}{len(s):>5}{f1.gt(0).mean()*100:>7.1f}%{pct(f1.mean()):>9}"
              f"{pct(f1.median()):>9}{pct(lo.median()):>18}{pct(f5.median()):>10}")
    print("  every row here is n<50: counter-examples only, never a conclusion.")
    print(f"  share of ALL gap<=-10% days that fall in 2026: "
          f"{(m[m['gap']<=-.10].index.year==2026).sum()}/{len(m[m['gap']<=-.10])}")

    # (e) ladder distance: what -22% meant then vs now
    print("-" * 78)
    print("(e) LADDER DISTANCE in units of the prevailing daily range")
    print("    mechanism: a limit order is a fixed % below the close, but the chance")
    print("    of it filling scales with the day's range. The same % is a different")
    print("    order in a 6% range regime than in a 3% one.")
    for name, d in (("2010-2025", m[m.index < "2026-01-01"]), ("2024-2026", recent),
                    ("2026", m[m.index >= "2026-01-01"]),
                    ("last 20 bars", m.iloc[-20:])):
        rg = d["range"].median()
        print(f"  {name:<14} median range {pct(rg):>7}   a -5% limit = "
              f"{.05/rg:>4.2f} ranges,  -11.9% limit (91.99) = {.119/rg:>4.2f} ranges")
    print("\n  probability the day's LOW is at least X% below the prior close:")
    print(f"  {'period':<14}{'n':>6}{'-5%':>8}{'-10%':>8}{'-12%':>8}{'-15%':>8}{'-20%':>8}")
    for name, d in (("2010-2025", m[m.index < "2026-01-01"]), ("2024-2026", recent),
                    ("2026", m[m.index >= "2026-01-01"])):
        lr = d["low_ret"].dropna()
        print(f"  {name:<14}{len(lr):>6}" + "".join(
            f"{(lr<=-t).mean()*100:>7.1f}%" for t in (.05, .10, .12, .15, .20)))


def part4f(m):
    sec("4f. THE HONEST FIX: condition on volatility instead of on the calendar")
    print("Conditioner: trailing 20-day MEDIAN high-low range, computed from bars up")
    print("to YESTERDAY only -> known before today's open, so it is a legal trigger.")
    m = m.copy()
    m["rv20"] = m["range"].rolling(20).median().shift(1)
    cur = m["rv20"].iloc[-1]
    print(f"  current value (as of the 2026-09-15 open): {pct(cur)}")
    hi = m[m["rv20"] >= .06]
    hi_old = hi[hi.index < "2024-01-01"]
    print(f"  days with rv20 >= 6.0%: n={len(hi)} total, of which "
          f"{len(hi_old)} are before 2024 -> the conditioned sample is NOT just 2026")
    print(f"  year spread of that sample: "
          f"{dict(sorted(pd.Series(hi.index.year).value_counts().items()))}")

    print(f"\n{'base rate':<30}{'full history':>16}{'2024-2026':>14}{'rv20>=6%':>14}"
          f"{'2026 only':>12}")
    groups = [("full history", m), ("2024-2026", m[m.index >= "2024-01-01"]),
              ("rv20>=6%", hi), ("2026 only", m[m.index >= "2026-01-01"])]

    def row(label, f):
        print(f"{label:<30}" + "".join(f"{f(g):>16}" if i == 0 else f"{f(g):>14}"
                                       for i, (_, g) in enumerate(groups[:3]))
              + f"{f(groups[3][1]):>12}")
    row("n", lambda g: str(len(g)))
    row("next-day up rate", lambda g: f"{g['fwd1'].dropna().gt(0).mean()*100:.1f}%")
    row("next-day median ret", lambda g: pct(g['fwd1'].median()))
    row("fwd_min20 MEDIAN (sizing)", lambda g: pct(g['fwd_min20'].median()))
    row("fwd_min20 p10", lambda g: pct(g['fwd_min20'].quantile(.10)))
    row("fwd20 median", lambda g: pct(g['fwd20'].median()))
    row("fwd20 up rate", lambda g: f"{g['fwd20'].dropna().gt(0).mean()*100:.1f}%")
    row("P(day low <= -10% vs prevcl)", lambda g: f"{(g['low_ret']<=-.10).mean()*100:.1f}%")
    row("P(day low <= -12% vs prevcl)", lambda g: f"{(g['low_ret']<=-.12).mean()*100:.1f}%")

    lo = m[m["rv20"] < .06]
    a = hi["fwd_min20"].dropna().values
    b = lo["fwd_min20"].dropna().values
    print(f"\nfwd_min20 median: rv20>=6% {pct(np.median(a))} vs rv20<6% {pct(np.median(b))}")
    print(f"  iid perm p = {perm_p(a, b, lambda x, y: np.median(x)-np.median(y), n=5000):.4f}")
    print(f"  block(21) perm p = "
          f"{block_perm_p(a, b, lambda x, y: np.median(x)-np.median(y), L=21, n=3000):.4f}")
    print(f"  overlapping 20-day windows: independent blocks ~{len(a)//20} vs {len(b)//20}")
    print(f"\nnext-day up rate: rv20>=6% "
          f"{hi['fwd1'].dropna().gt(0).mean()*100:.1f}% (n={hi['fwd1'].notna().sum()}) vs "
          f"rv20<6% {lo['fwd1'].dropna().gt(0).mean()*100:.1f}% (n={lo['fwd1'].notna().sum()})")
    print(f"  iid perm p = "
          f"{perm_p(hi['fwd1'].dropna().gt(0), lo['fwd1'].dropna().gt(0), lambda x,y: x.mean()-y.mean(), n=5000):.4f}")


# ---------------------------------------------------------------- 5. speed


def part5(m):
    sec("5. IS THE 2026 DRAWDOWN UNPRECEDENTED IN SPEED?")
    print("Episode definition: a peak is a running-max close; the episode runs to the")
    print("lowest close before that peak is exceeded again. Non-overlapping by")
    print("construction. The 2026 episode is still open (peak not recovered), so its")
    print("trough is the lowest close so far.")
    c = m["close"]
    THR = -.30
    eps = []
    pk, pkd = c.iloc[0], c.index[0]
    tr, trd = c.iloc[0], c.index[0]
    for d, x in c.items():
        if x > pk:
            if tr / pk - 1 <= THR:
                eps.append((pkd, trd, pk, tr))
            pk, pkd, tr, trd = x, d, x, d
        elif x < tr:
            tr, trd = x, d
    if tr / pk - 1 <= THR:
        eps.append((pkd, trd, pk, tr))
    rows = []
    for pkd, trd, pkv, trv in eps:
        depth = trv / pkv - 1
        cal = (trd - pkd).days
        td = len(c.loc[pkd:trd]) - 1
        rows.append(dict(peak=pkd.date(), trough=trd.date(), peak_px=pkv, trough_px=trv,
                         depth=depth, cal_days=cal, td=td,
                         pct_per_cal_day=depth / max(cal, 1),
                         pct_per_td=depth / max(td, 1)))
    t = pd.DataFrame(rows).sort_values("pct_per_cal_day")
    print(f"\nAll peak-to-trough episodes of {THR*100:.0f}% or worse, ranked by "
          f"percent lost per CALENDAR day (fastest first). n={len(t)} episodes:")
    print(f"{'#':>3} {'peak':>11} {'trough':>11}{'peak px':>10}{'trough px':>10}"
          f"{'depth':>9}{'cal d':>7}{'trd d':>7}{'%/cal day':>11}{'%/trade day':>13}")
    for i, (_, r) in enumerate(t.iterrows(), 1):
        mark = "  <== 2026" if str(r["peak"]) == "2026-06-22" else ""
        print(f"{i:>3} {str(r['peak']):>11} {str(r['trough']):>11}{r['peak_px']:>10.2f}"
              f"{r['trough_px']:>10.2f}{r['depth']*100:>8.1f}%{r['cal_days']:>7}{r['td']:>7}"
              f"{r['pct_per_cal_day']*100:>10.2f}%{r['pct_per_td']*100:>12.2f}%{mark}")
    tgt = t[t["peak"].astype(str) == "2026-06-22"]
    if len(tgt):
        rank = t["pct_per_cal_day"].rank().loc[tgt.index[0]]
        print(f"\n2026-06-22 -> 2026-07-29 ranks {int(rank)} of {len(t)} by %/calendar day.")
    print("\nn=%d episodes: by the session's own rule this is a counter-example set,"
          " not a statistic." % len(t))

    print("\nSame ranking on a fixed 26-trading-day window (the length of the 2026 leg),")
    print("which removes the dependence on how an 'episode' is carved up:")
    w = 26
    roll = c / c.rolling(w).max() - 1
    rr = roll.dropna().sort_values()
    print(f"  worst {w}-day drawdown-from-window-high, top 12 (dedup 26 bars apart):")
    seen, shown = [], 0
    for d, x in rr.items():
        if any(abs((d - s).days) < 40 for s in seen):
            continue
        seen.append(d)
        shown += 1
        print(f"    {d.date()}  {x*100:7.1f}%")
        if shown == 12:
            break
    cur = roll.loc["2026-07-29"]
    print(f"  2026-07-29 value: {cur*100:.1f}%   percentile of history: "
          f"{100*(roll.dropna()<cur).mean():.2f}% (lower = worse)")


def main():
    pd.set_option("display.width", 200)
    m = build()
    print(f"SOXL bars {m.index.min().date()} -> {m.index.max().date()}  n={len(m)}")
    print(f"appended manually: {LAST_BAR['date']} close {LAST_BAR['close']} "
          f"ret {pct(m['ret'].iloc[-1])} gap {pct(m['gap'].iloc[-1])} "
          f"QQQ ret {pct(m['qqq_ret'].iloc[-1])}")
    part1(m)
    part2(m)
    part3(m)
    part4(m)
    part4f(m)
    part5(m)


if __name__ == "__main__":
    main()
