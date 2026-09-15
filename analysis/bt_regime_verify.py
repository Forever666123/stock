"""Adversarial verification of analysis/bt_regime.py.

Mechanisms stated BEFORE the numbers:

V1 TRUNCATED FORWARD WINDOWS. fwd_min20 is built with min_periods=1 on a
   reversed rolling min, so the last 19 rows of the frame get forward windows of
   length 19,18,...,1 but are NOT NaN. They are pooled with genuine 20-day
   windows. Those 19 rows are ~11% of the 2026 sample and 0.5% of full history,
   so the bias is asymmetric between the two groups being compared. A shorter
   window can only produce a SHALLOWER minimum, so the 2026 median should be
   biased upward (less negative). Recompute with min_periods=h.

V2 WRONG BASELINE FOR THE SIZING NUMBER. fwd_min20 is measured from a CLOSE.
   The plan does not enter at a close; it enters on resting limits at 96 and 92,
   i.e. 5.07% and 9.03% below the reference close. An order that fills only
   because price fell already has the fall behind it. Applying a from-close
   excursion distribution to a from-limit-fill entry double counts the drop.
   Simulate the actual ladder instead.

V3 CLOSE vs INTRADAY UNITS. fwd_min20 compares an intraday LOW to a CLOSE. That
   is right for an intraday mark-to-market tolerance and wrong for a tolerance
   measured on closes. Report both so the reader knows which -25% is meant.

V4 NESTED SAMPLES IN A TWO-SAMPLE TEST. part4 sets full = m, which CONTAINS the
   2024-2026 and 2026 rows it is compared against. A permutation test on nested
   samples is biased toward p -> 1. Redo disjointly (2010-2023 vs 2024-2026).

V5 MIXTURE ARTIFACT IN THE SHAPE TEST. Pooled 2010-2025 kurtosis of 5.27 is not
   a shape property; pooling subperiods whose vol ranges 52%-131% mechanically
   fattens the pooled tail. Same for standardising the whole 16 years by ONE
   median-abs-deviation. Redo standardising WITHIN each year, and run the
   standardised KS against the matched-vol years (2020, 2022) rather than
   against the mixture.

V6 OFF-BY-ONE IN THE CONDITIONER. rv20 = range.rolling(20).median().shift(1).
   m['rv20'].iloc[-1] is the value that was known before the 2026-09-14 OPEN,
   not before the 2026-09-15 open. The value available for tomorrow's decision
   is the UNSHIFTED rolling median at the last bar.

V7 HORIZON MISMATCH IN THE FILL RATES. P(day low <= -X% of PRIOR close) is a
   one-day statistic. The order rests for 20 days against a FIXED price. The
   decision-relevant number is P(min low over the next 20 days <= the limit),
   measured from today's close, which is far larger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.bt_regime import build, perm_p, block_perm_p, ks_stat, exkurt, ann, sec, pct

REF_CLOSE = 101.13
L1, L2 = 96.0, 92.0
R1, R2 = L1 / REF_CLOSE - 1, L2 / REF_CLOSE - 1   # -5.07%, -9.03%


def strict_fwd(m, h=20):
    """fwd_min{h} with a FULL window only; truncated tails become NaN."""
    low, c = m["low"], m["close"]
    fm = low[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1) / c - 1
    fmc = c[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1) / c - 1
    return fm, fmc


def v1_v3(m):
    sec("V1/V3. fwd_min20: truncated windows, and close-based vs intraday-low")
    m = m.copy()
    m["fm20_strict"], m["fmc20_strict"] = strict_fwd(m, 20)
    grp = [("full 2010-2026", m), ("2010-2023 (disjoint)", m[m.index < "2024-01-01"]),
           ("2024-2026", m[m.index >= "2024-01-01"]), ("2026 only", m[m.index >= "2026-01-01"])]
    print(f"{'sample':<22}{'n_theirs':>9}{'THEIR med':>11}{'n_strict':>9}"
          f"{'STRICT med':>12}{'delta':>8}{'CLOSE-based med':>17}")
    for nm, g in grp:
        a = g["fwd_min20"].dropna()
        b = g["fm20_strict"].dropna()
        cc = g["fmc20_strict"].dropna()
        print(f"{nm:<22}{len(a):>9}{pct(a.median()):>11}{len(b):>9}{pct(b.median()):>12}"
              f"{(b.median()-a.median())*100:>7.2f}{pct(cc.median()):>17}")
    print("\n  A truncated window can only be shallower, so their 2026 median is")
    print("  biased toward zero. Direction of the bug FAVOURS their conclusion.")
    print("  The close-based column is the one to use if the -25% tolerance is a")
    print("  closing-basis tolerance rather than an intraday mark.")
    a26 = m.loc[m.index >= "2026-01-01", "fm20_strict"].dropna()
    afull = m["fm20_strict"].dropna()
    print(f"\n  STRICT ratio 2026/full = {a26.median()/afull.median():.2f}x  "
          f"(they claimed 1.91x)")


def v2_ladder(m):
    sec("V2. INDEPENDENT RE-DERIVATION: simulate the actual 96/92 ladder")
    print("Different route entirely. For every historical day t, rescale so that")
    print(f"close_t = {REF_CLOSE}. Rest limits at {L1} ({R1*100:.2f}%) and {L2} "
          f"({R2*100:.2f}%),")
    print("half the position each, for the next 20 sessions. Fill a rung when the")
    print("day's LOW touches it, at the limit price, or at the OPEN if the bar")
    print("gapped straight through it. Then track the worst intraday mark against")
    print("the RUNNING average cost. No stop. This never compares a low to a close.")
    lo = m["low"].values
    op = m["open"].values
    cl = m["close"].values
    hi = m["high"].values
    idx = m.index
    n = len(m)
    H = 20
    out = []
    for t in range(n - H):
        base = cl[t]
        l1, l2 = base * (1 + R1), base * (1 + R2)
        sh1 = sh2 = 0.0
        cost = 0.0
        worst = 0.0
        filled_any = False
        for d in range(t + 1, t + 1 + H):
            # fills happen at the open first, then intraday
            if sh1 == 0.0 and (op[d] <= l1 or lo[d] <= l1):
                px = min(l1, op[d])
                sh1, cost = 0.5, cost + 0.5 * px
                filled_any = True
            if sh2 == 0.0 and (op[d] <= l2 or lo[d] <= l2):
                px = min(l2, op[d])
                sh2, cost = 0.5, cost + 0.5 * px
                filled_any = True
            sh = sh1 + sh2
            if sh > 0:
                avg = cost / sh
                dd = lo[d] / avg - 1
                if dd < worst:
                    worst = dd
        out.append(dict(date=idx[t], filled=filled_any, both=(sh1 > 0 and sh2 > 0),
                        only1=(sh1 > 0 and sh2 == 0),
                        worst=worst if filled_any else np.nan,
                        avgfill=(cost / (sh1 + sh2) / base) if filled_any else np.nan,
                        end=(cl[t + H] / (cost / (sh1 + sh2)) - 1) if filled_any else np.nan))
    r = pd.DataFrame(out).set_index("date")
    grp = [("full 2010-2026", r), ("2010-2023", r[r.index < "2024-01-01"]),
           ("2024-2026", r[r.index >= "2024-01-01"]), ("2026 only", r[r.index >= "2026-01-01"])]
    print(f"\n{'sample':<18}{'n':>6}{'P(any fill)':>12}{'P(both)':>9}"
          f"{'worst-MTM med':>15}{'mean':>9}{'p10':>9}{'p05':>9}{'P(dd<=-25%)':>13}")
    for nm, g in grp:
        w = g["worst"].dropna()
        if len(w) == 0:
            continue
        print(f"{nm:<18}{len(g):>6}{g['filled'].mean()*100:>11.1f}%"
              f"{g['both'].mean()*100:>8.1f}%{pct(w.median()):>15}{pct(w.mean()):>9}"
              f"{pct(w.quantile(.10)):>9}{pct(w.quantile(.05)):>9}"
              f"{(w<=-.25).mean()*100:>12.1f}%")
    print("\n  Same table conditioned on the observable trailing state (rv20 legal):")
    rv = m["range"].rolling(20).median().shift(1)
    rvr = rv.reindex(r.index)
    cur_legal = m["range"].rolling(20).median().iloc[-1]
    band = r[(rvr >= cur_legal - .01) & (rvr <= cur_legal + .01)]
    for nm, g in (("rv20 in band, 2010-2025", band[band.index < "2026-01-01"]),
                  ("rv20 in band, 2026", band[band.index >= "2026-01-01"])):
        w = g["worst"].dropna()
        if len(w) == 0:
            continue
        flag = "" if len(g) >= 50 else "  <- n<50, counter-example only"
        print(f"  {nm:<26}n={len(g):>5}  P(fill) {g['filled'].mean()*100:5.1f}%  "
              f"med {pct(w.median()):>8}  p10 {pct(w.quantile(.10)):>8}  "
              f"P(<=-25%) {(w<=-.25).mean()*100:5.1f}%{flag}")
    print("\n  Their headline maps -23.65% (from-close excursion) onto an average")
    print("  fill of 94 and concludes -25% is the median 2026 outcome. Compare that")
    print("  with the P(dd<=-25%) column above, which is the same question asked")
    print("  from the actual fill price.")
    w26 = r[r.index >= "2026-01-01"]["worst"].dropna()
    wfull = r["worst"].dropna()
    if len(w26) and len(wfull):
        print(f"\n  ladder-based ratio 2026/full = {w26.median()/wfull.median():.2f}x")
        print(f"  perm p on the median difference (overlapping 20d windows, so this")
        print(f"  p is an UPPER BOUND on evidence, not a result): "
              f"{perm_p(w26.values, wfull.values, lambda x,y: np.median(x)-np.median(y), n=5000):.4f}")
    return r


def v4_direction(m):
    sec("V4. DIRECTION TRANSFER, with DISJOINT samples (their test was nested)")
    f = m["fwd1"]
    a = f[m.index >= "2024-01-01"].dropna()
    b = f[m.index < "2024-01-01"].dropna()
    c26 = f[m.index >= "2026-01-01"].dropna()
    b25 = f[m.index < "2026-01-01"].dropna()
    print(f"  2010-2023 up {b.gt(0).mean()*100:.1f}% (n={len(b)})   "
          f"2024-2026 up {a.gt(0).mean()*100:.1f}% (n={len(a)})")
    print(f"    disjoint iid perm p = "
          f"{perm_p(a.gt(0), b.gt(0), lambda x,y: x.mean()-y.mean(), n=20000):.4f}"
          f"   (theirs, nested: 0.6143)")
    print(f"    disjoint block(21) p = "
          f"{block_perm_p(a.gt(0), b.gt(0), lambda x,y: x.mean()-y.mean(), L=21, n=5000):.4f}")
    print(f"\n  2010-2025 up {b25.gt(0).mean()*100:.1f}% (n={len(b25)})   "
          f"2026 up {c26.gt(0).mean()*100:.1f}% (n={len(c26)})  "
          f"gap {(c26.gt(0).mean()-b25.gt(0).mean())*100:+.1f}pp")
    print(f"    iid perm p = "
          f"{perm_p(c26.gt(0), b25.gt(0), lambda x,y: x.mean()-y.mean(), n=20000):.4f}")
    print(f"    block(21) p = "
          f"{block_perm_p(c26.gt(0), b25.gt(0), lambda x,y: x.mean()-y.mean(), L=21, n=5000):.4f}")
    se = np.sqrt(.25 / len(a))
    print(f"\n  POWER: with n={len(a)} the 1-sigma band on a 2024-26 up rate is "
          f"{se*100:.1f}pp, so the")
    print(f"  test cannot see anything smaller than about {2*se*100:.1f}pp. "
          f"'No difference' here means")
    print(f"  'no difference larger than ~{2*se*100:.0f}pp', not 'the same number'.")


def v5_shape(m):
    sec("V5. SHAPE TEST REDONE: pooled kurtosis and one-MAD scaling are mixtures")
    m = m.copy()
    m["yr"] = m.index.year
    r = m["ret"].dropna()
    ky = m.groupby("yr")["ret"].apply(lambda x: exkurt(x.dropna()))
    old_y = ky[ky.index < 2026]
    print(f"  pooled 2010-2025 excess kurtosis        : {exkurt(r[r.index<'2026-01-01']):.2f}"
          f"   <- what they compared against")
    print(f"  MEDIAN of the 16 per-YEAR kurtoses      : {old_y.median():.2f}")
    print(f"  mean of the 16 per-year kurtoses        : {old_y.mean():.2f}"
          f"  (dragged by 2025={ky.loc[2025]:.1f}, 2014={ky.loc[2014]:.1f})")
    print(f"  2026 excess kurtosis                    : {ky.loc[2026]:.2f}")
    print(f"  2026 rank among the 17 yearly values    : "
          f"{int(ky.rank().loc[2026])} of {len(ky)} (1 = thinnest)")
    print("  => the 5.27 they quote is a REGIME-MIXTURE artifact, not a shape.")
    print("     2026 is thin-tailed for a year, but not an outlier among years.")

    # standardise WITHIN year, then pool
    z = m.groupby("yr")["ret"].transform(lambda x: x / np.median(np.abs(x.dropna())))
    za = z[m.index >= "2026-01-01"].dropna().values
    zb = z[m.index < "2026-01-01"].dropna().values
    print(f"\n  KS, returns standardised WITHIN EACH YEAR (kills the mixture):")
    print(f"    D = {ks_stat(za,zb):.4f}   iid perm p = {perm_p(za,zb,ks_stat,n=20000):.4f}"
          f"   block(21) p = {block_perm_p(za,zb,ks_stat,L=21,n=5000):.4f}")
    print(f"    (theirs, one MAD for all 16 years: D=0.0895 p=0.1257)")

    print("\n  Standardised KS against the MATCHED-VOL years instead of the mixture:")
    for yr in (2020, 2022, 2024, 2025):
        x = m.loc[m.index.year == yr, "ret"].dropna().values
        sx = x / np.median(np.abs(x))
        y = m.loc[m.index >= "2026-01-01", "ret"].dropna().values
        sy = y / np.median(np.abs(y))
        print(f"    2026 vs {yr}: D={ks_stat(sy,sx):.4f}  iid p={perm_p(sy,sx,ks_stat,n=20000):.4f}"
              f"  block(21) p={block_perm_p(sy,sx,ks_stat,L=21,n=5000):.4f}"
              f"   (n={len(x)})")
    hv = m[(m.index.year.isin([2020, 2022, 2024, 2025]))]["ret"].dropna()
    zhv = m[(m.index.year.isin([2020, 2022, 2024, 2025]))].groupby(
        m[(m.index.year.isin([2020, 2022, 2024, 2025]))].index.year)["ret"].transform(
        lambda x: x / np.median(np.abs(x.dropna()))).dropna().values
    y = m.loc[m.index >= "2026-01-01", "ret"].dropna().values
    sy = y / np.median(np.abs(y))
    print(f"    2026 vs 2020+2022+2024+2025 standardised within year: "
          f"D={ks_stat(sy,zhv):.4f} iid p={perm_p(sy,zhv,ks_stat,n=20000):.4f} "
          f"block(21) p={block_perm_p(sy,zhv,ks_stat,L=21,n=5000):.4f} (n={len(zhv)})")


def v6_conditioner(m):
    sec("V6. OFF-BY-ONE: which rv20 is actually known at the 2026-09-15 open?")
    rv_shift = m["range"].rolling(20).median().shift(1)
    rv_now = m["range"].rolling(20).median()
    print(f"  their value, m['rv20'].iloc[-1]  = {pct(rv_shift.iloc[-1])}  "
          f"(bars {m.index[-21].date()} .. {m.index[-2].date()})")
    print(f"  value known at the 9/15 open     = {pct(rv_now.iloc[-1])}  "
          f"(bars {m.index[-20].date()} .. {m.index[-1].date()})")
    print(f"  2026-09-14 own range = {pct(m['range'].iloc[-1])} of the PRIOR close "
          f"(a -17% day whose high-low span was narrow because the gap did the work)")
    cur = rv_now.iloc[-1]
    old = rv_shift[rv_shift.index < "2026-01-01"].dropna()
    new = rv_shift[rv_shift.index >= "2026-01-01"].dropna()
    print(f"  percentile of the CORRECT value in 2010-2025 = {100*(old<cur).mean():.1f}% "
          f"(they said 63.6%)")
    print(f"  percentile of the CORRECT value within 2026  = {100*(new<cur).mean():.1f}% "
          f"(they said 11.4%)")
    print("  The brief's own '6.0% trailing 20-day median range' matches the")
    print("  unshifted value, not theirs; the direction of the error is small here")
    print("  but the conditioner they tested is one day stale.")


def v7_fillrate(m):
    sec("V7. FILL RATES AT THE RIGHT HORIZON (theirs are one-day rates)")
    print(f"  The order rests for ~20 sessions against FIXED prices "
          f"{L1} ({R1*100:.2f}%) and {L2} ({R2*100:.2f}%).")
    fm = {}
    for h in (1, 5, 10, 20):
        fm[h] = m["low"][::-1].rolling(h, min_periods=h).min()[::-1].shift(-1) / m["close"] - 1
    print(f"\n{'sample':<16}{'n':>6}" + "".join(f"{f'<= {R1*100:.1f}% in {h}d':>17}" for h in (1, 5, 20)))
    for nm, msk in (("2010-2025", m.index < "2026-01-01"), ("2024-2026", m.index >= "2024-01-01"),
                    ("2026 only", m.index >= "2026-01-01")):
        row = f"{nm:<16}{int(msk.sum()):>6}"
        for h in (1, 5, 20):
            s = fm[h][msk].dropna()
            row += f"{(s<=R1).mean()*100:>16.1f}%"
        print(row)
    print(f"\n{'sample':<16}{'n':>6}" + "".join(f"{f'<= {R2*100:.1f}% in {h}d':>17}" for h in (1, 5, 20)))
    for nm, msk in (("2010-2025", m.index < "2026-01-01"), ("2024-2026", m.index >= "2024-01-01"),
                    ("2026 only", m.index >= "2026-01-01")):
        row = f"{nm:<16}{int(msk.sum()):>6}"
        for h in (1, 5, 20):
            s = fm[h][msk].dropna()
            row += f"{(s<=R2).mean()*100:>16.1f}%"
        print(row)
    print("\n  Their table reads the 92 rung off the '-10%' row, but 92 is -9.03%,")
    print("  not -10%, so even their one-day number understates that rung.")
    print("  At the real horizon both rungs fill in a large majority of 2026 starts,")
    print("  which is the opposite of a tail event and is the number the ladder EV")
    print("  should be built on.")


def v8_misc(m):
    sec("V8. ASSORTED CHECKS ON THE REMAINING CLAIMS")
    # claim 5: rv20>=6 year spread mis-stated
    rv = m["range"].rolling(20).median().shift(1)
    hi = m[rv >= .06]
    yrs = sorted(set(hi.index.year))
    print(f"  claim 5 says the rv20>=6% sample is 'spread across 2010, 2011, 2018-2026'.")
    print(f"  actual years present: {yrs}")
    print(f"  share of that sample that is 2024-2026: "
          f"{(hi.index.year>=2024).mean()*100:.1f}%  (n={len(hi)})")
    # claim 6: 'full history' contains 2026
    f_all = m["fwd_min20"].dropna()
    f_dis = m.loc[m.index < "2026-01-01", "fwd_min20"].dropna()
    print(f"\n  claim 6 'full history' median {pct(f_all.median())} INCLUDES 2026.")
    print(f"  excluding 2026 it is {pct(f_dis.median())} -> the gap they report is")
    print(f"  understated, not overstated. Ratio 2026/(2010-2025) = "
          f"{m.loc[m.index>='2026-01-01','fwd_min20'].median()/f_dis.median():.2f}x")
    # claim 9: episode threshold sensitivity
    print("\n  claim 9 rank sensitivity to the -30% episode threshold:")
    c = m["close"]
    for THR in (-.20, -.25, -.30, -.35, -.40, -.50):
        eps = []
        pk, pkd, tr, trd = c.iloc[0], c.index[0], c.iloc[0], c.index[0]
        for d, x in c.items():
            if x > pk:
                if tr / pk - 1 <= THR:
                    eps.append((pkd, trd, pk, tr))
                pk, pkd, tr, trd = x, d, x, d
            elif x < tr:
                tr, trd = x, d
        if tr / pk - 1 <= THR:
            eps.append((pkd, trd, pk, tr))
        rows = [dict(peak=str(p.date()), spd=(t / pv - 1) / max((td_ - p).days, 1))
                for p, td_, pv, t in eps]
        t_ = pd.DataFrame(rows).sort_values("spd")
        tg = t_[t_["peak"] == "2026-06-22"]
        rk = int(t_["spd"].rank().loc[tg.index[0]]) if len(tg) else -1
        print(f"    threshold {THR*100:>4.0f}%: n={len(t_):>3} episodes, "
              f"2026-06-22 ranks {rk if rk>0 else 'n/a':>4} "
              f"({rk/len(t_)*100 if rk>0 else float('nan'):.0f}th pct)")
    print("  The '6th of 14' headline moves with an arbitrary cutoff; report the")
    print("  percentile, not the rank.")
    # claim 3: does the 9/14 bar change the SOXL/SOXX story
    print("\n  claim 3: SOXL 2026 vol used in the index comparison EXCLUDES 2026-09-14")
    a_all = m.loc[m.index >= "2026-01-01", "ret"].dropna()
    a_ex = a_all[a_all.index < "2026-09-14"]
    print(f"    SOXL 2026 ann vol incl 9/14 {ann(a_all)*100:.1f}%  excl 9/14 "
          f"{ann(a_ex)*100:.1f}%  -> the single worst day adds "
          f"{(ann(a_all)/ann(a_ex)-1)*100:.1f}%")
    print(f"    so 'SOXL vol rise 1.41x vs SOXX 1.42x' is computed on a 2026 that is")
    print(f"    missing its most extreme observation. With 9/14 the SOXL rise is "
          f"{ann(a_all)/ann(m.loc[(m.index<'2026-01-01')&(m.index>='2018-01-01'),'ret'].dropna())*100/100:.2f}x")
    print(f"    vs SOXX 1.42x, still not a leverage-feedback signature.")


def main():
    pd.set_option("display.width", 200)
    m = build()
    print(f"SOXL bars {m.index.min().date()} -> {m.index.max().date()}  n={len(m)}")
    v1_v3(m)
    v2_ladder(m)
    v4_direction(m)
    v5_shape(m)
    v6_conditioner(m)
    v7_fillrate(m)
    v8_misc(m)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- V9


def v9_headline(m):
    """The headline arithmetic, checked, plus robustness of the ladder number."""
    sec("V9. THE HEADLINE SENTENCE, CHECKED ARITHMETICALLY")
    ref, fill = REF_CLOSE, 94.0
    exc = -0.2365
    print(f"  Their sentence: '-25% from an average fill near 94 is 70.5, while the")
    print(f"  2026 median 20-day excursion of {exc*100:.2f}% lands at 71.8.'")
    print(f"    94 * 0.75                         = {fill*0.75:.2f}   (their 70.5, correct)")
    print(f"    94 * (1{exc:+.4f})                    = {fill*(1+exc):.2f}   (their 71.8)")
    print(f"  But fwd_min20 is measured FROM A CLOSE, not from a fill. The close is")
    print(f"  {ref}, so a {exc*100:.2f}% excursion lands at {ref*(1+exc):.2f}, not {fill*(1+exc):.2f}.")
    print(f"    {ref*(1+exc):.2f} from a {fill:.0f} fill = {(ref*(1+exc)/fill-1)*100:.2f}%, "
          f"not {exc*100:.2f}%.")
    print(f"  Applying a from-close percentage to a fill price double counts the")
    print(f"  {(fill/ref-1)*100:.1f}% the position already waited out. That is the same")
    print(f"  mismatched-baseline error class the session was told to hunt for.")

    sec("V9b. ROBUSTNESS OF THE LADDER NUMBER (overlapping starts)")
    lo, op, cl = m["low"].values, m["open"].values, m["close"].values
    idx, n, H = m.index, len(m), 20
    rec = []
    for t in range(n - H):
        base = cl[t]
        l1, l2 = base * (1 + R1), base * (1 + R2)
        sh1 = sh2 = 0.0
        cost = 0.0
        worst = 0.0
        filled = False
        for d in range(t + 1, t + 1 + H):
            if sh1 == 0.0 and (op[d] <= l1 or lo[d] <= l1):
                sh1, cost, filled = 0.5, cost + 0.5 * min(l1, op[d]), True
            if sh2 == 0.0 and (op[d] <= l2 or lo[d] <= l2):
                sh2, cost, filled = 0.5, cost + 0.5 * min(l2, op[d]), True
            if sh1 + sh2 > 0:
                worst = min(worst, lo[d] / (cost / (sh1 + sh2)) - 1)
        rec.append((idx[t], filled, worst if filled else np.nan))
    r = pd.DataFrame(rec, columns=["date", "filled", "worst"]).set_index("date")
    w26 = r[r.index >= "2026-01-01"]["worst"].dropna()
    print(f"  2026 filled starts n={len(w26)} (but only ~{len(w26)//21} independent")
    print(f"  monthly blocks; 20-day windows overlap ~20-fold).")
    print(f"    median {pct(w26.median())}   P(<=-25%) {(w26<=-.25).mean()*100:.1f}%")
    print("  Non-overlapping starts only (every 20th bar), 2026:")
    for off in range(0, 20, 4):
        s = r[r.index >= "2026-01-01"].iloc[off::20]["worst"].dropna()
        print(f"    offset {off:>2}: n={len(s):>2}  median {pct(s.median()):>8}  "
              f"P(<=-25%) {(s<=-.25).mean()*100:>5.1f}%   <- n<50, counter-example only")
    print("  Month-by-month 2026 (median worst-MTM of starts in that month):")
    g = w26.groupby([w26.index.year, w26.index.month])
    for k, v in g:
        print(f"    {k[0]}-{k[1]:02d}  n={len(v):>2}  median {pct(v.median()):>8}  "
              f"P(<=-25%) {(v<=-.25).mean()*100:>5.1f}%")
    print("\n  The 2026 ladder drawdown is NOT a stable property of the year: it is")
    print("  concentrated in starts that sit just before the June-July leg. Starts")
    print("  from the most recent months look different. Report the spread.")


if __name__ != "__main__":
    pass
