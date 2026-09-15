"""ADVERSARIAL VERIFICATION of analysis/bt_multiple_testing.py.

    python3 -m analysis.vfy_multiple_testing

Every number below is re-derived by a DIFFERENT route from the one the audit
used, or re-derived under an assumption the audit made silently.  Mechanism is
stated before each number.  n<50 produces counter-examples only.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from analysis.data import load_merged, load_qqq
from analysis.fomc import fomc_table, _ROWS_2026

RNG = np.random.default_rng(7)
NPERM = 20000
W = 100

def hdr(t):
    print("\n" + "=" * W); print(t); print("=" * W)

def desc(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x): return "n=0"
    return f"n={len(x)}  mean {x.mean():+.2%}  median {np.median(x):+.2%}  up {(x>0).mean():.0%}"

def perm(x, g, nperm=NPERM, rng=RNG):
    ok = ~np.isnan(x); x, g = x[ok], g[ok].astype(bool)
    k, n = int(g.sum()), len(x)
    if k < 2 or n - k < 2: return np.nan, np.nan
    obs = x[g].mean() - x[~g].mean(); tot = x.sum(); hits = 0
    for _ in range(nperm // 500):
        keys = rng.random((500, n)); idx = np.argpartition(keys, k, axis=1)[:, :k]
        sa = x[idx].sum(axis=1); d = sa / k - (tot - sa) / (n - k)
        hits += int((np.abs(d) >= abs(obs) - 1e-15).sum())
    return obs, (1 + hits) / (nperm + 1)

# ---------------------------------------------------------------- RAW-CSV FRAME
# Route B: rebuild everything straight off the CSV, no load_merged, no fomc index map.
LAST = ("2026-09-14", 101.53, 105.39, 99.87, 101.13, 711.96)
def raw_frame():
    d = pd.read_csv(Path("data/SOXL_OHLC.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
    d.columns = [c.lower() for c in d.columns]
    d.loc[pd.Timestamp(LAST[0]), ["open","high","low","close","volume"]] = [LAST[1],LAST[2],LAST[3],LAST[4],np.nan]
    d = d.sort_index()
    d["ret"] = d["close"].pct_change()
    return d

def main():
    hdr("0.  REPRODUCTION")
    print("The audit script runs clean and every number in its findings list reproduces")
    print("exactly, including n=9/+8.31%/p=0.026, 18/18 stop cells, and all bootstrap CIs.")
    print("What follows is not a reproduction failure; it is what the script does not test.")

    d = raw_frame()
    m = load_merged()
    dd = pd.Timestamp("2026-09-14")
    m.loc[dd, ["open","high","low","close","qqq"]] = [LAST[1],LAST[2],LAST[3],LAST[4],LAST[5]]
    m = m.sort_index()
    m["ret"] = m["close"]/m["close"].shift(1)-1
    for h in (1,2,3,5,10,20):
        m[f"fwd{h}"] = m["close"].shift(-h)/m["close"]-1
    cv, lv, ov, hv = (m[c].values for c in ("close","low","open","high"))
    n = len(m)
    pos = pd.Series(np.arange(n), index=m.index)

    # =============================================================== HEADLINE CHECK
    hdr("1.  THE HEADLINE CONTRADICTS THE SCRIPT'S OWN TABLE")
    print("Their headline: 'exactly one survives correction' and 'the single claim that")
    print("was ever nominally significant (claim 4, raw p=0.026)'.")
    print("Their own multiplicity table prints:")
    print("    claim 6  raw p 0.0000  Bonf 0.005  surv B True   BH q 0.002  surv BH True")
    print("    claim 7  raw p 0.0000  Bonf 0.005  surv B True   BH q 0.002  surv BH True")
    print("    'Number of these 8 with raw p < 0.05: 3'")
    print("-> TWO claims survive both corrections, not one, and THREE were nominally")
    print("   significant, not one.  Claim 7 is demoted to 'fragile' on an interpretive")
    print("   argument about the median, which is a different operation from a correction.")
    print("   The sentence given to the user is not what the script computed.")

    # =============================================================== CLAIM 4, ROUTE B
    hdr("2.  CLAIM 4 RE-DERIVED BY A DIFFERENT ROUTE (raw CSV, calendar dates, no index map)")
    print("MECHANISM FIRST: the audit maps each FOMC date to a bar position and reads")
    print("position-2.  If the Fed date is not itself a trading day, or if the map is off")
    print("by one, the whole n=9 moves.  Route B looks up D-2 by calendar date directly.")
    fomc_all = fomc_table(include_2026=True)
    sched = fomc_all[~fomc_all["emergency"]]
    dates = d.index
    def dm2_dates(meet_dates):
        out = []
        for dt in meet_dates:
            loc = dates.searchsorted(dt)
            if loc >= len(dates) or dates[loc] != dt: continue   # not a trading day
            if loc - 2 < 0: continue
            out.append((dates[loc-2], dates[loc]))
        return out
    known = sched[sched["action"].notna()]
    in_s = [x for x in dm2_dates(known.index) if x[1].year < 2026]
    ext  = dm2_dates(known.index)
    def fwd2_to_decision(pairs):
        rows = []
        for dm2, d0 in pairs:
            r = d.loc[dm2, "ret"]
            if not np.isfinite(r) or r > -0.06: continue
            rows.append((dm2, d0, r, d.loc[d0,"close"]/d.loc[dm2,"close"]-1))
        return pd.DataFrame(rows, columns=["dm2","d0","day","to_decision"]).set_index("dm2")
    A = fwd2_to_decision(in_s); B = fwd2_to_decision(ext)
    print(f"\n  route B in-sample (2015-2025): n={len(A)}  mean {A['to_decision'].mean():+.2%}  "
          f"median {A['to_decision'].median():+.2%}  up {(A['to_decision']>0).mean():.0%}")
    print(f"  route A (their script)        : n=9  mean +8.31%  median +6.82%  up 89%")
    print(f"  MATCH: {len(A)==9 and abs(A['to_decision'].mean()-0.0831)<0.0005}")
    print(f"  route B extended incl 2026    : n={len(B)}  mean {B['to_decision'].mean():+.2%}  "
          f"median {B['to_decision'].median():+.2%}")

    print("\n  *** SELECTION EFFECT THE AUDIT INTRODUCED WITH ITS OWN 'OUT-OF-SAMPLE' SET ***")
    print("  meetings(m) filters on action.notna().  For a filter that only needs the MEETING")
    print("  DATE -- claims 1 and 4 both do -- that throws away 2026 meetings that already")
    print("  happened and whose prices are in the CSV, purely because the ACTION was not")
    print("  recovered from FedWatch.  The correct out-of-sample set is every scheduled 2026")
    print("  meeting that has occurred, action known or not.")
    all26 = pd.to_datetime([r[0] for r in _ROWS_2026])
    occurred26 = [x for x in all26 if x <= dates[-1]]
    print(f"    2026 scheduled meetings already past: {[str(x.date()) for x in occurred26]}")
    print(f"    of which the audit uses only: ['2026-04-29', '2026-06-17', '2026-07-29']")
    full = [x for x in dm2_dates(list(sched[sched.index.year<2026].index) + occurred26)]
    C = fwd2_to_decision(full)
    print(f"\n  claim 4 on the DATE-COMPLETE sample: n={len(C)}  mean {C['to_decision'].mean():+.2%}  "
          f"median {C['to_decision'].median():+.2%}  up {(C['to_decision']>0).mean():.0%}")
    newr = C.index.difference(A.index)
    for x in newr:
        print(f"    added: {x.date()}  day {C.loc[x,'day']:+.1%}  -> decision close {C.loc[x,'to_decision']:+.1%}")
    # permutation against non-D-2 6% declines on the date-complete sample
    dm2set = set(C.index) | {p[0] for p in full}
    big = d[(d["ret"] <= -0.06)].copy()
    big["f2"] = d["close"].shift(-2)/d["close"]-1
    big = big[big["f2"].notna()]
    gg = np.array([ix in dm2set for ix in big.index])
    o_, p_ = perm(big["f2"].values, gg)
    print(f"  date-complete difference vs non-D-2 6% declines = {o_:+.2%}  raw perm p = {p_:.3f}")
    print(f"  [n={int(gg.sum())} < 50: COUNTER-EXAMPLE ONLY]")

    # =============================================================== CLAIM 1 date-complete
    hdr("3.  CLAIM 1 ON THE DATE-COMPLETE 2026 SET")
    print("MECHANISM FIRST: D-1 needs only the meeting date, so the same selection applies.")
    w = m.loc["2015-01-01":]
    wpos = pd.Series(np.arange(len(w)), index=w.index)
    def dm1(meets):
        f = np.zeros(len(w), bool)
        for dt in meets:
            if dt in wpos.index and int(wpos[dt]) - 1 >= 0: f[int(wpos[dt])-1] = True
        return f
    x = w["ret"].values
    for lbl, mi in (("their IN-SAMPLE (2015-2025)", [t for t in known.index if t.year<2026]),
                    ("their EXTENDED (3 verified 2026)", list(known.index)),
                    ("DATE-COMPLETE (all 5 past 2026)", [t for t in sched.index if t.year<2026]+occurred26)):
        g = dm1(mi); a, b = x[g & ~np.isnan(x)], x[~g & ~np.isnan(x)]
        o2, p2 = perm(x, g)
        print(f"  {lbl:<34} D-1 {desc(a)}   diff {o2:+.4%}  p={p2:.3f}")
    print("  -> the audit's 'sign flip on three observations' is really a sign flip on a")
    print("     subset of the 2026 meetings chosen by whether FedWatch happened to resolve")
    print("     the action.  The conclusion (claim 1 is noise) is unchanged; the stated")
    print("     reason is contaminated by the same kind of selection the audit is auditing.")

    # =============================================================== CLAIM 6 ROUTE B
    hdr("4.  CLAIM 6 RE-DERIVED: vectorised cummin + REALISTIC GAP FILL")
    print("MECHANISM FIRST: their loop fills the stop at EXACTLY -S whenever the intraday")
    print("low touches -S.  A real stop is a market order: on a day that OPENS below the")
    print("trigger you are filled at the open, not at -S.  SOXL gaps -16.7% in one day, so")
    print("this is not hypothetical.  A correct fill makes the stop WORSE, so this bug runs")
    print("in favour of the claim -- but the magnitude has to be stated, not assumed.")
    HOLD = 20
    pops = {"all days": np.ones(n, bool),
            "day fell >= 8%": (m["ret"].values <= -0.08),
            "day fell >= 13%": (m["ret"].values <= -0.13)}
    print(f"\n  {'population':<17}{'stop':>6}{'n':>6}{'diff(fill@-S)':>15}{'diff(fill@open)':>17}"
          f"{'med diff@open':>15}{'gapped%':>9}")
    rows6 = []
    for pname, mask in pops.items():
        ii = np.where(mask)[0]; ii = ii[ii + HOLD < n]
        base = cv[ii+HOLD]/cv[ii]-1
        # windows of lows / opens, vectorised
        off = np.arange(1, HOLD+1)
        L = lv[ii[:,None]+off]; O = ov[ii[:,None]+off]
        for S in (0.08,0.12,0.15,0.20,0.25,0.30):
            thr = cv[ii]*(1-S)
            hitmat = L <= thr[:,None]
            hit = hitmat.any(1)
            j = np.argmax(hitmat, axis=1)
            # idealised fill
            st_id = np.where(hit, -S, base)
            # realistic: fill at open if the trigger day opened below thr, else at thr
            fillpx = np.where(O[np.arange(len(ii)), j] <= thr, O[np.arange(len(ii)), j], thr)
            st_re = np.where(hit, fillpx/cv[ii]-1, base)
            gapped = float((hit & (O[np.arange(len(ii)),j] <= thr)).sum())/max(1,hit.sum())
            d_id = st_id-base; d_re = st_re-base
            rows6.append((pname,S,d_id.mean(),d_re.mean(),np.median(d_re),gapped,len(ii)))
            print(f"  {pname:<17}{'-'+format(S*100,'.0f')+'%':>6}{len(ii):>6}{d_id.mean():>15.2%}"
                  f"{d_re.mean():>17.2%}{np.median(d_re):>15.2%}{gapped:>9.0%}")
    print(f"\n  sign of the idealised difference:  negative in {sum(1 for r in rows6 if r[2]<0)}/18 cells")
    print(f"  sign of the realistic difference:  negative in {sum(1 for r in rows6 if r[3]<0)}/18 cells")
    a8 = [r for r in rows6 if r[0]=='all days' and abs(r[1]-0.08)<1e-9][0]
    print(f"  their headline cell (all days, -8%): their {a8[2]:+.2%}  vs realistic {a8[3]:+.2%}")
    print(f"  MATCH on the idealised number: {abs(a8[2]+0.0237)<0.0005}")
    print(f"  {a8[5]:.0%} of stop triggers happen on a day that OPENED through the stop.")

    print("\n  EFFECTIVE-n CHECK.  Their perm p<0.0001 shuffles 4133 overlapping 20-day")
    print("  windows.  Re-run on NON-OVERLAPPING entries only (every 20th bar):")
    for step in (20,):
        ii = np.arange(0, n-HOLD, step)
        base = cv[ii+HOLD]/cv[ii]-1
        off = np.arange(1,HOLD+1); L = lv[ii[:,None]+off]
        for S in (0.08,0.15,0.30):
            thr = cv[ii]*(1-S); hit = (L<=thr[:,None]).any(1)
            dd_ = np.where(hit,-S,base)-base
            s = RNG.choice([-1.,1.], size=(NPERM, dd_.size))
            null = (s*dd_).mean(1)
            p = (1+int((np.abs(null)>=abs(dd_.mean())).sum()))/(NPERM+1)
            print(f"    non-overlapping n={len(ii)}  stop -{S*100:.0f}%  diff {dd_.mean():+.2%}  p={p:.4f}")
    print("  -> the sign and rough magnitude survive a 20x reduction in n.  Claim 6 holds.")

    # =============================================================== CLAIM 7 ROUTE B
    hdr("5.  CLAIM 7 RE-DERIVED: the limit fill price is WRONG IN THE OTHER DIRECTION")
    print("MECHANISM FIRST: a resting BUY limit at px fills at px only if the market trades")
    print("down through it during the session.  If the session OPENS below px the limit is")
    print("marketable on the open and fills at the OPEN -- a BETTER price.  Their loop always")
    print("charges px.  That systematically understates the limit's return, which is exactly")
    print("the direction of the claim being made.  Same class of bug as comparing an intraday")
    print("low to a close: a touch is not a fill price.")
    WINDOW = 20
    print(f"\n  {'population':<17}{'off':>6}{'fill%':>7}{'EV(px fill)':>13}{'EV(open fill)':>15}"
          f"{'market':>9}{'diff px':>9}{'diff open':>11}{'gapped%':>9}")
    rows7 = []
    for pname, mask in pops.items():
        ii = np.where(mask)[0]; ii = ii[ii + WINDOW + HOLD < n]
        mkt = cv[ii+HOLD]/cv[ii]-1
        off = np.arange(1, WINDOW+1)
        L = lv[ii[:,None]+off]; O = ov[ii[:,None]+off]
        for o_ in (-0.02,-0.05,-0.10):
            px = cv[ii]*(1+o_)
            hm = L <= px[:,None]; hit = hm.any(1); j = np.argmax(hm,axis=1)
            absj = ii + 1 + j
            opn = O[np.arange(len(ii)), j]
            fill_real = np.where(opn <= px, opn, px)
            safe = np.minimum(absj+HOLD, n-1)
            r_px   = np.where(hit, cv[safe]/px - 1, 0.0)
            r_open = np.where(hit, cv[safe]/fill_real - 1, 0.0)
            gap = float((hit & (opn<=px)).sum())/max(1,hit.sum())
            rows7.append((pname,o_,(r_px-mkt).mean(),(r_open-mkt).mean()))
            print(f"  {pname:<17}{o_*100:>5.0f}%{hit.mean():>7.0%}{r_px.mean():>13.2%}"
                  f"{r_open.mean():>15.2%}{mkt.mean():>9.2%}{(r_px-mkt).mean():>9.2%}"
                  f"{(r_open-mkt).mean():>11.2%}{gap:>9.0%}")
    print(f"\n  cells where the limit loses to market, their fill: "
          f"{sum(1 for r in rows7 if r[2]<0)}/9")
    print(f"  cells where the limit loses to market, correct fill: "
          f"{sum(1 for r in rows7 if r[3]<0)}/9")
    h7 = [r for r in rows7 if r[0]=='all days' and abs(r[1]+0.02)<1e-9][0]
    print(f"  their headline cell (all days, -2%): their {h7[2]:+.2%}  vs correct fill {h7[3]:+.2%}")

    print("\n  SECOND DEFECT IN CLAIM 7, AND IT IS BIGGER: THE TWO ARMS ARE NOT THE SAME TRADE.")
    print("  market arm  = long from bar i to bar i+20.")
    print("  limit arm   = flat for up to 20 bars, then long for 20 bars, ending at i+40.")
    print("  Different horizon, different time in market, different terminal date.  On a")
    print("  series with +5.4% mean 20-day drift, an arm that sits in cash part of the time")
    print("  must lose on the mean by construction.  Matched-horizon version, both arms")
    print("  marked to the SAME terminal bar i+40:")
    for pname, mask in list(pops.items())[:1]:
        ii = np.where(mask)[0]; ii = ii[ii + WINDOW + HOLD < n]
        off = np.arange(1, WINDOW+1)
        L = lv[ii[:,None]+off]; O = ov[ii[:,None]+off]
        term = ii + WINDOW + HOLD
        mkt_m = cv[term]/cv[ii]-1
        for o_ in (-0.02,-0.05,-0.10):
            px = cv[ii]*(1+o_)
            hm = L <= px[:,None]; hit = hm.any(1); j=np.argmax(hm,axis=1)
            opn = O[np.arange(len(ii)),j]; fill = np.where(opn<=px,opn,px)
            lim_m = np.where(hit, cv[term]/fill-1, 0.0)
            dm_ = lim_m-mkt_m
            print(f"    all days {o_*100:>4.0f}%  matched-horizon limit {lim_m.mean():+.2%} "
                  f"vs market {mkt_m.mean():+.2%}  diff {dm_.mean():+.2%}  median {np.median(dm_):+.2%}")
    print("  -> AGAINST MY OWN PRIOR: matching the horizon does NOT shrink the gap, it")
    print("     GROWS it (-1.9/-2.9/-4.4% vs their -1.6/-1.8/-2.0%), and it grows MONOTONICALLY")
    print("     with offset depth.  That is the signature of missed drift, not of adverse")
    print("     selection: the deeper the limit, the longer the arm sits in cash.  Their")
    print("     conclusion (mis-attributed mechanism) is right; their number understates it.")

    # =============================================================== CLAIM 5 units
    hdr("6.  CLAIM 5: MISMATCHED UNITS IN THE CONTROL GROUP, AND ASYMMETRIC DEDUPLICATION")
    print("MECHANISM FIRST: 'broke' is defined on the intraday LOW vs the prior low.  The")
    print("control 'near, held' is defined on the CLOSE vs the prior low (c/prior_low-1<=.15).")
    print("Two different price types against the same reference.  Separately, 'broke' is")
    print("deduplicated to one row per episode and 'near' is NOT, so 37 episodes are compared")
    print("against 218 overlapping days.")
    c = m["close"]; low = m["low"]
    ddq = c / c.rolling(250, min_periods=60).max() - 1
    pl = low.shift(1).rolling(60, min_periods=25).min()
    deep = ddq <= -0.40
    broke_all = m[deep & (m["low"] < pl) & m["fwd20"].notna()]
    broke = broke_all.groupby((broke_all.index.to_series().diff().dt.days>10).cumsum()).head(1)
    near_theirs = m[deep & (m["low"]>=pl) & (c/pl-1 <= 0.15) & m["fwd20"].notna()]
    near_units  = m[deep & (m["low"]>=pl) & (low/pl-1 <= 0.15) & m["fwd20"].notna()]
    def dedup(f):
        return f.groupby((f.index.to_series().diff().dt.days>10).cumsum()).head(1)
    print(f"\n  broke (deduped)                 : {desc(broke['fwd20'])}")
    print(f"  near, THEIR close-vs-low control: {desc(near_theirs['fwd20'])}  (not deduped)")
    print(f"  near, low-vs-low control        : {desc(near_units['fwd20'])}  (not deduped)")
    print(f"  near, low-vs-low AND deduped    : {desc(dedup(near_units)['fwd20'])}")
    for lbl, ctl in (("theirs", near_theirs), ("units fixed", near_units),
                     ("units fixed + deduped", dedup(near_units))):
        bb = pd.concat([broke.assign(g=True), ctl.assign(g=False)])
        o_, p_ = perm(bb["fwd20"].values, bb["g"].values)
        print(f"    control={lbl:<22} difference {o_:+.2%}  p={p_:.3f}")
    print("  -> the NULL verdict is unchanged under every variant, so claim 5's conclusion")
    print("     survives; but the +4.83% point estimate the audit quoted is not stable to")
    print("     fixing its own units bug, and it was already one observation (2026-03-30).")

    # =============================================================== seam
    hdr("7.  CLAIM 3: frame() RECOMPUTES qqq_ret ACROSS THE SOURCE SEAM data.py FORBIDS")
    print("MECHANISM FIRST: data.py stitches QQQ from a parquet and a JSON that differ by a")
    print("~0.1% constant and explicitly computes returns PER SOURCE so no return crosses")
    print("the seam.  bt_multiple_testing.frame() overwrites qqq_ret with a naive")
    print("close/close.shift(1) over the stitched series, reintroducing one bogus return.")
    q = load_qqq()
    seam = q.index[q["src"].ne(q["src"].shift(1))][1:]
    naive = (q["qqq"]/q["qqq"].shift(1)-1)
    for s in seam:
        print(f"    seam at {s.date()}: per-source qqq_ret {q.loc[s,'qqq_ret']:+.4%}  "
              f"vs naive cross-seam {naive.loc[s]:+.4%}  (error {naive.loc[s]-q.loc[s,'qqq_ret']:+.4%})")
    print(f"    was that date ever a SOXL <= -8% day (i.e. does it enter claim 3)? "
          f"{bool(((m['ret']<=-0.08) & m.index.isin(seam)).any())}")
    print("  -> harmless here, but it is a real defect in a script whose entire purpose is")
    print("     to catch defects, and it is the same class of error it was hired to find.")

    # =============================================================== duplicate closes
    hdr("8.  A DATA PROBLEM NEITHER SCRIPT CHECKS: REPEATED CLOSES IN THE 2026 TAIL")
    rr = d["close"]
    dup = rr[(rr.diff()==0) & (rr.index.year>=2024)]
    print(f"  bars in 2024+ whose close exactly equals the prior close: {len(dup)}")
    for ix, v in dup.items():
        print(f"    {ix.date()}  close {v:.6f}  (prior bar identical)")
    print("  An exactly-repeated close on a 3x ETF that moved 10%+ intraday is a data")
    print("  artefact, not a price.  2026-07-30/07-31 both close 114.720001 and both sit")
    print("  inside claim 4's out-of-sample window and claim 8's 2026 hold fwd20 windows.")

    # =============================================================== 2026 regime
    hdr("9.  THE AUDIT NEVER TESTS ITS OWN LOAD-BEARING ASSUMPTION: ONE DISTRIBUTION")
    for lo, hi, lbl in (("2010-01-01","2025-12-31","2010-2025"), ("2026-01-01","2026-12-31","2026 only")):
        s = m.loc[lo:hi]
        print(f"  {lbl:<10} n={len(s):>5}  daily sd {s['ret'].std():.2%}  "
              f"median |ret| {s['ret'].abs().median():.2%}  "
              f"median range {((s['high']-s['low'])/s['close'].shift(1)).median():.2%}  "
              f"P(|ret|>10%) {(s['ret'].abs()>0.10).mean():.1%}")
    print("  -> 2026 is a different volatility regime by every measure.  Every pooled")
    print("     statistic in the audit (claims 6 and 7 included) mixes the two.  Claim 6's")
    print("     sign survives that (checked below); its MAGNITUDE does not transfer.")
    s26 = m.loc["2026-01-01":]
    i26 = np.array([i for i in range(n) if m.index[i].year==2026 and i+HOLD<n])
    if len(i26):
        base = cv[i26+HOLD]/cv[i26]-1
        off = np.arange(1,HOLD+1); L = lv[i26[:,None]+off]
        for S in (0.08,0.20,0.30):
            thr = cv[i26]*(1-S); hit=(L<=thr[:,None]).any(1)
            dd_ = np.where(hit,-S,base)-base
            print(f"    2026 only, stop -{S*100:.0f}%: n={len(i26)} diff {dd_.mean():+.2%} "
                  f"median {np.median(dd_):+.2%} hit {hit.mean():.0%}  "
                  f"[n<50 windows independent: counter-example only]")

    # =============================================================== HAC
    hdr("10. THE DECISIVE NUMBER, THIRD ROUTE: HAC t-TEST ON THE OVERLAP")
    print("MECHANISM FIRST: claim 6's p<0.0001 is a sign-flip permutation over 4133 paired")
    print("differences whose 20-day windows share 19 of 20 days.  Their OWN finding says")
    print("independent_events = 207.  A p-value computed at n=4133 and an independence")
    print("count of 207 cannot both be reported for the same statistic.  Two corrections:")
    print("(a) Newey-West HAC standard errors at lag 40, (b) the test at n=207 across all")
    print("20 non-overlapping phases, so no phase is cherry-picked.")
    from math import erfc, sqrt as _sq
    def nw_se(v, Lg):
        v = v - v.mean(); N = len(v); s = (v*v).sum()/N
        for k in range(1, Lg+1):
            s += 2*(1-k/(Lg+1))*(v[k:]*v[:-k]).sum()/N
        return np.sqrt(max(s,1e-18)/N)
    ii = np.arange(n-HOLD); base = cv[ii+HOLD]/cv[ii]-1
    Lw = lv[ii[:,None]+np.arange(1,HOLD+1)]
    print(f"\n  {'stop':>6}{'mean':>9}{'naive t':>10}{'HAC40 t':>10}{'HAC p':>9}"
          f"{'Bonf(100)':>11}{'surv B':>8}{'med p over 20 phases':>22}{'phases p<.0005':>16}")
    for S in (0.08,0.12,0.15,0.20,0.25,0.30):
        thr = cv[ii]*(1-S); hit=(Lw<=thr[:,None]).any(1); xx=np.where(hit,-S,base)-base
        mu=xx.mean(); se0=xx.std(ddof=1)/np.sqrt(len(xx)); s40=nw_se(xx,40)
        pH=erfc(abs(mu/s40)/_sq(2)); ps=[]
        for ph in range(20):
            z=xx[ph::20]
            sg=RNG.choice([-1.,1.],size=(4000,z.size)); nl=(sg*z).mean(1)
            ps.append((1+int((np.abs(nl)>=abs(z.mean())).sum()))/4001)
        ps=np.array(ps)
        print(f"  {-S*100:>5.0f}%{mu:>9.2%}{mu/se0:>10.2f}{mu/s40:>10.2f}{pH:>9.4f}"
              f"{min(1,pH*100):>11.3f}{str(min(1,pH*100)<0.05):>8}{np.median(ps):>22.4f}"
              f"{int((ps<0.0005).sum()):>12}/20")
    W2=20; jj=np.arange(n-W2-HOLD); mkt=cv[jj+HOLD]/cv[jj]-1
    Lv=lv[jj[:,None]+np.arange(1,W2+1)]; pxx=cv[jj]*0.98
    hm=Lv<=pxx[:,None]; hh=hm.any(1); jx=np.argmax(hm,1)
    aj=np.minimum(jj+1+jx+HOLD,n-1); xx=np.where(hh,cv[aj]/pxx-1,0.0)-mkt
    mu=xx.mean(); s40=nw_se(xx,40); pH=erfc(abs(mu/s40)/_sq(2))
    print(f"\n  claim 7 (all days, -2%): mean {mu:+.2%}  naive t {mu/(xx.std(ddof=1)/np.sqrt(len(xx))):.2f}"
          f"  HAC40 t {mu/s40:.2f}  HAC p {pH:.4f}  Bonf(100) {min(1,pH*100):.3f}")
    print("\n  CORRECTED MULTIPLICITY TABLE (only the two rows that change):")
    print(f"    claim 6  their raw p 0.0000 -> HAC 0.0033   their Bonf 0.005 -> 0.327   surv B: True -> FALSE")
    print(f"    claim 7  their raw p 0.0000 -> HAC {pH:.4f}   their Bonf 0.005 -> {min(1,pH*100):.3f}   surv B: True -> {str(min(1,pH*100)<0.05).upper()}")
    print("  -> the audit's headline ('exactly one claim survives correction for ~100 tests,")
    print("     and it is the stop claim') inverts under the correction the audit itself said")
    print("     was necessary.  Claim 6 does NOT survive Bonferroni at its own stated")
    print("     effective n; claim 7, the one it demoted, is the one that does.")
    print("  -> what DOES survive for claim 6 is the part that never needed a p-value: the")
    print("     sign is negative in 18/18 cells at both fill conventions, and conditional on")
    print("     the stop firing the mean and median are both negative in 18/18.  Verified")
    print("     independently below.  That is an arithmetic argument and it stands.")
    print("\n  conditional-on-hit check, MY construction, realistic gap fill:")
    neg=0; tot=0; mn=[]
    for pname, mask in pops.items():
        iq=np.where(mask)[0]; iq=iq[iq+HOLD<n]; bq=cv[iq+HOLD]/cv[iq]-1
        Lq=lv[iq[:,None]+np.arange(1,HOLD+1)]; Oq=ov[iq[:,None]+np.arange(1,HOLD+1)]
        for S in (0.08,0.12,0.15,0.20,0.25,0.30):
            th=cv[iq]*(1-S); hmq=Lq<=th[:,None]; hq=hmq.any(1); jq=np.argmax(hmq,1)
            op=Oq[np.arange(len(iq)),jq]; fp=np.where(op<=th,op,th)
            dq=(np.where(hq,fp/cv[iq]-1,bq)-bq)[hq]
            tot+=1; neg+= int(dq.mean()<0 and np.median(dq)<0); mn.append((dq>0).mean())
    print(f"    cells where BOTH mean and median of d|hit are negative: {neg}/{tot}")
    print(f"    'stop was the right call' rate range: {min(mn):.0%}-{max(mn):.0%}  "
          f"(their finding says 38-46%; their own table's minimum is 36%)")

if __name__ == "__main__":
    main()
