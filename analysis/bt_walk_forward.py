"""Walk-forward test of the two-tranche limit ladder.

    python3 -m analysis.bt_walk_forward

Plan under test: on a signal day, rest two limit orders at o1 and o2 below that
day's CLOSE, half the position each. Orders rest WINDOW trading days then cancel.
A filled tranche is held HOLD trading days FROM ITS OWN FILL. An unfilled tranche
contributes 0.0 (no position, not a loss).

Walk-forward: the pair (o1, o2) is chosen by grid search on 2010-2017 signal days
only, then applied unchanged to 2018-2026. The in-sample-best minus out-of-sample
gap is the overfitting estimate.

Every level is an offset from the signal close, so it is knowable at the signal
close and the order can actually be placed. Nothing here uses a close-derived
feature to TRIGGER anything; the signals are (a) a close-to-close fall, which is
known at the close of the signal day and the order is placed for the NEXT session
onward, (b) a gap, known at the open, (c) a drawdown/20-day return, known at the
close. Fills are tested against the intraday low of days strictly AFTER the
signal day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import load_merged

HOLD = 20      # trading days held after a fill
WINDOW = 20    # trading days the order rests before cancelling
GRID = [-i / 100 for i in range(2, 16)]   # -2% .. -15%, 1% steps
PROPOSED = (-0.05, -0.09)
SPLIT = pd.Timestamp("2018-01-01")
RNG = np.random.default_rng(20260915)
NPERM = 4000

# The last bar is not in the CSV; append it manually.
LAST_BAR = dict(date="2026-09-14", open=101.53, high=105.39, low=99.87, close=101.13, qqq=711.96)


# ----------------------------------------------------------------- data ------
def frame() -> pd.DataFrame:
    m = load_merged()
    d = pd.Timestamp(LAST_BAR["date"])
    assert d not in m.index, "last bar already present"
    row = pd.Series({"open": LAST_BAR["open"], "high": LAST_BAR["high"],
                     "low": LAST_BAR["low"], "close": LAST_BAR["close"],
                     "volume": np.nan, "qqq": LAST_BAR["qqq"], "src": "manual"}, name=d)
    m = pd.concat([m, row.to_frame().T]).sort_index()
    for c in ("open", "high", "low", "close", "qqq"):
        m[c] = m[c].astype(float)
    m = m[m.index >= "2010-03-12"].reset_index().rename(columns={"index": "date", "Date": "date"})
    c = m["close"]
    m["ret"] = c.pct_change()
    m["gap"] = m["open"] / c.shift(1) - 1
    m["dd250"] = c / c.rolling(250, min_periods=60).max() - 1
    m["ret20"] = c.pct_change(20)
    return m


def signals(m: pd.DataFrame) -> dict:
    return {
        "(a) daily fall <= -13%": (m["ret"] <= -0.13).values,
        "(b) gap down <= -12%": (m["gap"] <= -0.12).values,
        "(c) dd250<=-55% & ret20<=-25%": ((m["dd250"] <= -0.55) & (m["ret20"] <= -0.25)).values,
    }


def independent(dates, gap_days=20):
    """Count episodes: consecutive signal days closer than gap_days collapse to one."""
    if len(dates) == 0:
        return 0
    k, prev = 1, dates[0]
    for d in dates[1:]:
        if (d - prev).days > gap_days * 1.6:   # ~20 trading days
            k += 1
        prev = d
    return k


# ------------------------------------------------------------- mechanics -----
GAP_FILL = False   # True -> a limit that gaps through fills at the OPEN, not the limit


def tranche(c, low, n, i, off, op=None):
    """One tranche resting at c[i]*(1+off). Returns (ret, filled, fill_idx)."""
    px = c[i] * (1 + off)
    for j in range(i + 1, min(i + 1 + WINDOW, n)):
        if low[j] <= px:
            fp = min(px, op[j]) if (GAP_FILL and op is not None) else px
            return c[j + HOLD] / fp - 1, True, j
    return 0.0, False, -1


OPENS = None


def event_returns(c, low, n, idx, o1, o2):
    """Per-event half-and-half ladder return, plus fill flags."""
    out, f1, f2 = [], [], []
    for i in idx:
        r1, a, _ = tranche(c, low, n, i, o1, OPENS)
        r2, b, _ = tranche(c, low, n, i, o2, OPENS)
        out.append(0.5 * r1 + 0.5 * r2)
        f1.append(a); f2.append(b)
    return np.array(out), np.array(f1), np.array(f2)


def mae_of_fills(c, low, n, idx, o1, o2):
    """Worst mark-to-market of each filled tranche over its own holding period."""
    out = []
    for i in idx:
        for off in (o1, o2):
            px = c[i] * (1 + off)
            for j in range(i + 1, min(i + 1 + WINDOW, n)):
                if low[j] <= px:
                    out.append(low[j:j + HOLD + 1].min() / px - 1)
                    break
    return np.array(out)


def portfolio_mae(c, low, n, idx, o1, o2):
    """Worst mark of the WHOLE position (half in each tranche, unfilled half = cash)."""
    out = []
    for i in idx:
        legs = []
        for off in (o1, o2):
            px = c[i] * (1 + off)
            for j in range(i + 1, min(i + 1 + WINDOW, n)):
                if low[j] <= px:
                    legs.append((px, j, j + HOLD))
                    break
        if not legs:
            continue
        lo_, hi_ = min(f for _, f, _ in legs), max(e for _, _, e in legs)
        worst = 0.0
        for t in range(lo_, hi_ + 1):
            v = 0.0
            for px, f, e in legs:
                if f <= t <= e:
                    v += 0.5 * (low[t] / px - 1)
            worst = min(worst, v)
        out.append(worst)
    return np.array(out)


def stat(v):
    if len(v) == 0:
        return "n=0"
    return (f"mean {v.mean()*100:+6.2f}%  median {np.median(v)*100:+6.2f}%  "
            f"win {(v > 0).mean()*100:3.0f}%  n={len(v)}")


def usable(mask, n):
    """Signal rows whose whole 20d window + 20d hold fits inside the data."""
    return [i for i in np.where(mask)[0] if i + WINDOW + HOLD < n]


# ------------------------------------------------------------ permutation ----
def perm_vs_random(c, low, n, idx, o1, o2, pool, nperm=NPERM):
    """H0: signal days carry no ladder information. Draw same-size random day sets."""
    obs, _, _ = event_returns(c, low, n, idx, o1, o2)
    if len(obs) == 0:
        return np.nan, np.nan
    om = obs.mean()
    cnt = 0
    for _ in range(nperm):
        s = RNG.choice(pool, size=len(idx), replace=False)
        v, _, _ = event_returns(c, low, n, s, o1, o2)
        cnt += v.mean() >= om
    return om, (cnt + 1) / (nperm + 1)


def perm_paired(a, b, nperm=NPERM):
    """Paired sign-flip permutation on the per-event difference a-b (two-sided)."""
    d = a - b
    if len(d) == 0:
        return np.nan, np.nan
    obs = d.mean()
    cnt = 0
    for _ in range(nperm):
        s = RNG.choice([-1.0, 1.0], size=len(d))
        cnt += abs((d * s).mean()) >= abs(obs)
    return obs, (cnt + 1) / (nperm + 1)


# ------------------------------------------------------------------ main -----
def episode_ids(dates_arr, idx, gap_days=20):
    """Group signal rows into episodes separated by more than ~gap_days trading days."""
    ids, k = [], 0
    prev = None
    for i in idx:
        d = pd.Timestamp(dates_arr[i])
        if prev is not None and (d - prev).days > gap_days * 1.6:
            k += 1
        ids.append(k); prev = d
    return np.array(ids)


def block_perm(v, ids, nperm=NPERM):
    """Mean over EPISODE means, with a sign-flip permutation at the episode level.
    This is the honest n: overlapping signals inside one selloff are one observation."""
    ep = np.array([v[ids == k].mean() for k in np.unique(ids)])
    obs = ep.mean()
    cnt = sum(1 for _ in range(nperm)
              if abs((ep * RNG.choice([-1.0, 1.0], size=len(ep))).mean()) >= abs(obs))
    return obs, (cnt + 1) / (nperm + 1), len(ep)


def main():
    global OPENS
    m = frame()
    c, low, op, n = m["close"].values, m["low"].values, m["open"].values, len(m)
    OPENS = op
    dates = m["date"].values
    sigs = signals(m)

    print("=" * 96)
    print("WALK-FORWARD TEST OF THE TWO-TRANCHE LADDER")
    print(f"data {pd.Timestamp(dates[0]).date()} .. {pd.Timestamp(dates[-1]).date()}  "
          f"({n} bars, last bar appended manually)")
    print(f"hold {HOLD}d from own fill, orders rest {WINDOW}d, unfilled tranche = 0.0, "
          f"grid {GRID[0]*100:.0f}%..{GRID[-1]*100:.0f}% step 1%")
    print("=" * 96)

    # ---- signal census
    print("\n[1] SIGNAL CENSUS")
    print(f"{'signal':<32} {'all':>5} {'usable':>7} {'IS<=2017':>9} {'OOS>=2018':>10} "
          f"{'indep(all)':>11}")
    IDX = {}
    for name, mask in sigs.items():
        u = usable(mask, n)
        d = pd.to_datetime(dates[u])
        ins = [i for i in u if pd.Timestamp(dates[i]) < SPLIT]
        oos = [i for i in u if pd.Timestamp(dates[i]) >= SPLIT]
        IDX[name] = dict(all=u, ins=ins, oos=oos)
        print(f"{name:<32} {int(mask.sum()):>5} {len(u):>7} {len(ins):>9} {len(oos):>10} "
              f"{independent(list(d)):>11}")
    print("  'usable' drops signals whose 20d order window + 20d hold runs past the data end,")
    print("  including 2026-09-14 itself. 'indep' collapses signals inside ~20 trading days.")

    # ---- dates, so clustering is visible
    print("\n[2] SIGNAL DATES (clustering is the reason n overstates the evidence)")
    for name, mask in sigs.items():
        d = [str(pd.Timestamp(x).date()) for x in dates[np.where(mask)[0]]]
        print(f"  {name}: " + (", ".join(d) if d else "none"))

    # ---- walk-forward
    print("\n[3] WALK-FORWARD  (grid-search on 2010-2017 ONLY, apply unchanged to 2018-2026)")
    chosen = {}
    for name in sigs:
        ins, oos = IDX[name]["ins"], IDX[name]["oos"]
        print(f"\n  --- {name} ---")
        if len(ins) == 0:
            print("    no in-sample signals at all -> nothing to select on. SKIP.")
            chosen[name] = None
            continue
        rows = []
        for a_i, o1 in enumerate(GRID):
            for o2 in GRID[a_i + 1:]:
                v, _, _ = event_returns(c, low, n, ins, o1, o2)
                rows.append((o1, o2, v.mean(), np.median(v), (v > 0).mean()))
        rows.sort(key=lambda r: -r[2])
        best = rows[0]
        chosen[name] = (best[0], best[1])
        print(f"    in-sample n={len(ins)} events, {len(rows)} grid pairs searched")
        print("    top 5 in-sample pairs:")
        for r in rows[:5]:
            print(f"      {r[0]*100:5.0f}% / {r[1]*100:5.0f}%   IS mean {r[2]*100:+7.2f}%  "
                  f"median {r[3]*100:+7.2f}%  win {r[4]*100:3.0f}%")
        prop = next(r for r in rows if abs(r[0] - PROPOSED[0]) < 1e-9 and abs(r[1] - PROPOSED[1]) < 1e-9)
        print(f"    proposed -5%/-9% in-sample: mean {prop[2]*100:+.2f}%  "
              f"rank {rows.index(prop)+1}/{len(rows)}")

        if len(oos) == 0:
            print("    no out-of-sample signals -> walk-forward is UNTESTABLE for this signal.")
            continue
        o1, o2 = best[0], best[1]
        vo, f1, f2 = event_returns(c, low, n, oos, o1, o2)
        vp, p1, p2 = event_returns(c, low, n, oos, *PROPOSED)
        print(f"    OUT OF SAMPLE n={len(oos)} events "
              f"({independent(list(pd.to_datetime(dates[oos])))} independent episodes)")
        print(f"      IS-best pair {o1*100:.0f}%/{o2*100:.0f}%  : {stat(vo)}  "
              f"fill {f1.mean()*100:.0f}%/{f2.mean()*100:.0f}%")
        print(f"      proposed     -5%/-9%     : {stat(vp)}  "
              f"fill {p1.mean()*100:.0f}%/{p2.mean()*100:.0f}%")
        print(f"      OVERFIT GAP  IS mean {best[2]*100:+.2f}%  ->  OOS mean {vo.mean()*100:+.2f}%  "
              f"= {(vo.mean()-best[2])*100:+.2f} pp")
        # Did the SELECTION transfer, or only the period differ? Rank the IS-winner OOS.
        oos_grid = {}
        for a_i, g1 in enumerate(GRID):
            for g2 in GRID[a_i + 1:]:
                vv, _, _ = event_returns(c, low, n, oos, g1, g2)
                oos_grid[(g1, g2)] = vv.mean()
        ov = np.array(list(oos_grid.values()))
        r_best = 1 + sum(1 for x in ov if x > oos_grid[(o1, o2)])
        r_prop = 1 + sum(1 for x in ov if x > oos_grid[PROPOSED])
        se = vo.std(ddof=1) / np.sqrt(len(vo))
        print(f"      SELECTION TRANSFER: IS-winner ranks {r_best}/{len(ov)} out of sample "
              f"(random would average {len(ov)/2:.0f}); OOS grid best "
              f"{max(ov)*100:+.2f}% median {np.median(ov)*100:+.2f}% worst {min(ov)*100:+.2f}%")
        print(f"      proposed -5/-9 ranks {r_prop}/{len(ov)} out of sample.  "
              f"OOS s.e. of the mean = {se*100:.2f}pp -> the whole OOS grid spans "
              f"{(max(ov)-min(ov))/se:.2f} s.e.")

    # ---- benchmarks
    print("\n[4] BENCHMARKS on the SAME events (full position, same 20d hold)")
    print(f"{'signal':<32} {'period':<5} {'ladder -5/-9':<38} {'buy at close':<38} {'buy next open'}")
    BENCH = {}
    for name in sigs:
        for per in ("IS", "OOS", "ALL"):
            idx = IDX[name]["ins" if per == "IS" else "oos" if per == "OOS" else "all"]
            if not idx:
                continue
            v, _, _ = event_returns(c, low, n, idx, *PROPOSED)
            bc = np.array([c[i + HOLD] / c[i] - 1 for i in idx])
            bo = np.array([c[i + 1 + HOLD] / op[i + 1] - 1 for i in idx])
            BENCH[(name, per)] = (v, bc, bo)
            print(f"{name:<32} {per:<5} {stat(v):<38} {stat(bc):<38} {stat(bo)}")

    print("\n  Paired permutation, ladder(-5/-9) minus buy-at-close, per event, two-sided:")
    for (name, per), (v, bc, bo) in BENCH.items():
        d1, p1 = perm_paired(v, bc)
        d2, p2 = perm_paired(v, bo)
        print(f"    {name:<32} {per:<4} vs close {d1*100:+6.2f}pp p={p1:.3f}   "
              f"vs next-open {d2*100:+6.2f}pp p={p2:.3f}")

    # ---- does the SIGNAL matter at all?
    print("\n[5] DOES THE SIGNAL MATTER? ladder(-5/-9) on signal days vs random days")
    print("    (permutation: same number of days drawn at random from the same period)")
    for name in sigs:
        for per, lo_, hi_ in (("IS", 0, SPLIT), ("OOS", SPLIT, None), ("ALL", None, None)):
            idx = IDX[name]["ins" if per == "IS" else "oos" if per == "OOS" else "all"]
            if not idx:
                continue
            ok = np.arange(n - WINDOW - HOLD - 1)
            dd = pd.to_datetime(dates[ok])
            if per == "IS":
                ok = ok[dd < SPLIT]
            elif per == "OOS":
                ok = ok[dd >= SPLIT]
            om, p = perm_vs_random(c, low, n, idx, *PROPOSED, ok)
            base, _, _ = event_returns(c, low, n, ok, *PROPOSED)
            print(f"    {name:<32} {per:<4} signal mean {om*100:+6.2f}%  "
                  f"all-days mean {base.mean()*100:+6.2f}%  p={p:.4f}  n={len(idx)}")

    # ---- sensitivity surface
    print("\n[6] PARAMETER-SENSITIVITY SURFACE (FULL history, mean EV %, rows=tranche1, cols=tranche2)")
    for name in sigs:
        idx = IDX[name]["all"]
        if len(idx) < 3:
            print(f"\n  --- {name} --- n={len(idx)}: too few events for a surface.")
            continue
        surf = {}
        for a_i, o1 in enumerate(GRID):
            for o2 in GRID[a_i + 1:]:
                v, _, _ = event_returns(c, low, n, idx, o1, o2)
                surf[(o1, o2)] = v.mean()
        vals = np.array(list(surf.values()))
        peak = max(surf, key=surf.get)
        print(f"\n  --- {name} --- n={len(idx)} events, {len(surf)} pairs")
        hdr = "        " + "".join(f"{o*100:7.0f}" for o in GRID[1:])
        print(hdr)
        for a_i, o1 in enumerate(GRID[:-1]):
            cells = []
            for o2 in GRID[1:]:
                cells.append(f"{surf[(o1,o2)]*100:7.1f}" if (o1, o2) in surf else "      .")
            print(f"  {o1*100:5.0f}%" + "".join(cells))
        prop = surf[PROPOSED]
        rank = 1 + sum(1 for v in vals if v > prop)
        # plateau test: neighbours of the peak and of the proposal within +-1 grid step
        def neigh(p):
            o1, o2 = p
            out = []
            for k in (-0.01, 0.0, 0.01):
                for l in (-0.01, 0.0, 0.01):
                    q = (round(o1 + k, 4), round(o2 + l, 4))
                    if q in surf:
                        out.append(surf[q])
            return np.array(out)
        np_ = neigh(PROPOSED); nk = neigh(peak)
        spread = vals.max() - vals.min()
        print(f"    peak {peak[0]*100:.0f}%/{peak[1]*100:.0f}% = {surf[peak]*100:+.2f}%   "
              f"grid median {np.median(vals)*100:+.2f}%   grid min {vals.min()*100:+.2f}%   "
              f"spread {spread*100:.2f}pp")
        print(f"    proposed -5%/-9% = {prop*100:+.2f}%  rank {rank}/{len(vals)}  "
              f"({(1-rank/len(vals))*100:.0f}th pct of the grid)")
        print(f"    neighbourhood of -5/-9 (+-1 step, n={len(np_)}): "
              f"min {np_.min()*100:+.2f}% max {np_.max()*100:+.2f}% "
              f"range {(np_.max()-np_.min())*100:.2f}pp")
        print(f"    neighbourhood of the peak   (+-1 step, n={len(nk)}): "
              f"min {nk.min()*100:+.2f}% max {nk.max()*100:+.2f}% "
              f"range {(nk.max()-nk.min())*100:.2f}pp")
        vref, _, _ = event_returns(c, low, n, idx, *PROPOSED)
        se = vref.std(ddof=1) / np.sqrt(len(vref))
        print(f"    s.e. of the mean at -5/-9 = {se*100:.2f}pp. "
              f"Whole-grid spread = {spread/se:.2f} s.e.; peak minus -5/-9 = "
              f"{(surf[peak]-prop)/se:.2f} s.e.")
        shape = ("BROAD PLATEAU - the grid cannot distinguish its own cells"
                 if spread < 1.0 * se else
                 ("SHAPED but within noise" if spread < 2.0 * se else
                  "REAL STRUCTURE across the grid"))
        print(f"    -> {shape}")
        onstep = np.array([surf[q] for q in surf
                           if abs(q[0] - PROPOSED[0]) <= 0.0101 and abs(q[1] - PROPOSED[1]) <= 0.0101])
        print(f"    is -5/-9 on a plateau? its +-1-step neighbourhood spans "
              f"{(onstep.max()-onstep.min())/se:.2f} s.e. and sits "
              f"{(prop-np.median(vals))/se:+.2f} s.e. from the grid median -> "
              f"{'YES, indistinguishable from its neighbours' if (onstep.max()-onstep.min()) < se else 'neighbourhood itself is not flat'}")

    # ---- outlier audit + drawdown tolerance
    print("\n[7] ADVERSARIAL AUDIT")
    for name in sigs:
        for per in ("IS", "OOS", "ALL"):
            idx = IDX[name]["ins" if per == "IS" else "oos" if per == "OOS" else "all"]
            if not idx:
                continue
            v, _, _ = event_returns(c, low, n, idx, *PROPOSED)
            o = np.argsort(v)[::-1]
            drop1 = v[o[1:]].mean() if len(v) > 1 else np.nan
            top_d = pd.Timestamp(dates[idx[o[0]]]).date()
            print(f"  {name:<32} {per:<4} mean {v.mean()*100:+6.2f}%  "
                  f"drop best obs ({top_d}, {v[o[0]]*100:+.1f}%) -> {drop1*100:+6.2f}%  "
                  f"zeros(no fill at all) {(v == 0).sum()}/{len(v)}")
    print()
    for name in sigs:
        idx = IDX[name]["all"]
        if not idx:
            continue
        mae = mae_of_fills(c, low, n, idx, *PROPOSED)
        if len(mae) == 0:
            continue
        print(f"  {name:<32} filled tranches n={len(mae)}  MAE median {np.median(mae)*100:.1f}%  "
              f"5th pct {np.percentile(mae,5)*100:.1f}%  worst {mae.min()*100:.1f}%  "
              f"share breaching -25% {(mae <= -0.25).mean()*100:.0f}%")

    print("\n[7b] PORTFOLIO-LEVEL DRAWDOWN (half in each tranche; unfilled half is cash)")
    print("     The plan is 'sized to tolerate -25%'. This is what the position actually saw.")
    for name in sigs:
        for per in ("IS", "OOS", "ALL"):
            idx = IDX[name]["ins" if per == "IS" else "oos" if per == "OOS" else "all"]
            if not idx:
                continue
            pm = portfolio_mae(c, low, n, idx, *PROPOSED)
            if len(pm) == 0:
                continue
            print(f"  {name:<32} {per:<4} n={len(pm):>3}  median {np.median(pm)*100:6.1f}%  "
                  f"5th pct {np.percentile(pm,5)*100:6.1f}%  worst {pm.min()*100:6.1f}%  "
                  f"breached -25%: {(pm <= -0.25).mean()*100:3.0f}%")

    print("\n[7c] UNCONDITIONAL BASELINE (every day, no signal at all)")
    allidx = list(range(n - WINDOW - HOLD - 1))
    bh = np.array([c[i + HOLD] / c[i] - 1 for i in allidx])
    lad, _, _ = event_returns(c, low, n, allidx, *PROPOSED)
    print(f"  buy-and-hold 20d any day : mean {bh.mean()*100:+.2f}%  median {np.median(bh)*100:+.2f}%  "
          f"win {(bh>0).mean()*100:.0f}%  n={len(bh)}")
    print(f"  ladder -5/-9 any day     : mean {lad.mean()*100:+.2f}%  median {np.median(lad)*100:+.2f}%  "
          f"win {(lad>0).mean()*100:.0f}%  n={len(lad)}")
    print("  Any signal EV must clear these, not zero.")

    print("\n[7d] THE HONEST n: EPISODE-LEVEL TEST (overlapping signals inside one selloff = 1 obs)")
    print("     mean of episode means, sign-flip permutation across episodes, two-sided vs 0,")
    print("     and the same for ladder minus buy-at-close.")
    for name in sigs:
        for per in ("OOS", "ALL"):
            idx = IDX[name]["oos" if per == "OOS" else "all"]
            if not idx:
                continue
            ids = episode_ids(dates, idx)
            v, _, _ = event_returns(c, low, n, idx, *PROPOSED)
            bc = np.array([c[i + HOLD] / c[i] - 1 for i in idx])
            m1, p1, ne = block_perm(v, ids)
            m2, p2, _ = block_perm(v - bc, ids)
            print(f"  {name:<32} {per:<4} episodes={ne:<3} ladder {m1*100:+6.2f}% p={p1:.3f}   "
                  f"ladder-minus-close {m2*100:+6.2f}% p={p2:.3f}")
    print("  Every episode count here is far below 50: these can only be counter-examples.")

    print("\n[7e] FILL REALISM: limit price vs open price when the market gaps through the limit")
    global GAP_FILL
    for flag, lbl in ((False, "fill at the limit (as modelled above)"),
                      (True, "fill at the open when it gaps through (more realistic)")):
        GAP_FILL = flag
        row = []
        for name in sigs:
            v, _, _ = event_returns(c, low, n, IDX[name]["all"], *PROPOSED)
            row.append(f"{name.split()[0]} {v.mean()*100:+.2f}%/med {np.median(v)*100:+.2f}%")
        print(f"  {lbl:<52} " + "  ".join(row))
    GAP_FILL = False

    print("\n[8] MULTIPLE-TESTING BOOKKEEPING")
    npairs = sum(1 for a_i, _ in enumerate(GRID) for _ in GRID[a_i + 1:])
    print(f"  grid pairs per signal: {npairs}; signals: {len(sigs)}; "
          f"periods reported: 3 -> {npairs*len(sigs)*3} cells computed here.")
    print(f"  A Bonferroni floor for one grid search alone is p < {0.05/npairs:.5f}.")
    print("  n<50 anywhere below can only produce a counter-example, never a conclusion.")


if __name__ == "__main__":
    main()
