"""ADVERSARIAL VERIFICATION of analysis/bt_walk_forward.py.

Re-derives the headline numbers by a DIFFERENT code path and stress-tests the
four load-bearing claims.  Nothing here reuses their tranche()/event_returns().
"""
from __future__ import annotations
import numpy as np, pandas as pd
from analysis.data import load_merged

HOLD, WINDOW = 20, 20
PROP = (-0.05, -0.09)
SPLIT = pd.Timestamp("2018-01-01")
RNG = np.random.default_rng(7777)
NPERM = 20000
LAST = dict(date="2026-09-14", open=101.53, high=105.39, low=99.87, close=101.13, qqq=711.96)


def frame():
    m = load_merged()
    d = pd.Timestamp(LAST["date"])
    row = pd.Series({"open": LAST["open"], "high": LAST["high"], "low": LAST["low"],
                     "close": LAST["close"], "volume": np.nan, "qqq": LAST["qqq"]}, name=d)
    m = pd.concat([m, row.to_frame().T]).sort_index()
    for c in ("open", "high", "low", "close"):
        m[c] = m[c].astype(float)
    m = m[m.index >= "2010-03-12"]
    return m


# ---- DIFFERENT fill engine: vectorised argmax on a boolean window, not a for-loop
def fill_idx_vec(close, low, i_arr, off):
    """First j in (i, i+WINDOW] with low[j] <= close[i]*(1+off); -1 if none."""
    n = len(close)
    out = np.full(len(i_arr), -1, dtype=int)
    for k, i in enumerate(i_arr):
        px = close[i] * (1 + off)
        hi = min(i + 1 + WINDOW, n)
        w = low[i + 1:hi] <= px
        if w.any():
            out[k] = i + 1 + int(np.argmax(w))
    return out


def ladder_ret(close, low, i_arr, o1, o2):
    j1 = fill_idx_vec(close, low, i_arr, o1)
    j2 = fill_idx_vec(close, low, i_arr, o2)
    p1 = close[i_arr] * (1 + o1); p2 = close[i_arr] * (1 + o2)
    r1 = np.where(j1 >= 0, close[np.clip(j1 + HOLD, 0, len(close) - 1)] / p1 - 1, 0.0)
    r2 = np.where(j2 >= 0, close[np.clip(j2 + HOLD, 0, len(close) - 1)] / p2 - 1, 0.0)
    return 0.5 * r1 + 0.5 * r2, j1, j2


def paired_p(d, nperm=NPERM):
    obs = d.mean()
    s = RNG.choice([-1.0, 1.0], size=(nperm, len(d)))
    null = (s * d).mean(axis=1)
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1)


def episodes(dates, idx, gap_trading_days):
    """Group by TRADING-day gap (their version used calendar days * 1.6)."""
    ids, k, prev = [], 0, None
    for i in idx:
        if prev is not None and (i - prev) > gap_trading_days:
            k += 1
        ids.append(k); prev = i
    return np.array(ids)


def block_p(v, ids, nperm=NPERM):
    ep = np.array([v[ids == k].mean() for k in np.unique(ids)])
    obs = ep.mean()
    s = RNG.choice([-1.0, 1.0], size=(nperm, len(ep)))
    null = (s * ep).mean(axis=1)
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1), len(ep)


def main():
    m = frame()
    close = m["close"].values; low = m["low"].values; op = m["open"].values
    dates = m.index.values; n = len(m)

    # signals rebuilt from the MERGED frame's own columns where possible
    cl = m["close"]
    ret = cl.pct_change().values                       # data.py computes the same from prev_close
    gap = (m["open"].values / np.r_[np.nan, close[:-1]]) - 1
    dd250 = (cl / cl.rolling(250, min_periods=60).max() - 1).values
    ret20 = cl.pct_change(20).values
    S = {"a": ret <= -0.13, "b": gap <= -0.12, "c": (dd250 <= -0.55) & (ret20 <= -0.25)}
    U = {k: np.array([i for i in np.where(v)[0] if i + WINDOW + HOLD < n]) for k, v in S.items()}
    OOS = {k: v[pd.to_datetime(dates[v]) >= SPLIT] for k, v in U.items()}
    INS = {k: v[pd.to_datetime(dates[v]) < SPLIT] for k, v in U.items()}

    print("=" * 92)
    print("A. INDEPENDENT RE-DERIVATION (different fill engine, different signal construction)")
    print("=" * 92)
    print(f"bars {n}, last {pd.Timestamp(dates[-1]).date()}")
    for k in "abc":
        print(f"  signal({k}): all={int(S[k].sum())} usable={len(U[k])} IS={len(INS[k])} OOS={len(OOS[k])}")

    print("\n  headline cell -- (c) OOS ladder(-5/-9) vs buy-at-close, hold 20d:")
    idx = OOS["c"]
    v, j1, j2 = ladder_ret(close, low, idx, *PROP)
    bc = close[idx + HOLD] / close[idx] - 1
    bo = close[idx + 1 + HOLD] / op[idx + 1] - 1
    d, p = paired_p(v - bc)
    d2, p2 = paired_p(v - bo)
    print(f"    ladder  mean {v.mean()*100:+.2f}%  median {np.median(v)*100:+.2f}%  win {(v>0).mean()*100:.0f}%  n={len(v)}")
    print(f"    close   mean {bc.mean()*100:+.2f}%  median {np.median(bc)*100:+.2f}%  win {(bc>0).mean()*100:.0f}%")
    print(f"    nextopen mean {bo.mean()*100:+.2f}%  median {np.median(bo)*100:+.2f}%")
    print(f"    diff vs close  {d*100:+.2f}pp  p={p:.3f}   [they report -4.50pp p=0.087]")
    print(f"    diff vs open   {d2*100:+.2f}pp  p={p2:.3f}   [they report -5.46pp p=0.017]")
    print(f"    MEDIAN diff vs close {np.median(v-bc)*100:+.2f}pp  (mean-vs-median check)")

    print("\n  how many DISTINCT trades do those 142 'events' actually contain?")
    pair = set(zip(j1.tolist(), j2.tolist()))
    print(f"    distinct (fill1,fill2) index pairs = {len(pair)} out of {len(idx)} events")
    print(f"    distinct fill-1 days = {len(set(j1.tolist()))}, distinct fill-2 days = {len(set(j2.tolist()))}")

    print("\n" + "=" * 92)
    print("B. CLAIM 3 STRESS TEST -- s.e. of the LEVEL vs s.e. of the DIFFERENCE")
    print("   Their flatness test compares a cell-to-cell GAP against the s.e. of one cell's MEAN.")
    print("   The cells share the same events, so the honest yardstick is the PAIRED s.e.")
    print("=" * 92)
    GRID = [-i / 100 for i in range(2, 16)]
    for k, peak in (("a", (-0.11, -0.12)), ("b", (-0.10, -0.11)), ("c", (-0.06, -0.09))):
        i_all = U[k]
        vp, _, _ = ladder_ret(close, low, i_all, *PROP)
        vk, _, _ = ladder_ret(close, low, i_all, *peak)
        se_lvl = vp.std(ddof=1) / np.sqrt(len(vp))
        dd = vk - vp
        se_pair = dd.std(ddof=1) / np.sqrt(len(dd))
        obs, pv = paired_p(dd)
        # full grid, paired against proposal
        best_t, best_p = None, 1.0
        cells = {}
        for ai, o1 in enumerate(GRID):
            for o2 in GRID[ai + 1:]:
                vv, _, _ = ladder_ret(close, low, i_all, o1, o2)
                cells[(o1, o2)] = vv
        vals = np.array([c_.mean() for c_ in cells.values()])
        # paired t-like z for every cell vs proposal
        zs = []
        for key, vv in cells.items():
            dq = vv - vp
            if dq.std(ddof=1) == 0:
                zs.append((0.0, key)); continue
            zs.append((dq.mean() / (dq.std(ddof=1) / np.sqrt(len(dq))), key))
        zs.sort()
        print(f"\n  signal({k})  n={len(i_all)}  grid spread {(vals.max()-vals.min())*100:.2f}pp")
        print(f"    s.e. of the LEVEL at -5/-9        = {se_lvl*100:5.2f}pp   (their yardstick)")
        print(f"    PAIRED s.e. of (peak minus -5/-9) = {se_pair*100:5.2f}pp   ({se_lvl/se_pair:.1f}x smaller)")
        print(f"    peak-minus-proposal = {obs*100:+.2f}pp = {obs/se_pair:.2f} PAIRED s.e., "
              f"sign-flip p={pv:.4f}  (they quote {(vk.mean()-vp.mean())/se_lvl:.2f} level-s.e.)")
        print(f"    grid spread in PAIRED s.e. units  = {(vals.max()-vals.min())/se_pair:.2f} s.e. "
              f"(they say {(vals.max()-vals.min())/se_lvl:.2f})")
        print(f"    most extreme cell vs -5/-9: z={zs[-1][0]:+.2f} at {zs[-1][1][0]*100:.0f}/{zs[-1][1][1]*100:.0f}, "
              f"z={zs[0][0]:+.2f} at {zs[0][1][0]*100:.0f}/{zs[0][1][1]*100:.0f}")

    print("\n" + "=" * 92)
    print("C. CLAIM 4 STRESS TEST -- breach rate denominator")
    print("   Their portfolio_mae() does `if not legs: continue`, dropping no-fill events.")
    print("=" * 92)

    def pmae(idx_):
        out = []
        for i in idx_:
            legs = []
            for off in PROP:
                px = close[i] * (1 + off)
                hi = min(i + 1 + WINDOW, n)
                w = low[i + 1:hi] <= px
                if w.any():
                    j = i + 1 + int(np.argmax(w))
                    legs.append((px, j, j + HOLD))
            if not legs:
                out.append(0.0)   # no fill -> position never existed -> 0% drawdown
                continue
            lo_ = min(f for _, f, _ in legs); hi_ = max(e for _, _, e in legs)
            worst = 0.0
            for t in range(lo_, hi_ + 1):
                vv = sum(0.5 * (low[t] / px - 1) for px, f, e in legs if f <= t <= e)
                worst = min(worst, vv)
            out.append(worst)
        return np.array(out)

    for k in "abc":
        for lbl, ii in (("IS", INS[k]), ("OOS", OOS[k]), ("ALL", U[k])):
            if len(ii) == 0:
                continue
            allp = pmae(ii)
            filled = allp[allp != 0.0]
            print(f"  ({k}) {lbl:<4} ALL events n={len(allp):>3} breach {(allp<=-0.25).mean()*100:3.0f}% "
                  f"median {np.median(allp)*100:6.1f}%   |   FILLED-ONLY n={len(filled):>3} "
                  f"breach {(filled<=-0.25).mean()*100:3.0f}% median {np.median(filled)*100:6.1f}%  <- theirs")

    print("\n" + "=" * 92)
    print("D. CLAIM 5 STRESS TEST -- period-matched baselines")
    print("   They compare OOS ladder EV (+5.10%) to a FULL-HISTORY buy-any-day mean (+5.44%).")
    print("=" * 92)
    pool_all = np.arange(n - WINDOW - HOLD - 1)
    dts = pd.to_datetime(dates[pool_all])
    for lbl, sel in (("2010-2026 (theirs)", np.ones(len(pool_all), bool)),
                     ("2010-2017 IS", np.asarray(dts < SPLIT)),
                     ("2018-2026 OOS", np.asarray(dts >= SPLIT))):
        pp = pool_all[sel]
        bh = close[pp + HOLD] / close[pp] - 1
        lad, _, _ = ladder_ret(close, low, pp, *PROP)
        print(f"  {lbl:<20} buy-any-day mean {bh.mean()*100:+.2f}% median {np.median(bh)*100:+.2f}% "
              f"win {(bh>0).mean()*100:.0f}%  |  ladder-any-day mean {lad.mean()*100:+.2f}% "
              f"median {np.median(lad)*100:+.2f}%  n={len(pp)}")
    va, _, _ = ladder_ret(close, low, OOS["a"], *PROP)
    print(f"  signal(a) OOS ladder = {va.mean()*100:+.2f}%  vs OOS ladder-any-day and OOS buy-any-day above.")

    print("\n" + "=" * 92)
    print("E. CLAIM 2 STRESS TEST -- does the episode-level p=0.046 survive a stricter episode?")
    print("   Theirs collapses signals >32 CALENDAR days apart. A 20d order window + 20d hold")
    print("   means two 'episodes' 33 days apart still share overlapping outcome windows.")
    print("=" * 92)
    for k in "abc":
        for lbl, ii in (("OOS", OOS[k]), ("ALL", U[k])):
            v, _, _ = ladder_ret(close, low, ii, *PROP)
            bc = close[ii + HOLD] / close[ii] - 1
            row = [f"  ({k}) {lbl:<4}"]
            for g in (20, 40, 60, 120):
                ids = episodes(dates, ii, g)
                mm, pp2, ne = block_p(v - bc, ids)
                row.append(f"gap{g:>3}d: ne={ne:<3} {mm*100:+6.2f}pp p={pp2:.3f}")
            print("  ".join(row))

    print("\n" + "=" * 92)
    print("F. DECOMPOSITION -- is the ladder's OOS loss the FILL RISK or the PRICE?")
    print("=" * 92)
    for k in "abc":
        ii = OOS[k]
        v, j1, j2 = ladder_ret(close, low, ii, *PROP)
        bc = close[ii + HOLD] / close[ii] - 1
        both = (j1 >= 0) & (j2 >= 0)
        none_ = (j1 < 0) & (j2 < 0)
        print(f"  ({k}) OOS n={len(ii)}: both filled {both.sum()}  no fill {none_.sum()}  partial {len(ii)-both.sum()-none_.sum()}")
        if both.sum():
            dfull = v[both] - bc[both]
            print(f"       conditional on BOTH tranches filling: ladder {v[both].mean()*100:+.2f}% "
                  f"vs close {bc[both].mean()*100:+.2f}%  diff {dfull.mean()*100:+.2f}pp "
                  f"(median {np.median(dfull)*100:+.2f}pp)")
        if none_.sum():
            print(f"       on NO-FILL events buy-at-close earned mean {bc[none_].mean()*100:+.2f}% "
                  f"median {np.median(bc[none_])*100:+.2f}% -- the ladder books 0 there")

    print("\n" + "=" * 92)
    print("G. OUTLIER / REGIME AUDIT of the (c) OOS -4.50pp")
    print("=" * 92)
    ii = OOS["c"]
    v, _, _ = ladder_ret(close, low, ii, *PROP)
    bc = close[ii + HOLD] / close[ii] - 1
    d = v - bc
    yr = pd.to_datetime(dates[ii]).year
    for y in sorted(set(yr)):
        s = yr == y
        print(f"  {y}: n={s.sum():>3}  ladder {v[s].mean()*100:+7.2f}%  close {bc[s].mean()*100:+7.2f}%  "
              f"diff {d[s].mean()*100:+7.2f}pp")
    o = np.argsort(d)
    print(f"  worst single event for the ladder: {pd.Timestamp(dates[ii[o[0]]]).date()} diff {d[o[0]]*100:+.1f}pp")
    print(f"  drop the 2020 block: diff = {d[yr != 2020].mean()*100:+.2f}pp (n={(yr!=2020).sum()})")
    print(f"  drop the 2022 block: diff = {d[yr != 2022].mean()*100:+.2f}pp (n={(yr!=2022).sum()})")
    print(f"  drop the 2025 block: diff = {d[yr != 2025].mean()*100:+.2f}pp (n={(yr!=2025).sum()})")

    print("\n" + "=" * 92)
    print("H. TODAY'S BAR -- does 2026-09-14 really fire all three?")
    print("=" * 92)
    i = n - 1
    print(f"  ret={ret[i]*100:.2f}%  gap={gap[i]*100:.2f}%  dd250={dd250[i]*100:.2f}%  ret20={ret20[i]*100:.2f}%")
    print(f"  (a) {ret[i]<=-0.13}   (b) {gap[i]<=-0.12}   (c) {(dd250[i]<=-0.55) and (ret20[i]<=-0.25)}")
    print(f"  250d rolling high = {cl.rolling(250,min_periods=60).max().values[i]:.2f}, close {close[i]:.2f}")


if __name__ == "__main__":
    main()
