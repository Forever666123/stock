"""The tails, not the medians: what the extreme case looks like for each of the
three actions available at the next session.

    python3 -m analysis.bt_tails

MECHANISM STATED BEFORE THE NUMBERS
-----------------------------------
A 3x daily-reset semiconductor ETF has no floor mechanism. Its one-day loss is
3x the index move plus the drag of daily rebalancing, and its multi-day loss
compounds that. So the prior is:

1. The left tail of a leveraged fund is fatter than the right tail is long, and
   the fat part is CLUSTERED: the worst 1-day, 5-day and 20-day outcomes come
   out of a handful of episodes (Aug-2015, Feb/Mar-2020, 2022, Apr-2025, the
   2026 selloff). Percentiles from a clustered sample are not 92 independent
   draws. Episode counts are printed next to every n for that reason.
2. Buying earlier means more exposure to that tail, not less: the open entry
   carries the whole path, the ladder carries only the part below its levels
   (and carries nothing at all if price never gets there), the post-decision
   close entry skips two sessions of it. Any ranking of the three by MEAN on
   ~30 episodes is noise; the ranking by TAIL DEPTH is mechanical and does not
   need a p-value.
3. Everything that classifies an event here is observable at the signal close
   (return, gap, drawdown). Nothing gates an entry on a number that could only
   be known later. Fills are decided on the intraday low, liquidations too --
   the same mechanic as analysis/stops.py.

WHAT THIS CANNOT DO
-------------------
n<50 populations (gap <= -12%, n=22 / 11 episodes) can produce a counter-example
and nothing else. A 1st percentile on n=22 IS the worst observation. Said again
in the output wherever it applies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import load_merged
from analysis.fomc import fomc_table

RNG = np.random.default_rng(20260915)
NPERM = 20000

# The most recent bar is not in the CSV.
LAST_BAR = dict(date="2026-09-14", open=101.53, high=105.39, low=99.87,
                close=101.13, qqq=711.96)
REF_CLOSE = 101.13          # 2026-09-14 close, the price everything is measured from
NEXT_OPEN_IND = 101.50      # overnight indication for 2026-09-15, ~flat
LADDER = (96.0, 92.0)       # resting limit orders, half the position each
WINDOW = 20                 # sessions the ladder orders rest before cancellation
HORIZONS = (1, 5, 20)
PCT = (1, 5, 25, 50, 75, 95, 99)

# From analysis/stops.py: liquidation fires on the intraday low at these drawdowns.
LIQ = [("10x", -0.095), ("5x", -0.195), ("3x", -0.328), ("2x", -0.495), ("1x cash", -1.0)]
# The same constants as a formula: equity is wiped at -(1/L) plus a ~0.5% cushion.
def liq_level(L: float) -> float:
    return -(1.0 / L) + 0.005
def max_leverage(mae: float) -> float:
    """Largest L whose liquidation level sits strictly below a drawdown of `mae`."""
    room = abs(mae) + 0.005
    return 1.0 / room if room > 0 else np.inf


# --------------------------------------------------------------------------- data
def frame() -> pd.DataFrame:
    m = load_merged()
    d = pd.Timestamp(LAST_BAR["date"])
    assert d not in m.index, "2026-09-14 already present -- do not double-append"
    m.loc[d, ["open", "high", "low", "close", "qqq"]] = [
        LAST_BAR["open"], LAST_BAR["high"], LAST_BAR["low"],
        LAST_BAR["close"], LAST_BAR["qqq"]]
    m = m.sort_index()
    for col in ("open", "high", "low", "close"):
        m[col] = m[col].astype(float)
    c = m["close"]
    m["ret"] = c / c.shift(1) - 1
    m["gap"] = m["open"] / c.shift(1) - 1
    m["dd250"] = c / c.rolling(250, min_periods=60).max() - 1
    m["r20"] = c.pct_change(20)
    return m.reset_index().rename(columns={"index": "Date"})


def populations(m: pd.DataFrame) -> dict:
    return {
        "day <= -13%": (m["ret"] <= -0.13).values,
        "gap <= -12%": (m["gap"] <= -0.12).values,
        "deep dd (dd250<=-55% & 20d<=-25%)": ((m["dd250"] <= -0.55) & (m["r20"] <= -0.25)).values,
    }


def episodes(ii, gap=20) -> int:
    """Distinct clusters: a new episode starts after `gap` quiet sessions."""
    ii = np.asarray(sorted(ii))
    if ii.size == 0:
        return 0
    return 1 + int((np.diff(ii) > gap).sum())


# ---------------------------------------------------------------------- actions
def act_open(c, o, lo, n, i, h):
    """(1) Buy the full position at the next open; hold h sessions incl. the entry day."""
    j = i + 1                      # entry bar
    e = j + h - 1                  # exit bar
    if e >= n:
        return None
    px = o[j]
    return dict(entry=px, ret=c[e] / px - 1, mae=lo[j:e + 1].min() / px - 1,
                exit_i=e, deployed=1.0)


def act_close_d0(c, lo, n, i, h):
    """(3) Buy nothing until after the decision, then buy the D0 close.
    2026-09-14 is the signal, 09-15 is D-1, 09-16 is D0 -> entry = close[i+2]."""
    j = i + 2
    e = j + h
    if e >= n:
        return None
    px = c[j]
    return dict(entry=px, ret=c[e] / px - 1, mae=lo[j + 1:e + 1].min() / px - 1,
                exit_i=e, deployed=1.0)


def act_ladder(c, lo, n, i, h, levels=LADDER, ref=REF_CLOSE, window=WINDOW):
    """(2) Two resting limit orders, half the position each, priced off the signal
    close. A tranche fills when the intraday low touches it, then is held h
    sessions FROM ITS OWN FILL. A tranche that never fills contributes 0 (cash).

    Two P&L conventions are returned:
      ret_acct     -- on the full intended position size, unfilled half = cash
      ret_deployed -- on the capital that actually got in (NaN if nothing filled)
    MAE is the account-equity drawdown, which is what a broker liquidates on.
    """
    offs = [p / ref - 1 for p in levels]
    w = 1.0 / len(offs)
    fills, rets, last = [], [], i
    for off in offs:
        px = c[i] * (1 + off)
        j = next((k for k in range(i + 1, min(i + 1 + window, n)) if lo[k] <= px), None)
        if j is None:
            fills.append(None); rets.append(0.0); continue
        e = j + h - 1              # h sessions of exposure incl. the fill day
        if e >= n:
            return None
        fills.append((j, px)); rets.append(c[e] / px - 1); last = max(last, e)
    filled = [f for f in fills if f is not None]
    deployed = w * len(filled)
    # account-equity path: cash for what has not filled yet, marked at the low
    worst = 0.0
    for t in range(i + 1, last + 1):
        eq = 0.0
        for f in fills:
            if f is not None and f[0] <= t:
                eq += w * (lo[t] / f[1] - 1)
        worst = min(worst, eq)
    return dict(entry=(np.mean([f[1] for f in filled]) if filled else np.nan),
                ret=float(np.sum([w * r for r in rets])),
                ret_deployed=(float(np.sum([w * r for r in rets])) / deployed if deployed else np.nan),
                mae=worst, deployed=deployed, nfill=len(filled),
                which=tuple(k for k, f in enumerate(fills) if f is not None))


# ---------------------------------------------------------------------- reporting
def qline(x):
    x = np.asarray(x, float)
    return np.percentile(x, PCT)


def dist_row(label, x, dates, extra=""):
    x = np.asarray(x, float)
    q = qline(x)
    k = int(np.argmin(x))
    cells = " | ".join(f"{v*100:+6.1f}%" for v in q)
    return (f"| {label} | {len(x)} | {cells} | {x.mean()*100:+.1f}% | "
            f"{x.min()*100:+.1f}% ({dates[k]}) |{extra}")


HEAD = ("| what | n | p1 | p5 | p25 | p50 | p75 | p95 | p99 | mean | worst (date) |")
SEP = "|---|---|---|---|---|---|---|---|---|---|---|"


def outlier_note(x, dates):
    """Which single observation is carrying the mean."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return ""
    k = int(np.argmax(np.abs(x - np.median(x))))
    drop = np.delete(x, k)
    return (f"mean {x.mean()*100:+.1f}% -> {drop.mean()*100:+.1f}% without "
            f"{dates[k]} ({x[k]*100:+.1f}%)")


def paired_perm(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if d.size < 3:
        return np.nan, np.nan, np.nan, d.size
    obs = d.mean()
    s = RNG.choice([-1.0, 1.0], size=(NPERM, d.size))
    null = (s * d).mean(axis=1)
    p = (1 + (np.abs(null) >= abs(obs)).sum()) / (NPERM + 1)
    return obs, float(np.median(d)), p, d.size


def holm(ps):
    order = sorted(range(len(ps)), key=lambda k: ps[k][1])
    k, adj, prev = len(ps), {}, 0.0
    for rank, idx in enumerate(order):
        name, p = ps[idx]
        v = max(prev, min(1.0, (k - rank) * p))
        adj[name] = v; prev = v
    return adj


def main():
    m = frame()
    c, o, hi, lo = (m["close"].values, m["open"].values, m["high"].values, m["low"].values)
    n = len(m)
    dates = m["Date"].dt.strftime("%Y-%m-%d").values
    pops = populations(m)
    sig = n - 1                       # 2026-09-14 itself

    print("=" * 96)
    print("SOXL TAIL BOOK -- signal bar 2026-09-14: close 101.13, day %.2f%%, gap %.2f%%, "
          % (m['ret'].iloc[-1] * 100, m['gap'].iloc[-1] * 100))
    print("  dd250 %.1f%%, 20d %.1f%%.  2026-09-15 = FOMC D-1, 2026-09-16 = D0."
          % (m['dd250'].iloc[-1] * 100, m['r20'].iloc[-1] * 100))
    print("  Today is a member of ALL THREE reference populations below.")
    print("=" * 96)

    print("\n### 0. Reference populations (the signal day is excluded from its own sample)\n")
    print("| population | n days | independent episodes | span |")
    print("|---|---|---|---|")
    idx = {}
    for k, mask in pops.items():
        ii = [i for i in np.where(mask)[0] if i < sig]
        idx[k] = ii
        yrs = sorted({dates[i][:4] for i in ii})
        print(f"| {k} | {len(ii)} | {episodes(ii)} | {yrs[0]}-{yrs[-1]} |")
    print("\nn<50 RULE: 'gap <= -12%' has n=22 over 11 episodes. Every number in that")
    print("column is a COUNTER-EXAMPLE, never a conclusion. Its p1 IS its worst")
    print("observation (1% of 22 is 0.2 observations). The other two columns are")
    print("n>=50 by day count but only 33 and 17 by episode -- treat their p1/p99 the")
    print("same way.")

    # ---------------------------------------------------------- 1. the price path
    print("\n" + "=" * 96)
    print("### 1. THE PRICE PATH ITSELF (action-independent), translated from 101.13")
    print("=" * 96)
    for k, ii in idx.items():
        print(f"\n-- {k}  (n={len(ii)}, {episodes(ii)} episodes)")
        print("| horizon | n | " + " | ".join(f"p{p}" for p in PCT) + " | worst (date) |")
        print("|---|---|" + "---|" * (len(PCT) + 1))
        for h in HORIZONS:
            jj = [i for i in ii if i + h < n]
            r = np.array([c[i + h] / c[i] - 1 for i in jj])
            q = qline(r); kk = int(np.argmin(r))
            print(f"| close +{h}d, %% | {len(jj)} | " + " | ".join(f"{v*100:+.1f}%" for v in q)
                  + f" | {r.min()*100:+.1f}% ({dates[jj[kk]]}) |")
            print(f"| close +{h}d, PRICE | | " + " | ".join(f"{REF_CLOSE*(1+v):.2f}" for v in q)
                  + f" | {REF_CLOSE*(1+r.min()):.2f} |")
        for h in HORIZONS:
            jj = [i for i in ii if i + h < n]
            r = np.array([lo[i + 1:i + h + 1].min() / c[i] - 1 for i in jj])
            q = qline(r); kk = int(np.argmin(r))
            print(f"| LOWEST LOW in {h}d, %% | {len(jj)} | " + " | ".join(f"{v*100:+.1f}%" for v in q)
                  + f" | {r.min()*100:+.1f}% ({dates[jj[kk]]}) |")
            print(f"| LOWEST LOW in {h}d, PRICE | | " + " | ".join(f"{REF_CLOSE*(1+v):.2f}" for v in q)
                  + f" | {REF_CLOSE*(1+r.min()):.2f} |")

    # ------------------------------------------------------------- 2. per action
    print("\n" + "=" * 96)
    print("### 2. OUTCOME DISTRIBUTION BY ACTION")
    print("Horizon h = h sessions of exposure counted from the entry bar.")
    print("(1) open   : entry = next open, exit = close of the h-th session")
    print("(2) ladder : 96 and 92, half each, orders rest 20 sessions, each tranche")
    print("             held h sessions from ITS OWN fill; unfilled half = cash = 0")
    print("(3) D0close: entry = close[i+2] (9/14 -> D-1 -> D0), exit = close[i+2+h]")
    print("=" * 96)

    store = {}
    for k, ii in idx.items():
        print(f"\n################ {k}  (n={len(ii)} days / {episodes(ii)} episodes)")
        for h in HORIZONS:
            print(f"\n-- hold {h} session(s)")
            print(HEAD); print(SEP)
            rows = {}
            for aname, fn in (("(1) open", lambda i: act_open(c, o, lo, n, i, h)),
                              ("(2) ladder", lambda i: act_ladder(c, lo, n, i, h)),
                              ("(3) D0close", lambda i: act_close_d0(c, lo, n, i, h))):
                res = [(i, fn(i)) for i in ii]
                res = [(i, r) for i, r in res if r is not None]
                if not res:
                    continue
                rr = np.array([r["ret"] for _, r in res])
                mm = np.array([r["mae"] for _, r in res])
                dd = [dates[i] for i, _ in res]
                rows[aname] = dict(ret=rr, mae=mm, dates=dd, res=res)
                print(dist_row(f"{aname} P&L", rr, dd))
                print(dist_row(f"{aname} MAE", mm, dd))
            store[(k, h)] = rows
            for aname, v in rows.items():
                note = outlier_note(v["ret"], v["dates"])
                print(f"   {aname}: {note}")
            if "(2) ladder" in rows:
                res = rows["(2) ladder"]["res"]
                dep = np.array([r["deployed"] for _, r in res])
                nf = np.array([r["nfill"] for _, r in res])
                both = (nf == 2).mean(); one = (nf == 1).mean(); none = (nf == 0).mean()
                dpl = np.array([r["ret_deployed"] for _, r in res], float)
                dpl = dpl[~np.isnan(dpl)]
                print(f"   (2) ladder fills: neither {none*100:.0f}% | 96 only {one*100:.0f}% | "
                      f"both {both*100:.0f}% | avg capital deployed {dep.mean()*100:.0f}%")
                if dpl.size:
                    q = np.percentile(dpl, PCT)
                    print("   (2) ladder P&L ON DEPLOYED CAPITAL ONLY (n=%d): " % dpl.size
                          + " ".join(f"p{p}={v*100:+.1f}%" for p, v in zip(PCT, q))
                          + f" | worst {dpl.min()*100:+.1f}%")

    # ------------------------------------------- 3. price translation of p1 / p5
    print("\n" + "=" * 96)
    print("### 3. THE 1st AND 5th PERCENTILE AS AN ACTUAL SOXL PRICE")
    print(f"Assumed entry prices: (1) next open {NEXT_OPEN_IND:.2f} (overnight indication);")
    print("(2) fills at 96.00 and 92.00 (average 94.00 when both fill);")
    print("(3) D0 close is unknown ex-ante, so its price column uses the p1/p5 of the")
    print("    close[i+2] distribution as the entry and compounds from there.")
    print("=" * 96)
    for k, ii in idx.items():
        print(f"\n-- {k}")
        print("| action | h | p1 P&L | p1 exit price | p5 P&L | p5 exit price | p1 MAE | p1 low touched |")
        print("|---|---|---|---|---|---|---|---|")
        for h in HORIZONS:
            rows = store[(k, h)]
            for aname, v in rows.items():
                q = np.percentile(v["ret"], [1, 5]); qm = np.percentile(v["mae"], [1])
                if aname == "(1) open":
                    base = NEXT_OPEN_IND
                elif aname == "(2) ladder":
                    base = float(np.nanmean([r["entry"] for _, r in v["res"]]))
                else:
                    jj = [i for i in ii if i + 2 < n]
                    base = REF_CLOSE * (1 + np.percentile([c[i + 2] / c[i] - 1 for i in jj], 5))
                print(f"| {aname} | {h} | {q[0]*100:+.1f}% | {base*(1+q[0]):.2f} | "
                      f"{q[1]*100:+.1f}% | {base*(1+q[1]):.2f} | {qm[0]*100:+.1f}% | "
                      f"{base*(1+qm[0]):.2f} |")

    # ----------------------------------------------- 4. the narrow next-day question
    print("\n" + "=" * 96)
    print("### 4. NARROW QUESTION: worst single-day move and worst gap AFTER a day like 9/14")
    print("=" * 96)
    for k, ii in idx.items():
        jj = [i for i in ii if i + 1 < n]
        nd = np.array([c[i + 1] / c[i] - 1 for i in jj])
        ng = np.array([o[i + 1] / c[i] - 1 for i in jj])
        nl = np.array([lo[i + 1] / c[i] - 1 for i in jj])
        print(f"\n-- {k} (n={len(jj)}, {episodes(jj)} episodes)")
        for label, arr in (("next-day close-to-close", nd), ("next-day GAP (open vs close)", ng),
                           ("next-day intraday LOW", nl)):
            q = np.percentile(arr, PCT)
            ordr = np.argsort(arr)[:3]
            worst = ", ".join(f"{dates[jj[t]]}->{dates[jj[t]+1]} {arr[t]*100:+.1f}%" for t in ordr)
            print(f"   {label:30s} " + " ".join(f"p{p}={v*100:+.1f}%" for p, v in zip(PCT, q)))
            print(f"   {'':30s} 3 worst: {worst}")
            print(f"   {'':30s} from 101.13 -> p1 {REF_CLOSE*(1+q[0]):.2f}, "
                  f"p5 {REF_CLOSE*(1+q[1]):.2f}, worst {REF_CLOSE*(1+arr.min()):.2f}")
        print(f"   share of next days that were themselves <= -13%: "
              f"{(nd <= -0.13).mean()*100:.0f}% ({int((nd<=-0.13).sum())}/{len(nd)})")

    # unconditional worst days ever, for scale
    r_all = m["ret"].values[1:]; g_all = m["gap"].values[1:]
    d_all = dates[1:]
    ko = np.argsort(r_all)[:5]; kg = np.argsort(g_all)[:5]
    print("\n-- for scale, the 5 worst days and 5 worst gaps in the whole 2010-2026 history")
    print("   worst days : " + ", ".join(f"{d_all[t]} {r_all[t]*100:+.1f}%" for t in ko))
    print("   worst gaps : " + ", ".join(f"{d_all[t]} {g_all[t]*100:+.1f}%" for t in kg))

    # back-to-back check
    ii13 = [i for i in np.where(pops["day <= -13%"])[0] if i < sig and i + 1 < n]
    b2b = [(dates[i], dates[i + 1], (c[i + 1] / c[i] - 1)) for i in ii13 if c[i + 1] / c[i] - 1 <= -0.13]
    print(f"\n   back-to-back -13% days (the thing that actually kills a levered entry): "
          f"{len(b2b)} of {len(ii13)}")
    for a, b, r in sorted(b2b, key=lambda t: t[2])[:8]:
        print(f"     {a} -> {b}  {r*100:+.1f}%")

    # ------------------------------------------------------- 5. leverage from MAE
    print("\n" + "=" * 96)
    print("### 5. LEVERAGE SIZING -- survive the 5th-percentile path without forced exit")
    print("Liquidation fires on the INTRADAY LOW (analysis/stops.py mechanic).")
    print("Check the formula reproduces stops.py: " +
          ", ".join(f"{nm}->{liq_level(float(nm[:-1])) if nm[-1]=='x' else -1.0:.3f}"
                    for nm, _ in LIQ[:4]))
    print("Max leverage L = 1 / (|MAE| + 0.005).")
    print("=" * 96)
    for k, ii in idx.items():
        print(f"\n-- {k}")
        print("| action | h | MAE p50 | MAE p5 | MAE p1 | MAE worst | Lmax @p5 | Lmax @p1 | Lmax @worst |")
        print("|---|---|---|---|---|---|---|---|---|")
        for h in HORIZONS:
            for aname, v in store[(k, h)].items():
                mm = v["mae"]
                p50, p5, p1 = np.percentile(mm, [50, 5, 1])
                w = mm.min()
                print(f"| {aname} | {h} | {p50*100:+.1f}% | {p5*100:+.1f}% | {p1*100:+.1f}% | "
                      f"{w*100:+.1f}% | {max_leverage(p5):.2f}x | {max_leverage(p1):.2f}x | "
                      f"{max_leverage(w):.2f}x |")
    print("\n-- the user's stated tolerance is a -25% drawdown. Probability the MAE is")
    print("   worse than -25% (h=20), i.e. the probability that plan is breached:")
    print("| action | " + " | ".join(idx) + " |")
    print("|---|" + "---|" * len(idx))
    for aname in ("(1) open", "(2) ladder", "(3) D0close"):
        cells = []
        for k in idx:
            rows = store[(k, 20)]
            if aname not in rows:
                cells.append("-"); continue
            mm = rows[aname]["mae"]
            cells.append(f"{(mm <= -0.25).mean()*100:.0f}% ({int((mm<=-0.25).sum())}/{len(mm)})")
        print(f"| {aname} | " + " | ".join(cells) + " |")
    print("\n-- liquidation probability at each leverage, h=20, using the same MAE samples:")
    for k in idx:
        print(f"\n   {k}")
        print("   | leverage | liq level | " + " | ".join(("(1) open", "(2) ladder", "(3) D0close")) + " |")
        print("   |---|---|---|---|---|")
        for nm, lv in LIQ:
            cells = []
            for aname in ("(1) open", "(2) ladder", "(3) D0close"):
                rows = store[(k, 20)]
                mm = rows[aname]["mae"] if aname in rows else None
                cells.append("-" if mm is None else f"{(mm <= lv).mean()*100:.0f}%")
            print(f"   | {nm} | {lv*100:.1f}% | " + " | ".join(cells) + " |")

    # --------------------------------------------------- 6. is any pair different?
    print("\n" + "=" * 96)
    print("### 6. DOES ANY PAIR ACTUALLY DIFFER? paired sign-flip permutation, common events")
    print("=" * 96)
    ps, detail = [], []
    for k, ii in idx.items():
        for h in HORIZONS:
            common = [i for i in ii if act_open(c, o, lo, n, i, h) and act_close_d0(c, lo, n, i, h)
                      and act_ladder(c, lo, n, i, h)]
            if len(common) < 5:
                continue
            a = np.array([act_open(c, o, lo, n, i, h)["ret"] for i in common])
            b = np.array([act_close_d0(c, lo, n, i, h)["ret"] for i in common])
            g = np.array([act_ladder(c, lo, n, i, h)["ret"] for i in common])
            for nm, x, y in (("open-D0close", a, b), ("open-ladder", a, g), ("ladder-D0close", g, b)):
                obs, med, p, sz = paired_perm(x, y)
                key = f"{k} h={h} {nm}"
                ps.append((key, p))
                detail.append((key, obs, med, p, sz))
    adj = holm(ps)
    print("| comparison | n | mean diff | median diff | p | Holm-adj p |")
    print("|---|---|---|---|---|---|")
    for key, obs, med, p, sz in detail:
        print(f"| {key} | {sz} | {obs*100:+.1f}% | {med*100:+.1f}% | {p:.4f} | {adj[key]:.4f} |")
    print(f"\n{len(ps)} tests. Bonferroni at 0.05 is {0.05/max(len(ps),1):.5f}. Read the Holm")
    print("column, not the raw p. Any comparison whose Holm-adj p is above 0.05 is a")
    print("null result and should be reported as one.")

    # --------------------------------------------------- 7. FOMC D0 counter-example
    print("\n" + "=" * 96)
    print("### 7. COUNTER-EXAMPLE SET: real FOMC decisions that followed a crash")
    print("=" * 96)
    ft = fomc_table()
    ft = ft[~ft["emergency"] & ft["action"].notna()]
    dmap = {d: i for i, d in enumerate(m["Date"])}
    hits = []
    for dt, r in ft.iterrows():
        i0 = dmap.get(dt)
        if i0 is None or i0 < 3:
            continue
        pre = m["ret"].values[i0 - 2:i0]     # D-2, D-1
        if np.nanmin(pre) <= -0.10 or m["dd250"].values[i0 - 2] <= -0.55:
            hits.append((dt.strftime("%Y-%m-%d"), float(r["action"]), str(r["kind"]),
                         i0, float(np.nanmin(pre)), float(m["dd250"].values[i0 - 2])))
    print(f"n = {len(hits)} meetings where D-2 or D-1 fell >=10%, or D-2 drawdown <=-55%.")
    print("| meeting | action bp | kind | worst pre-day | dd250 at D-2 | D0 close +1 | +5 | +20 | min low 20d |")
    print("|---|---|---|---|---|---|---|---|---|")
    for dt, act, kind, i0, pre, dd in hits:
        cells = []
        for h in (1, 5, 20):
            cells.append(f"{(c[i0+h]/c[i0]-1)*100:+.1f}%" if i0 + h < n else "n/a")
        ml = f"{(lo[i0+1:i0+21].min()/c[i0]-1)*100:+.1f}%" if i0 + 20 < n else "n/a"
        print(f"| {dt} | {act:+.0f} | {kind} | {pre*100:+.1f}% | {dd*100:.0f}% | "
              + " | ".join(cells) + f" | {ml} |")
    print("\nThis set is far below n=50. It is a list of counter-examples, not a base rate.")
    print("\nDONE.")


if __name__ == "__main__":
    main()
