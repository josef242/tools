"""Gradient Productivity Metric (GPM) — analysis + retrofit tool.

WHAT IT MEASURES
  When grad-norm spikes ABOVE its local trend, does loss drop MORE than its local trend
  on the NEXT step? Positive => big gradients here are productive (strong learning signal);
  ~0 => spikes are noise; negative => spikes are anti-productive (norm spikes precede loss
  INCREASES — seen in DN2's first ~4k steps).

WHY IT'S NOT TRIVIAL
  Both nrm and loss-drops are large early / small late, so a NAIVE global correlation just
  confirms the shared training envelope ("norm tracks learning"), which is trivially true.
  We DETREND both against a rolling median (window W) and Spearman-correlate the RESIDUALS,
  so we measure whether LOCAL norm fluctuations predict LOCAL loss improvement, above the
  envelope. Spearman (rank) is robust to the heavy-tailed nrm spikes that are the point.

HONEST CAVEAT
  step N's ls and step N+1's ls are DIFFERENT batches, so this is not same-batch update
  effectiveness (which would need re-running the batch). It measures batch-to-batch
  TRANSFER: do high-norm updates help the NEXT (unseen) batch? — arguably a generalization
  signal; the window smooths batch-luck out. Lag-1: nrm[N] vs dloss[N]=ls[N]-ls[N+1].

WINDOW IS ITS OWN EXPERIMENT
  W too small -> batch-noise dominates -> GPM ~ 0;  W too large -> envelope tautology -> ->1.
  Meaningful signal lives in between. On our runs GPM is flat ~+0.25 (KeelHaul) from W=15..401,
  so we use a SHORT window (responsive) + LONG window (stable). The S-vs-L gap is itself a
  signal: S>L => productivity rising; S<L => falling.

  Validated result (W=51, steady-state): KeelHaul +0.25 vs mf +0.13 vs DN2 +0.07 — the
  tangent-projected regime has ~2-3.5x more productive gradients (radial shock-absorber
  removed => norm spikes are all loss-relevant).

MODES
  python gpm.py --log RUN/gen_log.txt                 # window-sweep + over-time (human)
  python gpm.py --compare keelhaul,dreadnought,mf-low-lr   # across runs
  python gpm.py --retrofit RUN/gen_log.txt [--out gpm.csv]  # per-step GPM-S/GPM-L for the
                                                             # Dashboard (matches LIVE tracker)
  python gpm.py --log RUN/gen_log.txt --json                 # machine-readable summary

RETROFIT == LIVE: --retrofit replays the log through the SAME trailing-window algorithm the
in-trainer GPMTracker uses, so historical (retrofitted) and future (live) runs are directly
comparable on the Dashboard. Defaults: short=15, long=101 (match the trainer defaults).
"""
import os, re, sys, argparse, json, statistics as st
from collections import deque


# ───────────────────────── parsing ─────────────────────────

def parse_log(path):
    """Return [(step, ls, nrm), ...] for steps with both ls and nrm present, sorted, deduped."""
    pat = re.compile(r"st:\s*(\d+).*?ls:\s*([0-9.]+).*?nrm:\s*([0-9.]+)")
    d = {}
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if m:
            d[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return [(s, d[s][0], d[s][1]) for s in sorted(d)]


# ───────────────────────── core math (shared by all modes AND the live tracker) ─────────────────────────

def _spearman(a, b):
    n = len(a)
    if n < 4:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def _resid_median(x):
    """Residual of x vs its median (the passed slice IS the local window)."""
    m = st.median(x)
    return [v - m for v in x]


def gpm_window(nr, ls, lag=1):
    """GPM over a window: Spearman(detrended nrm[i], detrended dloss[i]=ls[i]-ls[i+lag]).
    `nr`,`ls` are equal-length lists for ONE window. Returns float or None."""
    if len(ls) < lag + 4:
        return None
    dloss = [ls[i] - ls[i + lag] for i in range(len(ls) - lag)]
    nr_al = nr[:len(dloss)]
    return _spearman(_resid_median(nr_al), _resid_median(dloss))


# centered-window GPM for offline analysis (uses the rolling-median-residual detrend over a
# centered ±W//2 window per point — the original analysis form, kept for the sweep/compare).
def _rolling_resid(x, W):
    n = len(x); h = W // 2; out = [0.0] * n
    for i in range(n):
        lo = max(0, i - h); hi = min(n, i + h + 1)
        out[i] = x[i] - st.median(x[lo:hi])
    return out


def gpm_centered(rows, W, lag=1, step_lo=None, step_hi=None):
    if step_lo is not None or step_hi is not None:
        rows = [r for r in rows if (step_lo is None or r[0] >= step_lo) and (step_hi is None or r[0] <= step_hi)]
    if len(rows) < lag + 5:
        return None, 0
    ls = [r[1] for r in rows]; nr = [r[2] for r in rows]
    dloss = [ls[i] - ls[i + lag] for i in range(len(rows) - lag)]
    nr_al = nr[:len(dloss)]
    return _spearman(_rolling_resid(nr_al, W), _rolling_resid(dloss, W)), len(dloss)


# ───────────────────────── live-equivalent trailing tracker (== in-trainer GPMTracker) ─────────────────────────

class GPMReplay:
    """Trailing-window GPM identical to the in-trainer GPMTracker, for retrofit. At each step
    it holds the last `w_long` (nrm, ls) points and computes GPM over the trailing short/long
    windows. This is what makes retrofitted historical runs comparable to live runs."""
    def __init__(self, w_short=15, w_long=101, lag=1):
        self.ws, self.wl, self.lag = w_short, w_long, lag
        self.buf = deque(maxlen=w_long + 2)

    def push(self, nrm, ls):
        self.buf.append((float(nrm), float(ls)))

    def _g(self, W):
        pts = list(self.buf)[-(W + 1):]
        if len(pts) < 5:
            return None
        return gpm_window([p[0] for p in pts], [p[1] for p in pts], self.lag)

    def values(self):
        return self._g(self.ws), self._g(self.wl)


# ───────────────────────── modes ─────────────────────────

def mode_sweep(path, windows, lag):
    rows = parse_log(path)
    name = os.path.basename(os.path.dirname(path)) or path
    print(f"\n=== {name} : {len(rows)} steps — window sweep (centered, lag={lag}) ===")
    print(f"  {'W':>5} | {'GPM':>7} | n")
    for W in windows:
        g, n = gpm_centered(rows, W, lag)
        if g is not None:
            print(f"  {W:5d} | {g:+.4f} | {n}  {'+' if g>=0 else '-'}{'#'*int(abs(g)*40)}")
    midW = windows[len(windows) // 2]
    if len(rows) > 400:
        print(f"  over training (W={midW}):")
        nseg = min(6, len(rows) // 200); seg = len(rows) // nseg
        for k in range(nseg):
            lo = rows[k * seg][0]; hi = rows[min(len(rows) - 1, (k + 1) * seg)][0]
            g, n = gpm_centered(rows, midW, lag, lo, hi)
            if g is not None:
                print(f"    {lo:6d}-{hi:6d}: {g:+.4f} (n={n})")
    return rows


def mode_retrofit(path, w_short, w_long, lag, out):
    """Emit per-step (step, ls, nrm, gpm_s, gpm_l) replaying the LIVE trailing algorithm."""
    rows = parse_log(path)
    rep = GPMReplay(w_short, w_long, lag)
    recs = []
    for s, ls, nr in rows:
        rep.push(nr, ls)
        gs, gl = rep.values()
        recs.append({"step": s, "ls": ls, "nrm": nr,
                     "gpm_s": gs, "gpm_l": gl})
    if out:
        if out.endswith(".json"):
            json.dump(recs, open(out, "w"))
        else:  # csv
            with open(out, "w") as f:
                f.write("step,ls,nrm,gpm_s,gpm_l\n")
                for r in recs:
                    f.write(f"{r['step']},{r['ls']},{r['nrm']},"
                            f"{'' if r['gpm_s'] is None else round(r['gpm_s'],4)},"
                            f"{'' if r['gpm_l'] is None else round(r['gpm_l'],4)}\n")
        nz = sum(1 for r in recs if r["gpm_l"] is not None)
        print(f"wrote {out}: {len(recs)} steps ({nz} with GPM-L). "
              f"short={w_short} long={w_long} lag={lag} (== live tracker).")
    return recs


def main():
    ap = argparse.ArgumentParser(description="Gradient Productivity Metric")
    ap.add_argument("--log", help="one run's gen_log.txt (sweep + over-time)")
    ap.add_argument("--compare", help="comma-sep run names under --root")
    ap.add_argument("--retrofit", help="gen_log.txt -> per-step GPM (for Dashboard); pair with --out")
    ap.add_argument("--out", help="output path for --retrofit (.csv or .json)")
    ap.add_argument("--windows", default="5,9,15,25,51,101,201,401", help="sweep windows")
    ap.add_argument("--short", type=int, default=15, help="retrofit/live SHORT window")
    ap.add_argument("--long", type=int, default=101, help="retrofit/live LONG window")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="machine-readable summary for --log/--compare")
    ap.add_argument("--root", default=os.path.expanduser("~/brainbox/checkpoints/current"))
    a = ap.parse_args()
    windows = [int(w) for w in a.windows.split(",")]

    if a.retrofit:
        mode_retrofit(a.retrofit, a.short, a.long, a.lag, a.out)
    elif a.compare:
        summary = {}
        for nm in a.compare.split(","):
            p = os.path.join(a.root, nm.strip(), "gen_log.txt")
            if os.path.exists(p):
                rows = mode_sweep(p, windows, a.lag)
                summary[nm.strip()] = {str(W): gpm_centered(rows, W, a.lag)[0] for W in windows}
            else:
                print(f"  [skip] {p} not found")
        if a.json:
            print(json.dumps(summary, indent=1))
    elif a.log:
        rows = mode_sweep(a.log, windows, a.lag)
        if a.json:
            print(json.dumps({str(W): gpm_centered(rows, W, a.lag)[0] for W in windows}, indent=1))
    else:
        ap.error("pass --log, --compare, or --retrofit")


if __name__ == "__main__":
    main()
