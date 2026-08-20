"""pdr-overshoot monitor for the kv2 annealing experiment.

Question (Josef, 2026-06-25): kv2's body-LR anneal holds mult=1.0 until step 2680, then decays.
But DN2 (the target) peaked pdr ~2.7e-3 around step ~2000 then decayed on its own. If kv2's pdr is
on course to OVERSHOOT DN2's peak before the throttle engages at 2680, we may want to pull the first
transition back to ~1500-2000. This tool gives an OBJECTIVE trigger for that decision.

It reads body pdr=||dW||/||W|| (median over attn+ffn) from each run's diagnostics.jsonl and:
  - prints kv2 vs KH-v1 (unthrottled baseline) vs DN2 (target) pdr by step
  - finds DN2's pdr PEAK (value + step) — the level/timing kv2 should not blow past
  - fits kv2's recent pdr slope and PROJECTS its value at step 2680 (anneal onset)
  - verdict: is kv2 on track to overshoot DN2's peak before 2680?

Read-only. Re-run anytime: python pdr_overshoot_monitor.py [--anneal-step 2680]
"""
import os, json, re, argparse, statistics as st

ROOT = "/home/josef/brainbox/checkpoints/current"


def pdr_series(run):
    dj = os.path.join(ROOT, run, "diagnostics.jsonl")
    if not os.path.exists(dj):
        return []
    out = []
    for line in open(dj):
        try:
            o = json.loads(line)
        except Exception:
            continue
        pdrs = [layer.get(p, {}).get("param_delta_ratio")
                for layer in o.get("layers", []) for p in ("attn", "ffn")]
        pdrs = [v for v in pdrs if v is not None]
        if pdrs and o.get("step") is not None:
            pdrs.sort()
            out.append((o["step"], o.get("total_tokens", 0), pdrs[len(pdrs) // 2]))
    return out


def peak(series):
    if not series:
        return None
    s, t, p = max(series, key=lambda r: r[2])
    return {"step": s, "tok": t, "pdr": p}


def project_at(series, target_step, window=8):
    """Linear fit of pdr vs step over the last `window` points; project to target_step."""
    pts = series[-window:]
    if len(pts) < 3:
        return None
    xs = [r[0] for r in pts]
    ys = [r[2] for r in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / sxx
    inter = my - slope * mx
    return slope * target_step + inter, slope


# Math Q7 Option A forward prediction: this is the pdr trajectory kv2 SHOULD follow if the
# annealing model is right (Option A multiplier x KH-v1's measured pdr). kv2's live pdr tracking
# this curve = the hypothesis confirmed. (tok_M, predicted_pdr)
OPTION_A_PREDICTION = [
    (100, 1.45e-3), (190, 2.20e-3), (250, 2.74e-3), (300, 2.78e-3),
    (400, 2.71e-3), (500, 2.56e-3), (600, 2.56e-3), (800, 2.28e-3),
]


def _pred_at(tok_M):
    pts = OPTION_A_PREDICTION
    if tok_M <= pts[0][0]:
        return pts[0][1]
    if tok_M >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        if x0 <= tok_M <= x1:
            return y0 + (y1 - y0) * (tok_M - x0) / (x1 - x0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anneal-step", type=int, default=1500,
                    help="step where kv2's body-LR decay begins (Option A = 1500, at LR-cap)")
    a = ap.parse_args()

    KV = pdr_series("kv2")
    KH = pdr_series("keelhaul")
    DN = pdr_series("dreadnought_v2")

    if not KV:
        print("no kv2 diagnostics yet."); return
    cur = KV[-1]
    print("=== pdr-overshoot monitor (anneal onset = step %d) ===" % a.anneal_step)
    print("kv2 latest: step %d, tok %.0fM, pdr %.3e" % (cur[0], cur[1] / 1e6, cur[2]))
    print()

    # side-by-side at kv2's available steps, incl. Math's Option-A predicted pdr
    khd = {s: p for s, t, p in KH}
    dnd = {s: p for s, t, p in DN}
    print("kv2 vs KH-v1 (unthrottled twin) vs DN2 (target) + Math Q7 Option-A prediction:")
    print("%6s | %7s | %9s | %9s | %9s | %8s" % ("step", "tok(M)", "kv2", "Opt-A pred", "KH-v1", "DN2"))
    for s, t, p in KV[-10:]:
        kh = khd.get(s)
        dn = dnd.get(s)
        pred = _pred_at(t / 1e6)
        flag = ""
        if t / 1e6 > 200 and pred:  # only meaningful after the anneal would have engaged
            dev = (p - pred) / pred
            flag = " <-- ABOVE pred" if dev > 0.10 else (" <-- below pred" if dev < -0.10 else " on-track")
        print("%6d | %7.0f | %9.3e | %9.3e | %9s | %8s%s" % (
            s, t / 1e6, p, pred, ("%.3e" % kh) if kh else "-", ("%.3e" % dn) if dn else "-", flag))
    print()
    print("  Option-A success = kv2 pdr tracks the 'Opt-A pred' column: rise to ~2.2e-3 @190M,")
    print("  plateau ~2.7-2.8e-3 @250-400M, decay to ~2.3e-3 @800M. 'ABOVE pred' late = still too hot.")
    print()

    dn_peak = peak(DN)
    kh_peak = peak([r for r in KH if r[0] <= a.anneal_step])
    if dn_peak:
        print("DN2 pdr PEAK (the target ceiling): %.3e at step %d (%.0fM tok)" % (
            dn_peak["pdr"], dn_peak["step"], dn_peak["tok"] / 1e6))
    if kh_peak:
        print("KH-v1 pdr at/near anneal step %d (unthrottled): ~%.3e" % (a.anneal_step, kh_peak["pdr"]))
    print()

    # pdr climbs mechanically during warmup (LR is ramping). Projecting that ramp past the
    # warmup cap is meaningless — it predicts absurd values. Only project (and judge) once warmup
    # is done, when pdr is governed by dynamics, not the LR ramp.
    WARMUP_END = 1500
    if cur[0] < WARMUP_END:
        print("kv2 is still in WARMUP (step %d / %d, LR ramping). pdr climbs with LR here, so a" % (
            cur[0], WARMUP_END))
        print("projection to step %d is NOT meaningful yet — it would just extrapolate the LR ramp." % a.anneal_step)
        print()
        print("=== EARLY READ (no verdict until warmup completes) ===")
        if dn_peak:
            ratio_now = cur[2] / (dnd.get(cur[0]) or dn_peak["pdr"])
            print("  kv2 pdr now %.3e vs DN2-at-same-step %s (kv2 is %.0f%% of DN2 here)" % (
                cur[2], ("%.3e" % dnd[cur[0]]) if cur[0] in dnd else "?",
                100 * cur[2] / dnd[cur[0]] if cur[0] in dnd else 0))
            print("  DN2's eventual PEAK was %.3e at step %d. KH-v1 (unthrottled, kv2's twin so far)" % (
                dn_peak["pdr"], dn_peak["step"]))
            print("  reached %.3e by step %d — i.e. KH overshot DN2's peak by ~%.0f%%." % (
                kh_peak["pdr"] if kh_peak else 0, a.anneal_step,
                100 * (kh_peak["pdr"] / dn_peak["pdr"] - 1) if kh_peak else 0))
            print("  kv2 is tracking KH almost exactly, so EXPECT a similar ~20%% overshoot of DN2's")
            print("  peak, around step 2000-2680. DECISION POINT: re-run at step ~1500-2000 — if kv2")
            print("  pdr has flattened ABOVE %.2e and isn't decaying, pull the first transition back" % dn_peak["pdr"])
            print("  toward DN2's peak step (~%d)." % dn_peak["step"])
        return

    proj = project_at(KV, a.anneal_step)
    if proj:
        pval, slope = proj
        print("kv2 pdr projection -> at step %d (anneal onset): ~%.3e  (recent slope %.2e/step)" % (
            a.anneal_step, pval, slope))
        if dn_peak:
            over = pval - dn_peak["pdr"]
            ratio = pval / dn_peak["pdr"]
            print()
            print("=== VERDICT ===")
            if pval > dn_peak["pdr"] * 1.05:
                print("  ⚠ kv2 projected to OVERSHOOT DN2's peak by %.0f%% (%.3e vs %.3e) BEFORE the" % (
                    (ratio - 1) * 100, pval, dn_peak["pdr"]))
                print("    throttle engages. Consider pulling the first transition back to ~%d-2000" % (
                    dn_peak["step"]))
                print("    (near DN2's own peak step) so pdr tops out closer to the target, not above it.")
            elif pval > dn_peak["pdr"] * 0.95:
                print("  ~ kv2 projected to reach ~DN2's peak level (%.3e vs %.3e) right at the anneal" % (
                    pval, dn_peak["pdr"]))
                print("    onset. Borderline — 2680 timing is roughly OK but watch for steepening.")
            else:
                print("  ✓ kv2 projected BELOW DN2's peak (%.3e vs %.3e) at anneal onset — 2680 is fine," % (
                    pval, dn_peak["pdr"]))
                print("    pdr isn't overshooting; no need to pull the transition earlier.")
        # caveat
        print()
        print("  NOTE: linear projection over %d points; pdr may flatten on its own as warmup completes" % 8)
        print("  (LR caps at step 1500). Re-run near step 1500-2000 for the decisive read.")
    else:
        print("not enough kv2 points to project yet — re-run after a few more diag steps.")


if __name__ == "__main__":
    main()
