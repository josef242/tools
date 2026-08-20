"""Side-by-side body-diagnostics comparison: KeelHaul (tangent-projection) vs DN2 (normal NorMuon,
body-norm ramp). Answers: why does KH's gradient norm settle ~3x higher than DN2 at matched tokens,
and is KH behaving like a higher effective LR? Pulls from each run's diagnostics.jsonl + gen_log.

Read-only. Run: python kh_dn2_diag_compare.py
"""
import os, json, math, re, statistics as st

ROOT = "/home/josef/brainbox/checkpoints/current"


def diag_series(run):
    dj = os.path.join(ROOT, run, "diagnostics.jsonl")
    out = []
    for line in open(dj):
        try:
            o = json.loads(line)
        except Exception:
            continue
        L = o.get("layers", [])
        if not isinstance(L, list) or not L:
            continue
        acc = {k: [] for k in ("w_norm", "g_norm", "ratio", "update_rms", "param_delta_ratio", "w_rms")}
        gsq = wsq = 0.0
        for layer in L:
            for part in ("attn", "ffn"):
                d = layer.get(part, {})
                if d.get("g_norm") is not None:
                    gsq += d["g_norm"] ** 2
                if d.get("w_norm") is not None:
                    wsq += d["w_norm"] ** 2
                for k in acc:
                    if d.get(k) is not None:
                        acc[k].append(d[k])
        out.append({
            "tok": o.get("total_tokens"), "step": o.get("step"),
            "g_norm": math.sqrt(gsq), "w_norm": math.sqrt(wsq),
            "ratio": st.median(acc["ratio"]) if acc["ratio"] else None,
            "update_rms": st.median(acc["update_rms"]) if acc["update_rms"] else None,
            "pdr": st.median(acc["param_delta_ratio"]) if acc["param_delta_ratio"] else None,
            "w_rms": st.median(acc["w_rms"]) if acc["w_rms"] else None,
        })
    return out


def train_series(run):
    pat = re.compile(r"st:\s*(\d+).*?ls:\s*([0-9.]+).*?nrm:\s*([0-9.]+).*?t_tk:\s*([0-9,]+)")
    r = {}
    for line in open(os.path.join(ROOT, run, "gen_log.txt"), errors="ignore"):
        m = pat.search(line)
        if m:
            r[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), int(m.group(4).replace(",", "")))
    return [(s,) + r[s] for s in sorted(r)]


def val_series(run):
    out = []
    for line in open(os.path.join(ROOT, run, "gen_log.txt"), errors="ignore"):
        m = re.search(r"st:\s*(\d+).*?tok:\s*([0-9,]+).*?AVG:\s*([0-9.]+)", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2).replace(",", "")), float(m.group(3))))
    return out


def at(series, tok, key, w=30e6, tokkey="tok"):
    sel = [x for x in series if x.get(tokkey) and abs(x[tokkey] - tok) < w and x.get(key) is not None]
    return st.median([x[key] for x in sel]) if sel else None


def val_at(v, tok):
    sel = [x for x in v if abs(x[1] - tok) < 30e6]
    return st.median([x[2] for x in sel]) if sel else None


def train_at(t, tok):
    sel = [x for x in t if abs(x[3] - tok) < 20e6]
    return st.median([x[1] for x in sel]) if sel else None  # ls


def rx(a, b):
    return a / b if (a and b) else float("nan")


KH, DN = diag_series("keelhaul"), diag_series("dreadnought_v2")
KHt, DNt = train_series("keelhaul"), train_series("dreadnought_v2")
KHv, DNv = val_series("keelhaul"), val_series("dreadnought_v2")

print("=" * 110)
print("KEELHAUL (tangent-projection, WD 0.002) vs DN2/dreadnought_v2 (normal NorMuon ramp, WD 0.02)")
print("=" * 110)
print("\nBODY WEIGHT NORM (the ramp): KH projection holds it flat; DN2 ramps")
print(f"  {'tok(M)':>7} | {'KH w_norm':>10} | {'DN2 w_norm':>10} | {'KH/DN':>6}")
for tM in [100, 200, 300, 400, 500, 600, 700, 800]:
    t = tM * 1e6
    k, d = at(KH, t, "w_norm"), at(DN, t, "w_norm")
    if k and d:
        print(f"  {tM:>7} | {k:>10.1f} | {d:>10.1f} | {rx(k,d):>5.2f}x")

print("\nRELATIVE GRADIENT g/w (ratio): weight-norm-removed — KH stays high, DN2 decays")
print(f"  {'tok(M)':>7} | {'KH g/w':>11} | {'DN2 g/w':>11} | {'KH/DN':>6}")
for tM in [100, 200, 300, 400, 500, 600, 700, 800]:
    t = tM * 1e6
    k, d = at(KH, t, "ratio"), at(DN, t, "ratio")
    if k and d:
        print(f"  {tM:>7} | {k:>11.6f} | {d:>11.6f} | {rx(k,d):>5.2f}x")

print("\nRELATIVE STEP SIZE pdr=||dW||/||W|| (effective LR in a scale-invariant body — THE 'higher LR?' test)")
print("  (common to both runs; KH update_rms exists too but DN2 predates it)")
print(f"  {'tok(M)':>7} | {'KH pdr':>11} | {'DN2 pdr':>11} | {'KH/DN':>6}")
for tM in [100, 200, 300, 400, 500, 600, 700, 800]:
    t = tM * 1e6
    kp, dp = at(KH, t, "pdr"), at(DN, t, "pdr")
    if kp and dp:
        print(f"  {tM:>7} | {kp:>11.3e} | {dp:>11.3e} | {rx(kp,dp):>5.2f}x")

print("\nLOSS: val AVG + train ls @ matched tokens (KH-DN2 > 0 means KH worse)")
print(f"  {'tok(M)':>7} | {'KH val':>8} | {'DN2 val':>8} | {'dVal':>7} | {'KH trn':>8} | {'DN2 trn':>8} | {'dTrn':>7}")
for tM in [100, 200, 300, 400, 500, 600, 700, 800]:
    t = tM * 1e6
    kv, dv = val_at(KHv, t), val_at(DNv, t)
    kt, dt = train_at(KHt, t), train_at(DNt, t)
    row = f"  {tM:>7} | "
    row += f"{kv:>8.4f} | {dv:>8.4f} | {kv-dv:>+7.4f} | " if (kv and dv) else f"{'--':>8} | {'--':>8} | {'--':>7} | "
    row += f"{kt:>8.4f} | {dt:>8.4f} | {kt-dt:>+7.4f}" if (kt and dt) else f"{'--':>8} | {'--':>8} | {'--':>7}"
    print(row)
