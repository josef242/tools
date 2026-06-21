"""Row-centering diagnostic for the output head — is logZ a CE-invisible
common-mode offset? (Rook Nexus #139)

OFFLINE, forward-only, model weights only (no optimizer state) — runs against a
saved checkpoint without touching the training rig.

THE HYPOTHESIS
--------------
A logit is z_i = h . w_i. Split each output-head row into its vocab-mean plus a
remainder:
    mu  = (1/V) * sum_i w_i           # row-mean vector, shape [D]
    w_i = mu + w_i_centered
Then z_i = h.mu + h.w_i_centered. The term (h.mu) is a SINGLE SCALAR added to
EVERY vocab logit for that token. Softmax is shift-invariant, so CE,
probabilities, sampling, and the CE gradient through h are ALL unchanged by it —
but logZ = logsumexp absorbs it directly: logZ = h.mu + logZ_centered.

So dreadnought's logZ ~ 490 (vs ~7-11 normal) may be mostly a CE-INVISIBLE
common-mode gauge offset, not real classifier growth. This script measures it.

INTERPRETATION (calibrated against the MEASURED KEEL family baseline)
--------------------------------------------------------------------
The decision we reached (two reviews, Nexus thread 139/146): the large raw logZ
is dominated by a CE-invisible common-mode GAUGE (~78% of raw logZ; u1.ones~0.93
on every checkpoint), and the *centered* margin logZ_c is HEALTHY at the family
baseline. The action is GAUGE SUBTRACTION (row-centering, function-preserving),
NOT a head-norm brake — the dn1 10k->14k experiment showed a blunt head-WD brake
causes low-rank collapse (logZ_c +94%, spectral_conc_c 0.26->0.48), not a fix.

So we classify logZ_c against the empirically measured family band, NOT the old
conventional ~10 expectation:
  KEEL_LOGZC_FAMILY_BAND = (60, 130)  # coherent runs: dn2 83-108, mf 110, dn1@10k 73
1. logZ_c collapses to a small fraction of raw logZ -> logZ was mostly the gauge;
   logit scale was fine. Gauge subtraction cleans it up for free.
2. centering removes most of the weight norm / top mode -> the dominant mode is
   the common-offset gauge. Gauge subtraction.
3. logZ_c sits IN the family band -> normal KEEL centered margin (NOT pathology).
   Gauge subtraction only; do NOT brake the head.
4. logZ_c sits well ABOVE the family band -> anomalous centered-margin inflation
   (e.g. dn1@14k under lethal WD) -> investigate the cause; braking is what
   CAUSED this in the dn1 case, so a brake is NOT the remedy.

Run:
    python zloss_row_center_probe.py --ckpt <path-or-dir> [--ntokens 8192]
                                     [--out results.json] [--aux]
"""
import os
import re
import sys
import glob
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

# ---- repo wiring (mirror generate_neo.py) --------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("../common_fsdp2", "../saved_code"):
    _ap = os.path.normpath(os.path.join(_HERE, _p))
    if _ap not in sys.path:
        sys.path.insert(0, _ap)
import logger  # noqa: E402
logger._instance.set_logdir("./logs")
logger._instance.set_default_logfile("zloss_probe_log.txt")
logger._instance.set_rank(0)
import neo_common as nc  # noqa: E402

# Empirically measured logZ_c band for coherent KEEL-family runs (Nexus thread
# 139/146): dn2 83-108, mf-low-lr 110, healthy dn1@10k 73. The verdict compares
# logZ_c against THIS, not the old conventional ~10 expectation. Outside the band
# (high) flags anomaly to INVESTIGATE — not "brake the head" (braking is what
# inflated dn1@14k to 140 via low-rank collapse).
KEEL_LOGZC_FAMILY_BAND = (60.0, 130.0)


def resolve_ckpt(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pts = []
        for f in os.listdir(path):
            if f.startswith("model_") and f.endswith(".pt"):
                m = re.search(r"_(\d+)\.pt", f)
                if m:
                    pts.append((int(m.group(1)), os.path.join(path, f)))
        if not pts:
            raise FileNotFoundError(f"No model_*.pt in {path}")
        pts.sort(reverse=True)
        return pts[0][1]
    raise FileNotFoundError(path)


def build_val_batch(data_root, groups, ntokens, device, seed=0):
    """One long [1, ntokens] sequence assembled round-robin from val shards of
    the given groups (deterministic). Returns (tokens[1,N], targets[N]) where
    targets are the next-token labels (shifted)."""
    candidate = [data_root] if os.path.isabs(data_root) else [
        os.path.normpath(os.path.join(_HERE, "../mara_fsdp2", data_root)),
        os.path.abspath(data_root),
    ]
    root = next((p for p in candidate if os.path.isdir(p)), None)
    if root is None:
        raise FileNotFoundError(f"data root not found, tried {candidate}")
    rng = np.random.default_rng(seed)
    need = ntokens + 1                      # +1 so we can form shifted targets
    per_group = max(256, need // max(1, len(groups)))
    chunks = []
    for g in groups:
        shards = sorted(glob.glob(os.path.join(root, g, "*_val_*.npy")))
        if not shards:
            logger.print_and_log(f"  [warn] no val shards for '{g}', skipping")
            continue
        arr = np.load(shards[0], mmap_mode="r")
        if arr.shape[0] < per_group + 1:
            continue
        start = int(rng.integers(0, arr.shape[0] - per_group))
        c = np.asarray(arr[start:start + per_group])
        chunks.append(c.astype(np.int64))
        if sum(len(x) for x in chunks) >= need:
            break
    if not chunks:
        raise RuntimeError("no usable val shards for any group")
    flat = np.concatenate(chunks)[:need]
    toks = torch.from_numpy(flat).long().to(device)
    return toks[:-1].unsqueeze(0), toks[1:]   # [1, N], [N]


def capture_final_h(model, tokens, max_seq_len=None):
    """Run forward, capturing the post-final-norm hidden (the head input) via a
    forward hook on model.norm. Returns h_flat [N, D] (fp32).

    The probe sequence (ntokens ~8k) can exceed the model's max_seq_len (RoPE
    tables are precomputed to max_seq_len), so we forward in windows of
    <= max_seq_len and concatenate the captured h. Causal h for token i depends
    only on tokens <= i, and the model never trained on context longer than
    max_seq_len anyway, so windowing is the faithful thing to do."""
    cap = {}

    def hook(_m, _inp, out):
        cap["h"] = out.detach()

    handle = model.norm.register_forward_hook(hook)
    N = tokens.size(1)
    win = int(max_seq_len) if max_seq_len else N
    win = max(1, min(win, N))
    hs = []
    try:
        with torch.no_grad():
            for s in range(0, N, win):
                chunk = tokens[:, s:s + win]
                if torch.cuda.is_available():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        model(chunk)
                else:
                    model(chunk)
                hs.append(cap["h"].reshape(-1, cap["h"].size(-1)).float())
    finally:
        handle.remove()
    return torch.cat(hs, dim=0)


def dist_stats(x):
    x = x.float()
    return {
        "mean": x.mean().item(),
        "rms": x.pow(2).mean().sqrt().item(),
        "p50": torch.quantile(x, 0.50).item(),
        "p95": torch.quantile(x, 0.95).item(),
        "max": x.max().item(),
    }


def weight_metrics(W):
    """Pure-weight (no data) row-centering metrics for a head weight W [V, D]."""
    Wf = W.float()
    V, D = Wf.shape
    mu = Wf.mean(dim=0)                          # [D]
    Wc = Wf - mu.unsqueeze(0)                    # [V, D]
    fro = torch.linalg.matrix_norm(Wf, ord="fro").item()
    fro_c = torch.linalg.matrix_norm(Wc, ord="fro").item()
    # Top singular value + spectral concentration (s1^2 / sum si^2) for both.
    sv = torch.linalg.svdvals(Wf)
    sv_c = torch.linalg.svdvals(Wc)
    s1, s1_c = sv[0].item(), sv_c[0].item()
    conc = (sv[0] ** 2 / (sv ** 2).sum()).item()
    conc_c = (sv_c[0] ** 2 / (sv_c ** 2).sum()).item()
    # u1 alignment with the all-ones vocab direction (the common-mode signature):
    # u1 from W = U S V^T. svdvals doesn't return U, so get u1 via the top right
    # singular vector v1 (svd_lowrank) then u1 = W v1 / s1.
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
    u1 = U[:, 0]                                  # [V]
    ones = torch.ones(V, device=Wf.device) / (V ** 0.5)
    u1_dot_ones = abs(torch.dot(u1, ones).item())
    # u1 vs a smooth log-token-id ramp (a stand-in "frequency-like" smooth vector)
    ramp = torch.arange(V, device=Wf.device, dtype=torch.float32)
    ramp = (ramp - ramp.mean()) / (ramp.std() + 1e-8)
    u1_dot_ramp = abs(torch.dot(u1, ramp / (V ** 0.5)).item())
    return {
        "V": V, "D": D,
        "mu_norm": mu.norm().item(),
        "W_fro": fro, "Wc_fro": fro_c, "Wc_fro_frac": fro_c / fro,
        "s1": s1, "s1_c": s1_c, "s1_c_frac": s1_c / s1,
        "spectral_concentration": conc, "spectral_concentration_c": conc_c,
        "u1_dot_ones": u1_dot_ones, "u1_dot_idramp": u1_dot_ramp,
    }, mu, Wc


def data_metrics(h, W, mu, targets, pad_id, chunk_v=None):
    """logZ / logZ_c / h.mu distributional stats on the captured hidden h [N,D],
    plus the CE-invariance + decomposition SANITY checks."""
    N, D = h.shape
    V = W.shape[0]
    Wf = W.float()
    valid = targets != pad_id
    h_v = h[valid]
    tgt_v = targets[valid]
    # h.mu : per-token common offset (cheap, no [N,V])
    h_dot_mu = h_v @ mu                          # [Nv]
    # logits + centered logits.  [Nv, V] fp32 ~ Nv*32000*4 bytes.
    logits = h_v @ Wf.t()                        # [Nv, V]
    logZ = torch.logsumexp(logits, dim=-1)       # [Nv]
    logits_c = logits - h_dot_mu.unsqueeze(1)    # centered == h @ (W-mu).T
    logZ_c = torch.logsumexp(logits_c, dim=-1)
    # SANITY 1: CE identical from logits vs logits_c (centering is CE-invariant).
    ce = F.cross_entropy(logits, tgt_v, reduction="mean")
    ce_c = F.cross_entropy(logits_c, tgt_v, reduction="mean")
    # SANITY 2: logZ_c ~= logZ - h.mu elementwise.
    decomp_err = (logZ_c - (logZ - h_dot_mu)).abs().max().item()
    # EXCESS form (Rook #144): logZ_c = log V + KL(U||p), so log V is the HARD
    # FLOOR (Jensen) for ANY model. excess = logZ_c - log V = KL(U||p) is the
    # family-comparable concentration-above-uniform control variable. Compared
    # across KEEL checkpoints, NOT logZ_c raw (whose floor log V differs by V).
    logV = float(np.log(V))
    excess = logZ_c - logV
    return {
        "n_valid": int(valid.sum().item()),
        "logV": logV,
        "logZ": dist_stats(logZ),
        "logZ_c": dist_stats(logZ_c),
        "excess_logZc_minus_logV": dist_stats(excess),   # = KL(U||p), >= 0
        "h_dot_mu": dist_stats(h_dot_mu),
        "sanity_CE": ce.item(),
        "sanity_CE_centered": ce_c.item(),
        "sanity_CE_abs_diff": abs(ce.item() - ce_c.item()),
        "sanity_logZc_vs_logZ_minus_hmu_maxabs": decomp_err,
    }, logZ_c, tgt_v


def freq_buckets(logZ_c, tgt_v, nbucket=5):
    """logZ_c mean per target-id quantile bucket (cheap freq-structure proxy)."""
    order = torch.argsort(tgt_v.float())
    out = {}
    n = tgt_v.numel()
    for b in range(nbucket):
        lo, hi = b * n // nbucket, (b + 1) * n // nbucket
        idx = order[lo:hi]
        if idx.numel():
            out[f"bucket{b}"] = {
                "logZ_c_mean": logZ_c[idx].mean().item(),
                "tgt_id_range": [int(tgt_v[idx].min()), int(tgt_v[idx].max())],
                "n": int(idx.numel()),
            }
    return out


def _names_from_yaml(yaml_path):
    """Group NAMES from an explicit config_*.yaml (FullLoader handles the
    python/tuple tags that safe_load chokes on). Schedule/proportions are
    intentionally dropped — build_val_batch samples by name, not weight."""
    import yaml
    y = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    g = y.get("groups")
    return [e[0] for e in (g or []) if isinstance(e, (list, tuple)) and e]


def _resolve_own_groups(cfg, ckpt_path, config_path=None):
    """This model's training group NAMES, from its own config. Order:
    (0) an EXPLICIT --config yaml (ground truth you choose — decouples the .pt
    location from the config so the .pt can live anywhere, e.g. on valhalla);
    (1) the loaded cfg object; (2) the latest config_*.yaml next to the ckpt.
    Returns (names, source_str); names=[] if undetermined (caller fails loud
    rather than guessing a foreign mix)."""
    def names_from(g):
        return [e[0] for e in (g or []) if isinstance(e, (list, tuple)) and e]
    # 0) explicit --config (ground truth) — fail loud here, the user named it
    if config_path:
        names = _names_from_yaml(config_path)
        if names:
            return names, f"--config ({os.path.basename(config_path)})"
        raise RuntimeError(
            f"--config {config_path} parsed but has no usable 'groups' list."
        )
    # 1) loaded cfg object
    g = getattr(cfg, "groups", None)
    if g is None and isinstance(cfg, dict):
        g = cfg.get("groups")
    names = names_from(g)
    if names:
        return names, "cfg object"
    # 2) parse the config yaml in the checkpoint dir (FullLoader for tuple tags)
    import glob
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    yamls = sorted(glob.glob(os.path.join(ckpt_dir, "config_*.yaml")))
    if yamls:
        try:
            import yaml
            y = yaml.load(open(yamls[-1]), Loader=yaml.FullLoader)
            names = names_from(y.get("groups"))
            if names:
                return names, f"config yaml ({os.path.basename(yamls[-1])})"
        except Exception as ex:
            logger.print_and_log(f"  [warn] yaml group parse failed: {ex}")
    return [], "undetermined"


def extract_head_weights_mmap(path, device):
    """Fast path: pull ONLY output.weight (+ aux head linears) from the
    checkpoint via mmap, WITHOUT loading the 23GB full model. The weight-side
    row-centering metrics (||mu||, ||W_c||/||W||, sigma1(W_c), u1.ones) need
    only output.weight (~165MB) — seconds vs ~7min for a full network load."""
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("model") if isinstance(ck, dict) and "model" in ck else ck
    W = sd["output.weight"].to(device).float()
    aux = {}
    for k in sd:
        m = re.match(r"aux_heads\.(\d+)\.linear\.weight$", k)
        if m:
            aux[m.group(1)] = sd[k].to(device).float()
    return W, aux


def run(ckpt, ntokens, want_aux, out_path, weights_only=False, groups_override=None,
        config_path=None):
    device = nc.detect_device(None)
    path = resolve_ckpt(ckpt)
    step = int(re.search(r"_(\d+)\.pt", os.path.basename(path)).group(1)) \
        if re.search(r"_(\d+)\.pt", os.path.basename(path)) else None
    logger.print_and_log(f"=== row-center probe: {os.path.basename(path)} on {device} ===")

    # ---- WEIGHTS-ONLY fast path (no full model load, no forward pass) ----
    if weights_only:
        t0 = time.time()
        W, aux_W = extract_head_weights_mmap(path, device)
        logger.print_and_log(f"mmap-extracted head weights in {time.time()-t0:.1f}s "
                             f"(W {tuple(W.shape)})")
        wm, _, _ = weight_metrics(W)
        logger.print_and_log(f"[main head weights] {json.dumps(wm, indent=2)}")
        result = {"checkpoint": os.path.basename(path), "step": step,
                  "weights_only": True, "main_head_weights": wm}
        if want_aux and aux_W:
            result["aux_heads"] = {}
            for name, Wa in aux_W.items():
                awm, _, _ = weight_metrics(Wa)
                result["aux_heads"][name] = {"weights": awm}
            logger.print_and_log(f"[aux heads] {json.dumps(result['aux_heads'], indent=2)}")
        logger.print_and_log("\n=== WEIGHTS-ONLY VERDICT (no logZ_c data — needs full load) ===")
        logger.print_and_log(f"  ||mu||={wm['mu_norm']:.3f}  ||W_c||/||W||={wm['Wc_fro_frac']:.3f}  "
                             f"s1_c/s1={wm['s1_c_frac']:.3f}  u1.ones={wm['u1_dot_ones']:.3f}  "
                             f"spectral_conc {wm['spectral_concentration']:.3f}->{wm['spectral_concentration_c']:.3f}")
        if out_path:
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.print_and_log(f"wrote {out_path}")
        return result
    t0 = time.time()
    model, enc, cfg = nc.load_model_and_tokenizer(
        path, device=device, half_precision=True,
        shard_strategy="none", use_keel=None,   # auto-detect KEEL from ckpt
    )
    model.eval()
    pad_id = int(getattr(cfg, "pad_id", 0) or 0)
    logger.print_and_log(f"loaded in {time.time()-t0:.1f}s; pad_id={pad_id}")

    # --- pure-weight metrics on the main output head ---
    W = model.output.weight
    wm, mu, _ = weight_metrics(W)
    logger.print_and_log(f"[main head weights] {json.dumps(wm, indent=2)}")

    # --- data: capture post-final-norm h, compute logZ / logZ_c ---
    # Probe each model on ITS OWN training mix (groups from THIS checkpoint's
    # config) so logZ_c reflects the model's natural equilibrium on data it
    # actually saw — NOT a hardcoded foreign mix. For the dn1 before/after this
    # also guarantees both checkpoints use identical data (any logZ_c delta is
    # the weight change). `groups_override` (CLI --groups) forces a fixed common
    # probe set across runs if a controlled cross-run comparison is wanted.
    data_root = getattr(cfg, "data_root_path", "../../notebooks/datasets/tokenized/llama/")
    if groups_override:
        groups, groups_src = groups_override, "CLI override (fixed common set)"
    else:
        groups, groups_src = _resolve_own_groups(cfg, path, config_path=config_path)
    # FAIL LOUD rather than silently probe on a foreign/empty mix — a
    # silent-wrong data set would corrupt the cross-run logZ_c comparison.
    if not groups:
        raise RuntimeError(
            "Could not determine this model's training groups from the "
            "checkpoint config OR its config_*.yaml. Refusing to probe on a "
            "guessed mix (would silently corrupt logZ_c). Pass --groups explicitly."
        )
    logger.print_and_log(f"groups [{groups_src}]: {groups}")
    tokens, targets = build_val_batch(data_root, groups, ntokens, device)
    logger.print_and_log(f"val batch: {tuple(tokens.shape)} (+targets {tuple(targets.shape)})")
    # RoPE freqs are precomputed to params.max_seq_len, so forward in windows of
    # that length (an 8k probe sequence would otherwise overrun the rotary table).
    msl = getattr(getattr(model, "params", None), "max_seq_len", None) \
        or getattr(cfg, "max_seq_len", None)
    if msl:
        logger.print_and_log(f"forwarding in windows of max_seq_len={int(msl)}")
    h = capture_final_h(model, tokens, max_seq_len=msl)
    logger.print_and_log(f"captured final-norm h: {tuple(h.shape)} rms={h.pow(2).mean().sqrt().item():.4f}")
    dm, logZ_c, tgt_v = data_metrics(h, W, mu, targets, pad_id)
    fb = freq_buckets(logZ_c, tgt_v)
    logger.print_and_log(f"[main head data] {json.dumps(dm, indent=2)}")
    logger.print_and_log(f"[freq buckets logZ_c] {json.dumps(fb, indent=2)}")

    result = {
        "checkpoint": os.path.basename(path), "step": step,
        "ntokens": ntokens, "probe_groups": groups, "probe_groups_src": groups_src,
        "main_head_weights": wm,
        "main_head_data": dm, "freq_buckets": fb,
    }

    # --- bonus (b): aux heads (unregularized) ---
    if want_aux and hasattr(model, "aux_heads") and len(model.aux_heads):
        aux = {}
        for name, head in model.aux_heads.items():
            Wa = head.linear.weight
            awm, _, _ = weight_metrics(Wa)
            aux[name] = {"weights": awm}
        result["aux_heads"] = aux
        logger.print_and_log(f"[aux heads] {json.dumps(aux, indent=2)}")

    # --- verdict heuristic ---
    logZ_mean = dm["logZ"]["mean"]
    logZc_mean = dm["logZ_c"]["mean"]
    s1_frac = wm["s1_c_frac"]
    fro_frac = wm["Wc_fro_frac"]
    logger.print_and_log("\n=== VERDICT ===")
    logger.print_and_log(f"  CE-invariance OK: CE_diff={dm['sanity_CE_abs_diff']:.2e} "
                         f"(must be ~0); decomp_err={dm['sanity_logZc_vs_logZ_minus_hmu_maxabs']:.2e}")
    logger.print_and_log(f"  logZ mean {logZ_mean:.2f} -> logZ_c mean {logZc_mean:.2f} "
                         f"(centered/raw = {logZc_mean/logZ_mean:.3f})")
    logger.print_and_log(f"  ||W_c||/||W|| = {fro_frac:.3f}   s1_c/s1 = {s1_frac:.3f}   "
                         f"u1.ones = {wm['u1_dot_ones']:.3f}")
    # Classify the GAUGE share first, then place logZ_c against the measured
    # KEEL family band. Action is gauge subtraction (row-centering) in the normal
    # cases; we do NOT recommend head-norm braking (dn1@14k showed braking causes
    # low-rank collapse, not a fix — Nexus 139/146).
    # ABSOLUTE band classification takes priority — logZ_c vs the measured family
    # band is the definitive signal regardless of what fraction of raw logZ the
    # gauge happens to be (dn1@14k's gauge is 81% of raw logZ, yet its logZ_c=140
    # is the anomaly — a raw-fraction-first test would wrongly wave it through).
    lo, hi = KEEL_LOGZC_FAMILY_BAND
    gauge_share = 1.0 - (logZc_mean / logZ_mean)   # fraction of raw logZ that is the gauge
    logger.print_and_log(f"  gauge share of raw logZ = {gauge_share:.1%}   "
                         f"KEEL family logZ_c band = [{lo:.0f}, {hi:.0f}]")
    if logZc_mean > hi:
        logger.print_and_log(f"  => logZ_c {logZc_mean:.1f} is ABOVE the family band "
                             f"[{lo:.0f},{hi:.0f}] -> anomalous centered-margin inflation "
                             "(cf. dn1@14k=140 under lethal head-WD). ACTION: INVESTIGATE the "
                             "cause — note braking is what CAUSED this in dn1, so it is NOT the remedy.")
    elif logZc_mean < lo:
        logger.print_and_log(f"  => logZ_c {logZc_mean:.1f} is BELOW the family band "
                             f"[{lo:.0f},{hi:.0f}] -> margin not yet developed (early ckpt) or "
                             "actively compressed (e.g. z-loss on). ACTION: gauge subtraction; "
                             "expect logZ_c to drift up toward the band as the LM objective settles.")
    else:  # IN band -> normal KEEL centered margin
        logger.print_and_log(f"  => logZ_c {logZc_mean:.1f} is IN the KEEL family band "
                             f"[{lo:.0f},{hi:.0f}] -> NORMAL centered margin, not pathology "
                             f"(gauge is {gauge_share:.0%} of raw logZ). "
                             "ACTION: gauge subtraction only; do NOT brake the head.")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.print_and_log(f"wrote {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt or dir")
    ap.add_argument("--ntokens", type=int, default=8192)
    ap.add_argument("--aux", action="store_true", help="also analyze SCS aux heads")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--weights-only", action="store_true",
                    help="mmap-extract only output.weight (skip 23GB load + forward); "
                         "gives weight-side metrics in seconds, no logZ_c data metric")
    ap.add_argument("--groups", default=None,
                    help="comma-separated group override (fixed common probe set); "
                         "default = each model's OWN config groups")
    ap.add_argument("--config", default=None,
                    help="explicit ground-truth config_*.yaml to read group names "
                         "from (decouples .pt location from its config — point the "
                         "ckpt at valhalla, the config at the run dir). Beats "
                         "auto-detect; --groups still wins if both are given.")
    a = ap.parse_args()
    go = [g.strip() for g in a.groups.split(",")] if a.groups else None
    run(a.ckpt, a.ntokens, a.aux, a.out, weights_only=a.weights_only,
        groups_override=go, config_path=a.config)


if __name__ == "__main__":
    main()
