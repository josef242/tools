#!/usr/bin/env python
"""upgrade_checkpoint_v4.py — retrofit pre-4.0 checkpoints to the v4.0 schema.

For each checkpoint it:
  * bumps  checkpoint_version -> "4.0"
  * adds   rope_fixed: True
  * INJECTS the festival-feature fields (doc-mask / SWA / MTP) that older
    checkpoints did not self-describe, recovered from the run-dir config_*.yaml
    (the SAME source + extraction that inference uses).

After this a checkpoint loads with FIXED RoPE automatically (no --fixed_rope)
and needs no run-dir-yaml fallback at inference.

SAFETY:
  * DRY-RUN by default — prints the planned change; pass --apply to write.
  * Atomic write (temp file + os.replace); original is never left half-written.
  * --backup keeps <name>.v3.bak next to each upgraded checkpoint.
  * Idempotent: already-v4.0 checkpoints are skipped.
  * NEVER run against a checkpoint training is currently writing. Point it at
    COMPLETED checkpoints only (a half-written .pt just fails to load and is
    skipped — it is not corrupted — but don't tempt fate on the live step).

USAGE:
  python upgrade_checkpoint_v4.py <ckpt.pt | run_dir> [more ...] \
      [--apply] [--backup] [--config PATH] [--only-latest]

  # dry-run over a whole run dir:
  python upgrade_checkpoint_v4.py B:\\checkpoints\\current\\wizard101\\
  # then, once the diff looks right:
  python upgrade_checkpoint_v4.py B:\\checkpoints\\current\\wizard101\\ --apply --backup
"""
import argparse
import glob
import os
import shutil
import sys

import torch
import yaml

NEW_VERSION = "4.0"

# The festival fields a v4.0 checkpoint self-describes (checkpoint-config key ->
# yaml sub-block reader). Mirrors train_mara.save_model + neo_common recovery.
FESTIVAL_KEYS = (
    "doc_attn_mask", "doc_pos_reset", "bos_token_id",
    "swa_enabled", "swa_window", "swa_global_interleave", "mtp_enabled",
)


class _CfgLoader(yaml.SafeLoader):
    """SafeLoader tolerant of the trainer's !!python/tuple derived fields (e.g.
    an empty restart_steps) — same tag neo_common's recovery had to accept."""


_CfgLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda ld, node: ld.construct_sequence(node),
)


def newest_run_config(ckpt_path):
    """Newest config_*.yaml in the checkpoint's directory, or None."""
    ckdir = os.path.dirname(os.path.abspath(ckpt_path))
    cfgs = sorted(glob.glob(os.path.join(ckdir, "config_*.yaml")))
    return cfgs[-1] if cfgs else None


def extract_festival_fields(cfg_path):
    """Read the festival fields from a run config_*.yaml. Mirrors
    neo_common._build_model_from_checkpoint's recovery extraction exactly, so a
    tool-upgraded checkpoint is identical to one inference would have recovered."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        rc = yaml.load(f, Loader=_CfgLoader) or {}
    dm = rc.get("doc_attn_mask") or {}
    sw = rc.get("swa") or {}
    mt = rc.get("mtp") or {}
    if dm is True:
        dm = {"enabled": True}
    if sw is True:
        sw = {"enabled": True}
    if mt is True:
        mt = {"enabled": True}
    return {
        "doc_attn_mask": bool(dm.get("enabled", False)),
        "doc_pos_reset": bool(dm.get("reset_positions", False)),
        "bos_token_id": int(dm.get("bos_token_id", 1)),
        "swa_enabled": bool(sw.get("enabled", False)),
        "swa_window": int(sw.get("window", 512)),
        "swa_global_interleave": int(sw.get("global_interleave", 4)),
        "mtp_enabled": bool(mt.get("enabled", False)),
    }


def collect_checkpoints(paths, only_latest):
    """Expand file/dir args into a sorted list of checkpoint .pt paths."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "model_step_*.pt")))
            if only_latest and found:
                found = found[-1:]
            out.extend(found)
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f"  [skip] not a file or dir: {p}")
    # de-dup, keep order
    seen = set()
    uniq = []
    for c in out:
        a = os.path.abspath(c)
        if a not in seen:
            seen.add(a)
            uniq.append(c)
    return uniq


def plan_one(ckpt_path, cfg_override):
    """Load the checkpoint's METADATA (mmap, cheap) and compute the planned
    change. Returns (status, detail, festival, cfg_path). status in
    {'upgrade','already','no-config-key','no-run-yaml','error'}."""
    try:
        meta = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as e:
        return ("error", f"load failed: {type(e).__name__}: {e}", None, None)
    ver = str(meta.get("checkpoint_version", "?"))
    if ver == NEW_VERSION and meta.get("rope_fixed") is True \
            and "config" in meta and all(k in meta["config"] for k in FESTIVAL_KEYS):
        return ("already", f"already v{NEW_VERSION} + rope_fixed + festival fields", None, None)
    if "config" not in meta:
        return ("no-config-key", "checkpoint has no 'config' dict — cannot inject fields", None, None)
    cfg_path = cfg_override or newest_run_config(ckpt_path)
    if cfg_path is None:
        return ("no-run-yaml",
                "no config_*.yaml in the run dir and no --config given — "
                "cannot recover festival fields", None, None)
    try:
        festival = extract_festival_fields(cfg_path)
    except Exception as e:
        return ("error", f"config parse failed ({os.path.basename(cfg_path)}): "
                f"{type(e).__name__}: {e}", None, cfg_path)
    return ("upgrade", f"v{ver} -> v{NEW_VERSION}", festival, cfg_path)


def apply_one(ckpt_path, festival, backup):
    """Load the FULL checkpoint, inject fields, and write it back atomically.
    Returns (ok, message). Only injects festival keys that are MISSING; existing
    ones are left untouched (already self-describing)."""
    full = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = full["config"]
    injected, kept = [], []
    for k in FESTIVAL_KEYS:
        if k in cfg:
            kept.append(k)
        else:
            cfg[k] = festival[k]
            injected.append(k)
    full["checkpoint_version"] = NEW_VERSION
    full["rope_fixed"] = True

    tmp = ckpt_path + ".v4tmp"
    torch.save(full, tmp)
    # verify the temp file before we replace anything
    chk = torch.load(tmp, map_location="cpu", weights_only=False, mmap=True)
    assert str(chk.get("checkpoint_version")) == NEW_VERSION and chk.get("rope_fixed") is True
    assert all(k in chk["config"] for k in FESTIVAL_KEYS)
    del chk
    if backup:
        shutil.copy2(ckpt_path, ckpt_path + ".v3.bak")
    os.replace(tmp, ckpt_path)  # atomic on the same filesystem
    detail = f"injected {injected}" + (f", kept existing {kept}" if kept else "")
    return (True, detail)


def main():
    ap = argparse.ArgumentParser(description="Retrofit pre-4.0 checkpoints to the v4.0 schema.")
    ap.add_argument("paths", nargs="+", help="checkpoint .pt file(s) and/or run dir(s)")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry-run)")
    ap.add_argument("--backup", action="store_true", help="keep <name>.v3.bak next to each upgrade")
    ap.add_argument("--config", default=None, help="override the run config_*.yaml source")
    ap.add_argument("--only-latest", action="store_true",
                    help="for a dir, process only the newest model_step_*.pt")
    args = ap.parse_args()

    ckpts = collect_checkpoints(args.paths, args.only_latest)
    if not ckpts:
        print("No checkpoints found.")
        sys.exit(1)

    mode = "APPLY" if args.apply else "DRY-RUN (no files written; pass --apply to write)"
    print(f"=== upgrade_checkpoint_v4  [{mode}] ===")
    print(f"{len(ckpts)} checkpoint(s)\n")

    n_up = n_skip = n_err = 0
    for c in ckpts:
        status, detail, festival, cfg_path = plan_one(c, args.config)
        name = os.path.basename(c)
        if status == "already":
            print(f"  [skip] {name}: {detail}")
            n_skip += 1
            continue
        if status in ("no-config-key", "no-run-yaml", "error"):
            print(f"  [ERR ] {name}: {detail}")
            n_err += 1
            continue
        # status == 'upgrade'
        fvals = ", ".join(f"{k}={festival[k]}" for k in FESTIVAL_KEYS)
        print(f"  [plan] {name}: {detail}")
        print(f"         festival (from {os.path.basename(cfg_path)}): {fvals}")
        if args.apply:
            try:
                ok, msg = apply_one(c, festival, args.backup)
                print(f"         [done] {msg}"
                      + ("  (+.v3.bak)" if args.backup else ""))
                n_up += 1
            except Exception as e:
                print(f"         [FAIL] write failed: {type(e).__name__}: {e} "
                      f"(original untouched)")
                # clean up any temp file
                tmp = c + ".v4tmp"
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                n_err += 1
        else:
            n_up += 1

    print(f"\n=== {'upgraded' if args.apply else 'to upgrade'}: {n_up}"
          f"   skipped: {n_skip}   errors: {n_err} ===")
    if not args.apply and n_up:
        print("Re-run with --apply (and --backup) to write the changes.")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
