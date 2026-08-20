#!/usr/bin/env python
"""position_probe.py — zero-shot per-position loss probe (PE ablation phase 2 prelude).

Measures next-token loss AS A FUNCTION OF POSITION at a context length the
model was NOT trained on (e.g. T=4096/8192 for the T=2048 skiff-pe arms).
Separates "more context helps" from "unseen positions hurt" — the scalar val
loss cannot distinguish these. Positional tables are rebuilt mode-aware at the
probe length (rope extrapolates its rotation; envelope extends its cos table;
nope stays identity), honoring the checkpoint's persisted rope_mode.

Usage (per finished arm):
  python position_probe.py \
      --ckpt /home/josef/brainbox/checkpoints/current/skiff-pe-ctrl/model_step_011999.pt \
      --seq_len 4096 --windows 64 --batch 2

Writes <run_dir>/position_probe_T<seq_len>.json (bin means only — no text).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common_fsdp2"))

DEFAULT_SHARDS = [
    "/home/josef/valhalla/notebooks/datasets/tokenized/llama/stories/stories_val_000000.npy",
    "/home/josef/valhalla/notebooks/datasets/tokenized/llama/ao3/ao3_val_000000.npy",
    "/home/josef/valhalla/notebooks/datasets/tokenized/llama/books/books_val_000000.npy",
]


def build_model(ckpt_path, seq_len, device, dtype):
    from model_v2 import ModelArgs, Transformer
    chk = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    cfg = dict(chk["config"])
    native_T = cfg.get("max_seq_len")
    cfg["max_seq_len"] = seq_len          # tables rebuilt at probe length, mode-aware
    cfg["use_activation_checkpointing"] = False
    # Deployment semantics: no doc mask at probe time (see needle_probe.py —
    # weights-level measurement + uncompiled-flex OOM avoidance).
    cfg["doc_attn_mask"] = False
    cfg["doc_pos_reset"] = False
    known = set(ModelArgs.__dataclass_fields__.keys())
    args = ModelArgs(**{k: v for k, v in cfg.items() if k in known})
    model = Transformer(args)
    sd = chk["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # freqs are persistent=False (never in checkpoints); anything else missing is fatal
    real_missing = [k for k in missing if "freqs_" not in k]
    assert not real_missing, f"missing keys: {real_missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    model = model.to(dtype).to(device).eval()
    return model, cfg.get("rope_mode", "rope"), native_T, chk.get("step", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--windows", type=int, default=64, help="total windows across all shards")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--bin", type=int, default=128)
    ap.add_argument("--shards", type=str, default=",".join(DEFAULT_SHARDS))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--bin_by_bos", action="store_true",
                    help="bin loss by tokens-since-last-BOS (post-boundary "
                         "profile) instead of absolute position; tokens before "
                         "a window's first BOS are excluded")
    ap.add_argument("--bos_id", type=int, default=1)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model, rope_mode, native_T, step = build_model(args.ckpt, args.seq_len, args.device, dtype)
    pad_id = int(getattr(model.params, "pad_id", 0))
    print(f"[probe] ckpt step {step} | rope_mode={rope_mode} | native T={native_T} "
          f"| probe T={args.seq_len} | {args.windows} windows")

    shards = [s for s in args.shards.split(",") if s]
    per_shard = max(1, args.windows // len(shards))
    wins = []
    for sp in shards:
        toks = np.load(sp, mmap_mode="r")
        span = args.seq_len + 1
        # evenly spaced deterministic offsets, clear of the shard tail
        offs = np.linspace(0, len(toks) - span - 1, per_shard, dtype=np.int64)
        for o in offs:
            wins.append(np.asarray(toks[o:o + span], dtype=np.int64))
    print(f"[probe] {len(wins)} windows from {len(shards)} val shards")

    S = args.seq_len
    loss_sum = torch.zeros(S, dtype=torch.float64)
    loss_cnt = torch.zeros(S, dtype=torch.float64)
    # post-BOS profile: bin 0 = predicting the BOS itself (end-of-doc), bin
    # [1,5) = cold-start tokens right after a boundary, etc.
    REL_BINS = [(0, 1), (1, 5), (5, 17), (17, 65), (65, 257), (257, 1025),
                (1025, 10 ** 9)]
    rel_sum = torch.zeros(len(REL_BINS), dtype=torch.float64)
    rel_cnt = torch.zeros(len(REL_BINS), dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, len(wins), args.batch):
            w = torch.from_numpy(np.stack(wins[i:i + args.batch])).to(args.device)
            x, y = w[:, :-1], w[:, 1:]
            if args.bin_by_bos:
                idx = torch.arange(S, device=y.device).expand_as(y)
                last = torch.cummax(
                    torch.where(y == args.bos_id, idx,
                                torch.full_like(idx, -1)), dim=-1).values
                rel = torch.where(last >= 0, idx - last,
                                  torch.full_like(idx, -1))
            logits, _ = model(x)
            for lo in range(0, S, 512):  # chunk the fp32 CE to bound memory
                hi = min(lo + 512, S)
                lchunk = logits[:, lo:hi].float()
                ychunk = y[:, lo:hi]
                ce = F.cross_entropy(lchunk.reshape(-1, lchunk.shape[-1]),
                                     ychunk.reshape(-1), reduction="none")
                ce = ce.view(ychunk.shape)
                keep = (ychunk != pad_id)
                loss_sum[lo:hi] += (ce * keep).sum(0).double().cpu()
                loss_cnt[lo:hi] += keep.sum(0).double().cpu()
                if args.bin_by_bos:
                    rchunk = rel[:, lo:hi]
                    for bi, (blo, bhi) in enumerate(REL_BINS):
                        m = (rchunk >= blo) & (rchunk < bhi) & keep
                        rel_sum[bi] += (ce * m).sum().double().cpu()
                        rel_cnt[bi] += m.sum().double().cpu()
            del logits
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    mean = (loss_sum / loss_cnt.clamp(min=1)).numpy()
    bins = []
    for lo in range(0, S, args.bin):
        hi = min(lo + args.bin, S)
        bins.append({"start": lo, "end": hi,
                     "mean_loss": float(mean[lo:hi].mean()),
                     "tokens": float(loss_cnt[lo:hi].sum())})

    native = float(mean[:native_T].mean()) if native_T and native_T < S else None
    beyond = float(mean[native_T:].mean()) if native_T and native_T < S else None
    out = {
        "ckpt": args.ckpt, "step": step, "rope_mode": rope_mode,
        "native_T": native_T, "probe_T": S, "windows": len(wins),
        "overall_mean": float(mean.mean()),
        "mean_within_native_T": native,
        "mean_beyond_native_T": beyond,
        "bin_size": args.bin, "bins": bins,
    }
    if args.bin_by_bos:
        out["post_bos_bins"] = []
        labels = ["BOS-target", "1-4", "5-16", "17-64", "65-256", "257-1024", "1025+"]
        print("[probe] post-BOS profile (tokens since last boundary):")
        for lab, (blo, bhi), s, c in zip(labels, REL_BINS, rel_sum, rel_cnt):
            mean_l = float(s / max(c, 1))
            out["post_bos_bins"].append(
                {"label": lab, "mean_loss": mean_l, "tokens": float(c)})
            print(f"  {lab:>10}: {mean_l:.4f}  (n={int(c)})")

    out_path = args.out or os.path.join(os.path.dirname(args.ckpt),
                                        f"position_probe_T{S}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[probe] wrote {out_path}")
    print(f"[probe] overall {out['overall_mean']:.4f} | within native-T "
          f"{native if native is not None else float('nan'):.4f} | beyond "
          f"{beyond if beyond is not None else float('nan'):.4f}")
    for b in bins[:: max(1, len(bins) // 16)]:
        print(f"  pos {b['start']:5d}-{b['end']:5d}: {b['mean_loss']:.4f}")


if __name__ == "__main__":
    main()
