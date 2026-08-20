#!/usr/bin/env python
"""kda_horizon.py — read KDA's learned memory horizons out of the weights.

For safe_gate KDA (K3 parameterization) the per-channel log-decay is
    g = g_min * sigmoid(exp(A_head) * z),   z = f_proj(x) + dt_bias
which is input-dependent. At the RESTING point (x = 0, z = dt_bias) each
channel has a characteristic retention g_rest, and a half-life in tokens
    t_half = ln(2) / |g_rest|.
The resting-point histogram approximates the dependency lengths training
installed — channels with t_half >> probe length predict a flat extension
curve (no bend). CPU-only; reads A_log/dt_bias straight from a checkpoint.
"""
import argparse
import math

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--g_min", type=float, default=-5.0)
    args = ap.parse_args()

    chk = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    sd = chk["model"]
    layers = sorted({int(k.split(".")[1]) for k in sd if ".gdn_attn.A_log" in k})
    print(f"[horizon] {len(layers)} delta layers | g_min={args.g_min} "
          f"| step {chk.get('step', '?')}")

    BUCKETS = [(0, 256), (256, 1024), (1024, 4096), (4096, 16384),
               (16384, 65536), (65536, 262144), (262144, float("inf"))]
    total = torch.zeros(len(BUCKETS))
    all_half = []

    for li in layers:
        A = sd[f"layers.{li}.gdn_attn.A_log"].float()          # [heads]
        db = sd[f"layers.{li}.gdn_attn.dt_bias"].float()       # [gate_dim]
        H = A.numel()
        z = db.view(H, -1)                                     # [heads, d_k]
        g = args.g_min * torch.sigmoid(torch.exp(A)[:, None] * z)
        half = math.log(2) / g.abs().clamp(min=1e-9)           # tokens
        all_half.append(half.flatten())
        q = torch.quantile(half.flatten(),
                           torch.tensor([0.1, 0.5, 0.9, 0.99]))
        print(f"  L{li:2d}: t_half p10={q[0]:9.0f}  p50={q[1]:9.0f}  "
              f"p90={q[2]:11.0f}  p99={q[3]:13.0f}")
        for bi, (lo, hi) in enumerate(BUCKETS):
            total[bi] += ((half >= lo) & (half < hi)).sum()

    allh = torch.cat(all_half)
    n = allh.numel()
    print(f"\n[horizon] {n} channels total — half-life distribution:")
    labels = ["<256", "256-1k", "1k-4k", "4k-16k", "16k-64k", "64k-256k", ">256k"]
    for lab, cnt in zip(labels, total):
        bar = "#" * int(60 * cnt / n)
        print(f"  {lab:>9}: {cnt / n * 100:5.1f}%  {bar}")
    for p in (0.5, 0.9, 0.99):
        print(f"  p{int(p * 100):02d} = {torch.quantile(allh, p):,.0f} tokens")


if __name__ == "__main__":
    main()
