#!/usr/bin/env python
"""needle_probe.py — retrieval-at-depth for BASE models (cloze-margin needle test).

Distinguishes "processes long context without degrading" (position_probe)
from "actually RETRIEVES from long context" (this tool). For each
(context_len, depth) cell: splice a tokenized needle sentence into real
haystack tokens at the target depth, append a cue, and measure
    margin = logP(answer | haystack + needle + cue)
           - logP(answer | haystack        + cue)      # no-needle control
The control subtracts the answer's prior, so positive margin == retrieval.
Also reports argmax accuracy (answer is the top-1 next token). Scoring-only:
one forward per condition, no generation, no text stored.

Usage:
  python needle_probe.py --ckpt .../model_step_017999.pt \
      --ctx 2048,4096,8192,16384,32768 --depths 0.1,0.5,0.9 --samples 8
Writes <run_dir>/needle_probe.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common_fsdp2"))

HAYSTACK = "/home/josef/valhalla/notebooks/datasets/tokenized/llama/books/books_val_000000.npy"

# Cloze needles: narrative register (matches training data), single-token
# answers (verified at runtime; non-single-token pairs are skipped).
NEEDLES = [
    ("Later that evening she whispered the secret password: {}.",
     "When asked, he remembered that the secret password was", " seven"),
    ("Later that evening she whispered the secret password: {}.",
     "When asked, he remembered that the secret password was", " river"),
    ("The innkeeper wrote the number of the room on his hand: {}.",
     "He looked at his hand again; the number of the room was", " nine"),
    ("The old map marked the treasure beneath the ancient {}.",
     "They dug all night beneath the ancient", " oak"),
    ("Her grandmother's cat was named {}, after the winter month.",
     "She called out for the cat, whose name was", " December"),
    ("The captain hid the key inside a hollow {}.",
     "The key, he recalled, was hidden inside a hollow", " book"),
]


def build_model(ckpt_path, seq_len, device):
    from model_v2 import ModelArgs, Transformer
    chk = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    cfg = dict(chk["config"])
    cfg["max_seq_len"] = seq_len
    cfg["use_activation_checkpointing"] = False
    # Probe under DEPLOYMENT semantics: plain attention, no doc mask. The
    # firewall question is whether training burned boundary behavior into the
    # WEIGHTS — a runtime mask would block cross-doc attention structurally
    # (trivially, uninformatively). Also: the probe runs eager, and uncompiled
    # flex_attention materializes O(S^2) scores -> OOM at 32k (2026-08-01).
    cfg["doc_attn_mask"] = False
    cfg["doc_pos_reset"] = False
    known = set(ModelArgs.__dataclass_fields__.keys())
    model = Transformer(ModelArgs(**{k: v for k, v in cfg.items() if k in known}))
    missing, unexpected = model.load_state_dict(chk["model"], strict=False)
    assert not [k for k in missing if "freqs_" not in k] and not unexpected
    tok_meta = {"tok_kind": chk.get("tok_kind"), "tok_path": chk.get("tok_path"),
                "special_tokens": chk.get("special_tokens")}
    return (model.to(torch.bfloat16).to(device).eval(),
            cfg.get("rope_mode", "rope"), chk.get("step", "?"), tok_meta)


def get_tok(tok_meta, tok_path_arg):
    from tokenizer_abstraction import get_tokenizer
    path = tok_path_arg or tok_meta.get("tok_path")
    if path and not os.path.isabs(path):
        cand = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", path))
        if os.path.exists(cand):
            path = cand
    return get_tokenizer(tok_meta.get("tok_kind") or "llama", path=path,
                         special_tokens=tok_meta.get("special_tokens"))


@torch.no_grad()
def last_logprob(model, toks, answer_id, device):
    x = torch.as_tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x)
    last = logits[0, -1].float()
    lp = torch.log_softmax(last, dim=-1)[answer_id].item()
    top1 = int(last.argmax().item()) == answer_id
    del logits
    return lp, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ctx", type=str, default="2048,4096,8192,16384,32768")
    ap.add_argument("--depths", type=str, default="0.1,0.5,0.9")
    ap.add_argument("--samples", type=int, default=8, help="haystack windows per cell")
    ap.add_argument("--tok_path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--boundary_after_needle", action="store_true",
                    help="insert a BOS between needle and cue — tests CROSS-"
                         "document retrieval (does doc-reset training build "
                         "firewalls that inference inherits?)")
    ap.add_argument("--bos_id", type=int, default=1)
    args = ap.parse_args()

    ctxs = [int(c) for c in args.ctx.split(",")]
    depths = [float(d) for d in args.depths.split(",")]
    model, rope_mode, step, tok_meta = build_model(args.ckpt, max(ctxs), args.device)
    enc = get_tok(tok_meta, args.tok_path)

    needles = []
    for tmpl, cue, ans in NEEDLES:
        # SentencePiece merges a leading space INTO the word in context but
        # emits a standalone '▁' token when encoded in isolation — so derive
        # the answer id by tokenizing cue+answer jointly and diffing.
        c_ids = enc.encode(" " + cue, bos=False, eos=False)
        full = enc.encode(" " + cue + ans, bos=False, eos=False)
        if full[:len(c_ids)] != c_ids or len(full) != len(c_ids) + 1:
            continue  # boundary re-merge or multi-token answer: skip
        word = ans.strip()
        n_ids = enc.encode(" " + tmpl.format(word), bos=False, eos=False)
        needles.append((n_ids, c_ids, full[-1]))
    assert needles, "no single-token needles under this tokenizer"
    print(f"[needle] step {step} | rope_mode={rope_mode} | {len(needles)} needles "
          f"| ctx {ctxs} | depths {depths} | {args.samples} windows/cell")

    hay = np.load(HAYSTACK, mmap_mode="r")
    rng = np.random.default_rng(1234)
    offs = rng.integers(0, len(hay) - max(ctxs) - 8, size=args.samples)

    results = []
    for C in ctxs:
        for d in depths:
            margins, accs = [], []
            for si in range(args.samples):
                n_ids, c_ids, ans_id = needles[si % len(needles)]
                base = hay[offs[si]:offs[si] + C].astype(np.int64)
                bnd = [args.bos_id] if args.boundary_after_needle else []
                room = len(n_ids) + len(c_ids) + len(bnd)
                hs = base[: C - room]
                pos = max(1, int(len(hs) * d))
                # boundary mode: needle lives in the PREVIOUS document; the
                # BOS right after it starts a new doc containing the cue.
                with_n = np.concatenate([hs[:pos], n_ids, bnd, hs[pos:], c_ids])
                without = np.concatenate([hs[:pos], bnd, hs[pos:], c_ids])
                lp_w, top_w = last_logprob(model, with_n, ans_id, args.device)
                lp_o, _ = last_logprob(model, without, ans_id, args.device)
                margins.append(lp_w - lp_o)
                accs.append(top_w)
            results.append({"ctx": C, "depth": d,
                            "margin": float(np.mean(margins)),
                            "margin_std": float(np.std(margins)),
                            "top1_acc": float(np.mean(accs)),
                            "n": len(margins)})
            r = results[-1]
            print(f"  ctx {C:6d} depth {d:.1f}: margin {r['margin']:+6.2f} "
                  f"(±{r['margin_std']:.2f})  top1 {r['top1_acc']:.2f}")

    out = {"ckpt": args.ckpt, "step": step, "rope_mode": rope_mode,
           "samples_per_cell": args.samples, "cells": results}
    out_path = args.out or os.path.join(os.path.dirname(args.ckpt), "needle_probe.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[needle] wrote {out_path}")


if __name__ == "__main__":
    main()
