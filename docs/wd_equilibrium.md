# wd_equilibrium.py
> Computes the equilibrium body-weight norm each weight-decay strength would produce, given the measured per-step update magnitude — the "taming dial" for the body-norm ramp.

## What it does
This is **Test 2 of the WD-taming battery** (the weight-decay-waste investigation, Math Agent equilibrium model). For a pre-norm / scale-invariant matrix under decoupled WD plus a tangential optimizer update `U ⟂ W`, the norm reaches equilibrium when radial WD shrinkage balances tangential injection: `||W_eq|| ≈ ||U|| / sqrt(2·η·λ)`, where `||U|| = η·||update||` is the actual per-step `||ΔW||`. The tool reads a per-matrix update-norm probe output and current weight norms, then reports, per weight class, the equilibrium norm at several WD multiples and the exact `λ` that would pin each class at its current norm (stop further growth). Reach for it to size the body-only WD dial — i.e. "what WD bounds the body norm at a target scale?"

It is downstream of **`normuon_update_decomp.py`** (the `nud_*.json` source of `update_norm`/`w_norm`) and imports the tolerant JSONL reader from sibling **`wd_waste_probe.py`**. See also the broader WD-waste analysis doc (`mara_fsdp2/docs/WD_WASTE_ANALYSIS.md`).

## Usage
```bash
# Equilibrium analysis from a NorMuon update-decomp probe output
python wd_equilibrium.py --nud nud_<ckpt>.json --eta 3e-4 --wd 0.02

# Custom WD multiples + optional head w_norm from diagnostics, write JSON result
python wd_equilibrium.py --nud nud_<ckpt>.json --diag diagnostics.jsonl \
    --eta 3e-4 --wd 0.02 --mults 1,2,4,10 --out wdeq_<ckpt>.json
```
No GPU or checkpoint load is required — all inputs are already-computed files on disk.

## Key arguments
| flag | default | meaning |
| --- | --- | --- |
| `--nud` | (required) | `nud_*.json` from `normuon_update_decomp.py`; provides `per_matrix` rows with `update_norm` and `w_norm`. |
| `--eta` | (required) | Current learning rate η used to convert `||update||` → per-step `||ΔW||` and in the equilibrium formula. |
| `--wd` | `0.02` | Current decoupled WD strength λ (the `1x` baseline). |
| `--diag` | `None` | Optional `diagnostics.jsonl`; only used to pull the head (`output.weight`) `w_norm` from the last record. |
| `--mults` | `1,2,4,10` | Comma-separated WD multipliers; column `eq@Nx` is the equilibrium norm at `N·λ`. |
| `--out` | `None` | If set, writes the per-class result dict as JSON. |

## Notes
- Aggregation is **per weight class** via `_class_of`: `body_proj` (`attention.wo`, `feed_forward.w2`), `body_in` (`wq/wk/wv`, `feed_forward.w1/w3`), `head` (`output.weight`), `embedding` (`tok_embeddings`); class `other` is skipped. Each cell is the **median** over matrices in that class.
- Output columns: `cur||W||` (median current norm), `||U||/step` (= η·median update_norm), `eq@Nx` (equilibrium norm at each multiple), and `lambda_to_pin_current` (the λ making the current norm the equilibrium: `λ = (U/cur_w)² / (2η)`).
- Interpretation rule of thumb from the source: if `eq@1x` ≫ `cur||W||`, the class is still far below equilibrium and will keep ramping; pick a multiple whose `eq@Nx` matches a target norm. `lambda_to_pin_current` vs the current λ sizes the dial.
- Equilibrium scales as `||U||/sqrt(2ηλ) ∝ η^{1/2}/sqrt(λ)`, which is why the ramp is LR-coupled — `apply_normuon` preserves the update magnitude so `||U||` is roughly LR-scaled and only weakly gradient-dependent.
- The `--diag` head w_norm is read into a local dict but only the `nud` probe's own `w_norm` feeds the class aggregation; diag is effectively a convenience/fallback for the head and is otherwise non-load-bearing in the current code path.
