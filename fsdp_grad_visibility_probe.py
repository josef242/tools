"""Diagnostic — does FSDP2 fully_shard(reshard_after_forward=True) hide a wrapped param's
.grad from model.parameters() after backward? (Explains gn_body=0 in KeelHaul's clip telemetry.)

CONTEXT
  KeelHaul's per-group clip telemetry (Probe B) shows gn_body=0.0000 every step, and nrm exactly
  equals sqrt(gn_head^2+gn_emb^2+gn_other^2) — i.e. the Muon BODY contributes nothing to the global
  grad norm / clip. The clip loop and `nrm` both iterate model.parameters() and skip p.grad is None.
  Hypothesis: in production each TransformerBlock is wrapped in its OWN fully_shard(layer,
  reshard_after_forward=True), while embeddings/head/norms live at the ROOT
  fully_shard(model, reshard_after_forward=False). After backward, the per-layer reshard may DETACH
  the gradient from the outer nn.Parameter (the sharded grad lives in FSDP's internal FSDPParam
  handle), so layer.*.weight.grad reads None at clip time — exactly producing body-absent / root-
  present. Root params (no post-fwd reshard) keep .grad visible -> head/emb/other show up.

WHAT THIS DOES (no production touch, no big model, no torchrun needed — FSDP2 runs on 1 rank)
  Build a tiny model: a ROOT linear (mimics head/emb, root-sharded reshard_after_forward=False) +
  an inner "block" linear wrapped in its own fully_shard(reshard_after_forward=True/False) (mimics a
  body layer). One forward + backward. Then inspect, for the root param and the block param:
    - is .grad None on model.parameters()?
    - is the param a DTensor / FSDP-managed?
    - does the gradient exist somewhere (e.g. via fully_shard handle / _local_tensor)?
  Sweep reshard_after_forward in {True, False} for the block to confirm it's the trigger.

  If block.grad is None ONLY when reshard_after_forward=True  => CONFIRMED: reshard detaches .grad,
  so the production clip/nrm never sees the body. Benign for Muon (it reads grads via the optimizer's
  own param handles in step()), but means `nrm` and the global clip have always been body-free.

Run (single GPU is fine; FSDP2 works on world_size=1):
  torchrun --standalone --nproc_per_node=1 fsdp_grad_visibility_probe.py
  (or plain `python fsdp_grad_visibility_probe.py` — it self-bootstraps a 1-rank process group)
"""
import os, sys
import torch
import torch.nn as nn
import torch.distributed as dist


def _bootstrap_dist():
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return dist.is_initialized()


def _grad_state(param):
    """Describe where a param's gradient is/isn't visible."""
    from torch.distributed.tensor import DTensor
    is_dt = isinstance(param, DTensor)
    g = param.grad
    info = {
        "is_DTensor": is_dt,
        "param.grad is None": g is None,
    }
    if g is not None:
        gl = g._local_tensor if isinstance(g, DTensor) else g
        info["grad_norm"] = float(gl.float().norm().item())
        info["grad_is_DTensor"] = isinstance(g, DTensor)
    return info


def run():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    have_dist = _bootstrap_dist()
    print(f"=== FSDP2 grad-visibility probe (device={dev}, dist={have_dist}) ===")
    if not torch.cuda.is_available():
        print("  [warn] no CUDA — fully_shard needs a device mesh; results may be degenerate on CPU.")

    try:
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.device_mesh import init_device_mesh
    except Exception as e:
        print(f"  [abort] could not import FSDP2 fully_shard: {e}")
        return

    mesh = init_device_mesh(dev.split(":")[0], (dist.get_world_size() if have_dist else 1,))

    # Production uses PER-SUBMODULE torch.compile on the body blocks (not the root) AND
    # fully_shard. Sweep compile on/off for the block to see if compile is what hides the grad
    # from the OUTER model.named_parameters() reference (the clip iterates the compiled `model`,
    # while every other site reads `model._orig_mod`).
    for reshard in (True, False):
      for compile_block in (False, True):
        print(f"\n--- block: fully_shard(reshard_after_forward={reshard}), torch.compile={compile_block} ---")
        torch.manual_seed(0)

        class Block(nn.Module):
            def __init__(s):
                super().__init__()
                s.lin = nn.Linear(64, 64, bias=False)
            def forward(s, x):
                return s.lin(x)

        class Model(nn.Module):
            def __init__(s):
                super().__init__()
                s.root = nn.Linear(64, 64, bias=False)   # mimics head/emb (root-sharded, uncompiled)
                s.block = Block()                         # mimics a body TransformerBlock
            def forward(s, x):
                return s.block(s.root(x))

        m = Model().to(dev)
        # per-"layer" wrap with the swept reshard policy (mimics fully_shard(layer, ...))
        fully_shard(m.block, mesh=mesh, reshard_after_forward=reshard)
        # root wrap with reshard_after_forward=False (mimics fully_shard(model, ...) for head/emb)
        fully_shard(m, mesh=mesh, reshard_after_forward=False)
        if compile_block:
            # per-submodule compile, exactly like _apply_per_submodule_compile (body only)
            m.block = torch.compile(m.block)

        x = torch.randn(8, 64, device=dev)
        out = m(x)
        loss = out.float().pow(2).mean()
        loss.backward()

        # inspect via model.named_parameters() — EXACTLY what the clip loop iterates
        print(f"  {'param':<22} | {'grad None?':<10} | {'grad_norm':<10} | DTensor")
        seen = {}
        for name, p in m.named_parameters():
            st = _grad_state(p)
            tag = 'ROOT (head/emb-like)' if name.startswith('root') else 'BLOCK (body-like)'
            seen[tag] = st
            gn = f"{st.get('grad_norm', float('nan')):.4f}" if 'grad_norm' in st else "  --  "
            print(f"  {name:<22} | {str(st['param.grad is None']):<10} | {gn:<10} | {st['is_DTensor']}")

        block = seen.get('BLOCK (body-like)', {})
        root = seen.get('ROOT (head/emb-like)', {})
        verdict = ("BLOCK grad HIDDEN (None) while ROOT grad VISIBLE"
                   if block.get('param.grad is None') and not root.get('param.grad is None')
                   else "both visible" if not block.get('param.grad is None')
                   else "both hidden")
        print(f"  => {verdict}")

    print("\n=== READ ===")
    print("  If BLOCK grad is None ONLY when reshard_after_forward=True (and ROOT always visible),")
    print("  the production gn_body=0 is CONFIRMED: per-layer reshard detaches .grad from the outer")
    print("  param, so the clip loop / nrm (which iterate model.parameters() and skip grad-None)")
    print("  never see the Muon body. Muon itself still steps the body correctly (it reads grads via")
    print("  its own FSDPParam handles in step()), so training is fine — but nrm & the global clip")
    print("  have ALWAYS been body-free. That reframes every nrm chart + the clip/WD threads.")

    if have_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
