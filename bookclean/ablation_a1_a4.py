#!/usr/bin/env python3
"""Ablations A1 (context-window MLP, ±2 neighbors) and A4 (BiGRU sequence
tagger). Both trained on the same 30k-book sequence dataset, split by BOOK.
Metrics: paragraph AUC + boundary char-error vs silver on val books."""
import numpy as np, torch, torch.nn as nn, json, time

d = np.load('reports/train_sequences.npz')
X, y, starts = d['X'], d['y'], d['starts']
lengths, bounds = d['lengths'], d['boundaries']
nb = len(lengths)
offs = np.zeros(nb+1, dtype='int64'); offs[1:] = np.cumsum(lengths)
rng = np.random.default_rng(0)
perm = rng.permutation(nb)
val_books = set(perm[:nb//10].tolist())
dev = 'cuda:0'

# ---------- A1: context-window MLP ----------
def ctx_features(i0, i1):
    """features of para j concat with neighbors j-2..j+2 (zero-padded)."""
    F = X[i0:i1]
    n, f = F.shape
    out = np.zeros((n, f*5), dtype='float32')
    for k, off in enumerate((-2,-1,0,1,2)):
        lo, hi = max(0,-off), min(n, n-off)
        out[lo:hi, k*f:(k+1)*f] = F[lo+off:hi+off]
    return out

Xc = np.concatenate([ctx_features(offs[b], offs[b+1]) for b in range(nb)])
tr_mask = np.concatenate([(np.full(lengths[b], b not in val_books)) for b in range(nb)])
Xc_tr = torch.tensor(Xc[tr_mask]).to(dev); y_tr = torch.tensor(y[tr_mask]).to(dev)
Xc_va = torch.tensor(Xc[~tr_mask]).to(dev); y_va = torch.tensor(y[~tr_mask]).to(dev)

def auc_of(p, yv):
    order = p.argsort(); ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(len(p), device=p.device, dtype=torch.float)
    pos = yv > 0.5
    return ((ranks[pos].sum() - pos.sum()*(pos.sum()-1)/2) / (pos.sum()*(~pos).sum())).item()

print("=== A1: context-MLP (90-dim input, res-8 w192 body) ===", flush=True)
class ResBlock(nn.Module):
    def __init__(self, w, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(w); self.fc1 = nn.Linear(w, w); self.fc2 = nn.Linear(w, w)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        return x + self.fc2(self.drop(torch.relu(self.fc1(self.ln(x)))))
torch.manual_seed(1)
a1 = nn.Sequential(nn.Linear(90, 192), *[ResBlock(192) for _ in range(8)],
                   nn.LayerNorm(192), nn.Linear(192, 1)).to(dev)
opt = torch.optim.AdamW(a1.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss()
best = 0
for epoch in range(15):
    a1.train()
    p2 = torch.randperm(len(y_tr), device=dev)
    for i in range(0, len(y_tr), 8192):
        b = p2[i:i+8192]
        opt.zero_grad(); lossf(a1(Xc_tr[b]).squeeze(-1), y_tr[b]).backward(); opt.step()
    a1.eval()
    with torch.no_grad():
        auc = auc_of(torch.sigmoid(a1(Xc_va).squeeze(-1)), y_va)
    best = max(best, auc)
    if epoch % 3 == 2: print(f"  epoch {epoch+1}: val_auc {auc:.4f}", flush=True)
print(f"A1 best AUC: {best:.4f}")
torch.save(a1.state_dict(), 'reports/a1_context_mlp.pt')

# ---------- A4: BiGRU sequence tagger ----------
print("\n=== A4: BiGRU (2-layer, hidden 128, bidirectional) ===", flush=True)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
maxlen = int(lengths.max())
def make_batches(book_ids, bs=64):
    for i in range(0, len(book_ids), bs):
        chunk = book_ids[i:i+bs]
        L = [int(lengths[b]) for b in chunk]
        m = max(L)
        xb = np.zeros((len(chunk), m, 18), dtype='float32')
        yb = np.zeros((len(chunk), m), dtype='float32')
        mask = np.zeros((len(chunk), m), dtype=bool)
        for j, b in enumerate(chunk):
            xb[j,:L[j]] = X[offs[b]:offs[b+1]]; yb[j,:L[j]] = y[offs[b]:offs[b+1]]
            mask[j,:L[j]] = True
        yield (torch.tensor(xb), torch.tensor(yb), torch.tensor(mask), torch.tensor(L))

class BiGRU(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.inp = nn.Linear(18, h)
        self.gru = nn.GRU(h, h, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.out = nn.Linear(2*h, 1)
    def forward(self, x, L):
        h = torch.relu(self.inp(x))
        packed = pack_padded_sequence(h, L, batch_first=True, enforce_sorted=False)
        o, _ = self.gru(packed)
        o, _ = pad_packed_sequence(o, batch_first=True, total_length=x.shape[1])
        return self.out(o).squeeze(-1)

torch.manual_seed(1)
a4 = BiGRU().to(dev)
opt = torch.optim.AdamW(a4.parameters(), lr=1e-3, weight_decay=1e-4)
train_books = [b for b in range(nb) if b not in val_books]
va_books = [b for b in range(nb) if b in val_books]
best4 = 0
for epoch in range(8):
    a4.train(); rng.shuffle(train_books)
    for xb, yb, mask, L in make_batches(train_books):
        xb, yb, mask = xb.to(dev), yb.to(dev), mask.to(dev)
        opt.zero_grad()
        logits = a4(xb, L)
        loss = (nn.functional.binary_cross_entropy_with_logits(
            logits, yb, reduction='none') * mask).sum() / mask.sum()
        loss.backward(); opt.step()
    a4.eval(); ps, ys = [], []
    with torch.no_grad():
        for xb, yb, mask, L in make_batches(va_books):
            logits = a4(xb.to(dev), L)
            m = mask.to(dev)
            ps.append(torch.sigmoid(logits)[m]); ys.append(yb.to(dev)[m])
    auc = auc_of(torch.cat(ps), torch.cat(ys))
    best4 = max(best4, auc)
    print(f"  epoch {epoch+1}: val_auc {auc:.4f}", flush=True)
print(f"A4 best AUC: {best4:.4f}")
torch.save(a4.state_dict(), 'reports/a4_bigru.pt')

# ---------- boundary error vs silver on val books ----------
def boundary_from_probs(pj, sts, blen):
    for j in range(len(pj)):
        if pj[j] < 0.5 and (j+1 >= len(pj) or pj[j+1] < 0.5):
            return int(sts[j])
    return blen

errs1, errs4 = [], []
with torch.no_grad():
    for xb, yb, mask, L in make_batches(va_books):
        logits4 = torch.sigmoid(a4(xb.to(dev), L)).cpu().numpy()
        for j, b in enumerate(va_books[:0]): pass
    # per-book: recompute with contexts for a1
    for k, b in enumerate(va_books):
        i0, i1 = offs[b], offs[b+1]
        xc = torch.tensor(ctx_features(i0, i1)).to(dev)
        p1 = torch.sigmoid(a1(xc).squeeze(-1)).cpu().numpy()
        xb = torch.tensor(X[i0:i1][None]).to(dev)
        p4 = torch.sigmoid(a4(xb, torch.tensor([i1-i0]))).squeeze(0).cpu().numpy()[:i1-i0]
        sts = starts[i0:i1]; bl = int(bounds[b])
        errs1.append(abs(boundary_from_probs(p1, sts, 12000) - bl))
        errs4.append(abs(boundary_from_probs(p4, sts, 12000) - bl))
import statistics as st
print(f"\nboundary |err| vs silver on {len(errs1)} val books:")
print(f"  A1 context-MLP: median {st.median(errs1):.0f}  p90 {sorted(errs1)[int(len(errs1)*0.9)]}")
print(f"  A4 BiGRU:       median {st.median(errs4):.0f}  p90 {sorted(errs4)[int(len(errs4)*0.9)]}")
json.dump({'a1_auc': best, 'a4_auc': best4,
           'a1_boundary_median': st.median(errs1), 'a4_boundary_median': st.median(errs4)},
          open('reports/ablation_a1_a4.json','w'), indent=1)
