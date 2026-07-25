#!/usr/bin/env python3
"""Ablation A2: capacity sweep on the existing 18-feature dataset.
Hypothesis: flat — capacity is NOT the binding constraint. Configs share
recipe (AdamW 2e-3, wd 1e-4, BCE, bs 8192, 15 epochs); only shape varies."""
import numpy as np, torch, torch.nn as nn, json, time

d = np.load('reports/train_paragraphs.npz')
X, y = torch.tensor(d['X']), torch.tensor(d['y'])
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(y), generator=g)
X, y = X[idx], y[idx]
split = int(len(y) * 0.9)
dev = 'cuda:0'
Xtr, ytr = X[:split].to(dev), y[:split].to(dev)
Xva, yva = X[split:].to(dev), y[split:].to(dev)

def make(widths, dropout=0.1):
    layers, prev = [], X.shape[1]
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU(), nn.Dropout(dropout)]
        prev = w
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)

def auc_of(p, yv):
    order = p.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(len(p), device=p.device, dtype=torch.float)
    pos = yv > 0.5
    return ((ranks[pos].sum() - pos.sum()*(pos.sum()-1)/2) / (pos.sum()*(~pos).sum())).item()

CONFIGS = {
    'tiny-32':        [32],
    'baseline-96x96': [96, 96],
    'wide-384x384':   [384, 384],
    'deep-192x4':     [192, 192, 192, 192],
    'huge-768x768x768': [768, 768, 768],
}
results = {}
for name, widths in CONFIGS.items():
    torch.manual_seed(1)
    model = make(widths).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    t0 = time.time()
    best_auc = 0
    for epoch in range(15):
        model.train()
        perm = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(ytr), 8192):
            b = perm[i:i+8192]
            opt.zero_grad()
            lossf(model(Xtr[b]).squeeze(-1), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva).squeeze(-1))
            best_auc = max(best_auc, auc_of(p, yva))
    acc = ((p > 0.5) == (yva > 0.5)).float().mean().item()
    results[name] = {'params': nparams, 'val_acc': round(acc, 4),
                     'best_val_auc': round(best_auc, 4), 'sec': round(time.time()-t0, 1)}
    print(f"{name:<18} params={nparams:>8,} acc={acc:.4f} best_auc={best_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(results, open('reports/ablation_a2_capacity.json','w'), indent=1)
