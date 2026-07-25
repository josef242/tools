#!/usr/bin/env python3
"""Ablation A2b: depth scaling at fixed width 192 (the KEEL question).
Plain stacks 2->12 layers + residual/LayerNorm blocks 8->24 layers."""
import numpy as np, torch, torch.nn as nn, json, time

d = np.load('reports/train_paragraphs.npz')
X, y = torch.tensor(d['X']), torch.tensor(d['y'])
idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
X, y = X[idx], y[idx]
split = int(len(y) * 0.9)
dev = 'cuda:0'
Xtr, ytr = X[:split].to(dev), y[:split].to(dev)
Xva, yva = X[split:].to(dev), y[split:].to(dev)

class ResBlock(nn.Module):
    def __init__(self, w, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(w)
        self.fc1 = nn.Linear(w, w); self.fc2 = nn.Linear(w, w)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        h = self.drop(torch.relu(self.fc1(self.ln(x))))
        return x + self.fc2(h)

def plain(depth, w=192):
    layers, prev = [], 18
    for _ in range(depth):
        layers += [nn.Linear(prev, w), nn.ReLU(), nn.Dropout(0.1)]; prev = w
    return nn.Sequential(*layers, nn.Linear(w, 1))

def resnet(depth, w=192):
    return nn.Sequential(nn.Linear(18, w), *[ResBlock(w) for _ in range(depth)],
                         nn.LayerNorm(w), nn.Linear(w, 1))

def auc_of(p, yv):
    order = p.argsort(); ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(len(p), device=p.device, dtype=torch.float)
    pos = yv > 0.5
    return ((ranks[pos].sum() - pos.sum()*(pos.sum()-1)/2) / (pos.sum()*(~pos).sum())).item()

CONFIGS = [('plain-2', plain(2)), ('plain-4', plain(4)), ('plain-6', plain(6)),
           ('plain-8', plain(8)), ('plain-12', plain(12)),
           ('res-8', resnet(8)), ('res-16', resnet(16)), ('res-24', resnet(24))]
results = {}
for name, model in CONFIGS:
    torch.manual_seed(1)
    for m in model.modules():
        if isinstance(m, nn.Linear): nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    model = model.to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    t0 = time.time(); best = 0
    for epoch in range(18):
        model.train()
        perm = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(ytr), 8192):
            b = perm[i:i+8192]
            opt.zero_grad()
            lossf(model(Xtr[b]).squeeze(-1), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            best = max(best, auc_of(torch.sigmoid(model(Xva).squeeze(-1)), yva))
    acc = ((torch.sigmoid(model(Xva).squeeze(-1)) > 0.5) == (yva > 0.5)).float().mean().item()
    results[name] = {'params': nparams, 'val_acc': round(acc,4), 'best_auc': round(best,4),
                     'sec': round(time.time()-t0,1)}
    print(f"{name:<10} params={nparams:>9,} acc={acc:.4f} best_auc={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(results, open('reports/ablation_a2b_depth.json','w'), indent=1)
