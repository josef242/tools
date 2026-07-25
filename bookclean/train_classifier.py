#!/usr/bin/env python3
"""Round-3 contender: paragraph junk classifier (MLP on GTX 1070)."""
import numpy as np, torch, torch.nn as nn

d = np.load('reports/train_paragraphs.npz')
X, y = torch.tensor(d['X']), torch.tensor(d['y'])
n = len(y)
idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
X, y = X[idx], y[idx]
split = int(n * 0.9)
Xtr, ytr, Xva, yva = X[:split], y[:split], X[split:], y[split:]

dev = 'cuda:0'
model = nn.Sequential(
    nn.Linear(X.shape[1], 96), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(96, 96), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(96, 1)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss()
Xtr, ytr = Xtr.to(dev), ytr.to(dev)
Xva_d, yva_d = Xva.to(dev), yva.to(dev)

BS = 8192
for epoch in range(12):
    model.train()
    perm = torch.randperm(len(ytr), device=dev)
    tot = 0.0
    for i in range(0, len(ytr), BS):
        b = perm[i:i+BS]
        opt.zero_grad()
        loss = lossf(model(Xtr[b]).squeeze(-1), ytr[b])
        loss.backward(); opt.step()
        tot += loss.item() * len(b)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xva_d).squeeze(-1))
        acc = ((p > 0.5) == (yva_d > 0.5)).float().mean().item()
        # AUC (approximate via rank)
        order = p.argsort()
        ranks = torch.empty_like(order, dtype=torch.float); ranks[order] = torch.arange(len(p), device=dev, dtype=torch.float)
        pos = yva_d > 0.5
        auc = ((ranks[pos].sum() - pos.sum()*(pos.sum()-1)/2) / (pos.sum()*(~pos).sum())).item()
    print(f"epoch {epoch+1:2d} loss {tot/len(ytr):.4f} val_acc {acc:.4f} val_auc {auc:.4f}", flush=True)

torch.save(model.state_dict(), 'reports/junk_classifier.pt')
print("saved reports/junk_classifier.pt")
