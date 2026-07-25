#!/usr/bin/env python3
"""Fine-tune MiniLM-L6 as a paragraph junk classifier.

Role (strictly bounded by the zero-cut contract):
  the model may only promote NEUTRAL -> CUT at high confidence inside sentinel3.
  It may never override body-protection, KEEP-anchors, or the genre guard.
  It attacks RECALL; it is structurally forbidden from creating content-cuts.
"""
import json, random, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MAXLEN = 192
dev = 'cuda:0'

rows = json.load(open('reports/para_dataset.json'))
gold = [r for r in rows if r[2] != 'silver']
silver = [r for r in rows if r[2] == 'silver']
rng = random.Random(0)
rng.shuffle(gold); rng.shuffle(silver)

# hold out 20% of GOLD books' paragraphs for validation
nval = int(len(gold) * 0.2)
val = gold[:nval]
train = gold[nval:] * 4 + silver          # gold oversampled 4x
rng.shuffle(train)
print(f"train={len(train):,}  val={len(val):,} (gold only)")

tok = AutoTokenizer.from_pretrained(NAME)
enc = AutoModel.from_pretrained(NAME).to(dev)
head = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1)).to(dev)
opt = torch.optim.AdamW([{'params': enc.parameters(), 'lr': 2e-5},
                         {'params': head.parameters(), 'lr': 1e-3}], weight_decay=0.01)
lossf = nn.BCEWithLogitsLoss()

def batches(data, bs):
    for i in range(0, len(data), bs):
        chunk = data[i:i+bs]
        b = tok([c[0][:800] for c in chunk], padding=True, truncation=True,
                max_length=MAXLEN, return_tensors='pt').to(dev)
        y = torch.tensor([float(c[1]) for c in chunk], device=dev)
        yield b, y

def fwd(b):
    out = enc(**b).last_hidden_state
    mask = b['attention_mask'].unsqueeze(-1).float()
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
    return head(pooled).squeeze(-1)

BS = 96
for epoch in range(2):
    enc.train(); head.train()
    tot = 0; seen = 0
    for i, (b, y) in enumerate(batches(train, BS)):
        opt.zero_grad()
        loss = lossf(fwd(b), y)
        loss.backward(); opt.step()
        tot += loss.item()*len(y); seen += len(y)
        if i % 400 == 0:
            print(f"  ep{epoch+1} step {i} loss {tot/max(seen,1):.4f}", flush=True)
    # validate on held-out GOLD paragraphs
    enc.eval(); head.eval()
    ps, ys = [], []
    with torch.no_grad():
        for b, y in batches(val, 256):
            ps.append(torch.sigmoid(fwd(b)).cpu()); ys.append(y.cpu())
    p = torch.cat(ps); yv = torch.cat(ys)
    acc = ((p > 0.5) == (yv > 0.5)).float().mean().item()
    # precision of the JUNK class at high threshold (what the contract needs)
    for thr in (0.5, 0.9, 0.95, 0.99):
        sel = p > thr
        prec = (yv[sel] == 1).float().mean().item() if sel.sum() else float('nan')
        rec = (p[yv == 1] > thr).float().mean().item()
        print(f"  ep{epoch+1} thr={thr}: junk_precision={prec:.4f} junk_recall={rec:.4f} (n={int(sel.sum())})")
    print(f"  ep{epoch+1} val_acc={acc:.4f}", flush=True)

torch.save({'enc': enc.state_dict(), 'head': head.state_dict()}, 'reports/para_clf.pt')
print("saved reports/para_clf.pt")
