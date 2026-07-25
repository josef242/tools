#!/usr/bin/env python3
"""Gold-tuned BiGRU: silver base + 40 wave-2 gold books (paragraph labels
derived from gold boundaries, oversampled 25x in the loss). Test: 10 held-out
pilot golds. Also scores sentinel and silver-BiGRU on the same test set."""
import json, sys, numpy as np, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
sys.path.insert(0, '.')
from bakeoff import paragraphs, sentinel_head
from build_training import featurize

def book_seq(head, boundary):
    paras = [(s,e,t) for s,e,t in paragraphs(head) if len(t.strip()) >= 15][:60]
    if len(paras) < 3: return None
    F = np.array([featurize(t,s,e,len(head),i) for i,(s,e,t) in enumerate(paras)], dtype='float32')
    y = np.array([1.0 if (s+e)/2 < boundary else 0.0 for s,e,t in paras], dtype='float32')
    starts = np.array([s for s,e,t in paras], dtype='int32')
    return F, y, starts

# gold books
wave2 = {b['offset']: b for b in json.load(open('reports/gold_wave2_books.json'))}
pilot = {b['offset']: b for b in json.load(open('reports/gold_pilot_books.json'))}
gold = json.load(open('reports/gold_labels.json'))
train_gold, test_gold = [], []
for g in gold:
    src = wave2.get(g['offset']) or pilot.get(g['offset'])
    seq = book_seq(src['head'], g['content_start'])
    if seq is None: continue
    (train_gold if g['batch'] == 2 else test_gold).append((seq, src['head'], g['content_start']))
print(f"gold train books: {len(train_gold)}, gold test books: {len(test_gold)}")

# silver base
d = np.load('reports/train_sequences.npz')
X, y, starts = d['X'], d['y'], d['starts']
lengths = d['lengths']
offs = np.zeros(len(lengths)+1, dtype='int64'); offs[1:] = np.cumsum(lengths)

class BiGRU(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.inp = nn.Linear(18, h)
        self.gru = nn.GRU(h, h, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.out = nn.Linear(2*h, 1)
    def forward(self, x, L):
        hh = torch.relu(self.inp(x))
        packed = pack_padded_sequence(hh, L, batch_first=True, enforce_sorted=False)
        o, _ = self.gru(packed)
        o, _ = pad_packed_sequence(o, batch_first=True, total_length=x.shape[1])
        return self.out(o).squeeze(-1)

dev = 'cuda:0'
torch.manual_seed(1)
model = BiGRU().to(dev)
model.load_state_dict(torch.load('reports/a4_bigru.pt', weights_only=True))  # warm start from silver
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

GOLD_W = 25.0
rng = np.random.default_rng(0)
nb = len(lengths)
def batches(epoch_books, gold_items, bs=64):
    items = [('s', b) for b in epoch_books] + [('g', i) for i in range(len(gold_items))] * 25
    rng.shuffle(items)
    for i in range(0, len(items), bs):
        chunk = items[i:i+bs]
        seqs = []
        for kind, b in chunk:
            if kind == 's':
                seqs.append((X[offs[b]:offs[b+1]], y[offs[b]:offs[b+1]], 1.0))
            else:
                (F, yy, st), _, _ = gold_items[b]
                seqs.append((F, yy, GOLD_W))
        m = max(len(s[1]) for s in seqs)
        xb = np.zeros((len(seqs), m, 18), dtype='float32')
        yb = np.zeros((len(seqs), m), dtype='float32')
        wb = np.zeros((len(seqs), m), dtype='float32')
        L = []
        for j, (F, yy, w) in enumerate(seqs):
            xb[j,:len(yy)] = F; yb[j,:len(yy)] = yy; wb[j,:len(yy)] = w
            L.append(len(yy))
        yield torch.tensor(xb), torch.tensor(yb), torch.tensor(wb), torch.tensor(L)

# fine-tune: subsample silver each epoch to balance
for epoch in range(4):
    model.train()
    epoch_books = rng.choice(nb, 4000, replace=False)
    for xb, yb, wb, L in batches(list(epoch_books), train_gold):
        xb, yb, wb = xb.to(dev), yb.to(dev), wb.to(dev)
        opt.zero_grad()
        logits = model(xb, L)
        loss = (nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none') * wb).sum() / wb.sum()
        loss.backward(); opt.step()
    print(f"epoch {epoch+1} done", flush=True)
torch.save(model.state_dict(), 'reports/a5_bigru_gold.pt')

# evaluate boundary error on held-out pilot golds: gold-BiGRU vs silver-BiGRU vs sentinel
silver = BiGRU().to(dev)
silver.load_state_dict(torch.load('reports/a4_bigru.pt', weights_only=True))
def bnd(m, head):
    seq = book_seq(head, 0)
    if seq is None: return 0
    F, _, st = seq
    m.eval()
    with torch.no_grad():
        pj = torch.sigmoid(m(torch.tensor(F[None]).to(dev), torch.tensor([len(F)]))).squeeze(0).cpu().numpy()[:len(F)]
    if pj[0] < 0.5 and (len(pj) < 2 or pj[1] < 0.5): return 0
    for j in range(len(pj)):
        if pj[j] < 0.5 and (j+1 >= len(pj) or pj[j+1] < 0.5):
            return int(st[j])
    return 12000

rows = []
for (F, yy, st), head, truth in test_gold:
    s = sentinel_head(head); s = s if s is not None else len(head)
    rows.append((truth, s, bnd(silver, head), bnd(model, head)))
print(f"\n{'truth':>7} {'sentinel':>9} {'silverGRU':>10} {'goldGRU':>9}")
errs = {k: [] for k in ('sentinel','silver','gold')}
for truth, s, sv, gd in rows:
    print(f"{truth:>7} {s:>9} {sv:>10} {gd:>9}")
    errs['sentinel'].append(s-truth); errs['silver'].append(sv-truth); errs['gold'].append(gd-truth)
for k, es in errs.items():
    med = sorted(abs(e) for e in es)[len(es)//2]
    print(f"  {k:<9} medAbsErr={med:<7} within150: {sum(1 for e in es if abs(e)<=150)}/10  overcut>200: {sum(1 for e in es if e>200)}")
