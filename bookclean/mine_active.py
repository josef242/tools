#!/usr/bin/env python3
"""Active-learning miner: rank books by MiniLM-vs-rules DISAGREEMENT on tail
paragraphs. High disagreement = the model sees junk the rules miss (or vice
versa) = maximally informative to hand-label next."""
import json, sys, random, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '.')
from transformers import AutoTokenizer, AutoModel
from bakeoff import paragraphs
from sentinel3 import classify_tail_para
from span_cutter import body_genre_is_riskly

dev = 'cuda:0'
tok = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
enc = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(dev).eval()
head = nn.Sequential(nn.Linear(384,256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256,1)).to(dev).eval()
ck = torch.load('reports/para_clf.pt', weights_only=True)
enc.load_state_dict(ck['enc']); head.load_state_dict(ck['head'])

@torch.no_grad()
def probs(texts):
    if not texts: return np.array([])
    out=[]
    for i in range(0,len(texts),128):
        b = tok([t[:800] for t in texts[i:i+128]], padding=True, truncation=True,
                max_length=192, return_tensors='pt').to(dev)
        h = enc(**b).last_hidden_state
        m = b['attention_mask'].unsqueeze(-1).float()
        pooled = (h*m).sum(1)/m.sum(1).clamp(min=1)
        out.append(torch.sigmoid(head(pooled).squeeze(-1)).cpu().numpy())
    return np.concatenate(out)

used = set()
for f in ('reports/gold_tails_books.json','reports/gold_tails_test_books.json',
          'reports/gold_tails_val3_books.json','reports/gold_tails_strat_books.json'):
    used |= {b['offset'] for b in json.load(open(f))}

rng = random.Random(11)
offs = [json.loads(l)['offset'] for l in open('reports/silver_books3_heads.jsonl')]
offs = [o for o in offs if o not in used]
rng.shuffle(offs); offs = offs[:3000]

scored = []
with open('/home/josef/data/book.v1.jsonl','rb') as f:
    for k, off in enumerate(offs):
        f.seek(off); rec = json.loads(f.readline())
        t = rec['text']; tail = t[-10000:]
        ps = [(s,e,x) for s,e,x in paragraphs(tail) if len(x.strip())>=15]
        if len(ps) < 4: continue
        texts = [x for _,_,x in ps]
        pr = probs(texts)
        rk = [classify_tail_para(x) for x in texts]
        # disagreement: model-junk(>0.85) where rules say keep, OR model-content(<0.15) where rules cut
        dis = sum(1 for p,r in zip(pr,rk) if (p>0.85 and r=='keep') or (p<0.15 and r=='cut'))
        frac = dis/len(ps)
        scored.append((frac, off, rec['meta'].get('title','?')[:50], body_genre_is_riskly(t)))
        if k % 500 == 499: print(f"  scanned {k+1}", flush=True)

scored.sort(reverse=True)
top = scored[:180]
print(f"\nscanned {len(scored)} books; top-180 by disagreement")
print("disagreement histogram:", np.histogram([s[0] for s in scored], bins=[0,.05,.1,.2,.4,1])[0].tolist())
print("\ntop 12 most-informative books to label:")
for frac, off, title, risk in top[:12]:
    print(f"  {frac:.2f} {'[genre]' if risk else '       '} {title}")
json.dump([{'offset':o,'title':t,'disagreement':round(f,3),'genre_risk':r} for f,o,t,r in top],
          open('reports/active_candidates.json','w'))
print("\nsaved reports/active_candidates.json (180 books)")
