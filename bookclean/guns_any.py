import re, json, os, sys
MARKERS = {
 'copyright': re.compile(r'\bcopyright\b|©', re.I),
 'rights':    re.compile(r'all rights reserved', re.I),
 'isbn':      re.compile(r'\be?-?isbn\b', re.I),
 'loc':       re.compile(r'library of congress|cataloging?-in-publication|catalogue record', re.I),
 'reproduce': re.compile(r'reproduced.{0,40}(retrieval system|any form|written permission)|no part of this', re.I),
 'publisher': re.compile(r'\b(penguin|harpercollins|macmillan|hachette|random house|simon & schuster|bloomsbury|scholastic|routledge)\b', re.I),
}
def guns(s): return sum(1 for r in MARKERS.values() if r.search(s))
path=os.path.expanduser(sys.argv[1]); LIM=int(sys.argv[2]) if len(sys.argv)>2 else 1200
hist={}; ex={}; docs=0; lines=0
with open(path) as f:
    for line in f:
        docs+=1
        try: t=json.loads(line)['text']
        except Exception: continue
        for l in t.split('\n'):
            s=l.strip()
            if len(s)<12 or len(s)>250: continue
            lines+=1
            g=guns(s)
            if g>=2:
                hist[g]=hist.get(g,0)+1
                ex.setdefault(g,[]).append(s)
        if docs>=LIM: break
tot=sum(hist.values())
print(f"{os.path.basename(path)}: {docs:,} docs, {lines:,} lines")
for g in sorted(hist): print(f"   {g} markers: {hist[g]:,}")
print(f"   TOTAL >=2: {tot:,}   ({tot/max(docs,1):.3f} per doc)")
for g in sorted(ex, reverse=True)[:2]:
    for s in ex[g][:2]: print(f"   [{g}] {s[:120]}")
