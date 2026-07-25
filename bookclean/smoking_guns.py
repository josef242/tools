import re, json, os
# independent boilerplate markers -- count how many DISTINCT ones a line trips
MARKERS = {
 'copyright': re.compile(r'\bcopyright\b|©', re.I),
 'rights':    re.compile(r'all rights reserved', re.I),
 'isbn':      re.compile(r'\be?-?isbn\b', re.I),
 'loc':       re.compile(r'library of congress|cataloging?-in-publication|catalogue record', re.I),
 'reproduce': re.compile(r'reproduced.{0,40}(retrieval system|any form|written permission)|no part of this', re.I),
 'publisher': re.compile(r'\b(penguin|harpercollins|macmillan|hachette|random house|simon & schuster|bloomsbury|scholastic|routledge)\b', re.I),
}
def guns(s): return sum(1 for r in MARKERS.values() if r.search(s))
V11=os.path.expanduser('~/data/book.v11.jsonl')
hist={}; ex={2:[],3:[],4:[]}; docs=0; lines=0
with open(V11) as f:
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
                if g in ex and len(ex[g])<3: ex[g].append(s)
        if docs>=1200: break
print(f"scanned {docs:,} docs of SHIPPED v11, {lines:,} candidate lines")
print("lines tripping N independent boilerplate markers:")
for g in sorted(hist): print(f"   {g} markers: {hist[g]:,}")
tot=sum(hist.values())
print(f"   TOTAL >=2 markers: {tot:,}  ({tot/max(docs,1):.2f} per doc)")
for g in (4,3,2):
    for s in ex.get(g,[])[:2]: print(f"\n  [{g} guns] {s[:150]}")
