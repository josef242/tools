import json, os, sys, difflib
sys.path.insert(0,'.')
from dedup_filter import STRONG_LINE
A=os.path.expanduser('~/data/book.v9.jsonl'); B=os.path.expanduser('~/data/book.v9.1.jsonl')
junk=tot=junk_ch=tot_ch=n=0
with open(A) as fa, open(B) as fb:
    for la,lb in zip(fa,fb):
        if la==lb: continue
        try: ta=json.loads(la)['text']; tb=json.loads(lb)['text']
        except Exception: continue
        if ta==tb: continue
        sm=difflib.SequenceMatcher(None, ta.split('\n'), tb.split('\n'), autojunk=False)
        for tag,i1,i2,j1,j2 in sm.get_opcodes():
            if tag in ('insert','replace'):
                for l in tb.split('\n')[j1:j2]:
                    s=l.strip()
                    if not s: continue
                    tot+=1; tot_ch+=len(s)
                    if STRONG_LINE.search(s): junk+=1; junk_ch+=len(s)
        n+=1
        if n>=300: break
print(f"sampled {n} changed docs")
print(f"rescued nonblank lines : {tot:,}")
print(f"  match STRONG_LINE    : {junk:,} ({junk/max(tot,1)*100:.1f}% of lines)")
print(f"  by chars             : {junk_ch/max(tot_ch,1)*100:.1f}% of rescued text is convicted boilerplate")
