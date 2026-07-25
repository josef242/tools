#!/usr/bin/env python3
"""Produce book.v10.jsonl = v9 + corpus-wide boilerplate-line dedup. Parallel
shards. Also counts top duplicated lines removed (harm-proxy) + flagged books."""
import json, os, sys, pickle
from multiprocessing import Pool
from collections import Counter
sys.path.insert(0,'.')
SRC=os.path.expanduser('~/data/book.v9.jsonl'); DST=os.path.expanduser('~/data/book.v10.jsonl')
SH=os.path.expanduser('~/data/.v10d_shards')
def work(task):
    from dedup_filter import dedup_book
    import pickle as pk
    FREQ=pk.load(open('reports/line_freq.pkl','rb'))
    wid,start,end=task; st=Counter(); removed_lines=Counter()
    with open(SRC,'rb') as f, open(f"{SH}/s{wid:03d}.jsonl",'wb') as out, open(f"{SH}/l{wid:03d}.jsonl",'w') as lg:
        f.seek(start)
        if start: f.readline()
        pos=f.tell()
        while pos<end:
            raw=f.readline()
            if not raw: break
            pos+=len(raw); st['books']+=1
            rec=json.loads(raw); t2,info=dedup_book(rec['text'],FREQ)
            if info.get('flagged'):
                st['flagged']+=1; lg.write(json.dumps({'flag':info['reason']})+'\n'); out.write(raw)
            elif info['removed']:
                st['touched']+=1; st['lines']+=info['removed']; st['chars']+=len(rec['text'])-len(t2)
                rec['text']=t2; out.write(json.dumps(rec,ensure_ascii=True).encode()+b'\n')
            else:
                out.write(raw)
    return dict(st)
def main():
    os.makedirs(SH,exist_ok=True); size=os.path.getsize(SRC); n=24
    bounds=[size*i//n for i in range(n+1)]; tot=Counter()
    with Pool(10) as pool:
        for i,s in enumerate(pool.imap_unordered(work,[(i,bounds[i],bounds[i+1]) for i in range(n)])):
            tot.update(s)
            if (i+1)%6==0: print(f"[{i+1}/{n}] {dict(tot)}",flush=True)
    with open(DST,'wb') as out:
        for i in range(n):
            with open(f"{SH}/s{i:03d}.jsonl",'rb') as sh:
                while True:
                    c=sh.read(64_000_000)
                    if not c: break
                    out.write(c)
            os.remove(f"{SH}/s{i:03d}.jsonl")
    with open('reports/dedup_ledger.jsonl','w') as L:
        for i in range(n): L.write(open(f"{SH}/l{i:03d}.jsonl").read()); os.remove(f"{SH}/l{i:03d}.jsonl")
    print(f"DONE {dict(tot)} -> {DST}")
if __name__=='__main__': main()
