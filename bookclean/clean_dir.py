#!/usr/bin/env python3
"""clean_dir.py -- apply the FILE-LOCAL bookclean passes across a directory of .jsonl files.

Passes (all deletion-only; ladder order v9 -> v10 -> v11 is enforced regardless of --passes order):
  structural : clean_midbook -> clean_books3_v9   (midbook notes; front/back-matter boundary cut) [file-local]
  crossbook  : dedup_book (v10)                    (STRONG boilerplate line in >=N records)       [GLOBAL 2-pass]
  within     : within_dedup_book                   (repeated-line furniture + degenerate lines)    [file-local]

Cleaned files are MIRRORED to --out-dir (originals untouched). Every removed span is streamed to a
sidecar (--removed-out) as {file, record, pass, chars, text} so you can audit what's being thrown away
(tail -f during the run; shuf -n / group-by pass after). Capture is driver-side via difflib, so the
file-local engines are not modified.

crossbook is GLOBAL: pass 1 scans every file to build a distinct-record line-frequency table, pass 2
removes a line iff it matches the STRONG boilerplate pattern AND appears in >= --cross-book-min-dup
records. Two caveats: (1) the STRONG pattern is BOOKS-tuned (ISBN/publisher/copyright/Gutenberg), so on
a non-books corpus it may fire rarely -- the safe failure mode; (2) it's the pass most able to over-remove,
so ALWAYS --dry-run and inspect removed.jsonl before using the output. Our ablation pins crossbook as the
measured memorization-harm reducer; structural/within are hygiene (null interference harm at 132M).

Run with the project env:
  $HOME/miniconda3/envs/bookclean/bin/python clean_dir.py --in-dir DIR --out-dir DIR [--passes structural,crossbook,within] [--dry-run]
"""
import os, sys, json, argparse, glob, difflib, random
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from midbook_clean import clean_midbook
from books3_clean_v9 import clean_books3_v9
from within_dedup_filter import within_dedup_book
from dedup_filter import dedup_book, hh as _cb_hh, MIN_DUP as _CB_MIN_DEFAULT

# --- cross-book (global) state: freq table shared with forked workers, min-dup threshold ---
_FREQ = {}            # line-hash -> distinct-book count (set by main before the pass-2 Pool forks)
_CB_MIN_DUP = _CB_MIN_DEFAULT

def _crossbook_step(t):
    return dedup_book(t, _FREQ, min_dup=_CB_MIN_DUP)[0]

# atomic steps (name -> text->text); user-facing passes map onto these
_STEPS = {
    'midbook':   lambda t: clean_midbook(t)[0],
    'boundary':  lambda t: clean_books3_v9(t)[0],
    'crossbook': _crossbook_step,
    'within':    lambda t: within_dedup_book(t)[0],
}
_PASS_STEPS = {'structural': ['midbook', 'boundary'], 'crossbook': ['crossbook'], 'within': ['within']}
_CANON = ['midbook', 'boundary', 'crossbook', 'within']   # ladder order (v9 -> v10 -> v11)
_MAX_CHARDIFF = 20_000   # skip char-level diff on huge single-hunk replaces (perf guard)


def _scan_freq(task):
    """Pass-1 worker: distinct-book line frequency for one file (mirrors build_freq_table)."""
    in_path, text_key, limit_recs = task
    f = Counter()
    with open(in_path, errors='replace') as fh:
        for idx, line in enumerate(fh):
            if limit_recs and idx >= limit_recs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get(text_key)
            except Exception:
                continue
            if not isinstance(t, str):
                continue
            seen = set()
            for l in t.split('\n'):
                s = l.strip()
                if 12 <= len(s) <= 250:
                    seen.add(_cb_hh(s))
            for k in seen:
                f[k] += 1
    return f


def removed_chunks(before, after):
    """Removed text pieces for a deletion-only transform (before -> after).
    Line-level diff for block/line deletions; char-level diff on modified hunks for inline removals."""
    if before == after:
        return []
    bl, al = before.split('\n'), after.split('\n')
    sm = difflib.SequenceMatcher(None, bl, al, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'delete':
            out += [l.strip() for l in bl[i1:i2] if l.strip()]
        elif tag == 'replace':
            b, a = '\n'.join(bl[i1:i2]), '\n'.join(al[j1:j2])
            if len(b) > _MAX_CHARDIFF:          # perf guard: don't O(n^2) a giant line
                frag = b.strip()
                if frag:
                    out.append(frag[:500] + ' [..truncated capture..]')
                continue
            cm = difflib.SequenceMatcher(None, b, a, autojunk=False)
            for t2, x1, x2, y1, y2 in cm.get_opcodes():
                if t2 in ('delete', 'replace'):
                    frag = b[x1:x2].strip()
                    if frag:
                        out.append(frag)
    return out


def process_file(task):
    in_path, out_path, rem_path, steps, text_key, dry, limit_recs = task
    stats = Counter()
    fout = None if dry else open(out_path, 'w')
    with open(in_path, errors='replace') as fin, open(rem_path, 'w') as frem:
        for idx, line in enumerate(fin):
            if limit_recs and idx >= limit_recs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                stats['bad_json'] += 1
                continue
            if not isinstance(rec.get(text_key), str):
                stats['no_text'] += 1
                if fout:
                    fout.write(json.dumps(rec, ensure_ascii=True) + '\n')
                continue
            t = rec[text_key]
            orig_len = len(t)
            for name in steps:
                before = t
                try:
                    t = _STEPS[name](t)
                except Exception:
                    stats['err_' + name] += 1
                    t = before
                    continue
                if t != before:
                    for frag in removed_chunks(before, t):
                        frem.write(json.dumps(
                            {'file': os.path.basename(in_path), 'record': idx, 'pass': name,
                             'chars': len(frag), 'text': frag[:500]}, ensure_ascii=True) + '\n')
                        stats['rm_' + name] += 1
            stats['chars_removed'] += orig_len - len(t)
            stats['records'] += 1
            rec[text_key] = t
            if fout:
                fout.write(json.dumps(rec, ensure_ascii=True) + '\n')
    if fout:
        fout.close()
    return dict(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-dir', required=True, help='cleaned files mirrored here (created if absent)')
    ap.add_argument('--passes', default='structural,within',
                    help='comma list of {structural,crossbook,within} (default: structural,within). '
                         'crossbook is GLOBAL (two-pass, books-tuned pattern gate) -- see --cross-book-min-dup.')
    ap.add_argument('--cross-book-min-dup', type=int, default=3,
                    help='crossbook: remove a boilerplate line only if it appears in >= this many distinct '
                         'records (default 3 = the SHIPPED+AUDITED v10 recipe. >=2 is a v12 CANDIDATE: it '
                         'carries a real +0.33-nat fixed benefit but has NOT passed the validation gauntlet >=3 passed '
                         '(gold zero-damage, removed-line audit, breaker review, pristine-set FP). Promote only if it clears.')
    ap.add_argument('--freq-cache', default=None,
                    help='crossbook: reuse/persist the pass-1 line-freq table at this pickle path '
                         '(default <out-dir>/line_freq.pkl); reused if it already exists')
    ap.add_argument('--text-key', default='text')
    ap.add_argument('--glob', default='*.jsonl')
    ap.add_argument('--removed-out', default=None, help='sidecar of removed spans (default <out-dir>/removed.jsonl)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--dry-run', action='store_true', help='capture removals + stats but write no cleaned files')
    ap.add_argument('--limit-files', type=int, default=0, help='process only first N files (trial)')
    ap.add_argument('--limit-records', type=int, default=0, help='per file, process only first N records (trial)')
    ap.add_argument('--sample-removed', type=int, default=8, help='print N random removed spans at the end')
    a = ap.parse_args()

    in_dir = os.path.expanduser(a.in_dir)
    out_dir = os.path.expanduser(a.out_dir)
    sel = set()
    for p in a.passes.split(','):
        p = p.strip()
        if p not in _PASS_STEPS:
            ap.error(f'unknown pass {p!r}; choose from {list(_PASS_STEPS)}')
        sel.update(_PASS_STEPS[p])
    steps = [s for s in _CANON if s in sel]   # enforce ladder order regardless of input order

    files = sorted(glob.glob(os.path.join(in_dir, a.glob)))
    if a.limit_files:
        files = files[:a.limit_files]
    if not files:
        ap.error(f'no files matching {a.glob!r} in {in_dir}')
    os.makedirs(out_dir, exist_ok=True)

    # ---- crossbook pass-1: build (or load) the global distinct-book line-freq table ----
    if 'crossbook' in steps:
        global _FREQ, _CB_MIN_DUP
        _CB_MIN_DUP = a.cross_book_min_dup
        import pickle
        cache = os.path.expanduser(a.freq_cache) if a.freq_cache else os.path.join(out_dir, 'line_freq.pkl')
        if os.path.exists(cache):
            print(f"[crossbook] loading cached freq table: {cache}")
            _FREQ = pickle.load(open(cache, 'rb'))
        else:
            print(f"[crossbook] pass 1/2: scanning {len(files)} file(s) for cross-record line frequencies...")
            agg = Counter()
            with Pool(min(a.workers, len(files))) as pool:
                for i, fq in enumerate(pool.imap_unordered(
                        _scan_freq, [(f, a.text_key, a.limit_records) for f in files]), 1):
                    agg.update(fq)
                    print(f"    scanned {i}/{len(files)}", flush=True)
            _FREQ = {k: c for k, c in agg.items() if c >= 2}   # >=2 books can ever be a cross-book dup
            pickle.dump(_FREQ, open(cache, 'wb'), protocol=4)
            print(f"    freq table: {len(agg):,} distinct lines, {len(_FREQ):,} in >=2 records -> {cache}")
        print(f"[crossbook] threshold min_dup={_CB_MIN_DUP} (line removed iff STRONG pattern AND in >={_CB_MIN_DUP} records)")
    rem_dir = os.path.join(out_dir, '.removed_shards')
    os.makedirs(rem_dir, exist_ok=True)
    removed_out = os.path.expanduser(a.removed_out) if a.removed_out else os.path.join(out_dir, 'removed.jsonl')

    print(f"[clean_dir] {len(files)} file(s) | passes={a.passes} (steps={steps}) | "
          f"workers={a.workers}{' | DRY-RUN' if a.dry_run else ''}")
    tasks = [(f, os.path.join(out_dir, os.path.basename(f)),
              os.path.join(rem_dir, os.path.basename(f) + '.rm.jsonl'),
              steps, a.text_key, a.dry_run, a.limit_records) for f in files]

    total = Counter()
    with Pool(min(a.workers, len(tasks))) as pool:
        for i, st in enumerate(pool.imap_unordered(process_file, tasks), 1):
            total.update(st)
            print(f"  [{i}/{len(files)}] records={total['records']:,} chars_removed={total['chars_removed']:,}", flush=True)

    # merge removed shards -> single sidecar
    n_spans = 0
    with open(removed_out, 'w') as out:
        for f in files:
            shard = os.path.join(rem_dir, os.path.basename(f) + '.rm.jsonl')
            if os.path.exists(shard):
                with open(shard) as sh:
                    for l in sh:
                        out.write(l); n_spans += 1
                os.remove(shard)
    try:
        os.rmdir(rem_dir)
    except OSError:
        pass

    print("\n=== SUMMARY ===")
    print(f"records processed : {total['records']:,}")
    print(f"chars removed     : {total['chars_removed']:,}")
    for k in sorted(total):
        if k.startswith('rm_') or k.startswith('err_') or k in ('bad_json', 'no_text'):
            print(f"  {k:16s}: {total[k]:,}")
    print(f"removed spans     : {n_spans:,} -> {removed_out}")
    if not a.dry_run:
        print(f"cleaned files     : {out_dir}/")

    if a.sample_removed and n_spans:
        rows = [json.loads(l) for l in open(removed_out)]
        rng = random.Random(0)
        print(f"\n=== {min(a.sample_removed, len(rows))} random removed spans (audit) ===")
        for r in rng.sample(rows, min(a.sample_removed, len(rows))):
            disp = r['text'] if len(r['text']) < 140 else r['text'][:140] + '…'
            print(f"  [{r['pass']:9s} {r['chars']:>5}c] {disp.strip()[:140]}")


if __name__ == '__main__':
    main()
