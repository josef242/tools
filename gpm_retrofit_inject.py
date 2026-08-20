"""Retrofit the live ` | gpm: +S/+L` status-line field into a historical gen_log.txt,
IN PLACE, byte-identical to what the in-trainer GPMTracker emits going forward.

WHY
  The Dashboard parses fields out of each gen_log step line by independent regex. To show
  GPM for runs that finished BEFORE the metric existed (dreadnought/DN2, mf-low-lr), we splice
  the same `gpm:` field the live trainer now writes. A retrofitted historical run and a future
  live run are then computed IDENTICALLY (this tool replays GPMReplay from gpm.py, which shares
  the core `gpm_window` math with the in-trainer GPMTracker) and are directly comparable.

FORMAT (must match train_mara.py GPMTracker.status_tag EXACTLY)
  appended at end of each TRAINING step line:   ` | gpm: +0.31/+0.25`
  value formatter f(v):  f"{v:+.2f}" if v is not None else " -- "   (note the surrounding spaces)
  if BOTH gpm_s and gpm_l are None (first <5 steps) -> NO tag at all (matches live: status_tag
  returns "" so nothing is appended).
  NO trend arrow (dropped 2026-06-24: trending is read on the Dashboard, status line stays compact).

WHAT IS / ISN'T TOUCHED
  - TRAINING lines (have `nrm:` AND `t_tk:`, the Dashboard's pattern_mara) -> gpm appended.
  - EVAL/AVG lines, headers, blank lines -> untouched, copied verbatim.
  - A line that ALREADY has `gpm:` -> left as-is (idempotent; safe to re-run).

SAFETY
  - Always writes a timestamped .bak first (these are real training artifacts).
  - --dry-run prints what would change without writing.
  - Atomic replace via .tmp + os.replace.

USAGE
  python gpm_retrofit_inject.py PATH/gen_log.txt                 # in place, default 15/101
  python gpm_retrofit_inject.py PATH/gen_log.txt --dry-run       # preview, no write
  python gpm_retrofit_inject.py PATH/gen_log.txt --short 15 --long 101 --lag 1
"""
import os, re, sys, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpm import GPMReplay  # single source of truth for the trailing-window math (== live tracker)

# A training step line, the Dashboard's pattern_mara: needs st:, nrm:, t_tk:.
_TRAIN_RE = re.compile(r"st:\s*(\d+).*?ls:\s*([0-9.]+).*?nrm:\s*([0-9.]+).*?t_tk:")
_HAS_GPM_RE = re.compile(r"\bgpm:")


def _fmt(gs, gl):
    """EXACT replica of GPMTracker.status_tag: returns ' | gpm: +0.31/+0.25', or
    ' | gpm: pending' when both are None (the first <5 steps), matching live byte-for-byte.
    The Dashboard's numeric regex skips 'pending'."""
    if gs is None and gl is None:
        return " | gpm: pending"
    def f(v):
        return f"{v:+.2f}" if v is not None else " -- "
    return f" | gpm: {f(gs)}/{f(gl)}"


def retrofit(path, w_short=15, w_long=101, lag=1, dry_run=False):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        lines = fh.readlines()  # keep original newlines per line

    rep = GPMReplay(w_short, w_long, lag)
    out = []
    n_train = n_injected = n_skipped_have = 0

    for line in lines:
        m = _TRAIN_RE.search(line)
        if not m:
            out.append(line)              # eval/header/blank -> verbatim
            continue
        n_train += 1
        if _HAS_GPM_RE.search(line):
            # Already retrofitted: do NOT push again (would double-count) — but the replay
            # buffer still needs this step's (nrm, ls) to stay aligned. Push, skip injecting.
            rep.push(float(m.group(3)), float(m.group(2)))
            n_skipped_have += 1
            out.append(line)
            continue

        ls, nr = float(m.group(2)), float(m.group(3))
        rep.push(nr, ls)
        gs, gl = rep.values()
        tag = _fmt(gs, gl)                 # ' | gpm: +S/+L' or ' | gpm: pending' (first <5 steps)
        # splice tag at end of the line's text, BEFORE its newline(s)
        body = line.rstrip("\r\n")
        nl = line[len(body):]             # preserve the exact original line ending
        out.append(body + tag + nl)
        n_injected += 1

    sample = next((l for l in out if _HAS_GPM_RE.search(l)), None)
    print(f"{os.path.basename(os.path.dirname(path)) or path}: "
          f"{len(lines)} lines, {n_train} training lines, "
          f"{n_injected} gpm injected, {n_skipped_have} already had gpm "
          f"(short={w_short} long={w_long} lag={lag}).")
    if sample:
        print("  sample:", sample.rstrip()[-90:])

    if dry_run:
        print("  [dry-run] no file written.")
        return

    bak = f"{path}.pre-gpm.{time.strftime('%Y%m%d-%H%M%S')}.bak"
    # back up the original verbatim
    with open(bak, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(lines)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(out)
    os.replace(tmp, path)
    print(f"  backup -> {bak}")
    print(f"  rewrote {path}")


def main():
    ap = argparse.ArgumentParser(description="Inject live-format gpm field into a historical gen_log.txt")
    ap.add_argument("log", help="path to gen_log.txt (edited in place)")
    ap.add_argument("--short", type=int, default=15)
    ap.add_argument("--long", type=int, default=101)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    retrofit(a.log, a.short, a.long, a.lag, a.dry_run)


if __name__ == "__main__":
    main()
