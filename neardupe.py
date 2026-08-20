#!/usr/bin/env python3
"""Document-level near-duplicate detection for large pre-tokenized corpora.

Companion to dataset_explorer.py. Kept as a sibling module (rather than folded into
the explorer) so the algorithm can be unit-tested without opening a dataset.

PIPELINE
    token ids -> n-gram shingle hashes -> one-permutation MinHash (OPH)
              -> 1-bit b-bit signature -> all-pairs similarity -> union-find clusters

WHY THIS SHAPE, given book-sized documents on 8x3090-class hardware:

1. TOKEN SHINGLES, NOT TEXT SHINGLES. The corpora being deduped are already
   tokenized on disk. Decoding back to text to shingle it is wasted work AND lossy
   (detokenize -> re-tokenize does not round-trip), so we hash token n-grams directly.

2. ONE-PERMUTATION HASHING. Classic MinHash applies k independent permutations at
   O(k*m) per document. OPH bins a SINGLE hash of each shingle into k buckets and
   takes the per-bucket minimum: O(m). At k=1024 that is a ~1000x reduction in
   sketching work, and sketching is the term that scales with total corpus tokens.
   Book-length documents have m >> k, so empty buckets are vanishingly rare and the
   densification correction almost never fires (it is implemented for short docs).

3. ONE-BIT SIGNATURES (b-bit minwise hashing, Li & Konig). Keeping only the low bit
   of each bucket minimum costs 128 bytes/doc at k=1024. Two consequences:
     - the whole signature set stays resident even on an 8 GB card
       (1M docs = 128 MB packed), so a heterogeneous GPU pool is not a problem;
     - encoded as +/-1, the number of agreeing bits is a DOT PRODUCT, so all-pairs
       similarity is a matmul and runs on tensor cores.
   For 1-bit signatures P(bit agrees) = (1 + J)/2, hence J = dot / k exactly.
   That identity is the whole reason this is fast.

   Caveat: P(agree) = (1+J)/2 is the large-set limit. For documents with few shingles
   the estimator is biased; see MIN_SHINGLES_FOR_1BIT.

4. NO LSH BELOW THE CROSSOVER. LSH exists to dodge the N^2 term. At book-corpus
   document counts that term is cheap on a GPU, and skipping LSH removes banding
   parameters, S-curve recall loss, and the transitive over-merging that bucket
   collisions cause. Above the crossover, LSH generates candidates and the same
   dot-product kernel verifies them.
"""

import json
import time
from pathlib import Path

import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # CPU-only fallback; correctness identical, throughput is not
    TORCH_AVAILABLE = False

# --- hashing constants -------------------------------------------------------
_U64 = np.uint64
_MASK64 = (1 << 64) - 1
_MIX_A = 0xbf58476d1ce4e5b9          # splitmix64 finalizer multipliers
_MIX_B = 0x94d049bb133111eb
_POLY_P = 0x100000001b3               # FNV-1a 64-bit prime, used as the n-gram base

SENTINEL = np.uint64(0xFFFFFFFFFFFFFFFF)   # "bucket is empty" marker

# Below this many distinct shingles the 1-bit estimator's large-set assumption breaks down
# and most buckets end up empty, so densification fills them by probing the few occupied
# ones -- which can manufacture agreement between two short documents that share only a
# handful of shingles. Documents under this size are sketched normally but counted per
# cluster as 'n_short' so the caller can discount them.
MIN_SHINGLES_FOR_1BIT = 200

# A cluster is a connected component, so A~B and B~C pull in A~C. Edge density -- the share
# of possible within-cluster pairs that actually cleared the threshold -- separates a tight
# group (density 1.0) from a transitive chain (density ~2/n). Note that MEAN similarity
# cannot do this job: only pairs at or above the threshold are recorded, so the mean is
# bounded below by the threshold and stays reassuringly close to the max even on a chain.
CHAIN_DENSITY = 0.5


# =============================================================================
# Shingling
# =============================================================================

def _mix64_np(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer. The raw polynomial n-gram hash leaves structure in the
    low bits, and OPH derives its bucket index from exactly those bits, so the mix
    is required for correct bucket balance -- not merely cosmetic."""
    x = x.astype(_U64, copy=True)
    with np.errstate(over='ignore'):
        x ^= x >> _U64(30)
        x *= _U64(_MIX_A)
        x ^= x >> _U64(27)
        x *= _U64(_MIX_B)
        x ^= x >> _U64(31)
    return x


def token_shingle_hashes(tokens: np.ndarray, n: int = 13) -> np.ndarray:
    """Unique 64-bit hashes of a document's token n-grams, sorted ascending.

    n defaults to 13 following SlimPajama's token-level choice: longer grams are more
    distinctive, which matters for book-length documents where short grams collide on
    ordinary language rather than on shared provenance.
    """
    tokens = np.asarray(tokens)
    if tokens.size == 0:
        return np.empty(0, dtype=_U64)
    if tokens.size < n:
        n = int(tokens.size)          # short doc: one whole-document shingle
    t = tokens.astype(_U64, copy=False)
    m = t.size - n + 1
    h = np.zeros(m, dtype=_U64)
    P = _U64(_POLY_P)
    with np.errstate(over='ignore'):
        for j in range(n):
            h = h * P + t[j:j + m]
    return np.unique(_mix64_np(h))


# =============================================================================
# One-permutation MinHash
# =============================================================================

def _mix64_scalar(x: int) -> int:
    x &= _MASK64
    x ^= x >> 30
    x = (x * _MIX_A) & _MASK64
    x ^= x >> 27
    x = (x * _MIX_B) & _MASK64
    x ^= x >> 31
    return x


def oph_sketch(shingles: np.ndarray, k: int = 1024) -> np.ndarray:
    """One-permutation MinHash: (k,) array of per-bucket minima, SENTINEL where empty.

    k must be a power of two so the bucket index is a mask rather than a modulo.
    Buckets come from the low log2(k) bits and the retained value from the rest.
    """
    if k & (k - 1):
        raise ValueError(f"k must be a power of two, got {k}")
    sig = np.full(k, SENTINEL, dtype=_U64)
    if shingles.size == 0:
        return sig

    shift = int(k).bit_length() - 1
    bins = (shingles & _U64(k - 1)).astype(np.int64)
    vals = shingles >> _U64(shift)

    # Per-bucket minimum WITHOUT np.minimum.at, which is pathologically slow.
    # Sorting by (bucket, value) puts each bucket's minimum first in its run.
    order = np.lexsort((vals, bins))
    b_sorted = bins[order]
    v_sorted = vals[order]
    first = np.empty(b_sorted.size, dtype=bool)
    first[0] = True
    np.not_equal(b_sorted[1:], b_sorted[:-1], out=first[1:])
    sig[b_sorted[first]] = v_sorted[first]
    return sig


def densify(sig: np.ndarray, k: int) -> np.ndarray:
    """Fill empty buckets by unbiased probing into occupied ones.

    Shrivastava's optimal densification: an empty bucket borrows from a bucket chosen
    by hashing (bucket_index, attempt), retrying while it lands on another empty. This
    keeps the Jaccard estimator unbiased, unlike simply leaving empties as a shared
    sentinel (which would make sparse documents look similar to each other).

    For book-length documents this is effectively a no-op -- with m >> k the expected
    number of empty buckets is k*exp(-m/k), which is ~0 at m=100k, k=1024.
    """
    empties = np.flatnonzero(sig == SENTINEL)
    if empties.size == 0 or empties.size == k:
        return sig
    out = sig.copy()
    # Vectorized over ATTEMPT ROUNDS rather than buckets. The probe target depends
    # only on (bucket, attempt) -- never on sig -- so a whole round of probes for
    # every unresolved bucket is one _mix64_np call. Bit-identical to the original
    # scalar loop: same probe sequence, donors read from the ORIGINAL sig (an empty
    # bucket never borrows from another densified one), same 4k attempt cap.
    #
    # This is not a micro-optimization. A one-shingle document (any doc shorter than
    # the n-gram window) has k-1 empty buckets, each probing ~k times before finding
    # the single occupied one: ~4M scalar-Python hash calls, ~800 ms PER DOCUMENT.
    # A corpus with a few percent of short docs turned an hours-long sketch into
    # days. Vectorized, the same case is a few milliseconds.
    j = empties.astype(np.int64)              # still-unresolved bucket indices
    chunk = 64                                # attempts probed per round
    attempt = 0
    while j.size and attempt <= 4 * k:
        n_att = min(chunk, 4 * k + 1 - attempt)
        att = np.arange(attempt, attempt + n_att, dtype=np.int64)
        probes = (_mix64_np((j[None, :] + att[:, None] * k).astype(_U64))
                  % _U64(k)).astype(np.int64)
        hits = sig[probes] != SENTINEL        # (n_att, |j|)
        any_hit = hits.any(axis=0)
        if any_hit.any():
            first = hits.argmax(axis=0)       # first hitting attempt within the chunk
            res = np.flatnonzero(any_hit)
            out[j[res]] = sig[probes[first[res], res]]
            j = j[~any_hit]
        attempt += n_att
    return out


def sketch_to_bits(sig: np.ndarray) -> np.ndarray:
    """Low bit of each bucket minimum, packed little-endian into uint8 (k/8 bytes)."""
    bits = (sig & _U64(1)).astype(np.uint8)
    return np.packbits(bits, bitorder='little')


def sketch_document(tokens: np.ndarray, n: int = 13, k: int = 1024
                    ) -> Tuple[np.ndarray, int]:
    """Full single-document path: tokens -> (packed 1-bit signature, shingle count)."""
    sh = token_shingle_hashes(tokens, n)
    sig = densify(oph_sketch(sh, k), k)
    return sketch_to_bits(sig), int(sh.size)


def sketch_corpus(docs: Iterable[np.ndarray], n: int = 13, k: int = 1024,
                  progress=None, out: Optional[np.ndarray] = None,
                  out_cards: Optional[np.ndarray] = None,
                  start: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Sketch an iterable of token arrays.

    Returns (signatures, cardinalities):
        signatures    (N, k/8) uint8, packed bits
        cardinalities (N,)     int64, distinct shingle count per document
    Cardinalities are retained because containment |A^B|/min(|A|,|B|) cannot be
    recovered from the signatures alone, and costs 8 bytes per document to keep.

    Pass `out` / `out_cards` (arrays or np.memmaps with at least N rows) to have results
    written IN PLACE. At corpus scale this is the difference between working and not: the
    accumulate-then-stack path holds one Python object per document -- at 13M docs that is
    gigabytes of interpreter overhead, it makes every GC pass progressively slower, and the
    final stack briefly doubles peak memory. Writing into a memmap holds none of it.
    """
    if start and out is None:
        raise ValueError("start (sketch resume) requires streaming output arrays")
    rows: List[np.ndarray] = []
    cards: List[int] = []
    count = start          # resume: docs iterator starts at this corpus position
    for toks in docs:
        packed, card = sketch_document(toks, n, k)
        if out is not None:
            out[count] = packed
            out_cards[count] = card
        else:
            rows.append(packed)
            cards.append(card)
        count += 1
        if progress is not None:
            progress(count)
    if out is not None:
        return out[:count], out_cards[:count]
    if not rows:
        return np.empty((0, k // 8), dtype=np.uint8), np.empty(0, dtype=np.int64)
    return np.vstack(rows), np.asarray(cards, dtype=np.int64)


# =============================================================================
# GPU sketching
#
# Sketching is the stage that scales with TOTAL CORPUS TOKENS, and measurement showed
# it -- not the all-pairs match -- is what bounds the pipeline: ~3.2 M tok/s on one CPU
# core, ~64 M tok/s across 20 cores, versus an all-pairs match that finishes 1M docs in
# seconds. The rolling hash is 13 streaming passes over the token array plus one
# scatter_reduce, so on a GPU it is purely bandwidth-bound and should run ~2 orders of
# magnitude faster, leaving the pipeline bound by NAS read instead.
#
# torch has no usable uint64, so the 64-bit hash arithmetic runs on int64. Two's
# complement multiplication produces identical low-64 bits to unsigned, so products
# match numpy exactly; only right shifts differ (arithmetic vs logical) and are
# emulated below. GPU and CPU signatures are asserted bit-identical in the tests.
# =============================================================================

def _as_i64(u: int) -> int:
    """Reinterpret a 64-bit unsigned constant as the int64 with the same bit pattern."""
    return u - (1 << 64) if u >= (1 << 63) else u


def _lshr(x: "torch.Tensor", s: int) -> "torch.Tensor":
    """Logical (zero-filling) right shift on int64.

    torch's >> is arithmetic on signed types, so a negative value -- which is simply a
    hash with the high bit set -- would sign-extend and corrupt the result. Masking to
    the surviving 64-s bits restores unsigned semantics. Valid for s >= 1; every shift
    used here is >= 10.
    """
    return (x >> s) & ((1 << (64 - s)) - 1)


def _mix64_torch(x: "torch.Tensor") -> "torch.Tensor":
    """splitmix64 finalizer, int64 mirror of _mix64_np."""
    x = x ^ _lshr(x, 30)
    x = x * _as_i64(_MIX_A)
    x = x ^ _lshr(x, 27)
    x = x * _as_i64(_MIX_B)
    x = x ^ _lshr(x, 31)
    return x


def _pack_bits_gpu(sig: "torch.Tensor", k: int) -> "torch.Tensor":
    """(B, k) bucket minima -> (B, k/8) uint8, little-endian, matching np.packbits."""
    bits = (sig & 1).to(torch.int32).view(sig.shape[0], k // 8, 8)
    weights = (1 << torch.arange(8, device=sig.device, dtype=torch.int32))
    return (bits * weights).sum(-1).to(torch.uint8)


def _sketch_batch_gpu(tok_list: Sequence[np.ndarray], n: int, k: int, dev,
                      exact_cardinality: bool):
    """Sketch one batch of documents. Every document must have len >= n."""
    shift = int(k).bit_length() - 1
    sent = 1 << (64 - shift)          # strictly greater than any post-shift value

    # Host-side prep dominated an earlier version of this kernel: profiling showed 78 ms
    # of GPU math behind 560 ms of wall clock. Two rules keep it off the critical path.
    #   (1) Cross PCIe in the NARROWEST dtype. Token ids fit in int32, so widening to
    #       int64 happens on the device -- halving transfer -- not on the host.
    #   (2) Never ship a per-token array that the device can synthesize. doc_id is
    #       repeat_interleave of the lengths, so only the lengths are transferred.
    lens = np.fromiter((len(t) for t in tok_list), dtype=np.int64, count=len(tok_list))
    flat = np.concatenate(tok_list, dtype=np.int32)

    t = torch.from_numpy(flat).to(dev, non_blocking=True).to(torch.int64)
    d = torch.repeat_interleave(
        torch.arange(len(tok_list), device=dev, dtype=torch.int64),
        torch.from_numpy(lens).to(dev, non_blocking=True))
    m = t.numel() - n + 1

    h = torch.zeros(m, dtype=torch.int64, device=dev)
    P = _as_i64(_POLY_P)
    for j in range(n):
        h = h * P + t[j:j + m]

    # An n-gram is real only if its first and last token belong to the same document;
    # windows that straddle a document boundary are artifacts of the concatenation.
    keep = d[:m] == d[n - 1:n - 1 + m]
    h = _mix64_torch(h[keep])
    dv = d[:m][keep]

    # Cardinality deliberately avoids torch.bincount. Profiling put bincount at 165 ms of
    # a 310 ms kernel -- 53% -- because binning 20M values into ~200 bins is pure atomic
    # contention on a tiny histogram. Both paths below replace it with an O(B) computation.
    if exact_cardinality:
        # Distinct shingles per document. Duplicates never change a bucket MINIMUM, so
        # this is needed only for containment's |A|, not for the signature itself --
        # which is why it is opt-in: the sort is the most expensive step in the kernel.
        pairs = torch.unique(torch.stack([dv, h], dim=1), dim=0)
        # .contiguous(): these are column slices of `pairs`, and searchsorted silently
        # copies a non-contiguous boundary tensor on every call.
        dv_u, h_u = pairs[:, 0].contiguous(), pairs[:, 1].contiguous()
        # unique(dim=0) returns lexicographically sorted rows, so dv_u is non-decreasing
        # and run boundaries fall out of a binary search.
        edges = torch.searchsorted(
            dv_u, torch.arange(len(tok_list) + 1, device=dev, dtype=dv_u.dtype))
        cards = edges[1:] - edges[:-1]
    else:
        dv_u, h_u = dv, h
        # Documents are contiguous and every window of a doc with len >= n is valid, so
        # the raw shingle count is exactly len - n + 1. No device work required.
        cards = torch.from_numpy(lens - n + 1).to(dev, non_blocking=True)

    idx = dv_u * k + (h_u & (k - 1))
    sig = torch.full((len(tok_list) * k,), sent, dtype=torch.int64, device=dev)
    sig.scatter_reduce_(0, idx, _lshr(h_u, shift), reduce='amin', include_self=True)
    sig = sig.view(len(tok_list), k)

    # Densification is a CPU fallback: at book length k*exp(-m/k) is ~0 empty buckets,
    # so this branch effectively never fires and is not worth a GPU kernel.
    empty_per_doc = (sig == sent).sum(dim=1)
    needs_fix = torch.nonzero(empty_per_doc > 0, as_tuple=True)[0]
    if needs_fix.numel() == 0:
        return _pack_bits_gpu(sig, k).cpu().numpy(), cards.cpu().numpy()

    packed = _pack_bits_gpu(sig, k).cpu().numpy()
    fix_idx = needs_fix.cpu().numpy()
    host = sig[needs_fix].cpu().numpy().astype(_U64)
    host[host == sent] = SENTINEL
    for row, doc in enumerate(fix_idx):
        packed[doc] = sketch_to_bits(densify(host[row], k))
    return packed, cards.cpu().numpy()


def _auto_batch_tokens(dev, target_frac: float = 0.25) -> int:
    """Tokens per batch, sized from free VRAM (~64 B/token of working set)."""
    if not TORCH_AVAILABLE or dev.type != 'cuda':
        return 4_000_000
    torch.cuda.empty_cache()   # see _auto_tile: cached blocks masquerade as used VRAM
    free, _total = torch.cuda.mem_get_info(dev)
    return max(1_000_000, min(64_000_000, int(free * target_frac / 64)))


def sketch_corpus_gpu(docs: Iterable[np.ndarray], n: int = 13, k: int = 1024,
                      device: str = 'cuda', batch_tokens: Optional[int] = None,
                      exact_cardinality: bool = False, progress=None,
                      out: Optional[np.ndarray] = None,
                      out_cards: Optional[np.ndarray] = None,
                      start: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """GPU equivalent of sketch_corpus. Produces bit-identical signatures.

    Documents shorter than n are routed to the CPU path, which shrinks n to the document
    length so a short document still yields one whole-document shingle instead of none.
    Reproducing that per-document adjustment on GPU would need a ragged kernel for a case
    that is both rare and trivially cheap.

    `out` / `out_cards` behave as in sketch_corpus and are strongly preferred at scale --
    see that docstring. Without them results are held in a dict keyed by document index,
    which is fine for small corpora but is the wrong shape for millions of documents.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("sketch_corpus_gpu requires torch")
    dev = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    if batch_tokens is None:
        batch_tokens = _auto_batch_tokens(dev)

    streaming = out is not None
    if start and not streaming:
        raise ValueError("start (sketch resume) requires streaming output arrays")
    sig_map: Dict[int, np.ndarray] = {}
    card_map: Dict[int, int] = {}
    batch: List[np.ndarray] = []
    batch_ids: List[int] = []
    pending = 0
    count = start          # resume: docs iterator starts at this corpus position

    def emit(idx: int, packed_row: np.ndarray, card: int):
        if streaming:
            out[idx] = packed_row
            out_cards[idx] = card
        else:
            sig_map[idx] = packed_row
            card_map[idx] = card

    # Live wall-time accounting, readable by the caller between progress callbacks:
    # time inside flush() is batch work (GPU kernel + transfers + result writes);
    # everything else is the caller's document feed. Distinguishing the two is the
    # first question whenever a sketch is slower than expected.
    stats = {'flush_s': 0.0}
    sketch_corpus_gpu.live_stats = stats

    def flush():
        nonlocal batch, batch_ids, pending
        if not batch:
            return
        t_flush = time.time()
        packed, cards = _sketch_batch_gpu(batch, n, k, dev, exact_cardinality)
        for row, doc_id in enumerate(batch_ids):
            emit(doc_id, packed[row], int(cards[row]))
        batch, batch_ids, pending = [], [], 0
        stats['flush_s'] += time.time() - t_flush

    # Short documents bypass the GPU batch and sketch individually; identical stubs
    # (empty/deleted records) are extremely common in scraped corpora, so their
    # signatures are memoized by content.
    short_cache: Dict[bytes, Tuple[np.ndarray, int]] = {}

    for toks in docs:
        toks = np.asarray(toks)
        idx = count
        count += 1
        if toks.size < n:
            key = toks.tobytes()
            hit = short_cache.get(key)
            if hit is None:
                hit = sketch_document(toks, n, k)
                if len(short_cache) < 1_000_000:
                    short_cache[key] = hit
            emit(idx, hit[0], hit[1])
        else:
            batch.append(toks)
            batch_ids.append(idx)
            pending += toks.size
            if pending >= batch_tokens:
                flush()
                if progress is not None:
                    progress(count)
    flush()
    if progress is not None:
        progress(count)

    if streaming:
        return out[:count], out_cards[:count]
    if not sig_map:  # noqa: E501 (non-streaming path; start is always 0 here)
        return np.empty((0, k // 8), dtype=np.uint8), np.empty(0, dtype=np.int64)
    order = sorted(sig_map)
    return (np.vstack([sig_map[i] for i in order]),
            np.asarray([card_map[i] for i in order], dtype=np.int64))


# =============================================================================
# Similarity search
# =============================================================================

def unpack_pm1(packed: np.ndarray, k: int) -> np.ndarray:
    """Packed bits -> (N, k) float16 in {-1, +1}.

    With this encoding the dot product of two signatures is
        dot = (#agreeing bits) - (#disagreeing bits) = 2*agree - k,
    and since P(agree) = (1 + J)/2 for 1-bit signatures, E[dot]/k = J exactly.
    So Jaccard is recovered by a matmul followed by a scale -- no popcount needed,
    and the matmul lands on tensor cores. float16 represents integers up to 2048
    exactly, so k <= 2048 accumulates without rounding error.
    """
    bits = np.unpackbits(packed, axis=-1, count=k, bitorder='little')
    return (bits.astype(np.float16) * np.float16(2.0)) - np.float16(1.0)


def _pairs_from_tile(sim: "torch.Tensor", row_off: int, col_off: int,
                     threshold: float, same_block: bool):
    """Extract (i, j, similarity) above threshold from one similarity tile.

    On the diagonal tile each pair would otherwise appear twice (plus self-matches on
    the diagonal), so the lower triangle is suppressed IN PLACE with triu_. The obvious
    alternative, torch.triu_indices, allocates two int64 tensors of tile^2/2 elements --
    ~2 GB at tile=16384 -- which costs an order of magnitude more than the matmul it
    is filtering. Requires threshold > 0, since triu_ zeroes rather than masks.
    """
    if same_block:
        sim.triu_(diagonal=1)
    rows, cols = torch.nonzero(sim >= threshold, as_tuple=True)
    return rows + row_off, cols + col_off, sim[rows, cols]


class MatchCheckpoint:
    """Periodic checkpoint for the MATCHING phase, so a killed process resumes
    instead of restarting hours of tile work.

    Saves the full cluster state (per-doc stats + union-find parent) plus the
    set of COMPLETED row-block offsets, atomically (.part + rename). State is
    snapshotted only at row-block boundaries, so a resumed run re-processes at
    most the block that was in flight -- never double-counts a finished one.
    Parameters (including the tile size, which otherwise varies with free VRAM)
    are pinned in the file; a mismatch ignores the checkpoint rather than
    corrupting a differently-shaped run.
    """

    def __init__(self, path, params: Dict[str, Any], n_docs: int,
                 interval_s: float = 600.0):
        self.path = Path(path)
        self.params = dict(params, n_docs=int(n_docs))
        self.interval_s = interval_s
        self.done: set = set()
        self.baseline: Optional[Dict[str, Any]] = None
        self.tile: Optional[int] = None
        self._last_save = time.time()

    def _params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True, default=str)

    def try_load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with np.load(self.path, allow_pickle=False) as z:
                if str(z['params_json']) != self._params_json():
                    print("  match checkpoint exists but parameters differ; "
                          "starting matching fresh")
                    return False
                self.done = {int(x) for x in z['done_starts']}
                self.tile = int(z['tile'])
                self.baseline = {
                    'deg': z['deg'], 'ssum': z['ssum'], 'smax': z['smax'],
                    'cmax': z['cmax'] if 'cmax' in z.files else None,
                    'parent': z['parent'], 'total_pairs': int(z['total_pairs']),
                }
            return True
        except Exception as e:
            print(f"  could not read match checkpoint ({e}); starting fresh")
            return False

    def save(self, sink: 'StreamingClusterer'):
        state = sink.snapshot()
        arrays = {
            'params_json': np.asarray(self._params_json()),
            'done_starts': np.asarray(sorted(self.done), dtype=np.int64),
            'tile': np.int64(self.tile or 0),
            'total_pairs': np.int64(state['total_pairs']),
            'deg': state['deg'], 'ssum': state['ssum'], 'smax': state['smax'],
            'parent': state['parent'],
        }
        if state.get('cmax') is not None:
            arrays['cmax'] = state['cmax']
        tmp = self.path.with_name(self.path.name + '.part')
        with open(tmp, 'wb') as f:
            np.savez(f, **arrays)
        tmp.replace(self.path)
        self._last_save = time.time()

    def due(self) -> bool:
        return time.time() - self._last_save >= self.interval_s

    def delete(self):
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass


def parse_devices(device: str) -> List[str]:
    """Expand a device spec: 'cpu', 'cuda', 'cuda:1', 'cuda:all', 'cuda:0,cuda:1'."""
    if device == 'cpu' or not TORCH_AVAILABLE or not torch.cuda.is_available():
        return ['cpu'] if device == 'cpu' else [device]
    if device == 'cuda:all':
        return [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cuda']
    if ',' in device:
        return [d.strip() for d in device.split(',') if d.strip()]
    return [device]


def all_pairs_bruteforce(packed: np.ndarray, k: int, threshold: float = 0.8,
                         device: str = 'cuda', tile: Optional[int] = None,
                         progress=None, cards: Optional[np.ndarray] = None,
                         band_slack: float = 0.9, pair_sink=None,
                         min_card: int = 0, _shard: Tuple[int, int] = (0, 1),
                         checkpoint: Optional[MatchCheckpoint] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact all-pairs search over 1-bit signatures. Returns (i, j, jaccard) with i < j.

    Complete above `threshold`: unlike LSH there is no probabilistic recall loss and
    nothing to tune.

    `cards` enables LENGTH BANDING -- an exact pruning that skips block pairs whose
    shingle counts are too far apart to possibly reach `threshold`. Strongly recommended:
    it costs one sort and removes most of the triangle on any corpus with varied document
    lengths.

    `pair_sink`, if given, is called as sink(i, j, sim) with host numpy arrays per tile and
    the function returns empty arrays. Use it with StreamingClusterer so memory stays
    O(documents) instead of O(pairs) -- required on corpora where duplication is heavy.

    `min_card` excludes documents with fewer than that many shingles from matching -- they
    are below the estimator's reliable range and are the usual cause of a runaway pair
    count. Requires `cards`.

    `progress`, if given, is called as progress(tiles_done, tiles_total, pairs_found).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("all_pairs_bruteforce requires torch")
    if threshold <= 0:
        raise ValueError("threshold must be > 0 (the diagonal tile is masked with triu_)")
    n_docs = packed.shape[0]
    if n_docs < 2:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0, dtype=np.float32)

    devs = parse_devices(device)
    if len(devs) > 1:
        # Multi-GPU needs a mergeable sink (per-device cluster state is folded
        # together at the end); without one, silently use the first device.
        if pair_sink is not None and hasattr(pair_sink, 'merge_from'):
            return _all_pairs_multi(packed, k, threshold, devs, tile, progress,
                                    cards, band_slack, pair_sink, min_card)
        devs = devs[:1]
    device = devs[0]

    dev = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    dtype = _matmul_dtype(dev)

    # Signatures stay PACKED on the device (128 B/doc) and each tile is expanded to +/-1
    # only when it is about to be multiplied. Unpacking the whole corpus up front costs
    # 16x more -- 1 bit becomes one fp16 -- which is 20 GB at 10M docs and OOMs any card
    # we have. Packed, 10M docs is 1.28 GB and the tiles are tens of MB.
    # A memory-mapped signature file opens read-only, and torch.from_numpy on a
    # non-writable array is undefined behavior (it warns, then shares the buffer). Copy
    # only in that case -- an in-RAM array is handed over without duplication.
    # LENGTH BANDING (exact, no recall loss). Jaccard is bounded by the size ratio:
    #     J = |A^B|/|AuB| <= min(|A|,|B|) / max(|A|,|B|)
    # so at threshold t two documents cannot match unless their shingle counts are within
    # a factor of t. Sorting by cardinality turns the full N^2/2 triangle into a narrow
    # band around the diagonal, and because the sort is monotonic the inner loop can BREAK
    # the moment a block is out of range. On a corpus with a wide length spread this is
    # the difference between hours and minutes, and it discards nothing.
    order = None
    cards_sorted = None
    if cards is not None:
        cards = np.asarray(cards)
        order = np.argsort(cards, kind='stable')
        packed = np.asarray(packed)[order]
        cards_sorted = cards[order].astype(np.float64)

    host = np.ascontiguousarray(packed)
    if not host.flags.writeable:
        host = host.copy()
    packed_t = torch.from_numpy(host).to(dev)
    del host
    shifts = torch.arange(8, device=dev, dtype=torch.uint8)

    # Device-resident sink: keep the banding order on the GPU so the
    # sorted -> original id remap happens where the pairs are born.
    gpu_sink = (pair_sink is not None
                and getattr(pair_sink, '_dev', None) is not None
                and dev.type == 'cuda')
    order_t = (torch.from_numpy(order).to(dev)
               if (gpu_sink and order is not None) else None)

    # Tile size is chosen AFTER the signatures are resident, so free-VRAM reflects what is
    # actually left. Sizing first would over-commit on a large corpus: 50M docs is 6.4 GB
    # of packed signatures that _auto_tile would otherwise still count as available.
    # A resumed checkpoint PINS the tile: block boundaries derive from it, so a
    # different auto-sized tile would misalign every completed-block offset.
    if checkpoint is not None and checkpoint.tile:
        tile = checkpoint.tile
    if tile is None:
        tile = _auto_tile(dev, k)
    if checkpoint is not None:
        checkpoint.tile = tile

    def tile_pm1(lo: int, hi: int) -> "torch.Tensor":
        chunk = packed_t[lo:hi]
        bits = (chunk.unsqueeze(-1) >> shifts) & 1        # little-endian, matches packbits
        return bits.reshape(chunk.shape[0], k).to(dtype).mul_(2).sub_(1)

    scale = 1.0 / float(k)

    # MINIMUM LENGTH. Documents below min_card are excluded from matching entirely.
    # Because banding has already sorted by cardinality they form a contiguous prefix, so
    # this costs one searchsorted and removes them from every tile at once. They are worth
    # removing on two counts: the 1-bit estimator is unreliable below MIN_SHINGLES_FOR_1BIT
    # (densification can invent the similarity), and short documents match each other
    # readily -- so a large short tail dominates the pair count and can stall the run.
    lo_doc = 0
    if cards_sorted is not None and min_card > 1:
        lo_doc = int(np.searchsorted(cards_sorted, min_card, side='left'))
    excluded = lo_doc

    # Row blocks may be SHARDED across workers (multi-GPU); columns never are --
    # each row block still scans its full band of column blocks.
    col_starts = list(range(lo_doc, n_docs, tile))
    starts = col_starts[_shard[0]::_shard[1]]

    def band_limit(a: int) -> int:
        """Last column block that can still contain a match for row block `a`."""
        if cards_sorted is None:
            return n_docs
        # Best possible ratio between block A and block B (a <= b, sorted ascending) is
        # largest-in-A / smallest-in-B. band_slack absorbs the gap between the raw shingle
        # counts we store by default and true DISTINCT counts, which are what the bound
        # actually needs; with --nd-exact-card the bound is tight and slack can be 1.0.
        max_a = cards_sorted[min(a + tile, n_docs) - 1]
        cutoff = max_a / (threshold * band_slack)
        return int(np.searchsorted(cards_sorted, cutoff, side='right'))

    tiles_by_start = {a: len([b for b in col_starts if a <= b < band_limit(a)])
                      for a in starts}
    total_tiles = sum(tiles_by_start.values())
    all_starts = list(range(0, n_docs, tile))
    full_tiles = len(all_starts) * (len(all_starts) + 1) // 2
    _ = (full_tiles, excluded)   # reported by the caller via last_run_stats

    out_i: List[np.ndarray] = []
    out_j: List[np.ndarray] = []
    out_s: List[np.ndarray] = []
    tiles_done = 0
    n_pairs = 0
    ckpt_done = checkpoint.done if checkpoint is not None else ()
    if ckpt_done:
        tiles_done = sum(tiles_by_start.get(a, 0) for a in ckpt_done)
        n_pairs = int(getattr(pair_sink, 'total_pairs', 0))
        print(f"  resuming matching: {tiles_done:,}/{total_tiles:,} tiles already "
              f"done ({tiles_done / max(total_tiles, 1) * 100:.1f}%), "
              f"{n_pairs:,} pairs carried over")
    if progress is not None:
        progress(tiles_done, max(total_tiles, 1), n_pairs)

    for a in starts:
        if a in ckpt_done:
            continue
        limit_b = band_limit(a)
        if limit_b <= a:
            continue
        A = tile_pm1(a, a + tile)
        for b in col_starts:
            if b < a or b >= limit_b:
                continue
            B = A if b == a else tile_pm1(b, b + tile)
            sim = (A @ B.T).float() * scale
            rows, cols, vals = _pairs_from_tile(sim, a, b, threshold, same_block=(a == b))
            # The similarity tile is ~1 GB at tile=16384 and is DONE once the
            # pairs are extracted. Freeing it before clustering matters more
            # than it looks: a dense tile's pair tensors are GB-scale, and on
            # an 8 GB card holding sim through the sink call pushed peak VRAM
            # over the edge -- the allocator then thrashes free/malloc cycles
            # and a 1-second tile takes 35.
            del sim
            if rows.numel():
                if gpu_sink:
                    # Pairs never touch the host: remap through the banding
                    # order on-device and fold straight into the clusterer's
                    # GPU accumulators, in bounded CHUNKS so the remap/stat
                    # intermediates stay ~100 MB and come from the allocator
                    # cache instead of fresh cudaMallocs.
                    nn = rows.numel()
                    n_pairs += int(nn)
                    CHUNK = 16_000_000
                    for lo in range(0, nn, CHUNK):
                        r_c = rows[lo:lo + CHUNK]
                        c_c = cols[lo:lo + CHUNK]
                        v_c = vals[lo:lo + CHUNK]
                        if order_t is not None:
                            oi, oj = order_t[r_c], order_t[c_c]
                            di = torch.minimum(oi, oj)
                            dj = torch.maximum(oi, oj)
                        else:
                            di, dj = r_c, c_c
                        pair_sink.consume_device(di, dj, v_c)
                        del di, dj
                else:
                    # Move results to HOST immediately. Holding them on device for
                    # the whole run is what starves the allocator on a small card:
                    # once accumulated pairs crowd VRAM, every ~1 GB tile buffer
                    # has to be freed and re-requested, and throughput collapses.
                    pi = rows.cpu().numpy().astype(np.int64)
                    pj = cols.cpu().numpy().astype(np.int64)
                    ps = vals.cpu().numpy().astype(np.float32)
                    if order is not None:            # map sorted -> original indices
                        oi, oj = order[pi], order[pj]
                        pi, pj = np.minimum(oi, oj), np.maximum(oi, oj)
                    n_pairs += int(pi.size)
                    if pair_sink is not None:
                        pair_sink(pi, pj, ps)
                    else:
                        out_i.append(pi); out_j.append(pj); out_s.append(ps)
            del rows, cols, vals
            if b != a:
                del B
            tiles_done += 1
            if progress is not None:
                progress(tiles_done, total_tiles, n_pairs)
        del A
        if checkpoint is not None:
            # Row block complete: everything the sink holds is consistent with
            # `done` -- the only safe moment to snapshot.
            checkpoint.done.add(a)
            if checkpoint.due():
                checkpoint.save(pair_sink)

    all_pairs_bruteforce.last_run_stats = {
        'tiles': total_tiles, 'tiles_full_triangle': full_tiles,
        'tiles_skipped_pct': (1 - total_tiles / full_tiles) * 100 if full_tiles else 0.0,
        'excluded_short': excluded, 'pairs': n_pairs, 'tile_size': tile,
    }
    if pair_sink is not None or not out_i:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0, dtype=np.float32)
    return np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_s)


def _all_pairs_multi(packed, k, threshold, devs, tile, progress, cards,
                     band_slack, pair_sink, min_card):
    """Shard row-blocks round-robin across GPUs; merge per-GPU cluster state.

    Round-robin (not contiguous split) because the banding sort front-loads the
    dense short-document blocks -- a contiguous split would give GPU 0 all the
    expensive tiles. Workers only update shared counters; progress is reported
    from the CALLING thread, which owns the host application's job/log context
    (thread-locals in the embedder don't survive into worker threads).
    """
    import threading

    n_docs = packed.shape[0]
    n = len(devs)
    done = [0] * n
    totals = [0] * n
    pairs = [0] * n
    sinks: List[Optional[StreamingClusterer]] = [None] * n
    errs: List[Optional[BaseException]] = [None] * n

    def run(idx: int, dv: str):
        try:
            sink = StreamingClusterer(n_docs, cards=cards, device=dv)
            sinks[idx] = sink

            def prog(d, t, p):
                done[idx], totals[idx], pairs[idx] = d, t, p

            all_pairs_bruteforce(packed, k, threshold, device=dv, tile=tile,
                                 progress=prog, cards=cards,
                                 band_slack=band_slack, pair_sink=sink,
                                 min_card=min_card, _shard=(idx, n))
        except BaseException as e:      # surfaced in the caller after join
            errs[idx] = e

    threads = [threading.Thread(target=run, args=(ix, dv), daemon=True)
               for ix, dv in enumerate(devs)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(0.5)
        if progress is not None:
            progress(sum(done), max(sum(totals), 1), sum(pairs))
    for t in threads:
        t.join()
    for e in errs:
        if e is not None:
            raise e

    per_dev_stats = getattr(all_pairs_bruteforce, 'last_run_stats', {})
    for s in sinks:
        if s is not None:
            pair_sink.merge_from(s)
    if progress is not None:
        progress(sum(done), max(sum(totals), 1), sum(pairs))
    all_pairs_bruteforce.last_run_stats = {
        'tiles': sum(totals), 'tiles_full_triangle': sum(totals),
        'tiles_skipped_pct': 0.0,
        'excluded_short': per_dev_stats.get('excluded_short', 0),
        'pairs': pair_sink.total_pairs,
        'tile_size': per_dev_stats.get('tile_size', 0),
        'devices': list(devs),
    }
    return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32))


def _matmul_dtype(dev) -> "torch.dtype":
    """float16 only where tensor cores exist (sm_70+).

    Measured on a GTX 1070 (sm_61): fp32 2.67 TMAC/s vs fp16 2.42 TMAC/s -- Pascal runs
    fp16 arithmetic at 1/64 rate, so half precision is a pessimization there. Every
    Ampere/Ada card (3060, 3090, 4080) wants fp16. k <= 2048 accumulates exactly in
    either, since both represent the integer range involved without rounding.
    """
    if dev.type != 'cuda':
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(dev)
    return torch.float16 if major >= 7 else torch.float32


def _auto_tile(dev, k: int, target_frac: float = 0.25) -> int:
    """Pick a tile size from the device's actual free VRAM.

    The similarity tile (tile^2 floats) dominates, so this is what lets the same code
    run on an 8 GB 3060 and a 24 GB 3090 without per-machine tuning.
    """
    if dev.type != 'cuda':
        return 2048
    # Release the caching allocator's unused blocks first. After a sketch phase in the
    # SAME process, torch still holds those gigabytes as cached-but-free, and
    # mem_get_info counts them as used -- which once shrank the tile here from ~12k to
    # ~1.5k and inflated a 1-hour match into a 70-DAY one (tile count grows with the
    # square of the shrink, and small tiles are all fixed overhead).
    with torch.cuda.device(dev):        # empty_cache acts on the CURRENT device
        torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(dev)
    budget = free * target_frac
    # Per tile element: 4 B (fp32 similarity) + 1 B (the >= threshold bool mask) plus
    # slack for nonzero()'s output. Budget ~6 B/element rather than 4.
    t = int(((budget / 6.0) ** 0.5))
    return max(512, min(16384, (t // 512) * 512))


# =============================================================================
# Clustering
# =============================================================================

class UnionFind:
    """Union-find over document indices, used to turn verified similar PAIRS into
    clusters without ever materializing the full pair graph.

    numpy-backed so find_many can resolve MILLIONS of queries per call: a dense
    similarity tile yields tens of millions of pairs, and any per-pair Python
    step at that scale turns a one-hour match into a multi-day one.
    """

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = int(p[x])
        return int(x)

    def find_many(self, x: np.ndarray) -> np.ndarray:
        """Vectorized root lookup with path halving (one numpy pass per tree hop)."""
        p = self.parent
        x = np.asarray(x, dtype=np.int64)
        cur = p[x]
        while True:
            nxt = p[cur]
            if np.array_equal(nxt, cur):        # all roots reached
                return cur
            p[cur] = p[nxt]                     # halve paths as we go
            cur = nxt

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


class StreamingClusterer:
    """Consumes duplicate pairs tile-by-tile and keeps only O(documents) of state.

    The accumulate-everything approach holds every pair until the end, which on a corpus
    with heavy duplication is unbounded -- one cluster of N identical documents alone
    contributes N(N-1)/2 pairs. Here pairs are folded into a union-find immediately and
    then discarded; per-cluster statistics are reconstructed at the end from per-DOCUMENT
    accumulators, which are exact:

        n_pairs  = sum(degree over members) / 2      each pair counted at both endpoints
        mean_sim = sum(sim over members) / 2 / n_pairs
        max_sim  = max(per-member max)

    Memory is 16 bytes per document regardless of how many pairs are found.
    """

    def __init__(self, n_docs: int, cards: Optional[np.ndarray] = None,
                 device: Optional[str] = None):
        self.n_docs = n_docs
        self.uf = UnionFind(n_docs)
        self.deg = np.zeros(n_docs, dtype=np.int64)
        self.ssum = np.zeros(n_docs, dtype=np.float64)
        self.smax = np.zeros(n_docs, dtype=np.float32)
        # Containment is computed per PAIR as it streams by -- exact, and impossible to
        # reconstruct afterwards once the pairs are discarded.
        self.cards = np.asarray(cards) if cards is not None else None
        self.cmax = np.zeros(n_docs, dtype=np.float32) if cards is not None else None
        self.total_pairs = 0

        # Optional GPU residency. Pairs are BORN on the device in all_pairs'
        # nonzero(); a boilerplate-dense tile yields ~10^8 of them, and any
        # per-pair host cost -- even vectorized numpy at ~100 ns -- turns into
        # minutes per tile with the GPU idle. With device set, per-document
        # stats accumulate via scatter ops and union-find runs its Boruvka
        # rounds on-device; the host sees only the final O(n_docs) arrays.
        self._dev = None
        if device is not None and TORCH_AVAILABLE:
            dev = torch.device(
                device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
            if dev.type == 'cuda':
                self._dev = dev
                self._deg_t = torch.zeros(n_docs, dtype=torch.int64, device=dev)
                self._ssum_t = torch.zeros(n_docs, dtype=torch.float64, device=dev)
                self._smax_t = torch.zeros(n_docs, dtype=torch.float32, device=dev)
                self._parent_t = torch.arange(n_docs, dtype=torch.int64, device=dev)
                if self.cards is not None:
                    # cards usually arrive as a read-only .npy memmap, and
                    # torch.from_numpy on a non-writable array is UB (it also
                    # warns). Copy only in that case, same as the signature
                    # block in all_pairs_bruteforce.
                    cards_host = np.ascontiguousarray(self.cards)
                    if not cards_host.flags.writeable:
                        cards_host = cards_host.copy()
                    self._cards_t = torch.from_numpy(cards_host).to(dev, torch.float64)
                else:
                    self._cards_t = None
                self._cmax_t = (torch.zeros(n_docs, dtype=torch.float32, device=dev)
                                if self.cards is not None else None)
                self._finalized = False

    def _find_many_t(self, x: "torch.Tensor") -> "torch.Tensor":
        """find_many on the device parent, with path halving."""
        p = self._parent_t
        cur = p[x]
        while True:
            nxt = p[cur]
            if torch.equal(nxt, cur):
                return cur
            p[cur] = p[nxt]
            cur = nxt

    def consume_device(self, i: "torch.Tensor", j: "torch.Tensor",
                       s: "torch.Tensor"):
        """Fold one tile's pairs without leaving the GPU. i < j, global doc ids."""
        n = i.numel()
        if n == 0:
            return
        self.total_pairs += int(n)
        ones = torch.ones(n, dtype=torch.int64, device=self._dev)
        self._deg_t.scatter_add_(0, i, ones)
        self._deg_t.scatter_add_(0, j, ones)
        s64 = s.to(torch.float64)
        self._ssum_t.scatter_add_(0, i, s64)
        self._ssum_t.scatter_add_(0, j, s64)
        s32 = s.to(torch.float32)
        self._smax_t.scatter_reduce_(0, i, s32, reduce='amax')
        self._smax_t.scatter_reduce_(0, j, s32, reduce='amax')

        if self._cards_t is not None:
            ca = self._cards_t[i]
            cb = self._cards_t[j]
            inter = s64 * (ca + cb) / (1.0 + s64)
            cont = torch.clamp(
                inter / torch.clamp(torch.minimum(ca, cb), min=1.0), max=1.0
            ).to(torch.float32)
            self._cmax_t.scatter_reduce_(0, i, cont, reduce='amax')
            self._cmax_t.scatter_reduce_(0, j, cont, reduce='amax')

        # Boruvka union rounds, entirely on-device: attach every larger root to
        # its smallest neighboring root (scatter-amin is exactly min-neighbor
        # attach, and larger-to-smaller can't create cycles), re-resolve, repeat.
        p = self._parent_t
        ri = self._find_many_t(i)
        rj = self._find_many_t(j)
        while True:
            m = ri != rj
            if not bool(m.any()):
                break
            ri, rj = ri[m], rj[m]
            a = torch.minimum(ri, rj)
            b = torch.maximum(ri, rj)
            p.scatter_reduce_(0, b, a, reduce='amin', include_self=True)
            ri = self._find_many_t(ri)
            rj = self._find_many_t(rj)

    def snapshot(self) -> Dict[str, Any]:
        """Host copies of the full cluster state WITHOUT finalizing.

        Safe to call between row blocks (never mid-tile: a partially consumed
        tile would double-count when that block re-runs after resume)."""
        if self._dev is not None and not self._finalized:
            return {'deg': self._deg_t.cpu().numpy(),
                    'ssum': self._ssum_t.cpu().numpy(),
                    'smax': self._smax_t.cpu().numpy(),
                    'cmax': (self._cmax_t.cpu().numpy()
                             if self._cmax_t is not None else None),
                    'parent': self._parent_t.cpu().numpy(),
                    'total_pairs': self.total_pairs}
        return {'deg': self.deg.copy(), 'ssum': self.ssum.copy(),
                'smax': self.smax.copy(),
                'cmax': self.cmax.copy() if self.cmax is not None else None,
                'parent': self.uf.parent.copy(), 'total_pairs': self.total_pairs}

    def seed(self, state: Dict[str, Any]):
        """Initialize from a checkpoint snapshot (call before consuming pairs)."""
        self.total_pairs = int(state['total_pairs'])
        if self._dev is not None:
            dev = self._dev
            self._deg_t = torch.from_numpy(state['deg'].copy()).to(dev)
            self._ssum_t = torch.from_numpy(state['ssum'].copy()).to(dev)
            self._smax_t = torch.from_numpy(state['smax'].copy()).to(dev)
            if state.get('cmax') is not None and self._cmax_t is not None:
                self._cmax_t = torch.from_numpy(state['cmax'].copy()).to(dev)
            self._parent_t = torch.from_numpy(state['parent'].copy()).to(dev)
        else:
            self.deg[:] = state['deg']
            self.ssum[:] = state['ssum']
            self.smax[:] = state['smax']
            if state.get('cmax') is not None and self.cmax is not None:
                self.cmax[:] = state['cmax']
            self.uf.parent[:] = state['parent']

    def _union_np(self, i: np.ndarray, j: np.ndarray):
        """Vectorized Boruvka union rounds on the host parent array."""
        ri = self.uf.find_many(i)
        rj = self.uf.find_many(j)
        p = self.uf.parent
        while True:
            m = ri != rj
            if not m.any():
                return
            ri, rj = ri[m], rj[m]
            a = np.minimum(ri, rj)
            b = np.maximum(ri, rj)
            order = np.lexsort((a, b))
            b_s, a_s = b[order], a[order]
            first = np.empty(b_s.size, dtype=bool)
            first[0] = True
            np.not_equal(b_s[1:], b_s[:-1], out=first[1:])
            p[b_s[first]] = a_s[first]
            ri = self.uf.find_many(ri)
            rj = self.uf.find_many(rj)

    def merge_from(self, other: 'StreamingClusterer'):
        """Fold another clusterer's state into this one (multi-GPU merge).

        Stats merge element-wise; connectivity merges by treating every
        non-root parent link in `other` as an edge. Exact: union-find state is
        precisely the set of x -> parent[x] links.
        """
        other._finalize()
        self._finalize()
        self.total_pairs += other.total_pairs
        self.deg += other.deg
        self.ssum += other.ssum
        np.maximum(self.smax, other.smax, out=self.smax)
        if self.cmax is not None and other.cmax is not None:
            np.maximum(self.cmax, other.cmax, out=self.cmax)
        op = other.uf.parent
        linked = np.flatnonzero(op != np.arange(op.size, dtype=op.dtype))
        if linked.size:
            self._union_np(linked, op[linked])

    def _finalize(self):
        """Pull device state into the numpy fields the reporting path reads."""
        if self._dev is None or self._finalized:
            return
        self._finalized = True
        self.deg = self._deg_t.cpu().numpy()
        self.ssum = self._ssum_t.cpu().numpy()
        self.smax = self._smax_t.cpu().numpy()
        if self._cmax_t is not None:
            self.cmax = self._cmax_t.cpu().numpy()
        self.uf.parent = self._parent_t.cpu().numpy()
        self._deg_t = self._ssum_t = self._smax_t = None
        self._cmax_t = self._cards_t = self._parent_t = None

    def _fold(self, idx: np.ndarray, s: np.ndarray, cont: Optional[np.ndarray]):
        """Group-by-document accumulation via one sort + reduceat.

        Replaces np.add.at / np.maximum.at, whose unbuffered per-element path
        (~100ns each, six calls per batch) dominated dense tiles. One argsort
        amortized over count, sum, and max is ~5x cheaper and scales with the
        batch, not with corpus size.
        """
        order = np.argsort(idx, kind='stable')
        idx_s = idx[order]
        s_s = s[order]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(idx_s)) + 1))
        uniq = idx_s[starts]
        self.deg[uniq] += np.diff(np.append(starts, idx_s.size))
        self.ssum[uniq] += np.add.reduceat(s_s.astype(np.float64), starts)
        self.smax[uniq] = np.maximum(self.smax[uniq],
                                     np.maximum.reduceat(s_s, starts))
        if cont is not None:
            self.cmax[uniq] = np.maximum(self.cmax[uniq],
                                         np.maximum.reduceat(cont[order], starts))

    def __call__(self, i: np.ndarray, j: np.ndarray, s: np.ndarray):
        if i.size == 0:
            return
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        s = np.asarray(s, dtype=np.float32)
        self.total_pairs += int(i.size)

        cont32 = None
        if self.cards is not None:
            ca = self.cards[i].astype(np.float64)
            cb = self.cards[j].astype(np.float64)
            sd = s.astype(np.float64)
            # |A^B| = J*(|A|+|B|)/(1+J);  containment = |A^B| / min(|A|,|B|)
            inter = sd * (ca + cb) / (1.0 + sd)
            cont = np.minimum(1.0, inter / np.maximum(np.minimum(ca, cb), 1.0))
            cont32 = cont.astype(np.float32)

        self._fold(i, s, cont32)
        self._fold(j, s, cont32)

        # Union WITHOUT a per-pair Python loop, at any density: vectorized
        # Boruvka rounds (see _union_np). The per-pair Python union this
        # replaces was the reason a dense tile took minutes of host work.
        self._union_np(i, j)

    def clusters(self) -> List[Dict]:
        self._finalize()
        touched = np.flatnonzero(self.deg > 0)
        groups: Dict[int, List[int]] = {}
        if touched.size:
            # Vectorized root resolution; the Python find() loop here was
            # minutes on a corpus with millions of duplicated documents.
            roots = self.uf.find_many(touched)
            for d, r in zip(touched.tolist(), roots.tolist()):
                groups.setdefault(r, []).append(d)

        out = []
        for root, members in groups.items():
            m = np.asarray(members)
            n_pairs = int(self.deg[m].sum() // 2)
            sim_sum = float(self.ssum[m].sum() / 2.0)
            size = len(members)
            possible = size * (size - 1) / 2
            out.append({
                'members': sorted(members),
                'size': size,
                'n_pairs': n_pairs,
                'max_sim': float(self.smax[m].max()),
                'mean_sim': (sim_sum / n_pairs) if n_pairs else 0.0,
                'density': (n_pairs / possible) if possible else 1.0,
                'max_containment': (float(self.cmax[m].max())
                                    if self.cmax is not None else 0.0),
            })
        out.sort(key=lambda c: (c['size'], c['max_sim']), reverse=True)
        return out


def cluster_pairs(n_docs: int, pi: np.ndarray, pj: np.ndarray, ps: np.ndarray
                  ) -> List[Dict]:
    """Group verified pairs into clusters, ranked by match count.

    Returns dicts with 'members', 'size', 'max_sim', 'mean_sim', 'n_pairs', sorted by
    size then max similarity -- "sort them by matches", which is what makes the output
    reviewable rather than just a filter decision.

    NOTE these are connected components of the VERIFIED graph. Components still merge
    transitively (A~B, B~C pulls in A~C), so a large cluster is not a claim that every
    member resembles every other. 'mean_sim' is the signal for that: a big cluster with
    low mean similarity is a transitive chain and should be inspected before cutting.
    """
    uf = UnionFind(n_docs)
    for a, b in zip(pi.tolist(), pj.tolist()):
        uf.union(a, b)

    groups: Dict[int, List[int]] = {}
    for a, b in zip(pi.tolist(), pj.tolist()):
        root = uf.find(a)
        g = groups.setdefault(root, [])
        g.append(a)
        g.append(b)

    stats: Dict[int, List[float]] = {}
    for a, s in zip(pi.tolist(), ps.tolist()):
        stats.setdefault(uf.find(a), []).append(float(s))

    clusters = []
    for root, members in groups.items():
        uniq = sorted(set(members))
        sims = stats.get(root, [])
        possible = len(uniq) * (len(uniq) - 1) / 2
        clusters.append({
            'members': uniq,
            'size': len(uniq),
            'n_pairs': len(sims),
            'max_sim': max(sims) if sims else 0.0,
            'mean_sim': float(np.mean(sims)) if sims else 0.0,
            # 1.0 = every pair is similar to every other; ~2/n = a chain
            'density': (len(sims) / possible) if possible else 1.0,
        })
    clusters.sort(key=lambda c: (c['size'], c['max_sim']), reverse=True)
    return clusters


# =============================================================================
# Prune selection
# =============================================================================

def select_kill_set(clusters: Sequence[Dict], cards: Optional[np.ndarray] = None,
                    protected: frozenset = frozenset(), skip_chains: bool = True,
                    skip_short: bool = True, keep: str = 'longest'
                    ) -> Tuple[set, List[Dict], Dict[str, int]]:
    """Decide which documents to remove. Returns (kill_ids, decisions, skip_counts).

    `decisions` is one record per removal -- the audit trail that makes a prune
    reviewable and, because the source is never modified, reversible.

    PROTECTED documents are never removed, and their protection overrides the keep rule.
    The caller uses this for validation shards: when a cluster spans train and val, every
    train copy is cut and the val copy survives. That is eval decontamination, and it is
    the more valuable direction -- a train document duplicating a val document inflates
    your eval scores, whereas the reverse merely wastes a little training compute.

    Two classes are skipped by default because collapsing them destroys real content:
      - CHAINS (low edge density): members are not mutually duplicate, so keeping one and
        cutting the rest deletes documents that were never duplicates of the survivor.
      - SHORT-DOC clusters: the 1-bit estimator is unreliable below MIN_SHINGLES_FOR_1BIT
        and densification can invent the similarity that put them in a cluster at all.
    """
    kill: set = set()
    decisions: List[Dict] = []
    skipped = {'chain': 0, 'short': 0, 'all_protected': 0}

    for rank, c in enumerate(clusters):
        members = list(c['members'])
        if skip_chains and c['size'] > 2 and c.get('density', 1.0) < CHAIN_DENSITY:
            skipped['chain'] += 1
            continue
        if skip_short and c.get('n_short', 0):
            skipped['short'] += 1
            continue

        prot = [m for m in members if m in protected]
        if prot and len(prot) == len(members):
            skipped['all_protected'] += 1
            continue

        if prot:
            # Cross-split cluster: every protected member survives, everything else goes.
            survivors, doomed = prot, [m for m in members if m not in protected]
            reason = 'contamination'
        else:
            if keep == 'first':
                winner = members[0]
            elif keep == 'longest' and cards is not None:
                winner = max(members, key=lambda m: int(cards[m]))
            else:
                winner = members[0]
            survivors, doomed = [winner], [m for m in members if m != winner]
            reason = 'duplicate'

        for d in doomed:
            kill.add(d)
            decisions.append({
                'doc': d, 'cluster_rank': rank + 1, 'reason': reason,
                'kept': survivors[0], 'cluster_size': c['size'],
                'max_sim': round(float(c['max_sim']), 4),
                'density': round(float(c.get('density', 1.0)), 3),
            })
    return kill, decisions, skipped


# =============================================================================
# Strategy selection
# =============================================================================

# Where exact all-pairs stops being comfortable. This is ADVISORY ONLY -- nothing refuses
# to run above it. Raised from 2M after a measured 500k run on an RTX 4080 matched in
# seconds rather than the minutes extrapolated from a GTX 1070: brute force stays viable
# far longer than the original estimate, and it is strictly more accurate than LSH
# (complete recall, no S-curve, no banding parameters).
BRUTEFORCE_ADVISORY = 10_000_000


def recommend_strategy(n_docs: int, k: int = 1024,
                       crossover: int = BRUTEFORCE_ADVISORY) -> Dict:
    """Describe the cost of an exact all-pairs match. ADVISORY -- imposes no limit.

    Memory is NOT the binding constraint: signatures stay packed at 128 bytes/doc, so 10M
    docs is 1.28 GB on the device and tiles are expanded one at a time. What grows is
    time, quadratically, so the estimate below is what to plan around.
    """
    macs = (n_docs * (n_docs - 1) / 2.0) * k
    # ~25 TMAC/s is an Ampere/Ada-class card on this shape (measured 4080 territory).
    est_s = macs / 25e12
    return {
        'strategy': 'bruteforce' if n_docs <= crossover else 'lsh',
        'n_docs': n_docs,
        'pair_macs': macs,
        'est_seconds': est_s,
        'signature_mb': n_docs * (k / 8) / (1024 ** 2),
        'reason': (f"{n_docs:,} docs: exact all-pairs is ~{macs/1e12:.1f} TMAC "
                   f"(~{est_s/60:.1f} min on one modern GPU), complete recall, "
                   f"nothing to tune"
                   if n_docs <= crossover else
                   f"{n_docs:,} docs: exact all-pairs is ~{macs/1e12:.1f} TMAC "
                   f"(~{est_s/3600:.1f} h on one modern GPU). Still exact and still "
                   f"runs; consider --sample, more GPUs, or LSH candidates if that is "
                   f"too slow"),
    }
