# Dataset Explorer Web — Session Handoff

*Written 2026-08-07 for a fresh session to resume from. Author: Code.*

This is the running state of the **Dataset Explorer web app** (`tools/explorer_web/`),
a FastAPI + vanilla-JS wrapper around `dataset_explorer.py` that turned into Josef's
end-to-end training-data curation tool over the past several days.

---

## 0. How to run / test

**Server** (the data stack lives in the `bookclean` conda env; base env has no numpy):
```bash
~/miniconda3/envs/bookclean/bin/python explorer_web/server.py \
    --host 0.0.0.0 --port 8765 \
    --registry ~/whynot/traindata/registry.json \
    [--tokenized-root <dir>]
```
- On the Windows box the tools run live off the `\\192.168.1.3\valhalla` share, so
  editing files here is enough; a **server restart** picks up `server.py` /
  `dataset_explorer.py` / `neardupe.py`, a **browser hard-refresh (Ctrl-F5)** picks up
  `static/`.
- Registry default: `~/whynot/traindata/registry.json` (NAS-shareable across rigs).

**Test loop used all session** (headless Chromium needs a conda-forge lib):
```bash
# start server on 8799 with a scratch registry, then:
LD_LIBRARY_PATH=~/miniconda3/envs/bookclean/lib \
  ~/miniconda3/envs/bookclean/bin/python  # + playwright script
```
- `~/miniconda3/envs/bookclean/bin/python -m playwright` + `chromium` are installed;
  `alsa-lib` came from conda-forge (`LD_LIBRARY_PATH` points at the env's `lib`).
- Server is detached with `setsid ... < /dev/null &`; killed via
  `kill $(ss -tlnp | grep <port> | grep -oP 'pid=\K[0-9]+')`.
- Scratch dir this session:
  `/tmp/claude-1000/-home-josef-valhalla-code-tools/b4521dec-.../scratchpad`
  (has `wiki_ab.jsonl` = 20k synthetic wiki docs for the filter A/B, plus bigset/
  multiset/ao3_* fixtures).
- Two GPUs on this box (GTX 1070 ×2); the real rig is a 2080.

**Sandbox note:** `python -u` subprocess reads on Windows default to cp1252 — the
server forces `PYTHONUTF8=1` + lenient decode. A single `→` in pre_tokenize once
crashed a 21.9B-token run at the final print; it's now `->`.

**Watchdog note (2026-08-09):** the quiet-child heartbeat used to report
`('tokenize', 0, 0, note)` — clobbering the docs-stage pct/ETA every time the
subprocess went silent. It now re-emits the last known docs progress with the
quiet note appended, so ETA survives silent phases (and honestly inflates
during stalls). Josef's nit, fixed same day.

---

## 1. Architecture (stable, don't relitigate)

- **One `DatasetExplorer` per loaded dataset**, each owned by a dedicated **worker
  thread** (`DatasetWorker`); all mutating/slow ops run as **Jobs** on that thread,
  preserving the explorer's single-threaded assumptions. Fast reads (record fetch via
  line index) bypass the queue.
- **`UTILITY_WORKER`** runs dataset-less jobs (cache migration).
- **Progress protocol:** `dx.report_progress(stage, done, total, main=False, note=None,
  unit=None)` → thread-local `_progress_hook` → per-job `progress` dict (`main`/`stage`
  levels, each with pct/eta/note/unit) → SSE `progress` events → UI progress line.
  `unit='bytes'` renders as `114.3 GB / 120.1 GB` (line-index/scan/decompress stages
  report byte offsets, not item counts).
- **Job log capture:** worker stdout → thread-local proxy → per-job `JobLogWriter` that
  understands `\r` rewrites (collapses progress-rewriting lines). SSE streams log lines
  in chunks + a `replace` event for `\r`-mutated last lines. UI log sink buffers,
  caps at 4000 lines, autoscrolls only when pinned, and has a live `\r` line element.
- **Jobs are cancellable:** `JobCancelled(BaseException)` raised from the progress hook
  (escapes the hook's `except Exception` guard — the same property sketch-resume relies
  on). Subprocess jobs (tokenize) kill their **process tree** (`taskkill /F /T` on
  Windows, killpg on POSIX) via a watchdog that also fires during silent phases (cancel
  latency ≤5s). Queued jobs cancel instantly. States: queued|running|done|error|
  cancelled. Cancelled/failed **loads remove their placeholder entry** (no ghost).

### Registry (the "Library")
- JSON file, atomic writes, reload-before-mutate (NAS-shareable). Entries carry
  `id, name, path (absolute!), kind (text|tokenized), tags, notes, derived_from,
  open_opts, stats, recipe, created, last_opened`.
- `find_by_path` matches by `str(Path.absolute())` — **not `resolve()`** (mapped-drive/
  symlink trap: `resolve()` rewrites `W:\...`→`\\nas\...`, orphaning caches).
- **Filters** now live here too (see §3): `data['filters']` keyed by `flt-<slug>`.

### Cache layout & moves (all working, heavily battle-tested)
- **Consolidated layout:** everything derived from a dataset lives in
  `<root>/.dataset_explorer_cache/` including `tmp/` (decompressions/decodes). Legacy
  scattered layouts (per-source-dir `tmp/` + `.dataset_explorer_cache/`)
  **auto-consolidate on open** (`consolidate_cache_layout`, rename-only).
- **Cache keys** embed an 8-hex md5 of the **absolute path as given**. Moving a dataset
  orphans caches → **auto-adopt on load**: `pathkey.json` marker records the spelling
  caches are keyed for; a mismatch triggers `migrate_cache` inside the load job
  (rename-only, idempotent). Ambiguous artifacts (>1 candidate) raise
  `CacheConflictError` → UI **conflict chooser modal** (keep A / keep B with dates+
  sizes; losers → `.superseded`).
- **Self-healing:** atomic temp writes (`.part`+rename); `_decompressed_temp_complete`
  (truncation check) + `_line_index_matches` (index vs file EOF) re-derive poisoned
  caches on open. This fixed the "truncated temp reported wrong record count → sets
  rejected" incident.
- CLI: `python dataset_explorer.py <path> --migrate-cache`.

---

## 2. Feature inventory (all shipped & tested this session)

- **Library sidebar** = registry list (text datasets); **Result sets** below;
  **Tokenized** section below that (separate phase). Click a dataset → Info page;
  edit/unregister on the Info page. `open` button on unopened entries (auto-selects).
  ☆ promotes ad-hoc loads (registration takes vitals from the live explorer — **no
  filesystem sweep**, was a 5,843-file SMB stall).
- **Browse** (paged, set-scoped, full-record view), **Search** (findall text/regex/
  multi-term with live per-term tallies; metadata queries via metaindex),
  **Sets** (union/intersect/subtract, rename, delete, search-within-set),
  **Export** (own tab: recipe → live plan w/ exact byte sizes + per-file table →
  single / shard-mirror / **split-to-size**; parts named `<dataset>_00000.jsonl`;
  atomic; auto-register with lineage).
- **Dedup** (neardupe): GPU-resident clustering, multi-GPU (`cuda:all`, round-robin),
  sketch-resume (`*-sketchprog.json` sidecar) + match-resume (`MatchCheckpoint`),
  `min_tokens`, prune (dry-run/write). The big perf saga (1957h→~18min ETA) is done:
  `empty_cache` before VRAM sizing, GPU scatter/Borůvka clustering, tile freed before
  clustering + 16M-pair chunking, densify vectorized.
- **Tokenize tab** (Phase 2): full `pre_tokenize.py` orchestration — extraction preview
  vs real records, preflight (tokenizer→vocab→dtype, file globs, template render,
  resume-manifest detection), subprocess job w/ 4 progress signals + quiet-child
  watchdog, auto-register child with structured **`recipe`** (one source → N tokenized
  versions, "clone settings" prefill, "Tokenized versions" table). Output path suggestion
  matches Josef's layout: `.../source/<name>` → `.../tokenized/<tokenizer>/<name>`
  (or `--tokenized-root`).
- **Recursive directory load** (opt-in `include subdirectories`; changes record
  numbering so it's off by default). **Schema pre-flight** (reads 1st line of every file
  before indexing — fails fast on mismatch; skips `.json` sidecars in mixed dirs).
- **File browser** uses `os.scandir` (was per-entry SMB stats = minutes); 500-entry cap
  with note; UI shows `listing…` immediately + sequence-guards stale responses.
- **Adaptive metaindex discovery** (stops ~2k records after schema stabilizes).
- **Progress everywhere** — every long phase reports; "action/reaction violation" =
  Josef's term for work done silently in a click path (the recurring bug class).

---

## 3. WHERE WE ARE RIGHT NOW: Filters (the current build)

### The conceptual model (agreed with Josef — this is the north star)
**2026-08-09: now a FOUR-noun system.** Transforms split out of filters into their
own first-class noun + tab (Josef's call, and the code agreed — the rules-hash
freshness patch existed only because two objects shared one version counter):
- **Transforms** = *intensional rewrites* — ordered scrub chains (record → record′,
  composes by SEQUENCING, optional fixpoint) at library level beside filters
  (record → bool, composes as set algebra). Registry `data['transforms']`,
  `tfm-<slug>` ids, independent versioning; startup `migrate_filter_scrubs()`
  moved filter-embedded scrubs into `<filter>-scrubs` transforms (idempotent, no
  filter version bump, r= stamps stay valid). Transforms never materialize alone —
  only as components of a composition `(source, filter@v, transform@v, mode)`;
  future overlay/delta tier keys on (source, transform), reusable across filters.
- **Transforms tab**: table (chain shown in order + ⟳ fixpoint marker, stored-def
  tooltip), editor, preview (sample + optional "survivors of <filter>" context,
  diffs + convergence stats), per-corpus preview history panel.
- **Compose**: `transform_id` beside `filter_id`; recipe records both ids+versions.
  View manifest unchanged in shape (resolved scrubs + fixpoint) → **pre_tokenize
  untouched**, hashes continuous. **Export** (all three modes) also takes
  `transform_id` — rewrites while writing, `[scrub]` counts printed, recipe noted.
- Tested: migration verified; preview with/without filter context exact; composed
  stream with filter+transform token-identical to the pre-split anchor (231,800);
  export bytes scrubbed; UI smoke green.
- **Literal mode** (2026-08-09): per-scrub `literal` flag — pattern treated as
  an exact substring (`re.escape` at compile; view manifests carry the escaped
  regex so pre_tokenize stays literal-unaware; stress vet skipped — escaped
  literals are provably linear). UI: per-row "literal" checkbox; tooltip shows
  `'pattern' (literal)` vs `/pattern/`. The foot-gun it kills: `(disambiguation)`
  as a REGEX is a group matching the bare word — deleting it leaves `()` husks.
- **Line mode** (2026-08-09): per-scrub `line` flag — the match selects its
  ENTIRE line (wrapped `[^\n]*(?:…)[^\n]*\n?` by `_scrub_regex_source`, the
  single authority both the compiler and the view-manifest resolver use).
  Composes with literal: check both, paste a string, the line dies. Handles
  final lines without trailing newlines. Vet runs on the WRAPPED source.
- **Glob mode** (2026-08-10): third pattern mode — exact substring except
  `*` = SHORTEST stretch of anything within the line (`[^\n]*?` join of
  escaped pieces; lazy so deletions take the minimum span). UI is now a
  three-way mode select (regex | literal | glob) + the line checkbox.
  Deliberately NOT folded into literal: `*` is a real char in wiki text
  (birth-stars `(* 1305…`). literal+glob mutually exclusive; glob is vetted.
  A "? pattern help" toggle in the editor documents all modes with examples.
  Per-scrub **aA (nocase)** checkbox: case-insensitive matching in any mode
  (inline-flag prefix composed with line mode's m-flag: `(?mi)` — flags must
  lead the pattern, built once in `_scrub_regex_source`).
- **Anchored line-mode wrapper** (2026-08-10, after Josef asked "is glob
  slow?"): glob itself is ~6µs/doc (literal prefix scan), but the original
  line wrapper `[^\n]*(?:…)[^\n]*\n?` re-attempted at EVERY offset with
  per-split backtracking — O(L²) per non-matching line: 1.6ms/18KB doc,
  179ms on one 20KB line (and the corpus has 587KB records). Now
  `(?m)^(?=[^\n]*?(?:…))[^\n]*\n?` — attempt only at line starts, test once
  via lookahead, consume unconditionally. 4x typical, 513x adversarial,
  semantics verified identical (both fixtures byte-exact).
- **Escapes checkbox** (2026-08-11): per-scrub `\n` flag — literal/glob
  interpret exactly `\n` `\t` `\\` in pattern AND replacement (e.g. literal
  `>>>>>\n` = marker at line end). OPT-IN so the paste-anything contract
  stays default; unknown escapes and trailing backslashes are save-time
  errors, and line-mode + a pattern `\n` is rejected (can never match).
- **Expand/paging acknowledgment** (2026-08-10, action/reaction): expand
  flips to a disabled "loading…" the instant it's clicked (restored on error);
  next/prev swap the counter to "loading N record(s)…" with buttons disabled.
  Verified under 900ms-latency network emulation.
- **Expand button** (2026-08-10): each preview example toggles between region
  fragments and the FULL modified document with every change highlighted
  inline — a real before/after sequence diff (`_full_diff_segments`,
  eq/del/ins segments; docs >400KB skip highlighting and show plain
  after-text, guarding SequenceMatcher's quadratic worst case). Fetched on
  demand via `/diff {full:true}`, cached on the record. Segments verified to
  reconstruct source byte-exactly (eq+del == before, eq+ins == after).
- **Scrub pills** (2026-08-10): the preview also returns `changed_masks`
  (per-record bitmask, bit i = chain-order scrub i hit it — sourced from
  `_apply_scrubs`' hit-names, now returned as the 4th tuple element). UI pills
  (one per scrub, all-on, labeled with doc counts) narrow the pageable record
  list client-side (mask ∧ enabled-bits) AND hide disabled scrubs' regions
  with a "(+N hidden)" note. Per-RECORD example cache — re-filtering never
  refetches. Bit-counts verified == per-scrub docs exactly.
- **Example paging** (2026-08-10, Josef needs volume for damage spot-checks):
  the preview result carries `changed_indices` (every changed record's index,
  evenly thinned past 5,000) and the UI pages through them 8 at a time with
  prev/next — pages fetched on demand via synchronous
  `POST .../transforms/{tid}/diff {indices}` (≤32 fast-path record reads +
  regex, no job, no re-sample) and cached client-side.
- **Per-scrub diff regions** (2026-08-10, Josef's catch): the preview's old
  flat excerpt anchored at the FIRST changed byte — with a chain, a 2-char
  husk deletion at doc top hid the 300-char tail chop at doc end. Preview
  examples now return `regions`: every match with context, attributed to its
  scrub, computed by walking the chain in execution order (contexts reflect
  predecessors' output). UI renders red-strikethrough removals / green
  insertions per `[scrub]` line (`_scrub_diff_regions`; styles in style.css).
  Also fixed while writing the help: REPLACEMENT strings in literal/glob mode
  are now backslash-literalized (`_scrub_repl_source`) — re.subn processes
  \1/\g escapes, so a verbatim `C:\Users` replacement used to bad-escape;
  paste-safe modes are now paste-safe end to end (regex mode keeps backrefs).
- **Backtracking guard** (2026-08-09, after a pattern I wrote wedged Josef's
  preview): every scrub pattern is stress-tested at validation time against
  adversarial documents in a killable subprocess (2s budget) — catastrophic
  backtracking is a 400 at save, never a hung worker (a pathological subn()
  blocks the dataset worker in C where cancellation can't reach; restart is
  the only cure, so prevention is the fix). Vetted patterns cached per
  process. The lesson pattern: repeated line-groups must consume a MANDATORY
  newline — `(?:[^\n]*\n){0,40}[^\n]*$`, never `(?:[^\n]*\n?){0,40}$`.

The pipeline has a ~~three~~ four-noun system:
- **Datasets** = raw material (the only root; immutable source of truth).
- **Sets** = *extensional* — frozen index lists; observations about one corpus
  ("what matched", "what dedup found"). Corpus-bound. Small (indices). Stored.
- **Filters** = *intensional* — named, ordered rule **definitions** that yield sets when
  evaluated against ANY corpus. Corpus-**agnostic** → they live at **library level**,
  beside datasets, not under them. "Search discovers; filters legislate."

**Materialization** (the unifying concept, mostly future):
- Identity of any derived thing = `(source fingerprint, composition recipe)`. Bytes are
  a **cache with a policy**, not identity. Grades: **ephemeral** (cleaned stream feeding
  tokenize; dies after) / **cached** (evictable, staleness-checkable) / **pinned**
  (explicit; tokenizations default here — expensive to regen; training-consumed artifacts
  auto-pin for ledger reproducibility).
- **Copy-on-write tiers:** `pointers` (a pure set — Tier 1, *exists today*) / `delta`
  (overlay: rewritten records + `output_idx→(source_idx|delta_offset)` map — Tier 2,
  **deferred with the transform system**) / `full` (representation change e.g. tokens,
  or high-touch scrubs). The `source_map` doubles as per-record provenance (set algebra
  commutes across materializations; trace any token back to raw byte offset).
- **Deferred explicitly:** scrubs/transforms ("find line with substring X → action Y",
  a data-migration system). Schema **reserves `scrubs: []`** on every filter now, so
  stored recipes never need migration when it lands. Transforms will live at the export
  pass (only moment records are rewritten); per-transform hit-counts join the ledger row.

### What's BUILT and TESTED at the API level (verified via curl this session)
Filters as registry objects + rule engine + shallow/materialize evaluation. In
`server.py`:
- `Registry.list_filters/get_filter/upsert_filter/delete_filter` — filters stored in
  `data['filters']`.
- **Rule engine:** `FILTER_RULE_KINDS = (contains, startswith, len_lt, len_gt, regex,
  python)`. `_compile_rule(rule, default_field)` → `callable(rec)->bool`. `python` kind
  eval's `expr` with a locked-down env (`_RULE_ENV`: `re,len,str,int,float,any,all,min,
  max,sum,abs`; no builtins) and `rec` in scope — honest eval, LAN single-user tool.
  Missing field never matches (safe for drops).
- Models: `FilterRule`, `FilterDef` (has reserved `scrubs: []`), `_validate_filter`
  (unique non-empty rule names; compiles each rule to catch bad kind/regex/expr).
- Endpoints: `GET/POST /api/filters`, `DELETE /api/filters/{fid}`,
  `POST /api/datasets/{id}/filters/{fid}/evaluate` (body `{sample, materialize}`).
  - **Shallow** (sampled, evenly-spaced via linspace): per-rule counts + **per-10k
    rates** + `ANY` row + median chars → *this is the wiki A/B table*. Runs as a job.
  - **Materialize** (full scan, `sample:null materialize:true`): additionally stores each
    rule's matches as an ordinary **set** (`<filter>.<rule>`) plus `<filter>.any` — so
    drops are BROWSABLE before destructive, and `.any` feeds export/tokenize exclusion.
  - Versioning: `upsert` bumps `version` on edit (identity = recipe+version).

**Verified numbers** (20k `wiki_ab.jsonl`: planted disambig 5% / stub 10% / husk 8% /
list 3%): shallow 10k-sample gave disambig 508/10k, stub 1543/10k, list_page 311/10k,
husk 782/10k (after fixing husk regex to match the scrub pattern), ANY 2636/10k.
Materialize full-scan produced sets `wiki-clean.disambig` (1016), `.stub` (3102),
`.list_page` (622), `.any` (3724). All correct vs planted rates.

### UI (built; last test was passing once driven correctly)
**2026-08-08: filters moved to their OWN tab** ("Filters", between Library and
Browse — the second library-level tab; Library keeps registry + tokenized only).
Added the **Evaluations panel**: every eval is persisted to the registry
(`data['filter_evals'][fid]`, keyed by corpus path+version+sample — re-runs
replace their column, cap 24, cleared on filter delete; endpoints
`GET/DELETE /api/filters/{fid}/evals`) and rendered as a side-by-side
comparison — rows = rules per-10k + ANY + median chars + corpus records,
one column per recorded run. **This is the wiki A/B screen**: evaluate
`wiki-clean` on both corpora, read adjacent columns. Tested: two-corpus
side-by-side + history survives page reload and server restart.

Original panel description (now lives in the Filters tab, unchanged):
`static/index.html`: **Filters panel** on the ~~Library~~ Filters tab —
filters table (name/rules/version/notes + evaluate/edit/delete), inline rule editor
(`RULE_KINDS` map drives which arg/num inputs show per kind), eval controls
(sample size, "materialize sets" checkbox, "Evaluate on open dataset"), progress line,
result table. `static/app.js`: `refreshFilters`, `addRuleRow`, `collectFilterDef`,
editor save/cancel, `btn-flt-eval` handler (calls evaluate, `watchJob` → renders the
per-10k table, `refreshSets` on materialize). `refreshFilters()` called on boot and on
Library tab show.

### ~~The ONE loose end~~ — RESOLVED 2026-08-07 (evening session)
The prior failure was confirmed spurious (test never selected the dataset, so
`state.current` was null and the eval button correctly alerted). Re-ran the full UI
smoke test driving the real click path — load form → auto-select → Library tab →
wiki-clean `evaluate` → `btn-flt-eval` — **PASS, zero dialogs**:
- **Shallow** (10k sample): disambig 508 / stub 1543 / list_page 311 / husk 782 /
  ANY 2636 per-10k — byte-identical to the API-verified numbers.
- **Materialize** (full 20k scan): sets `wiki-clean.{disambig,stub,list_page,husk,any}`
  = 1016/3102/622/1595/5319, rendered in the sidebar via `refreshSets`. Bonus verified:
  the v1 sets from the earlier materialize (pre-husk, `.any`=3724) were **overwritten by
  name in place** — no duplicate sets after re-materializing a bumped filter version.
- Screenshots: `<this-session scratchpad>/filters_{shallow,materialize}.png`; test
  script `ui_filter_test.py` alongside them.

Still worth: a real server-restart + Ctrl-F5 sanity pass by Josef on his registry.

Note: `flt-` id minting uses `re.sub(...).strip('-')`; a filter named "wiki-clean"
→ `flt-wiki-clean`. Editing sends `id` back to bump version.

---

## 4. NEXT (agreed scaffolding order — filters track)
1. ✅ Filter definitions (registry objects, rule engine) — **done, API-tested**.
2. ✅ Shallow evaluation (per-10k A/B table) — **done, this is the wiki A/B deliverable**.
3. ✅ Materialize-to-sets (per-rule + combined sets) — **done, UI-verified end-to-end**.
4. ✅ **Tokenize composition** — **built & E2E-tested 2026-08-07 (evening)**. Sets
   include/exclude + filter pickers on the Tokenize tab; live plan line via export/plan.
   Server: `POST /api/datasets/{id}/tokenize/composed` (`ComposedTokenizeRequest`), one
   `tokenize-composed` job on the dataset worker = materialize (skipped when the filter's
   current-version sets exist — set descriptions end in `vN`) → export single-JSONL
   `*_cleaned` intermediate (atomic, registered w/ lineage via `_register_export`) →
   `_make_tokenize_runner` (the refactored tokenize body; `extra_recipe` stores the
   composition, `lineage_path` re-parents when the intermediate is ephemeral).
   `keep_intermediate:false` = ephemeral grade: no registration, file deleted after
   success, tokenized child descends from the source. Refactors: `_filter_eval_impl`,
   `_iter_kept_lines(explorer, keep)`, `_make_tokenize_runner` all module-level now.
   Verified on wiki_ab: filter-only run kept 14,681/20,000 → 4.04M tokens, lineage
   `wiki-ab → wiki-ab-cleaned → wiki-ab-cleaned-tiktoken`, composition recipe on the
   child; ephemeral run (exclude set + unmaterialized filter) materialized on-run,
   kept 16,898, deleted intermediate, `derived_from` = source. Also fixed: tok-field
   defaults to `text` column when present (was column #0 — a silent 5-tokens/doc
   footgun when column #0 was `title`).

   **Stream mode (same evening, Josef's call):** materializing the intermediate is
   wasteful when the composition only SELECTS records (pointer tier!) — hours of SMB
   writes for untouched bytes. Added the **view-manifest bridge**: composed job's
   `intermediate_mode:'stream'` (new default in UI) writes `<label>.view.json` +
   `<label>.view.npz` (authoritative file list + per-file sorted skip-ordinal arrays)
   into the output dir; `pre_tokenize.py --view-manifest` uses that file list verbatim
   (never its own glob — the .json-sidecar misalignment trap) and drops ordinals
   in-stream. Guards, all tested: per-file raw record count verified at EOF (source
   drift fatal, file not marked done); composition content-hash stamped to
   `<label>.manifest.view` — identical composition resumes cleanly (hash covers
   arrays/counts, not manifest bytes), different composition REFUSES ("would mix
   compositions"), non-view resume with view (and vice versa) refuses. Restrictions:
   jsonl/parquet sources only; incompatible with scanned-book-jsonl/batch (their
   iterators renumber records). Token-identical to file mode by test: wiki_ab
   4,044,246 tokens exact match; multiset (400-file dir, per-file split via
   cum_record_counts) 370,968 exact match. **Overlay slot is the transform future:**
   when scrubs land, the view manifest gains `overlay` (rewritten records as plain
   jsonl + ordinal→overlay-line map) — the delta tier, streamed the same way.
5. ✅ **Scrubs/transforms — SHIPPED 2026-08-09** (Josef prioritized it ahead of the
   first real tokenization; right call — the wiki corpus's dirt is scrub-shaped).
   - **Model:** `ScrubDef {name, pattern, replacement, field}` on the filter (the
     reserved slot, now typed), ordered, applied to SURVIVORS at compose time only —
     source data never mutated. Order matters and is chain-aware (husk removal creates
     double spaces; `double_space` last cleans them — verified in tests).
   - **Preview** (`POST .../filters/{fid}/scrub-preview`): sampled dry-run on
     survivors — per-scrub docs/subs/chars-removed per-10k + before/after diff
     excerpts at the first change. UI: "preview scrubs" button on the Filters tab.
   - **Execution:** file mode rewrites records while writing the intermediate;
     stream mode ships scrubs in the view manifest and `pre_tokenize.py` applies
     them in-stream (per Josef's call: pervasive scrubs stream; the overlay/delta
     tier stays reserved for rare heavy rewrites). Scrubs join the view-manifest
     content hash → edited composition refuses to resume an old output dir.
     Exact per-scrub counts print as `[scrub] name: docs= subs= chars_removed=`
     lines; the web runner parses them into the job result + composition recipe.
   - **Fixpoint mode** (Josef's idea, independently re-derived): per-filter
     `scrub_fixpoint` flag — the chain re-runs on each record until a full pass
     changes nothing (cap 10). For chains whose rewrites expose patterns an
     EARLIER scrub would have caught. Preview reports max passes + a loud
     nonconverged count (oscillating chains announce themselves); pre_tokenize
     honors the flag from the view manifest and prints `[scrub] fixpoint
     nonconverged=N`. The flag joins the composition hash ONLY when on (old
     manifests keep their hash), so flipping it refuses to resume an old
     output dir. Tested: 3-pass chain exact (800→2,400 subs), oscillator
     detected 50/50 at cap, flag-flip resume refusal.
   - **Rules-hash freshness:** materialized sets are stamped `vN r=<md5(rules)[:8]>`;
     compose reuses sets when the RULES hash matches, so scrub-only edits (which
     bump the version) don't trigger a re-materialize pass. Legacy unstamped sets
     fall back to exact-version match (one extra pass after upgrading, then never).
   - **Tested** (planted fixture): preview counts exact incl. chain interaction;
     stream tokens == file tokens (231,800) with IDENTICAL per-scrub counts from
     the two independent implementations; intermediate verified artifact-free;
     scrub-edit→reuse / rule-edit→re-materialize / resume-mix-refusal all green.
6. **Deferred:** Per-set storage for result sets — the store is ONE gzip blob
   rewritten in full on every mutation, so delete/rename cost O(total store)
   not O(touched set); with an 11.8M-index probe set loaded that's seconds of
   SMB write per click. UI now updates optimistically (2026-08-09) so the lag
   is invisible, but the real fix is one file per set (delete=unlink,
   rename=rename) + migration of existing `.sets.gz`.
   Materialization delta tier (overlay slot reserved in the view
   manifest — for rare high-touch rewrites, not pervasive scrubs);
   cross-corpus dedup (needs a NEW **rectangular A×B match mode** — current matcher is
   triangular within-corpus; GPU clusterer/banding/checkpoint all reuse).

---

## 4b. Nested metadata (2026-08-08, found building the first real filter)
RedPajama wiki records are `{text, meta:{title,url,language,timestamp}}` — and THREE
layers of tooling were blind to nested dicts. All fixed + tested:
- **Filter rules**: `_compile_rule` was flat `rec.get(field)`; now dotted paths
  (`meta.title`) traverse nested dicts, exact flat key wins first. Missing-field
  semantics unchanged (never matches).
- **metaindex.flatten**: only the literal key `metadata` was flattened (AO3-ism,
  bare names — preserved for compat); now ANY dict column flattens one level under
  DOTTED names. Old field-poor indexes **self-heal**: `get_metaindex` checks record 0
  for non-`metadata` dict columns vs dotted fields in the index and auto-rebuilds
  (AO3-style datasets never rebuild — their field set didn't change).
- **metaquery lexer**: `.` now allowed in identifiers → `meta.language = bg` works
  in metadata search.
- **Info page**: `nested fields` row (from record 0) shows `meta.{...}` with the
  dotted addressing hint; Field statistics labeled top-level-only.
Also: Josef keeps ALL corpora multilingual on purpose (robustness) — never add
language-drop rules. The wiki-clean production filter is the brief's four rules
(disambig_text/disambig_title/stub/list_page), title rules on `meta.title`; the
English-shaped predicates knowingly let non-English nav pages through (acceptable
v1; extend with per-language needles if browsing shows it matters).

## 5. The driving work order (context for WHY filters exist)
`mara_fsdp2/docs/DATA_AGENT_BRIEF_amendment1_wiki.md` — wizard102 (6.4B KDA storyteller,
training on rig-30) gets "Amendment 1": world-knowledge pillars (history/science/
psych/logic), NOT general-STEM. Draft new groups (Josef still reflecting, NOT final):
`wikipedia 4%`, `essweb_stem 4%`, `essweb_med 2%`, `essweb_math 2%`, `stackex_craft 3%`;
narrative shrinks 66→57%.

**This week's blocker = the wiki A/B** (RedPajama wiki vs wikimedia/wikipedia 2023-11):
run the `WIKI_FILTERS` metrics over 10k samples of both, per-10k table. **The filter
shallow-eval built this session IS that tool.** Cleaning recipe drops disambig / stub /
"List of" / husks; expected ~15-25% docs, ~5% tokens dropped.

Review notes I gave on the brief (open, Josef's calls):
- Husk metric regex `\(\s*[;,]?\s*\)` ≠ scrub regex `\(\s*[;,]?\s*(;\s*)*\)` — multi-semi
  husks counted-vs-scrubbed mismatch; use the scrub regex for the metric (I did, in the
  test filter).
- `d.title` field may not exist on RedPajama wiki (embedded in text head) — schema-check
  both corpora before declaring the recipe portable, or title-rules silently no-op.
- Scrub dict has patterns, no replacements — the transform system needs
  `(pattern, replacement)` shape (deferred, but note it).
- Consider a `redirect_rate` (`#REDIRECT`) metric — extractors differ most there.
- Output conventions: tokenizer llama `../tokenizers/llama_tokenizer`; HEAD val
  convention (pre_tokenize `--val-holdout` already takes val from stream head — may make
  a separate rechunk pass redundant for fresh tokenizations; Josef to confirm once);
  dest `../../notebooks/datasets/tokenized/llama/<group>/`; `*_cleaned` suffix. **Store
  absolute paths in recipes** (relative-to-what rots — we paid the UNC tax already).

---

## 6. Files
- `tools/dataset_explorer.py` — engine (~5k lines). Key adds this session: consolidated
  cache layout, pathkey/migrate/heal, recursive load, schema pre-flight, sketch-resume,
  `report_progress` unit/note, adaptive metaindex, byte-unit stages.
- `tools/neardupe.py` — sketch/match; GPU-resident `StreamingClusterer` (`device=`),
  multi-GPU `_all_pairs_multi` + `merge_from`, `MatchCheckpoint`, `parse_devices`.
- `tools/pre_tokenize.py` — the `→`→`->` fix + `[writer] checking existing shards…`
  banner. Otherwise unchanged (shared with training; instrument sparingly).
- `tools/explorer_web/server.py` — the web layer (~2k lines). Filters engine + endpoints
  are the newest block (search `# ---- filters`).
- `tools/explorer_web/static/{index.html,app.js,style.css}` — UI.
- `tools/explorer_web/README.md` — user-facing feature doc (kept roughly current).
- Memory: `~/.claude/projects/.../memory/explorer-web-server.md` has the running log.
