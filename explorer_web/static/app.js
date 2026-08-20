/* Dataset Explorer Web — vanilla JS frontend */
'use strict';

const $ = (id) => document.getElementById(id);
const state = {
  datasets: [],           // live (open/loading) server-side dataset instances
  registry: [],           // library entries
  current: null,          // open dataset id (dsNN)
  selectedLib: null,      // selected library entry id (may not be open)
  sets: [],
  totalRecords: 0,
  browse: { set: '', start: 0, limit: 50, selected: null },
  selectedJob: null,
  jobEventSource: null,
};

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

function fmt(n) { return n == null ? '?' : Number(n).toLocaleString(); }
function esc(s) {
  const d = document.createElement('span'); d.textContent = String(s ?? ''); return d.innerHTML;
}

/* ---------------- job helper: submit, stream log, await completion -------- */

const MAX_LOG_LINES = 4000;   // DOM cap; the full log stays on the server

/* Buffered log writer: appends land in one DOM operation per flush interval,
   the pane keeps only a tail, and autoscroll engages only when the user is
   already at the bottom (so scrolling up to read is never fought). */
function makeLogSink(el) {
  let pending = [];
  let timer = null;
  let lineCount = el.textContent ? el.textContent.split('\n').length : 0;
  // Dedicated element for the current \r-rewriting line (e.g. "Scanning
  // 140/221 shards..."): updated in place, committed as a normal line when
  // ordinary output follows it.
  const live = document.createElement('span');
  el.appendChild(live);
  const commitLive = () => {
    if (live.textContent) { pending.unshift(live.textContent); live.textContent = ''; }
  };
  const flush = () => {
    if (timer) { clearTimeout(timer); timer = null; }
    if (!pending.length) return;
    const pinned = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    el.insertBefore(document.createTextNode(pending.join('\n') + '\n'), live);
    lineCount += pending.length;
    pending = [];
    if (lineCount > MAX_LOG_LINES) {
      const keep = MAX_LOG_LINES >> 1;
      const liveTxt = live.textContent;
      const lines = el.textContent.split('\n');
      el.textContent = `… (${fmt(lines.length - keep)} earlier lines hidden — full log kept server-side)\n`
        + lines.slice(-keep).join('\n');
      el.appendChild(live);            // rebuild destroyed it; restore
      live.textContent = liveTxt;
      lineCount = keep + 1;
    }
    if (pinned) el.scrollTop = el.scrollHeight;
  };
  return {
    add(lines) {
      commitLive();
      for (const l of lines) pending.push(l);
      if (!timer) timer = setTimeout(flush, 150);
    },
    line(s) { this.add([s]); },
    replaceLast(s) {
      const pinned = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      live.textContent = s;
      if (pinned) el.scrollTop = el.scrollHeight;
    },
    flush,
  };
}

/* ---- progress rendering: "<stage> [bar] pct% (done/total) eta" ------------ */

function fmtEta(s) {
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h${String(m).padStart(2, '0')}m`;
  if (m) return `${m}m${String(sec).padStart(2, '0')}s`;
  return `${sec}s`;
}

function textBar(pct) {
  const n = Math.max(0, Math.min(20, Math.round(pct / 5)));
  return '█'.repeat(n) + '░'.repeat(20 - n);
}

function renderProgress(el, p) {
  if (!el) return;
  if (!p) { el.textContent = ''; return; }
  const parts = [];
  for (const key of ['main', 'stage']) {
    const s = p[key];
    if (!s || !s.name) continue;
    const q = (v) => s.unit === 'bytes' ? fmtBytes(v) : fmt(v);
    let t = s.name;
    if (s.total) {
      t += ` ${textBar(s.pct)} ${s.pct.toFixed(1)}% (${q(s.done)}/${q(s.total)})`;
      if (s.eta_s > 1) t += ` eta ${fmtEta(s.eta_s)}`;
    } else if (s.done) {
      t += ` ${q(s.done)}`;
    }
    if (s.note) t += `  ·  ${s.note}`;
    parts.push(t);
  }
  el.textContent = parts.join('   ·   ');
}

function progressPct(p) {
  const s = p && (p.main?.total ? p.main : p.stage);
  return s && s.total ? s.pct : null;
}

function watchJob(jobId, logEl, onDone, skipLines = 0, progressEl = null) {
  refreshJobs();
  const sink = logEl ? makeLogSink(logEl) : null;
  let toSkip = skipLines;    // lines already rendered (selectJob preloads them)
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  es.addEventListener('progress', (ev) => renderProgress(progressEl, JSON.parse(ev.data)));
  es.addEventListener('replace', (ev) => {
    if (sink && toSkip <= 0) sink.replaceLast(JSON.parse(ev.data).line);
  });
  es.onmessage = (ev) => {
    if (!sink) return;
    let lines = JSON.parse(ev.data).lines || [];
    if (toSkip > 0) {
      const k = Math.min(toSkip, lines.length);
      lines = lines.slice(k);
      toSkip -= k;
    }
    if (lines.length) sink.add(lines);
  };
  es.addEventListener('status', (ev) => {
    const s = JSON.parse(ev.data).status;
    if (sink && s === 'running' && !skipLines) sink.line(`[${jobId}] running...`);
  });
  es.addEventListener('end', (ev) => {
    es.close();
    const job = JSON.parse(ev.data);
    if (sink) {
      sink.line(job.status === 'error' ? `[${jobId}] FAILED: ${job.error}`
        : job.status === 'cancelled' ? `[${jobId}] cancelled.`
        : `[${jobId}] done.`);
      sink.flush();
    }
    renderProgress(progressEl, null);
    refreshJobs();
    if (onDone) onDone(job);
  });
  es.onerror = () => es.close();
  return es;
}

/* ---------------- datasets ----------------------------------------------- */

function liveFor(entry) {   // the open/loading instance backing a library entry
  return state.datasets.find(d => d.path === entry.path);
}

async function refreshDatasets() {
  const [reg, live] = await Promise.all([
    api('/api/registry').catch(() => ({ datasets: [], registry_path: '' })),
    api('/api/datasets'),
  ]);
  state.registry = reg.datasets;
  state.datasets = live;
  state.registryPaths = new Set(reg.datasets.map(d => d.path));

  // Only touch the DOM when the rendered content would actually change.
  // The 2.5s poll during loads used to rebuild the list every tick, destroying
  // buttons out from under in-flight clicks (an "open" click landing on the
  // freshly recreated row selects instead of opening).
  const sig = JSON.stringify([
    state.registry.map(e => {
      const d = liveFor(e);
      return [e.id, e.name, e.kind, d?.status, d?.info?.num_rows,
              (d && d.id === state.current) || e.id === state.selectedLib];
    }),
    state.datasets.map(d => [d.id, d.status, d.path, d.info?.num_rows,
                             d.id === state.current]),
  ]);
  const anyLoading = state.datasets.some(d => d.status === 'loading');
  if (sig === state._sidebarSig) {
    if (anyLoading) setTimeout(refreshDatasets, 2500);
    return;
  }
  state._sidebarSig = sig;

  const ul = $('dataset-list');
  const ulTok = $('tokenized-list');
  ul.innerHTML = '';
  ulTok.innerHTML = '';

  // Library entries first — the sidebar IS the library. Tokenized products
  // live in their own section: a different phase of the pipeline.
  for (const e of state.registry) {
    const ds = liveFor(e);
    const li = document.createElement('li');
    const isSel = (ds && ds.id === state.current) || e.id === state.selectedLib;
    li.className = isSel ? 'active' : '';
    let right;
    if (ds && ds.status === 'ready') {
      right = `<span class="status-done" title="open">●</span>` +
              `<span class="count">${fmt(ds.info?.num_rows)}</span>`;
    } else if (ds) {
      right = `<span class="status-${ds.status}">${ds.status}</span>`;
    } else {
      right = `<button class="mini lib-open">open</button>`;
    }
    li.innerHTML = `<span title="${esc(e.path)}">${esc(e.name)}</span>` +
                   `<span class="row" style="gap:5px">${right}</span>`;
    li.onclick = () => selectLibrary(e.id);
    const openBtn = li.querySelector('.lib-open');
    if (openBtn) openBtn.onclick = (ev) => { ev.stopPropagation(); openRegistryEntry(e.id); };
    (e.kind === 'tokenized' ? ulTok : ul).appendChild(li);
  }
  $('tokenized-section').style.display = ulTok.children.length ? '' : 'none';

  // Ad-hoc loaded datasets that aren't registered yet (star to promote them).
  for (const ds of state.datasets) {
    if (state.registryPaths.has(ds.path)) continue;
    const li = document.createElement('li');
    li.className = ds.id === state.current ? 'active' : '';
    const name = ds.path.split(/[\\/]/).pop() || ds.path;
    const star = ds.status === 'ready'
      ? `<button class="mini star" title="add to library">☆</button>` : '';
    const right = ds.status === 'ready'
      ? `${star}<span class="count">${fmt(ds.info?.num_rows)}</span>`
      : `<span class="status-${ds.status}">${ds.status}</span>`;
    li.innerHTML = `<span title="${esc(ds.path)}">${esc(name)}</span>` +
                   `<span class="row" style="gap:5px">${right}</span>`;
    li.onclick = () => { state.selectedLib = null; showTab('info'); selectDataset(ds.id); };
    const starBtn = li.querySelector('.star');
    if (starBtn) starBtn.onclick = async (e) => {
      e.stopPropagation();
      const regName = prompt('Register in library as:', name.replace(/\.[^.]*$/, ''));
      if (!regName) return;
      try {
        await api(`/api/datasets/${ds.id}/register`,
                  { method: 'POST', body: { name: regName } });
        refreshDatasets();
      } catch (err) { alert(err.message); }
    };
    ul.appendChild(li);
  }

  if (anyLoading) setTimeout(refreshDatasets, 2500);
}

async function selectLibrary(rid) {
  state.selectedLib = rid;
  const e = state.registry.find(x => x.id === rid);
  if (!e) return;
  // Switch tabs IMMEDIATELY, then load data behind it. A trailing showTab
  // after the awaits would land seconds later and yank the user back from
  // whatever tab they clicked in the meantime.
  showTab('info');
  const ds = liveFor(e);
  if (ds && ds.status === 'ready') {
    await selectDataset(ds.id);
    return;
  }
  // Not open: registry-only detail view.
  state.current = null;
  await refreshDatasets();
  renderRegistryInfo(e);
  $('set-list').innerHTML = '<li><span class="dim">open the dataset to see its sets</span></li>';
}

async function openRegistryEntry(rid) {
  state.selectedLib = rid;        // opening implies selecting
  try {
    const r = await api(`/api/registry/${rid}/open`, { method: 'POST' });
    if (r.already_open) { await selectDataset(r.dataset_id); showTab('browse'); return; }
    await refreshDatasets();
    state.current = r.dataset_id;
    selectJob(r.job_id); showTab('jobs');
    watchJob(r.job_id, null, (job) => {
      if (maybeShowConflicts(job, () => openRegistryEntry(rid))) return;
      selectDataset(r.dataset_id); showTab('browse'); refreshDatasets();
    });
  } catch (err) { alert(err.message); }
}

async function selectDataset(id) {
  state.current = id;
  state.browse = { set: '', start: 0, limit: 50, selected: null };
  await refreshDatasets();
  const ds = state.datasets.find(d => d.id === id);
  if (ds?.status !== 'ready') {
    if (ds?.load_job) { selectJob(ds.load_job); showTab('jobs'); }
    return;
  }
  await refreshSets();
  await renderInfo();
  await loadBrowsePage(0);
}

$('btn-show-load').onclick = () => {
  const f = $('load-form');
  f.style.display = f.style.display === 'none' ? '' : 'none';
  if (f.style.display !== 'none') {
    browseTo($('load-path').value
      || localStorage.getItem('lastLoadPath') || '~');
  }
};

let browseSeq = 0;

async function browseTo(path) {
  const seq = ++browseSeq;               // newest navigation wins
  const pane = $('browse-pane');
  pane.innerHTML = '<div class="dim">listing…</div>';   // immediate reaction
  try {
    const before = $('load-path').value;   // don't clobber concurrent typing
    const res = await api(`/api/browse?path=${encodeURIComponent(path)}`);
    if (seq !== browseSeq) return;         // a newer click superseded this one
    try { localStorage.setItem('lastLoadPath', res.path); } catch (e) {}
    if ($('load-path').value === before) $('load-path').value = res.path;
    pane.innerHTML = '';
    if (res.truncated) {
      const note = document.createElement('div');
      note.className = 'dim';
      note.textContent = '(showing first 500 entries)';
      pane.appendChild(note);
    }
    const up = document.createElement('div');
    up.textContent = '⬑ ..';
    up.onclick = () => browseTo(res.parent);
    pane.appendChild(up);
    for (const e of res.entries) {
      const div = document.createElement('div');
      div.textContent = (e.dir ? '▸ ' : '· ') + e.name;
      div.onclick = () => e.dir ? browseTo(e.path) : ($('load-path').value = e.path);
      pane.appendChild(div);
    }
  } catch (err) {
    if (seq === browseSeq)
      pane.innerHTML = `<div class="dim">${esc(err.message)}</div>`;
  }
}
$('load-path').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') browseTo($('load-path').value);
});

$('btn-load').onclick = async () => {
  try {
    const body = {
      path: $('load-path').value.trim(),
      raw_shards: $('load-raw').checked,
      quick: $('load-quick').checked,
      recursive: $('load-recursive').checked,
      text_field: $('load-field').value.trim() || null,
      tok_kind: $('load-tok-kind').value.trim() || null,
      tok_path: $('load-tok-path').value.trim() || null,
      special_tokens: $('load-special').value.trim() || null,
    };
    const res = await api('/api/datasets', { method: 'POST', body });
    $('load-form').style.display = 'none';
    await refreshDatasets();
    state.current = res.dataset_id;
    selectJob(res.job_id);          // selectJob owns the log pane stream
    showTab('jobs');
    watchJob(res.job_id, null, (job) => {
      if (maybeShowConflicts(job, () => $('btn-load').onclick())) return;
      selectDataset(res.dataset_id);
    });
  } catch (err) { alert('Load failed: ' + err.message); }
};

/* ---- cache-conflict chooser: shown when a load stops on adoption conflicts ---- */

function maybeShowConflicts(job, retryLoad) {
  const conflicts = job.result?.conflicts;
  if (job.status !== 'error' || !conflicts?.length) return false;
  const list = $('conflict-list');
  list.innerHTML = '';
  conflicts.forEach((c, ci) => {
    const fs = document.createElement('fieldset');
    fs.innerHTML = `<legend>${esc(c.target.split(/[\\/]/).pop())}</legend>`;
    c.candidates.forEach((cand, i) => {
      const dt = new Date(cand.mtime * 1000).toLocaleString();
      const lbl = document.createElement('label');
      lbl.innerHTML =
        `<input type="radio" name="conflict-${ci}" value="${i}" ${i === 0 ? 'checked' : ''}>` +
        `<span>Keep <b>${esc(cand.name)}</b> <span class="dim">— modified ${dt} — ` +
        `${cand.size_mb >= 1 ? cand.size_mb.toFixed(1) + ' MB' : (cand.size_mb * 1000).toFixed(0) + ' KB'}</span></span>`;
      fs.appendChild(lbl);
    });
    list.appendChild(fs);
  });
  $('conflict-modal').style.display = '';
  $('btn-conflicts-cancel').onclick = () => { $('conflict-modal').style.display = 'none'; };
  $('btn-conflicts-apply').onclick = async () => {
    try {
      for (let ci = 0; ci < conflicts.length; ci++) {
        const c = conflicts[ci];
        const pick = parseInt(document.querySelector(`input[name=conflict-${ci}]:checked`).value, 10);
        await api('/api/resolve-conflict', { method: 'POST', body: {
          target: c.target,
          keep: c.candidates[pick].path,
          others: c.candidates.filter((_, i) => i !== pick).map(x => x.path),
        }});
      }
      $('conflict-modal').style.display = 'none';
      retryLoad();
    } catch (err) { alert('Could not apply choices: ' + err.message); }
  };
  return true;
}

/* ---------------- sets ---------------------------------------------------- */

async function refreshSets() {
  if (!state.current) return;
  const res = await api(`/api/datasets/${state.current}/sets`);
  state.sets = res.sets;
  state.totalRecords = res.total_records;
  renderSetLists();
}

// Render every set consumer from state.sets — separate from the fetch so
// optimistic mutations (delete/rename) can repaint INSTANTLY and reconcile
// with server truth afterwards. The sets file is one gzip blob rewritten in
// full on every mutation; with a multi-million-index set in the store that
// write takes seconds over SMB, and a click path must not sit silent on it.
function renderSetLists() {
  // sidebar
  const ul = $('set-list');
  ul.innerHTML = '';
  for (const s of state.sets) {
    const li = document.createElement('li');
    li.className = state.browse.set === s.name ? 'active' : '';
    li.innerHTML = `<span title="${esc(s.query)}">${esc(s.name)}</span>` +
                   `<span class="count">${fmt(s.count)}</span>`;
    li.onclick = () => { showTab('browse'); setBrowseSet(s.name); };
    ul.appendChild(li);
  }

  // browse selector
  const sel = $('browse-set');
  sel.innerHTML = `<option value="">All records (${fmt(state.totalRecords)})</option>`;
  for (const s of state.sets) {
    sel.innerHTML += `<option value="${esc(s.name)}" ${s.name === state.browse.set ? 'selected' : ''}>` +
                     `${esc(s.name)} (${fmt(s.count)})</option>`;
  }

  // search "within" selector + set-op / export selectors
  const within = $('search-within');
  const keep = within.value;
  within.innerHTML = '<option value="">whole dataset</option>';
  for (const s of state.sets) within.innerHTML += `<option>${esc(s.name)}</option>`;
  within.value = keep;
  for (const id of ['op-a', 'op-b']) {
    const el = $(id); const k = el.value; el.innerHTML = '';
    for (const s of state.sets) el.innerHTML += `<option>${esc(s.name)}</option>`;
    if (k) el.value = k;
  }
  // Guard each element: one missing id (e.g. mixed static versions mid-
  // upgrade) must not kill the rest of the rebuild below it.
  for (const id of ['export-include', 'export-exclude', 'tok-include', 'tok-exclude']) {
    const el = $(id);
    if (!el) continue;
    const sel2 = new Set([...el.selectedOptions].map(o => o.value));
    el.innerHTML = '';
    for (const s of state.sets)
      el.innerHTML += `<option${sel2.has(s.name) ? ' selected' : ''}>${esc(s.name)}</option>`;
  }
  try { tokComposeChanged(); } catch (e) { console.error('tokComposeChanged:', e); }

  // sets table
  const tb = $('sets-table').querySelector('tbody');
  tb.innerHTML = '';
  for (const s of state.sets) {
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td>${esc(s.name)}</td><td class="num">${fmt(s.count)}</td>` +
      `<td class="num">${s.pct.toFixed(2)}</td><td>${esc(s.kind)}</td>` +
      `<td class="dim">${esc(s.query.slice(0, 90))}</td>` +
      `<td><button class="mini" data-act="browse">browse</button> ` +
      `<button class="mini" data-act="rename">rename</button> ` +
      `<button class="mini" data-act="delete">delete</button></td>`;
    tr.querySelector('[data-act=browse]').onclick = () => { showTab('browse'); setBrowseSet(s.name); };
    tr.querySelector('[data-act=rename]').onclick = async () => {
      const old = s.name;
      const nn = prompt(`Rename set '${old}' to:`, old);
      if (!nn || nn === old) return;
      // optimistic: repaint now, persist behind it, reconcile after
      s.name = nn;
      if (state.browse.set === old) state.browse.set = nn;
      renderSetLists();
      try {
        await api(`/api/datasets/${state.current}/sets/rename`,
                  { method: 'POST', body: { old, new: nn } });
      } catch (err) { alert(err.message); }
      refreshSets();
    };
    tr.querySelector('[data-act=delete]').onclick = async () => {
      if (!confirm(`Delete result set '${s.name}'? (indices only; source data untouched)`)) return;
      // optimistic: vanish from every list NOW; the server's sets-file
      // rewrite can take seconds, and silence here reads as failure
      state.sets = state.sets.filter(x => x.name !== s.name);
      if (state.browse.set === s.name) state.browse.set = '';
      renderSetLists();
      try {
        await api(`/api/datasets/${state.current}/sets/${encodeURIComponent(s.name)}`,
                  { method: 'DELETE' });
      } catch (err) { alert(`Delete failed: ${err.message}`); }
      refreshSets();     // reconcile with server truth either way
    };
    tb.appendChild(tr);
  }
}

$('btn-op').onclick = async () => {
  const a = $('op-a').value, b = $('op-b').value;
  if (!a || !b) return alert('Pick two sets');
  try {
    const res = await api(`/api/datasets/${state.current}/sets/ops`, {
      method: 'POST',
      body: { op: $('op-kind').value, sets: [a, b], name: $('op-name').value.trim() || null },
    });
    $('op-name').value = '';
    await refreshSets();
    showTab('browse'); setBrowseSet(res.set);
  } catch (err) { alert(err.message); }
};

/* ---------------- export tab ---------------------------------------------- */

const exportSel = (id) => [...$(id).selectedOptions].map(o => o.value);
let planTimer = null;
let lastPlanBytes = null;

function updateSplitEstimate() {
  const el = $('export-split-est');
  const mode = document.querySelector('input[name=export-mode]:checked')?.value;
  const mb = parseFloat($('export-split-mb').value);
  if (mode !== 'split' || lastPlanBytes == null || !(mb > 0)) {
    el.textContent = ''; return;
  }
  el.textContent = `→ ≈ ${Math.max(1, Math.ceil(lastPlanBytes / (mb * 1e6)))} file(s)`;
}

function fmtBytes(b) {
  if (b == null) return '?';
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  return (b / 1e3).toFixed(0) + ' KB';
}

function schedulePlan() {
  if (planTimer) clearTimeout(planTimer);
  planTimer = setTimeout(refreshExportPlan, 250);
}

async function refreshExportPlan() {
  if (!state.current) return;
  try {
    const plan = await api(`/api/datasets/${state.current}/export/plan`, {
      method: 'POST',
      body: { include: exportSel('export-include'), exclude: exportSel('export-exclude') },
    });
    lastPlanBytes = plan.bytes;
    updateSplitEstimate();
    const sizeTxt = plan.bytes != null
      ? `${plan.bytes_exact ? '' : '~'}${fmtBytes(plan.bytes)}` : 'size unknown';
    $('export-plan-summary').innerHTML =
      `Keeps <b>${fmt(plan.kept_records)}</b> of ${fmt(plan.total_records)} records ` +
      `(${plan.pct.toFixed(2)}%) · drops ${fmt(plan.dropped_records)} · ` +
      `output ${sizeTxt}`;
    $('export-plan-summary').classList.remove('dim');
    const box = $('export-plan-files');
    if (plan.files.length <= 1) { box.innerHTML = ''; return; }
    let html = '<table class="data-table"><thead><tr><th>Source file</th>' +
      '<th>Kept</th><th></th><th>Size</th></tr></thead><tbody>';
    for (const f of plan.files) {
      const frac = f.records ? f.kept / f.records : 0;
      html += `<tr><td>${esc(f.name)}</td>` +
        `<td class="num">${fmt(f.kept)} / ${fmt(f.records)}</td>` +
        `<td><span class="dim">${textBar(frac * 100)}</span></td>` +
        `<td class="num">${f.bytes != null ? fmtBytes(f.bytes) : '—'}</td></tr>`;
    }
    box.innerHTML = html + '</tbody></table>';
  } catch (err) {
    $('export-plan-summary').textContent = 'Plan failed: ' + err.message;
  }
}

$('export-include').onchange = schedulePlan;
$('export-exclude').onchange = schedulePlan;
$('export-split-mb').oninput = updateSplitEstimate;
document.querySelectorAll('input[name=export-mode]').forEach(
  r => r.onchange = updateSplitEstimate);

$('btn-export').onclick = async () => {
  const path = $('export-path').value.trim();
  if (!path) return alert('Enter an output path');
  const mode = document.querySelector('input[name=export-mode]:checked').value;
  try {
    const res = await api(`/api/datasets/${state.current}/export`, {
      method: 'POST',
      body: { out_path: path, mode,
              split_mb: parseFloat($('export-split-mb').value) || 300,
              include: exportSel('export-include'),
              exclude: exportSel('export-exclude'),
              transform_id: $('export-transform').value || null,
              register_as: $('export-register').value.trim() || null },
    });
    selectJob(res.job_id); showTab('jobs');
    watchJob(res.job_id, null, () => refreshLibrary());
  } catch (err) { alert(err.message); }
};

/* ---------------- browse -------------------------------------------------- */

function setBrowseSet(name) {
  state.browse.set = name;
  state.browse.start = 0;
  refreshSets();
  loadBrowsePage(0);
}
$('browse-set').onchange = () => setBrowseSet($('browse-set').value);
$('browse-prev').onclick = () => loadBrowsePage(Math.max(0, state.browse.start - state.browse.limit));
$('browse-next').onclick = () => loadBrowsePage(state.browse.start + state.browse.limit);
$('browse-goto').addEventListener('keydown', (e) => {
  if (e.key !== 'Enter') return;
  const n = parseInt($('browse-goto').value, 10);
  if (!isNaN(n)) loadBrowsePage(Math.max(0, n));
});
$('browse-full').onchange = () => {
  if (state.browse.selected != null) showRecord(state.browse.selected);
};

async function loadBrowsePage(start) {
  if (!state.current) return;
  state.browse.start = start;
  const b = state.browse;
  const q = new URLSearchParams({ start: b.start, limit: b.limit, preview: 160 });
  if (b.set) q.set('set_name', b.set);
  const res = await api(`/api/datasets/${state.current}/records?${q}`);
  const end = Math.min(b.start + b.limit, res.total);
  $('browse-pageinfo').textContent =
    `${fmt(b.start)}–${fmt(Math.max(end - 1, 0))} of ${fmt(res.total)}` +
    (b.set ? ` in '${b.set}'` : '');
  const list = $('browse-list');
  list.innerHTML = '';
  for (const r of res.records) {
    const div = document.createElement('div');
    div.className = 'rec-row' + (r.index === b.selected ? ' active' : '');
    const body = r.error ? `⚠ ${r.error}`
      : Object.values(r.fields).map(v => String(v)).join(' · ');
    div.innerHTML = `<span class="idx">#${r.index}</span>${esc(body.slice(0, 240))}`;
    div.onclick = () => {
      b.selected = r.index;
      list.querySelectorAll('.rec-row').forEach(el => el.classList.remove('active'));
      div.classList.add('active');
      showRecord(r.index);
    };
    list.appendChild(div);
  }
}

async function showRecord(index) {
  const view = $('record-view');
  view.innerHTML = '<div class="dim pad">Loading…</div>';
  try {
    const res = await api(`/api/datasets/${state.current}/record/${index}`);
    const full = $('browse-full').checked;
    view.innerHTML = `<div class="field-block"><div class="field-name">record #${index}</div></div>`;
    for (const [k, v] of Object.entries(res.record)) {
      let text = typeof v === 'string' ? v : JSON.stringify(v, null, 2);
      if (!full && text.length > 4000) text = text.slice(0, 4000) + `\n… (${fmt(text.length)} chars, enable 'full text')`;
      view.innerHTML +=
        `<div class="field-block"><div class="field-name">${esc(k)}</div>` +
        `<div class="field-value">${esc(text)}</div></div>`;
    }
  } catch (err) {
    view.innerHTML = `<div class="pad status-error">${esc(err.message)}</div>`;
  }
}

/* ---------------- search -------------------------------------------------- */

$('search-mode').onchange = () => {
  $('search-text-opts').style.display = $('search-mode').value === 'text' ? '' : 'none';
};

$('search-form').onsubmit = async (e) => {
  e.preventDefault();
  if (!state.current) return alert('Load a dataset first');
  const mode = $('search-mode').value;
  const query = $('search-query').value.trim();
  if (!query) return;
  const multi = $('search-multi').checked;
  const limitRaw = $('search-limit').value.trim();
  const body = {
    mode, query,
    terms: (mode === 'text' && multi) ? query.split('|').map(s => s.trim()).filter(Boolean) : null,
    field: $('search-field').value.trim() || null,
    regex: $('search-regex').checked,
    limit: limitRaw ? parseInt(limitRaw, 10) : null,
    name: $('search-name').value.trim() || null,
    within_set: $('search-within').value || null,
  };
  const log = $('search-status');
  log.textContent = '';
  try {
    const res = await api(`/api/datasets/${state.current}/search`, { method: 'POST', body });
    watchJob(res.job_id, log, async (job) => {
      renderProgress($('search-progress'), null);
      await refreshSets();
      if (job.status === 'done' && job.result?.set) {
        log.textContent += `\n→ set '${job.result.set}': ${fmt(job.result.count)} records` +
          (job.result.limit_hit ? ' (limit hit, more may exist)' : '') + '\n';
        showTab('browse'); setBrowseSet(job.result.set);
      } else if (job.status === 'done') {
        log.textContent += '\nNo matches found.\n';
      }
    }, 0, $('search-progress'));
  } catch (err) { log.textContent = 'Error: ' + err.message; }
};

/* ---------------- dedup --------------------------------------------------- */

$('btn-neardupe').onclick = async () => {
  if (!state.current) return alert('Load a dataset first');
  const body = {
    threshold: parseFloat($('nd-threshold').value) || 0.8,
    ngram: parseInt($('nd-ngram').value, 10) || 13,
    perms: parseInt($('nd-perms').value, 10) || 1024,
    min_tokens: parseInt($('nd-min-tokens').value, 10) || 0,
    device: $('nd-cpu').checked ? 'cpu'
      : ($('nd-all-gpus').checked ? 'cuda:all' : 'cuda'),
    rebuild: $('nd-rebuild').checked,
    set_name: $('nd-name').value.trim() || null,
  };
  try {
    const res = await api(`/api/datasets/${state.current}/neardupe`, { method: 'POST', body });
    selectJob(res.job_id); showTab('jobs');
    watchJob(res.job_id, null, () => { showTab('dedup'); refreshClusters(); });
  } catch (err) { alert(err.message); }
};

async function refreshClusters() {
  if (!state.current) return;
  const res = await api(`/api/datasets/${state.current}/neardupe/clusters?top=100`);
  const tb = $('clusters-table').querySelector('tbody');
  tb.innerHTML = '';
  for (const c of res.clusters) {
    const tr = document.createElement('tr');
    const members = c.members.map(m =>
      `<a href="#" data-rec="${m}">#${m}</a>`).join(' ');
    tr.innerHTML = `<td class="num">${c.rank}</td><td class="num">${fmt(c.size)}</td>` +
      `<td class="num">${(c.max_jaccard ?? 0).toFixed(3)}</td>` +
      `<td class="num">${(c.mean_jaccard ?? 0).toFixed(3)}</td><td>${members}</td>`;
    tr.querySelectorAll('a[data-rec]').forEach(a => a.onclick = (e) => {
      e.preventDefault();
      showTab('browse');
      state.browse.selected = parseInt(a.dataset.rec, 10);
      loadBrowsePage(Math.max(0, state.browse.selected));
      showRecord(state.browse.selected);
    });
    tb.appendChild(tr);
  }
}

$('btn-prune').onclick = async () => {
  const out = $('prune-out').value.trim();
  if (!out) return alert('Enter an output directory');
  const write = $('prune-write').checked;
  if (write && !confirm(`Commit prune to ${out}? Source is never modified; a deduplicated copy is written.`)) return;
  try {
    const res = await api(`/api/datasets/${state.current}/neardupe/prune`, {
      method: 'POST',
      body: { out_dir: out, write, keep: $('prune-keep').value },
    });
    selectJob(res.job_id); showTab('jobs');
  } catch (err) { alert(err.message); }
};

/* ---------------- registry detail panel (Info tab) ------------------------- */

let infoEditRid = null;

function renderRegistryPanel(entry, isOpen, openDsId) {
  const panel = $('info-registry');
  $('info-edit').style.display = 'none';
  if (!entry) {
    if (isOpen && openDsId) {      // open but unregistered: offer promotion
      panel.style.display = '';
      $('info-registry-body').innerHTML =
        '<span class="dim">Not in the library yet.</span>';
      $('info-registry-actions').innerHTML =
        '<button class="mini" id="btn-info-register">☆ register this dataset</button>';
      $('btn-info-register').onclick = async () => {
        const ds = state.datasets.find(d => d.id === openDsId);
        const nm = prompt('Register in library as:',
          (ds?.path.split(/[\\/]/).pop() || 'dataset').replace(/\.[^.]*$/, ''));
        if (!nm) return;
        try {
          await api(`/api/datasets/${openDsId}/register`,
                    { method: 'POST', body: { name: nm } });
          await refreshDatasets(); renderInfo();
        } catch (err) { alert(err.message); }
      };
    } else panel.style.display = 'none';
    return;
  }
  panel.style.display = '';
  const byId = Object.fromEntries(state.registry.map(d => [d.id, d]));
  const children = state.registry.filter(d => d.derived_from === entry.id);
  const row = (k, v) => `<tr><td class="dim" style="width:110px">${esc(k)}</td><td>${v}</td></tr>`;
  $('info-registry-body').innerHTML = '<table class="data-table">' +
    row('name', esc(entry.name)) +
    row('kind', `<span class="badge ${esc(entry.kind || 'text')}">${esc(entry.kind || 'text')}</span>`) +
    row('tags', esc((entry.tags || []).join(', ') || '—')) +
    row('notes', esc(entry.notes || '—')) +
    row('lineage', entry.derived_from
      ? `← ${esc(byId[entry.derived_from]?.name || entry.derived_from)}` : '—') +
    (children.length ? row('derived sets', children.map(c => esc(c.name)).join(', ')) : '') +
    row('last opened', fmtAge(entry.last_opened)) +
    '</table>';
  const acts = $('info-registry-actions');
  acts.innerHTML =
    (isOpen ? '' : `<button class="primary" id="btn-info-open">Open</button>`) +
    `<button class="mini" id="btn-info-edit">edit</button>` +
    `<button class="mini" id="btn-info-unreg">unregister</button>`;
  if (!isOpen) $('btn-info-open').onclick = () => openRegistryEntry(entry.id);
  $('btn-info-unreg').onclick = async () => {
    if (!confirm(`Unregister '${entry.name}'? (library metadata only — no data files are touched)`)) return;
    await api(`/api/registry/${entry.id}`, { method: 'DELETE' });
    state.selectedLib = null;
    await refreshDatasets();
    renderRegistryPanel(null, isOpen, openDsId);
  };
  $('btn-info-edit').onclick = () => {
    infoEditRid = entry.id;
    $('info-edit').style.display = '';
    $('ie-name').value = entry.name;
    $('ie-tags').value = (entry.tags || []).join(', ');
    $('ie-kind').value = entry.kind || '';
    $('ie-notes').value = entry.notes || '';
    const oo = entry.open_opts || {};
    $('ie-field').value = oo.text_field || '';
    $('ie-tok-kind').value = oo.tok_kind || '';
    $('ie-tok-path').value = oo.tok_path || '';
    $('ie-raw').checked = !!oo.raw_shards;
    $('ie-recursive').checked = !!oo.recursive;
  };
}

$('btn-ie-save').onclick = async () => {
  if (!infoEditRid) return;
  try {
    await api(`/api/registry/${infoEditRid}`, { method: 'PATCH', body: {
      name: $('ie-name').value.trim() || null,
      tags: $('ie-tags').value.split(',').map(s => s.trim()).filter(Boolean),
      notes: $('ie-notes').value,
      kind: $('ie-kind').value || null,
      text_field: $('ie-field').value.trim() || null,
      tok_kind: $('ie-tok-kind').value.trim() || null,
      tok_path: $('ie-tok-path').value.trim() || null,
      raw_shards: $('ie-raw').checked,
      recursive: $('ie-recursive').checked,
    }});
    await refreshDatasets();
    const e = state.registry.find(x => x.id === infoEditRid);
    const ds = e && liveFor(e);
    renderRegistryPanel(e, !!(ds && ds.status === 'ready'), ds?.id);
    if (!ds) renderRegistryInfo(e);
  } catch (err) { alert(err.message); }
};

function renderRegistryInfo(e) {
  // Info page for a library entry that is NOT currently open.
  const s = e.stats || {};
  const row = (k, v) => `<tr><td class="dim" style="width:110px">${esc(k)}</td><td>${v}</td></tr>`;
  $('info-content').innerHTML = '<table class="data-table">' +
    row('path', esc(e.path)) +
    row('records', s.num_rows != null ? fmt(s.num_rows) : '—') +
    row('tokens', s.tokens != null ? fmt(s.tokens) : '—') +
    row('size', s.size_mb != null ? (s.size_mb >= 1000
      ? (s.size_mb / 1000).toFixed(1) + ' GB' : s.size_mb.toFixed(1) + ' MB') : '—') +
    row('files', s.num_files != null ? fmt(s.num_files) : '—') +
    row('status', '<span class="dim">not open — stats from last registry refresh</span>') +
    '</table>';
  $('info-content').classList.remove('dim');
  $('stats-field').innerHTML = '';
  renderRegistryPanel(e, false, null);
}

/* ---------------- info / stats / meta ------------------------------------- */

async function renderInfo() {
  if (!state.current) return;
  const info = await api(`/api/datasets/${state.current}`);
  const md = info.metadata || {};
  let html = `<table class="data-table">`;
  const row = (k, v) => `<tr><td class="dim">${esc(k)}</td><td>${v}</td></tr>`;
  html += row('path', esc(info.path));
  html += row('records', fmt(md.num_rows));
  html += row('type', esc(info.info?.file_type || ''));
  html += row('size', `${(md.file_size ?? 0).toFixed(1)} MB`);
  html += row('indexed', md.has_index ? 'yes' : 'no');
  if (info.files) html += row('files', info.files.length);
  if (md.schema) {
    html += row('schema', Object.entries(md.schema).map(
      ([k, v]) => `${esc(k)}: <span class="dim">${esc(v)}</span>`).join('<br>'));
  }
  if (info.nested_fields) {
    html += row('nested fields', Object.entries(info.nested_fields).map(
      ([k, keys]) => `${esc(k)}.{${keys.map(esc).join(', ')}} ` +
      `<span class="dim">— address as ${esc(k)}.${esc(keys[0])} in rules/templates; ` +
      `from record 0 — "field summary" below is exhaustive</span>`).join('<br>'));
  }
  html += `</table>`;
  if (info.files) {
    html += `<details><summary class="dim">files (${info.files.length})</summary><table class="data-table">`;
    for (const f of info.files) html += row(esc(f.name), fmt(f.records));
    html += `</table></details>`;
  }
  $('info-content').innerHTML = html;
  $('info-content').classList.remove('dim');

  const sf = $('stats-field');
  sf.innerHTML = '<option value="">(dataset overview)</option>';
  for (const c of (md.columns || [])) sf.innerHTML += `<option>${esc(c)}</option>`;

  const entry = state.registry.find(e => e.path === info.path);
  if (entry) state.selectedLib = entry.id;
  renderRegistryPanel(entry || null, true, state.current);
}

$('btn-stats').onclick = async () => {
  const res = await api(`/api/datasets/${state.current}/stats`, {
    method: 'POST', body: { field: $('stats-field').value || null },
  });
  $('stats-out').textContent = 'Computing…\n';
  watchJob(res.job_id, $('stats-out'), (job) => {
    if (job.status === 'done')
      $('stats-out').textContent = JSON.stringify(job.result, null, 2);
  });
};

async function metaFieldSummary(rebuild) {
  try {
    const res = await api(`/api/datasets/${state.current}/meta/fields`,
                          { method: 'POST', body: { rebuild: !!rebuild } });
    $('meta-out').textContent = rebuild
      ? 'Rebuilding metadata index (full corpus pass)…\n'
      : 'Building/loading metadata index…\n';
    watchJob(res.job_id, $('meta-out'), (job) => {
      if (job.status !== 'done') return;
      const fs = job.result.fields;
      const lines = fs.map(f =>
        `${f.field.padEnd(24)} ${String(f.non_empty).padStart(12)}  ` +
        `${f.fill_pct.toFixed(1).padStart(6)}%  ${f.exact ? '' : '~'}${f.distinct} distinct` +
        (f.derived ? '  [computed]' : ''));
      const sampled = fs.length && !fs[0].exact
        ? ` · cardinality (~) from a ${fmt(fs[0].sampled)}-row sample — "top values" is exact\n`
        : '\n';
      $('meta-out').textContent =
        `${fmt(job.result.n_rows)} records${sampled}\n` + lines.join('\n');
    });
  } catch (err) { $('meta-out').textContent = err.message; }
}
$('btn-meta-fields').onclick = () => metaFieldSummary(false);
$('btn-meta-rebuild').onclick = () => {
  if (!confirm('Rebuild the metadata index from scratch? This streams EVERY ' +
               'record — one full corpus pass (can take a while on big/remote data).'))
    return;
  metaFieldSummary(true);
};

$('btn-meta-values').onclick = async () => {
  const field = $('meta-values-field').value.trim();
  if (!field) return;
  try {
    const res = await api(`/api/datasets/${state.current}/meta/values`, {
      method: 'POST', body: { field, top: 25 },
    });
    watchJob(res.job_id, $('meta-out'), (job) => {
      if (job.status !== 'done') return;
      $('meta-out').textContent = job.result.values.map(v =>
        `${String(v.count).padStart(12)}  ${v.value.slice(0, 90)}`).join('\n');
    });
  } catch (err) { $('meta-out').textContent = err.message; }
};

/* ---------------- filters (library-level rule definitions) ----------------- */

let editingFilterId = null;
let selectedFilterId = null;

const RULE_KINDS = {
  contains:   { arg: 'needle',  argPh: 'substring',        num: 'first_n', numPh: 'first N chars (opt)' },
  startswith: { arg: 'prefix',  argPh: 'prefix',           num: null },
  len_lt:     { arg: null,      num: 'value', numPh: 'chars' },
  len_gt:     { arg: null,      num: 'value', numPh: 'chars' },
  regex:      { arg: 'pattern', argPh: 'regex (search)',   num: null },
  python:     { arg: 'expr',    argPh: "python expr over rec, e.g. len(rec.get('text','')) < 600", num: null },
};

function addRuleRow(rule) {
  const row = document.createElement('div');
  row.className = 'row flt-rule';
  row.innerHTML =
    `<input class="r-name" placeholder="rule name" size="12" value="${esc(rule?.name || '')}">` +
    `<select class="r-kind">${Object.keys(RULE_KINDS).map(k =>
      `<option ${rule?.kind === k ? 'selected' : ''}>${k}</option>`).join('')}</select>` +
    `<input class="r-field" placeholder="field (blank = text; dotted ok: meta.title)" size="20" value="${esc(rule?.field || '')}">` +
    `<input class="r-arg grow" value="${esc(rule ? (rule.needle ?? rule.prefix ?? rule.pattern ?? rule.expr ?? '') : '')}" spellcheck="false">` +
    `<input class="r-num" size="9" value="${rule?.first_n ?? rule?.value ?? ''}">` +
    `<button class="mini r-del">✕</button>`;
  const sync = () => {
    const spec = RULE_KINDS[row.querySelector('.r-kind').value];
    row.querySelector('.r-arg').style.display = spec.arg ? '' : 'none';
    row.querySelector('.r-arg').placeholder = spec.argPh || '';
    row.querySelector('.r-num').style.display = spec.num ? '' : 'none';
    row.querySelector('.r-num').placeholder = spec.numPh || '';
  };
  row.querySelector('.r-kind').onchange = sync;
  row.querySelector('.r-del').onclick = () => row.remove();
  sync();
  $('flt-rules').appendChild(row);
}

function addScrubRow(s) {
  const row = document.createElement('div');
  row.className = 'row tfm-scrub';
  row.innerHTML =
    `<input class="s-name" placeholder="scrub name" size="12" value="${esc(s?.name || '')}">` +
    `<input class="s-pattern grow" placeholder="regex pattern" spellcheck="false" value="${esc(s?.pattern || '')}">` +
    `<span class="dim">→</span>` +
    `<input class="s-repl" placeholder="replacement (blank = delete match)" size="20" spellcheck="false" value="${esc(s?.replacement || '')}">` +
    `<input class="s-field" placeholder="field (blank = text)" size="12" value="${esc(s?.field || '')}">` +
    `<select class="s-mode" title="regex: full Python regex · literal: exact substring, paste anything · glob: exact substring where * = shortest stretch of anything within the line">` +
    `<option value="regex"${!s?.literal && !s?.glob ? ' selected' : ''}>regex</option>` +
    `<option value="literal"${s?.literal ? ' selected' : ''}>literal</option>` +
    `<option value="glob"${s?.glob ? ' selected' : ''}>glob (*)</option></select>` +
    `<label title="the match selects its ENTIRE line — the replacement applies to the whole line including its newline (blank replacement = delete the line)"><input type="checkbox" class="s-line"${s?.line ? ' checked' : ''}> line</label>` +
    `<label title="case-insensitive matching (any mode)"><input type="checkbox" class="s-nocase"${s?.nocase ? ' checked' : ''}> aA</label>` +
    `<label title="literal/glob: interpret \\n \\t \\\\ escapes in the pattern AND replacement (off = every character verbatim, paste-safe)"><input type="checkbox" class="s-esc"${s?.escapes ? ' checked' : ''}> \\n</label>` +
    `<button class="mini s-del">✕</button>`;
  row.querySelector('.s-del').onclick = () => row.remove();
  $('tfm-scrubs').appendChild(row);
}

function collectFilterDef() {
  const rules = [];
  for (const row of document.querySelectorAll('.flt-rule')) {
    const kind = row.querySelector('.r-kind').value;
    const spec = RULE_KINDS[kind];
    const r = { name: row.querySelector('.r-name').value.trim(), kind,
                field: row.querySelector('.r-field').value.trim() || null };
    if (spec.arg) r[spec.arg] = row.querySelector('.r-arg').value;
    if (spec.num) {
      const v = parseInt(row.querySelector('.r-num').value);
      if (!isNaN(v)) r[spec.num] = v;
    }
    rules.push(r);
  }
  return { id: editingFilterId, name: $('flt-name').value.trim(),
           notes: $('flt-notes').value.trim(), rules, scrubs: [] };
}

async function refreshFilters() {
  let res;
  try { res = await api('/api/filters'); } catch (e) { return; }
  state.filters = res.filters;
  const tb = $('filters-table').querySelector('tbody');
  tb.innerHTML = '';
  for (const f of res.filters) {
    const tr = document.createElement('tr');
    if (f.id === selectedFilterId) tr.className = 'selected';
    // Tooltip shows the EXACT stored definition with quoted values — an
    // invisible trailing space in a needle reads as a gap before the quote.
    const ruleDump = f.rules.map(r => {
      const arg = r.needle ?? r.prefix ?? r.pattern ?? r.expr;
      const num = r.kind === 'contains' ? (r.first_n ? ` first ${r.first_n}` : '')
                                        : (r.value != null ? ` ${r.value}` : '');
      return `${r.name}: ${r.kind} on ${r.field || '(text field)'}` +
             (arg != null ? ` '${arg}'` : '') + num;
    }).join('\n');
    tr.innerHTML =
      `<td>${esc(f.name)}</td>` +
      `<td class="dim" title="${esc(ruleDump)}">${f.rules.map(r => esc(r.name)).join(', ')}</td>` +
      `<td class="num">v${f.version || 1}</td>` +
      `<td class="dim">${esc((f.notes || '').slice(0, 50))}</td>` +
      `<td><button class="mini" data-act="eval">evaluate</button> ` +
      `<button class="mini" data-act="edit">edit</button> ` +
      `<button class="mini" data-act="del">delete</button></td>`;
    tr.querySelector('[data-act=eval]').onclick = () => {
      selectedFilterId = f.id;
      $('filter-eval-controls').style.display = '';
      $('filter-eval-label').textContent = `evaluate '${f.name}':`;
      refreshFilters();
      renderFilterEvals(f.id);
    };
    tr.querySelector('[data-act=edit]').onclick = () => {
      editingFilterId = f.id;
      $('filter-editor').style.display = '';
      $('flt-name').value = f.name;
      $('flt-notes').value = f.notes || '';
      $('flt-rules').innerHTML = '';
      for (const r of f.rules) addRuleRow(r);
    };
    tr.querySelector('[data-act=del]').onclick = async () => {
      if (!confirm(`Delete filter '${f.name}'? (definition only — sets it materialized remain)`)) return;
      await api(`/api/filters/${f.id}`, { method: 'DELETE' });
      if (selectedFilterId === f.id) {
        selectedFilterId = null;
        $('filter-eval-controls').style.display = 'none';
        $('filter-evals-panel').style.display = 'none';
      }
      refreshFilters();
    };
    tb.appendChild(tr);
  }
}

/* Side-by-side eval history: rows = rules (per-10k), columns = recorded runs.
   THIS is the corpus A/B view — evaluate the same filter on two datasets and
   read the columns next to each other. */
async function renderFilterEvals(fid) {
  const panel = $('filter-evals-panel');
  if (!fid) { panel.style.display = 'none'; return; }
  let res;
  try { res = await api(`/api/filters/${fid}/evals`); }
  catch (e) { panel.style.display = 'none'; return; }
  const evals = res.evals || [];
  panel.style.display = '';
  const box = $('filter-evals-box');
  if (!evals.length) {
    box.innerHTML = '<div class="dim">no evaluations recorded yet — run one above and its column lands here</div>';
    return;
  }
  const ruleNames = [];
  for (const e of evals)
    for (const r of e.rules) if (!ruleNames.includes(r.name)) ruleNames.push(r.name);
  const cols = evals.map(e =>
    `<th>${esc(e.dataset)}<div class="dim" style="font-weight:normal">` +
    `v${e.version} · ${e.sample ? fmt(e.evaluated) + ' sampled' : 'full scan'}` +
    `${e.materialized ? ' · sets' : ''} · ${fmtAge(e.created)}</div></th>`).join('');
  let html = `<table class="data-table"><thead><tr><th>per 10k</th>${cols}</tr></thead><tbody>`;
  for (const rn of ruleNames) {
    html += `<tr><td>${esc(rn)}</td>` + evals.map(e => {
      const r = e.rules.find(x => x.name === rn);
      return `<td class="num">${r != null ? r.per_10k : '—'}</td>`;
    }).join('') + '</tr>';
  }
  html += `<tr style="font-weight:bold"><td>ANY</td>` +
    evals.map(e => `<td class="num">${e.any.per_10k}</td>`).join('') + '</tr>';
  html += `<tr><td class="dim">median chars</td>` +
    evals.map(e => `<td class="num">${fmt(e.median_chars)}</td>`).join('') + '</tr>';
  html += `<tr><td class="dim">corpus records</td>` +
    evals.map(e => `<td class="num">${fmt(e.total)}</td>`).join('') + '</tr>';
  box.innerHTML = html + '</tbody></table>';
}

$('btn-flt-evals-clear').onclick = async () => {
  if (!selectedFilterId) return;
  const f = (state.filters || []).find(x => x.id === selectedFilterId);
  if (!confirm(`Clear all recorded evaluations for '${f?.name || selectedFilterId}'?`)) return;
  await api(`/api/filters/${selectedFilterId}/evals`, { method: 'DELETE' });
  renderFilterEvals(selectedFilterId);
};

$('btn-new-filter').onclick = () => {
  editingFilterId = null;
  $('filter-editor').style.display = '';
  $('flt-name').value = ''; $('flt-notes').value = '';
  $('flt-rules').innerHTML = '';
  addRuleRow();
};
$('btn-flt-add-rule').onclick = () => addRuleRow();
$('btn-flt-cancel').onclick = () => { $('filter-editor').style.display = 'none'; };
$('btn-flt-save').onclick = async () => {
  const def = collectFilterDef();
  if (!def.name) return alert('Filter needs a name');
  try {
    await api('/api/filters', { method: 'POST', body: def });
    $('filter-editor').style.display = 'none';
    refreshFilters();
  } catch (err) { alert(err.message); }
};

$('btn-flt-eval').onclick = async () => {
  if (!state.current) return alert('Open a dataset first');
  if (!selectedFilterId) return;
  const materialize = $('flt-materialize').checked;
  const sample = materialize ? null : (parseInt($('flt-sample').value) || 10000);
  try {
    const r = await api(
      `/api/datasets/${state.current}/filters/${selectedFilterId}/evaluate`,
      { method: 'POST', body: { sample, materialize } });
    $('filter-eval-result').innerHTML = '<div class="dim">evaluating…</div>';
    watchJob(r.job_id, null, (job) => {
      if (job.status !== 'done') {
        $('filter-eval-result').innerHTML =
          `<div class="status-error">${esc(job.error || job.status)}</div>`;
        return;
      }
      const R = job.result;
      let html = `<div style="margin-top:6px">` +
        `<b>${esc(R.filter)}</b> v${R.version} on ${fmt(R.evaluated)} record(s)` +
        `${R.sampled ? ' (sampled)' : ' (full scan)'} · median ${fmt(R.median_chars)} chars` +
        `</div><table class="data-table"><thead><tr><th>Rule</th><th>Count</th>` +
        `<th>Per 10k</th><th>%</th></tr></thead><tbody>`;
      for (const r2 of R.rules.concat([{ name: 'ANY', ...R.any }])) {
        html += `<tr${r2.name === 'ANY' ? ' style="font-weight:bold"' : ''}>` +
          `<td>${esc(r2.name)}</td><td class="num">${fmt(r2.count)}</td>` +
          `<td class="num">${r2.per_10k}</td><td class="num">${r2.pct}%</td></tr>`;
      }
      html += '</tbody></table>';
      if (R.sets.length) {
        html += `<div class="dim">materialized sets: ${R.sets.map(esc).join(', ')} — browse them from the sidebar</div>`;
        refreshSets();
      }
      $('filter-eval-result').innerHTML = html;
      renderFilterEvals(selectedFilterId);
    }, 0, $('filter-eval-progress'));
  } catch (err) { alert(err.message); }
};

/* ---------------- transforms (the fourth noun: rewrite chains) ------------- */

let editingTransformId = null;
let selectedTransformId = null;

async function refreshTransforms() {
  let res;
  try { res = await api('/api/transforms'); } catch (e) { return; }
  state.transforms = res.transforms;
  const tb = $('transforms-table').querySelector('tbody');
  tb.innerHTML = '';
  for (const t of res.transforms) {
    const tr = document.createElement('tr');
    if (t.id === selectedTransformId) tr.className = 'selected';
    const dump = t.scrubs.map(s =>
      `${s.name}: ${s.literal ? `'${s.pattern}' (literal)`
                  : s.glob ? `«${s.pattern}» (glob)` : `/${s.pattern}/`}` +
      `${s.line ? ' (whole line)' : ''}${s.nocase ? ' (aA)' : ''}${s.escapes ? ' (\\n)' : ''}` +
      ` → '${s.replacement || ''}' on ${s.field || '(text field)'}`
    ).concat(t.fixpoint ? ['(chain runs to fixpoint)'] : []).join('\n');
    tr.innerHTML =
      `<td>${esc(t.name)}</td>` +
      `<td class="dim" title="${esc(dump)}">${t.scrubs.map(s => esc(s.name)).join(' → ')}` +
      `${t.fixpoint ? ' <span class="dim">⟳ fixpoint</span>' : ''}</td>` +
      `<td class="num">v${t.version || 1}</td>` +
      `<td class="dim">${esc((t.notes || '').slice(0, 50))}</td>` +
      `<td><button class="mini" data-act="preview">preview</button> ` +
      `<button class="mini" data-act="edit">edit</button> ` +
      `<button class="mini" data-act="del">delete</button></td>`;
    tr.querySelector('[data-act=preview]').onclick = () => {
      selectedTransformId = t.id;
      $('transform-preview-controls').style.display = '';
      $('transform-preview-label').textContent = `preview '${t.name}':`;
      populateTfmFilterCtx();
      refreshTransforms();
      renderTransformEvals(t.id);
    };
    tr.querySelector('[data-act=edit]').onclick = () => {
      editingTransformId = t.id;
      $('transform-editor').style.display = '';
      $('tfm-name').value = t.name;
      $('tfm-notes').value = t.notes || '';
      $('tfm-fixpoint').checked = !!t.fixpoint;
      $('tfm-scrubs').innerHTML = '';
      for (const s of t.scrubs) addScrubRow(s);
    };
    tr.querySelector('[data-act=del]').onclick = async () => {
      if (!confirm(`Delete transform '${t.name}'?`)) return;
      await api(`/api/transforms/${t.id}`, { method: 'DELETE' });
      if (selectedTransformId === t.id) {
        selectedTransformId = null;
        $('transform-preview-controls').style.display = 'none';
        $('transform-evals-panel').style.display = 'none';
      }
      refreshTransforms();
    };
    tb.appendChild(tr);
  }
  populateTransformPickers();
}

function populateTransformPickers() {
  for (const id of ['tok-transform', 'export-transform']) {
    const sel = $(id);
    if (!sel) continue;
    const keep = sel.value;
    sel.innerHTML = '<option value="">no transform</option>';
    for (const t of (state.transforms || []))
      sel.innerHTML += `<option value="${esc(t.id)}">${esc(t.name)} v${t.version || 1}</option>`;
    sel.value = keep;
    if (sel.value !== keep) sel.value = '';
  }
}

function populateTfmFilterCtx() {
  const sel = $('tfm-filter-ctx');
  const keep = sel.value;
  sel.innerHTML = '<option value="">all records</option>';
  for (const f of (state.filters || []))
    sel.innerHTML += `<option value="${esc(f.id)}">survivors of ${esc(f.name)}</option>`;
  sel.value = keep;
  if (sel.value !== keep) sel.value = '';
}

$('btn-new-transform').onclick = () => {
  editingTransformId = null;
  $('transform-editor').style.display = '';
  $('tfm-name').value = ''; $('tfm-notes').value = '';
  $('tfm-fixpoint').checked = false;
  $('tfm-scrubs').innerHTML = '';
  addScrubRow();
};
$('btn-tfm-add-scrub').onclick = () => addScrubRow();
$('btn-tfm-help').onclick = () => {
  const h = $('tfm-help');
  h.style.display = h.style.display === 'none' ? '' : 'none';
};
$('btn-tfm-cancel').onclick = () => { $('transform-editor').style.display = 'none'; };
$('btn-tfm-save').onclick = async () => {
  const scrubs = [];
  for (const row of document.querySelectorAll('.tfm-scrub')) {
    const name = row.querySelector('.s-name').value.trim();
    const pattern = row.querySelector('.s-pattern').value;   // NO trim: spaces matter
    if (!name && !pattern) continue;                          // stray empty row
    const mode = row.querySelector('.s-mode').value;
    scrubs.push({ name, pattern,
                  replacement: row.querySelector('.s-repl').value,
                  field: row.querySelector('.s-field').value.trim() || null,
                  literal: mode === 'literal',
                  glob: mode === 'glob',
                  line: row.querySelector('.s-line').checked,
                  nocase: row.querySelector('.s-nocase').checked,
                  escapes: row.querySelector('.s-esc').checked });
  }
  const def = { id: editingTransformId, name: $('tfm-name').value.trim(),
                notes: $('tfm-notes').value.trim(), scrubs,
                fixpoint: $('tfm-fixpoint').checked };
  if (!def.name) return alert('Transform needs a name');
  try {
    await api('/api/transforms', { method: 'POST', body: def });
    $('transform-editor').style.display = 'none';
    refreshTransforms();
  } catch (err) { alert(err.message); }
};

$('btn-tfm-preview').onclick = async () => {
  if (!state.current) return alert('Open a dataset first');
  if (!selectedTransformId) return;
  try {
    const r = await api(
      `/api/datasets/${state.current}/transforms/${selectedTransformId}/preview`,
      { method: 'POST', body: { sample: parseInt($('tfm-sample').value) || 10000,
                                filter_id: $('tfm-filter-ctx').value || null } });
    $('transform-preview-result').innerHTML = '<div class="dim">previewing…</div>';
    watchJob(r.job_id, null, (job) => {
      if (job.status !== 'done') {
        $('transform-preview-result').innerHTML =
          `<div class="status-error">${esc(job.error || job.status)}</div>`;
        return;
      }
      const R = job.result;
      let html = `<div style="margin-top:6px"><b>${esc(R.transform)}</b> v${R.version} ` +
        `preview — ${fmt(R.survivors)} record(s)` +
        (R.filter ? ` (survivors of ${esc(R.filter)})` : '') +
        ` of ${fmt(R.sampled)} sampled · <b>${fmt(R.changed)}</b> changed (${R.changed_per_10k}/10k)` +
        (R.fixpoint ? ` · fixpoint: max ${R.max_passes} pass(es)` +
          (R.nonconverged ? ` — <span class="status-error">${fmt(R.nonconverged)} did NOT converge (chain oscillates?)</span>`
                          : ', all converged') : '') +
        `</div>` +
        `<table class="data-table"><thead><tr><th>Scrub</th><th>Docs</th><th>Per 10k</th>` +
        `<th>Subs</th><th>Chars removed</th></tr></thead><tbody>`;
      for (const s of R.scrubs)
        html += `<tr><td>${esc(s.name)}</td><td class="num">${fmt(s.docs)}</td>` +
          `<td class="num">${s.docs_per_10k}</td><td class="num">${fmt(s.subs)}</td>` +
          `<td class="num">${fmt(s.chars_removed)}</td></tr>`;
      html += '</tbody></table>';
      if (R.examples?.length) {
        html += `<div class="row" style="margin-top:6px">` +
          `<span class="dim">every change, per scrub — <span class="diff-del">removed</span> <span class="diff-ins">inserted</span>, with context. Narrow by scrub:</span>` +
          `<span id="tfm-ex-pills"></span></div>` +
          `<div class="row">` +
          `<button id="btn-tfm-ex-prev" class="mini">‹ prev</button>` +
          `<span id="tfm-ex-count" class="dim"></span>` +
          `<button id="btn-tfm-ex-next" class="mini">next ›</button></div>` +
          `<div id="tfm-ex-body"></div>`;
      }
      $('transform-preview-result').innerHTML = html;
      if (R.examples?.length) {
        const recCache = {};
        for (const ex of R.examples) recCache[ex.index] = ex;
        tfmDiffState = {
          indices: R.changed_indices || [],
          masks: R.changed_masks || [],
          scrubNames: R.scrubs.map(s => s.name),   // chain order == bit order
          enabled: new Set(R.scrubs.map(s => s.name)),
          recCache, page: 0,
        };
        const pills = $('tfm-ex-pills');
        for (const s of R.scrubs) {
          const b = document.createElement('button');
          b.className = 'mini pill';
          b.textContent = `${s.name} ${fmt(s.docs)}`;
          b.title = `records touched by '${s.name}' — click to toggle; the record list narrows to records hit by any enabled scrub`;
          b.onclick = () => {
            const en = tfmDiffState.enabled;
            if (en.has(s.name)) en.delete(s.name); else en.add(s.name);
            b.classList.toggle('off', !en.has(s.name));
            tfmDiffState.page = 0;
            renderTfmExPage();
          };
          pills.appendChild(b);
        }
        $('btn-tfm-ex-prev').onclick = () => tfmExPage(-1);
        $('btn-tfm-ex-next').onclick = () => tfmExPage(1);
        renderTfmExPage();
      }
      renderTransformEvals(selectedTransformId);
    }, 0, $('transform-preview-progress'));
  } catch (err) { alert(err.message); }
};

/* preview example paging + scrub pills: spot-check MANY changed records and
   narrow to specific scrubs — masks came back with the preview, so filtering
   is client-side set algebra; pages are fetched on demand and cached
   per RECORD (so re-filtering never refetches) */
let tfmDiffState = null;
const TFM_EX_PAGE = 8;

function tfmEnabledBits(st) {
  let bits = 0;
  st.scrubNames.forEach((n, i) => { if (st.enabled.has(n)) bits |= (1 << i); });
  return bits;
}

function tfmFilteredIndices(st) {
  const bits = tfmEnabledBits(st);
  const out = [];
  for (let i = 0; i < st.indices.length; i++)
    if (st.masks.length === 0 || (st.masks[i] & bits)) out.push(st.indices[i]);
  return out;
}

function tfmExamplesHtml(examples, enabled) {
  let html = '';
  for (const ex of examples) {
    const show = ex.regions.filter(rg => enabled.has(rg.scrub));
    const hidden = ex.regions.length - show.length;
    html += `<pre class="log-pane" style="margin:4px 0"><span class="dim">— record #${fmt(ex.index)} · ${esc(ex.field)} · ${show.length} change region(s)` +
      (hidden ? ` (+${hidden} from disabled scrubs hidden)` : '') + ` — </span>` +
      `<button class="mini tfm-ex-expand" data-gi="${ex.index}" ` +
      `title="full modified content — the whole document with every change highlighted (all scrubs, regardless of pills)">` +
      `${ex._expanded ? 'collapse' : 'expand'}</button>\n`;
    if (ex._expanded && ex.segments) {
      if (!ex.segments_highlighted)
        html += `<div class="dim">(document too large to highlight — showing plain modified content)</div>`;
      html += `<div class="diff-region">` + ex.segments.map(sg =>
        sg.t === 'del' ? `<span class="diff-del">${esc(sg.s)}</span>` :
        sg.t === 'ins' ? `<span class="diff-ins">${esc(sg.s)}</span>` :
        esc(sg.s)).join('') + `</div>`;
    } else {
      for (const rg of show)
        html += `<div class="diff-region"><span class="dim">[${esc(rg.scrub)}] …</span>${esc(rg.ctx_before)}` +
          `<span class="diff-del">${esc(rg.removed)}</span>` +
          (rg.added ? `<span class="diff-ins">${esc(rg.added)}</span>` : '') +
          `${esc(rg.ctx_after)}<span class="dim">…</span></div>`;
    }
    html += `</pre>`;
  }
  return html || '<div class="dim">no records match the enabled scrubs</div>';
}

async function tfmToggleExpand(gi) {
  const st = tfmDiffState;
  const ex = st?.recCache[gi];
  if (!ex) return;
  if (!ex._expanded && !ex.segments) {
    // acknowledge the click NOW — record fetch + full diff can take seconds
    // (SMB, parquet decode, big docs) and a silent button reads as broken
    const btn = document.querySelector(`#tfm-ex-body .tfm-ex-expand[data-gi="${gi}"]`);
    if (btn) { btn.disabled = true; btn.textContent = 'loading…'; }
    try {
      const r = await api(
        `/api/datasets/${state.current}/transforms/${selectedTransformId}/diff`,
        { method: 'POST', body: { indices: [gi], full: true } });
      const full = r.examples.find(e => e.index === gi);
      if (full) st.recCache[gi] = Object.assign(ex, full);
    } catch (err) {
      if (btn) { btn.disabled = false; btn.textContent = 'expand'; }
      return alert(err.message);
    }
  }
  st.recCache[gi]._expanded = !st.recCache[gi]._expanded;
  renderTfmExPage();
}

async function renderTfmExPage() {
  const st = tfmDiffState;
  if (!st) return;
  const filtered = tfmFilteredIndices(st);
  const nPages = Math.max(1, Math.ceil(filtered.length / TFM_EX_PAGE));
  st.page = Math.min(st.page, nPages - 1);
  const slice = filtered.slice(st.page * TFM_EX_PAGE, (st.page + 1) * TFM_EX_PAGE);
  const missing = slice.filter(gi => !st.recCache[gi]);
  if (missing.length) {
    // acknowledge before the fetch: slow pages must not read as a dead click
    $('tfm-ex-count').textContent = `loading ${missing.length} record(s)…`;
    $('btn-tfm-ex-prev').disabled = true;
    $('btn-tfm-ex-next').disabled = true;
    try {
      const r = await api(
        `/api/datasets/${state.current}/transforms/${selectedTransformId}/diff`,
        { method: 'POST', body: { indices: missing } });
      for (const ex of r.examples) st.recCache[ex.index] = ex;
    } catch (err) {
      $('tfm-ex-count').textContent = 'page load failed';
      $('btn-tfm-ex-prev').disabled = false;
      $('btn-tfm-ex-next').disabled = false;
      return alert(err.message);
    }
  }
  $('tfm-ex-body').innerHTML =
    tfmExamplesHtml(slice.map(gi => st.recCache[gi]).filter(Boolean), st.enabled);
  for (const b of document.querySelectorAll('#tfm-ex-body .tfm-ex-expand'))
    b.onclick = () => tfmToggleExpand(parseInt(b.dataset.gi));
  $('tfm-ex-count').textContent =
    `${fmt(filtered.length)} record(s) match enabled scrubs · page ${st.page + 1}/${fmt(nPages)}`;
  $('btn-tfm-ex-prev').disabled = st.page === 0;
  $('btn-tfm-ex-next').disabled = st.page >= nPages - 1;
}

function tfmExPage(delta) {
  const st = tfmDiffState;
  if (!st) return;
  const nPages = Math.max(1, Math.ceil(tfmFilteredIndices(st).length / TFM_EX_PAGE));
  st.page = Math.min(Math.max(st.page + delta, 0), nPages - 1);
  renderTfmExPage();
}

async function renderTransformEvals(tid) {
  const panel = $('transform-evals-panel');
  if (!tid) { panel.style.display = 'none'; return; }
  let res;
  try { res = await api(`/api/transforms/${tid}/evals`); }
  catch (e) { panel.style.display = 'none'; return; }
  const evals = res.evals || [];
  panel.style.display = '';
  const box = $('transform-evals-box');
  if (!evals.length) {
    box.innerHTML = '<div class="dim">no previews recorded yet — run one above and its column lands here</div>';
    return;
  }
  const scrubNames = [];
  for (const e of evals)
    for (const s of e.scrubs) if (!scrubNames.includes(s.name)) scrubNames.push(s.name);
  const cols = evals.map(e =>
    `<th>${esc(e.dataset)}<div class="dim" style="font-weight:normal">` +
    `v${e.version}${e.filter ? ` · ∖ ${esc(e.filter)}` : ''} · ` +
    `${fmt(e.sampled)} sampled · ${fmtAge(e.created)}</div></th>`).join('');
  let html = `<table class="data-table"><thead><tr><th>docs per 10k</th>${cols}</tr></thead><tbody>`;
  for (const sn of scrubNames) {
    html += `<tr><td>${esc(sn)}</td>` + evals.map(e => {
      const s = e.scrubs.find(x => x.name === sn);
      return `<td class="num">${s != null ? s.docs_per_10k : '—'}</td>`;
    }).join('') + '</tr>';
  }
  html += `<tr style="font-weight:bold"><td>ANY CHANGE</td>` +
    evals.map(e => `<td class="num">${e.changed_per_10k}</td>`).join('') + '</tr>';
  box.innerHTML = html + '</tbody></table>';
}

/* ---------------- tokenize tab --------------------------------------------- */

function currentLibraryEntry() {
  const ds = state.datasets.find(d => d.id === state.current);
  return ds ? state.registry.find(e => e.path === ds.path) : null;
}

function renderTokenizeTab() {
  const ds = state.datasets.find(d => d.id === state.current && d.status === 'ready');
  if (!ds) {
    $('tok-source').textContent = 'Open a dataset to tokenize it.';
    $('tok-versions-panel').style.display = 'none';
    return;
  }
  const entry = currentLibraryEntry();
  const name = entry?.name || ds.path.split(/[\\/]/).pop();
  $('tok-source').innerHTML =
    `Source: <b>${esc(name)}</b> · ${fmt(ds.info?.num_rows)} records` +
    (entry ? '' : ' <span class="dim">(unregistered — lineage will be unlinked)</span>');
  $('tok-source').classList.remove('dim');

  // field dropdown from the dataset's actual columns
  const sel = $('tok-field');
  const keep = sel.value;
  sel.innerHTML = '';
  for (const c of (ds.info?.columns || [])) sel.innerHTML += `<option>${esc(c)}</option>`;
  if (keep) sel.value = keep;
  // fresh page: prefer 'text' (pre_tokenize's own default) over column #0 —
  // a silent field='title' run tokenizes 5 tokens/doc without complaint
  else if ([...sel.options].some(o => o.value === 'text')) sel.value = 'text';

  // sensible suggestions, only when untouched
  if (!$('tok-label').dataset.touched) {
    $('tok-label').value = (name || 'data').toLowerCase()
      .replace(/\.[^.]*$/, '').replace(/[^a-z0-9_-]+/g, '_');
  }
  suggestTokOut();

  // tokenized versions of THIS dataset (the linkage list)
  const kids = entry ? state.registry.filter(
    e => e.derived_from === entry.id && e.recipe?.kind === 'tokenize') : [];
  $('tok-versions-panel').style.display = kids.length ? '' : 'none';
  const tb = $('tok-versions').querySelector('tbody');
  tb.innerHTML = '';
  for (const k of kids) {
    const s = k.stats || {}, rp = k.recipe?.params || {};
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td>${esc(k.name)}</td><td>${esc(rp.tokenizer || '?')}` +
      `${rp.tokenizer_path ? ` <span class="dim">${esc(rp.tokenizer_path)}</span>` : ''}</td>` +
      `<td>${esc(rp.label || '')}</td>` +
      `<td class="num">${s.tokens != null ? fmt(s.tokens) : '—'}</td>` +
      `<td class="num">${s.num_files != null ? fmt(s.num_files) : '—'}</td>` +
      `<td class="dim">${fmtAge(k.recipe?.created || k.created)}</td>` +
      `<td><button class="mini" data-act="open">open</button> ` +
      `<button class="mini" data-act="clone" title="prefill the form with this run's exact settings">clone settings</button></td>`;
    tr.querySelector('[data-act=open]').onclick = () => openRegistryEntry(k.id);
    tr.querySelector('[data-act=clone]').onclick = () => cloneTokenizeRecipe(rp);
    tb.appendChild(tr);
  }

  populateTokFilter();
  tokComposeChanged();
}

function suggestTokOut() {
  if ($('tok-out').dataset.touched) return;
  const ds = state.datasets.find(d => d.id === state.current);
  if (!ds) return;
  const kind = $('tok-kind').value || 'tok';
  const sep = ds.path.includes('\\') ? '\\' : '/';
  const parts = ds.path.replace(/[\\/]+$/, '').split(/[\\/]/);
  let name = parts[parts.length - 1]
    .replace(/\.zst$/i, '').replace(/\.(jsonl|json|parquet)$/i, '');
  if (tokCompositionActive()) name += '_cleaned';
  const root = state.config?.tokenized_root;
  const si = parts.lastIndexOf('source');
  let out;
  if (root) {
    out = [root.replace(/[\\/]+$/, ''), kind, name].join(root.includes('\\') ? '\\' : '/');
  } else if (si > 0) {
    // .../source/<name>  ->  .../tokenized/<tokenizer>/<name>
    out = parts.slice(0, si).concat(['tokenized', kind, name]).join(sep);
  } else {
    let base = ds.path.replace(/[\\/]+$/, '').replace(/\.[^\\/.]*(\.zst)?$/, '');
    if (tokCompositionActive()) base += '_cleaned';
    out = `${base}-${kind}-tokens`;
  }
  $('tok-out').value = out;
  if (!$('tok-register').dataset.touched) {
    const entry = currentLibraryEntry();
    let base = entry?.name || name;
    if (tokCompositionActive() && !base.endsWith('_cleaned')) base += '_cleaned';
    $('tok-register').value = `${base} · ${kind}`;
  }
}

function cloneTokenizeRecipe(rp) {
  const set = (id, v) => { if (v != null && v !== '') { $(id).value = v; $(id).dataset.touched = '1'; } };
  set('tok-field', rp.field); set('tok-template', rp.template || '');
  set('tok-kind', rp.tokenizer); set('tok-path', rp.tokenizer_path || '');
  set('tok-shard-size', rp.shard_size); set('tok-val-holdout', rp.val_holdout);
  set('tok-coprime', rp.coprime); set('tok-min-shard', rp.min_shard);
  set('tok-label', rp.label); $('tok-format').value = rp.input_format || 'auto';
  $('tok-dtype').value = rp.dtype || 'auto';
  $('tok-filter-english').checked = !!rp.filter_english;
  $('tok-legacy').checked = !!rp.legacy_river;
  set('tok-workers', rp.workers || ''); set('tok-extra', rp.extra_args || '');
  set('tok-sb-nondict', rp.max_non_dict_ratio); set('tok-sb-alpha', rp.min_alpha_ratio);
  set('tok-sb-chars', rp.min_char_count); set('tok-sb-rep', rp.max_repetition_ratio);
  $('tok-sb-matter').checked = !!rp.include_matter;
  tokFormatChanged();
  alert('Settings cloned — adjust the output dir/name and go.');
}

/* ---- tokenize composition: sets/filter -> cleaned intermediate -> tokenize */

const tokSel = (id) => [...$(id).selectedOptions].map(o => o.value);

function tokCompositionActive() {
  return !!(tokSel('tok-include').length || tokSel('tok-exclude').length ||
            $('tok-filter').value || $('tok-transform').value);
}

function populateTokFilter() {
  const sel = $('tok-filter');
  const keep = sel.value;
  sel.innerHTML = '<option value="">no filter</option>';
  for (const f of (state.filters || []))
    sel.innerHTML += `<option value="${esc(f.id)}">${esc(f.name)} v${f.version || 1}</option>`;
  sel.value = keep;
  if (sel.value !== keep) sel.value = '';   // selected filter was deleted
}

function suggestTokIntermediate() {
  const ds = state.datasets.find(d => d.id === state.current);
  if (!ds) return;
  const sep = ds.path.includes('\\') ? '\\' : '/';
  const parts = ds.path.replace(/[\\/]+$/, '').split(/[\\/]/);
  const leaf = parts.pop()
    .replace(/\.zst$/i, '').replace(/\.(jsonl|json|parquet)$/i, '');
  if (!$('tok-int-path').dataset.touched)
    $('tok-int-path').value = parts.concat([`${leaf}_cleaned.jsonl`]).join(sep);
  if (!$('tok-int-register').dataset.touched) {
    const entry = currentLibraryEntry();
    $('tok-int-register').value = `${entry?.name || leaf}_cleaned`;
  }
}

// The filter's ANY set, but only if its RULES still match (r= stamp; scrub
// edits bump the version but keep drop sets fresh). Legacy sets without a
// stamp fall back to exact-version match. Stale/absent -> materialize on run.
function tokFilterAnySet() {
  const fid = $('tok-filter').value;
  if (!fid) return null;
  const f = (state.filters || []).find(x => x.id === fid);
  if (!f) return null;
  const s = (state.sets || []).find(x => x.name === `${f.name}.any`);
  if (!s) return null;
  const q = s.query || '';
  return (q.includes(`r=${f.rules_hash}`) || q.endsWith(`v${f.version || 1}`)) ? s : null;
}

let tokPlanTimer = null;
function tokComposeChanged() {
  if (!$('tok-int-mode')) return;   // mixed static versions: fail soft
  const active = tokCompositionActive();
  $('tok-int-row').style.display = active ? '' : 'none';
  const fileMode = $('tok-int-mode').value === 'file';
  for (const id of ['tok-int-path', 'tok-int-register', 'tok-int-keep-label'])
    $(id).style.display = fileMode ? '' : 'none';

  const fid = $('tok-filter').value;
  const f = fid ? (state.filters || []).find(x => x.id === fid) : null;
  const anySet = tokFilterAnySet();
  $('tok-filter-note').textContent = !f ? '' : (anySet
    ? `${f.name}.any: ${fmt(anySet.count)} record(s) will be dropped (sets already materialized)`
    : `v${f.version || 1} sets will be materialized on run (full scan; drops stay browsable)`);
  const tid = $('tok-transform').value;
  const t = tid ? (state.transforms || []).find(x => x.id === tid) : null;
  $('tok-transform-note').textContent = !t ? '' :
    `${t.scrubs.length} scrub(s)${t.fixpoint ? ' · fixpoint' : ''} — applied ` +
    `${$('tok-int-mode').value === 'file' ? 'while writing the intermediate' : 'in-stream'}`;

  if (active && fileMode) suggestTokIntermediate();
  if (!$('tok-out').dataset.touched) suggestTokOut();

  if (!active || !state.current) { $('tok-compose-plan').textContent = ''; return; }
  if (tokPlanTimer) clearTimeout(tokPlanTimer);
  tokPlanTimer = setTimeout(async () => {
    try {
      const exclude = tokSel('tok-exclude').concat(anySet ? [anySet.name] : []);
      const r = await api(`/api/datasets/${state.current}/export/plan`, {
        method: 'POST',
        body: { include: tokSel('tok-include'), exclude },
      });
      const kind = $('tok-int-mode').value === 'file' ? 'intermediate' : 'streams';
      $('tok-compose-plan').textContent =
        `plan: keeps ${fmt(r.kept_records)} of ${fmt(r.total_records)} records ` +
        `(${r.pct.toFixed(1)}%) · drops ${fmt(r.dropped_records)}` +
        (r.bytes != null ? ` · ${kind} ≈ ${fmtBytes(r.bytes)}` : '') +
        (f && !anySet ? ' · + filter drops (counted once materialized)' : '');
    } catch (err) {
      $('tok-compose-plan').textContent = 'plan failed: ' + err.message;
    }
  }, 250);
}

$('tok-include').onchange = tokComposeChanged;
$('tok-exclude').onchange = tokComposeChanged;
$('tok-filter').onchange = tokComposeChanged;
$('tok-transform').onchange = tokComposeChanged;
$('tok-int-mode').onchange = tokComposeChanged;
for (const id of ['tok-int-path', 'tok-int-register'])
  $(id).oninput = function () { this.dataset.touched = '1'; };

function collectTokenizeParams() {
  const num = (id, dflt) => parseInt($(id).value) || dflt;
  return {
    out_dir: $('tok-out').value.trim(),
    input_format: $('tok-format').value,
    field: $('tok-template').value.trim() ? null : ($('tok-field').value || 'text'),
    template: $('tok-template').value.trim() || null,
    tokenizer: $('tok-kind').value,
    tokenizer_path: $('tok-path').value.trim() || null,
    shard_size: num('tok-shard-size', 100000000),
    val_holdout: num('tok-val-holdout', 50000000),
    coprime: num('tok-coprime', 6),
    min_shard: num('tok-min-shard', 5000000),
    dtype: $('tok-dtype').value,
    label: $('tok-label').value.trim() || 'data',
    workers: parseInt($('tok-workers').value) || null,
    legacy_river: $('tok-legacy').checked,
    filter_english: $('tok-filter-english').checked,
    lang_threshold: parseFloat($('tok-lang-threshold').value) || 0.8,
    lang_backend: $('tok-lang-backend').value,
    max_non_dict_ratio: parseFloat($('tok-sb-nondict').value) || 0.5,
    min_alpha_ratio: parseFloat($('tok-sb-alpha').value) || 0.6,
    min_char_count: num('tok-sb-chars', 150),
    max_repetition_ratio: parseFloat($('tok-sb-rep').value) || 0.05,
    include_matter: $('tok-sb-matter').checked,
    extra_args: $('tok-extra').value.trim() || null,
    register_as: $('tok-register').value.trim() || null,
  };
}

function tokFormatChanged() {
  const f = $('tok-format').value;
  $('tok-sb-opts').style.display =
    (f === 'scanned-book-jsonl' || f === 'batch') ? '' : 'none';
}
$('tok-format').onchange = tokFormatChanged;
$('tok-filter-english').onchange = () => {
  $('tok-lang-opts').style.display = $('tok-filter-english').checked ? '' : 'none';
};
$('tok-kind').onchange = () => { $('tok-out').dataset.touched = ''; suggestTokOut(); };
for (const id of ['tok-out', 'tok-register', 'tok-label'])
  $(id).oninput = function () { this.dataset.touched = '1'; };

$('btn-tok-preview').onclick = async () => {
  if (!state.current) return alert('Open a dataset first');
  try {
    const r = await api(`/api/datasets/${state.current}/tokenize/preview`, {
      method: 'POST',
      body: { field: $('tok-field').value || null,
              template: $('tok-template').value.trim() || null, n: 3 },
    });
    const box = $('tok-preview');
    box.style.display = '';
    box.textContent = r.samples.map(s => s.ok
      ? `— record #${s.index} (${fmt(s.chars)} chars) —\n${s.preview}`
      : `— record #${s.index} — EXTRACTION FAILED: ${s.error}`).join('\n\n');
  } catch (err) { alert(err.message); }
};

$('btn-tok-preflight').onclick = async () => {
  if (!state.current) return alert('Open a dataset first');
  const p = collectTokenizeParams();
  if (!p.tokenizer) return alert('Choose a tokenizer');
  if (!p.out_dir) return alert('Set an output directory');
  try {
    const r = await api(`/api/datasets/${state.current}/tokenize/preflight`,
                        { method: 'POST', body: p });
    const rep = $('tok-preflight-report');
    rep.innerHTML = '<div class="dim">preflight running…</div>';
    watchJob(r.job_id, null, (job) => {
      if (job.status !== 'done') {
        rep.innerHTML = `<div class="status-error">preflight failed: ${esc(job.error)}</div>`;
        return;
      }
      const R = job.result;
      rep.innerHTML = R.checks.map(c =>
        `<div class="${c.ok ? 'status-done' : 'status-error'}">` +
        `${c.ok ? '✓' : '✗'} ${esc(c.name)}: ${esc(c.detail)}</div>`).join('') +
        `<div class="dim" style="margin-top:6px">${esc((R.argv || []).slice(2).join(' '))}</div>`;
    });
  } catch (err) { alert(err.message); }
};

$('btn-tok-start').onclick = async () => {
  if (!state.current) return alert('Open a dataset first');
  const p = collectTokenizeParams();
  if (!p.tokenizer) return alert('Choose a tokenizer');
  if (!p.out_dir) return alert('Set an output directory');
  const composed = tokCompositionActive();
  try {
    let r;
    if (composed) {
      const fileMode = $('tok-int-mode').value === 'file';
      if (fileMode && !$('tok-int-path').value.trim())
        return alert('Set the cleaned-intermediate path');
      r = await api(`/api/datasets/${state.current}/tokenize/composed`, {
        method: 'POST',
        body: {
          include_sets: tokSel('tok-include'),
          exclude_sets: tokSel('tok-exclude'),
          filter_id: $('tok-filter').value || null,
          transform_id: $('tok-transform').value || null,
          intermediate_mode: $('tok-int-mode').value,
          intermediate_path: fileMode ? $('tok-int-path').value.trim() : null,
          intermediate_register_as: fileMode ? ($('tok-int-register').value.trim() || null) : null,
          keep_intermediate: $('tok-int-keep').checked,
          tokenize: p,
        },
      });
    } else {
      r = await api(`/api/datasets/${state.current}/tokenize`,
                    { method: 'POST', body: p });
    }
    selectJob(r.job_id); showTab('jobs');
    watchJob(r.job_id, null, (job) => {
      refreshDatasets();
      if (composed) refreshSets();     // materialize may have added filter sets
      if (job.status === 'done' && job.result?.registered) {
        const R = job.result;
        alert(`Tokenization complete: ${fmt(R.tokens)} tokens, ` +
              `registered as '${p.register_as}'.` +
              (composed ? `\nComposition kept ${fmt(R.kept_records)} record(s), ` +
                          `dropped ${fmt(R.dropped_records)}.` : ''));
      }
    });
  } catch (err) { alert(err.message); }
};

/* ---------------- jobs ---------------------------------------------------- */

async function refreshJobs() {
  const jobs = await api('/api/jobs');
  const running = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;
  $('jobs-badge').textContent = running ? `(${running})` : '';
  const tb = $('jobs-table').querySelector('tbody');
  tb.innerHTML = '';
  for (const j of jobs) {
    const tr = document.createElement('tr');
    if (j.id === state.selectedJob) tr.className = 'selected';
    let dur = '';
    if (j.finished) {
      dur = `${(j.finished - j.created).toFixed(1)}s`;
    } else if (j.status === 'running') {
      const st = (j.progress?.main?.eta_s != null) ? j.progress.main
               : (j.progress?.stage?.eta_s != null) ? j.progress.stage : null;
      dur = st ? `eta ${fmtEta(st.eta_s)}`
               : `${fmtEta(Date.now() / 1000 - j.created)}…`;   // elapsed, no est. yet
    }
    let statusText = j.status;
    if (j.status === 'running') {
      const pct = progressPct(j.progress);
      if (pct != null) statusText += ` ${pct.toFixed(0)}%`;
    }
    let cancellable = j.status === 'running' || j.status === 'queued';
    let statusCls = j.status;
    if (cancellable && j.cancel_requested) {
      statusText = 'cancelling…';       // acknowledged: stop the ticking clock
      statusCls = 'cancelled';
      dur = '';
      cancellable = false;
    }
    tr.innerHTML = `<td>${j.id}</td><td>${esc(j.kind)}</td><td>${esc(j.dataset_id || '')}</td>` +
      `<td class="status-${statusCls}">${statusText}</td><td class="num">${dur}</td>` +
      `<td>${cancellable ? '<button class="mini job-cancel" title="cancel job">✕</button>' : ''}</td>`;
    tr.onclick = () => selectJob(j.id);
    const cbtn = tr.querySelector('.job-cancel');
    if (cbtn) cbtn.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Cancel ${j.id} (${j.kind})? Resumable work picks up where it left off.`)) return;
      try { await api(`/api/jobs/${j.id}/cancel`, { method: 'POST' }); }
      catch (err) { alert(err.message); }
      refreshJobs();
    };
    tb.appendChild(tr);
  }
  if (running) setTimeout(refreshJobs, 3000);
}

async function selectJob(jobId) {
  state.selectedJob = jobId;
  if (state.jobEventSource) { state.jobEventSource.close(); state.jobEventSource = null; }
  const log = $('job-log');
  log.textContent = 'Loading log…';
  const job = await api(`/api/jobs/${jobId}`);
  // Preload the existing log in ONE write, showing only the tail if huge.
  const tail = job.log.length > MAX_LOG_LINES
    ? [`… (${fmt(job.log.length - MAX_LOG_LINES)} earlier lines hidden — full log kept server-side)`,
       ...job.log.slice(-MAX_LOG_LINES)]
    : job.log;
  log.textContent = tail.join('\n') + '\n';
  log.scrollTop = log.scrollHeight;
  const active = job.status === 'running' || job.status === 'queued';
  renderProgress($('job-progress'), active ? job.progress : null);
  if (active) {
    // The event stream replays from the start; skip what we just rendered.
    state.jobEventSource = watchJob(jobId, log, null, job.log.length, $('job-progress'));
  }
  refreshJobs();
}

/* ---------------- library (managed-dataset registry) ----------------------- */

function fmtAge(ts) {
  if (!ts) return '—';
  const d = (Date.now() / 1000 - ts) / 86400;
  if (d < 1 / 24) return `${Math.round(d * 24 * 60)}m ago`;
  if (d < 1) return `${Math.round(d * 24)}h ago`;
  return `${Math.round(d)}d ago`;
}

async function refreshLibrary() {
  let res;
  try { res = await api('/api/registry'); }
  catch (err) { $('registry-info').textContent = 'Registry error: ' + err.message; return; }
  $('registry-info').textContent =
    `${res.datasets.length} managed dataset(s) · ${res.registry_path}`;
  state.registryPaths = new Set(res.datasets.map(d => d.path));
  const byId = Object.fromEntries(res.datasets.map(d => [d.id, d]));
  const tb = $('library-table').querySelector('tbody');
  const tbTok = $('library-table-tok').querySelector('tbody');
  tb.innerHTML = '';
  tbTok.innerHTML = '';
  for (const d of res.datasets) {
    const s = d.stats || {};
    const tr = document.createElement('tr');
    const kind = d.path_exists === false ? 'missing' : (d.kind || 'text');
    const lineage = d.derived_from
      ? `← ${esc(byId[d.derived_from]?.name || d.derived_from)}` : '';
    tr.innerHTML =
      `<td title="${esc(d.path)}">${esc(d.name)}` +
      (d.notes ? `<div class="dim" style="font-size:11px">${esc(d.notes.slice(0, 80))}</div>` : '') + `</td>` +
      `<td><span class="badge ${kind}">${kind}</span></td>` +
      `<td class="num">${s.num_rows != null ? fmt(s.num_rows) : '—'}</td>` +
      `<td class="num">${s.tokens != null ? fmt(s.tokens) : '—'}</td>` +
      `<td class="num">${s.size_mb != null ? (s.size_mb >= 1000 ? (s.size_mb / 1000).toFixed(1) + ' GB' : s.size_mb >= 10 ? s.size_mb.toFixed(0) + ' MB' : s.size_mb.toFixed(1) + ' MB') : '—'}</td>` +
      `<td class="dim">${esc((d.tags || []).join(', '))}</td>` +
      `<td class="dim">${lineage}</td>` +
      `<td class="dim">${d.open_as ? '<span class="status-done">open</span>' : fmtAge(d.last_opened)}</td>` +
      `<td><button class="mini" data-act="open">${d.open_as ? 'switch to' : 'open'}</button> ` +
      `<button class="mini" data-act="details">details</button></td>`;
    tr.querySelector('[data-act=open]').onclick = () => openRegistryEntry(d.id);
    tr.querySelector('[data-act=details]').onclick = () => selectLibrary(d.id);
    (d.kind === 'tokenized' ? tbTok : tb).appendChild(tr);
  }
  $('library-tok-panel').style.display = tbTok.children.length ? '' : 'none';
}

function resetRegisterForm() {
  $('register-title').textContent = 'Register dataset';
  $('reg-path').disabled = false;
  for (const id of ['reg-path', 'reg-name', 'reg-tags', 'reg-notes', 'reg-field',
                    'reg-tok-kind', 'reg-tok-path', 'reg-special']) $(id).value = '';
  $('reg-kind').value = ''; $('reg-raw').checked = false; $('reg-recursive').checked = false;
  $('scan-results').innerHTML = '';
}

$('btn-show-register').onclick = () => {
  const f = $('register-form');
  if (f.style.display === 'none') { resetRegisterForm(); f.style.display = ''; }
  else f.style.display = 'none';
};
$('btn-register-cancel').onclick = () => { $('register-form').style.display = 'none'; };

$('btn-scan').onclick = async () => {
  const path = $('reg-path').value.trim();
  if (!path) return alert('Enter a directory path to scan');
  try {
    const res = await api(`/api/registry/scan?path=${encodeURIComponent(path)}`);
    const box = $('scan-results');
    box.innerHTML = res.candidates.length ? '' : '<div class="dim">No candidates found.</div>';
    for (const c of res.candidates) {
      const div = document.createElement('div');
      div.className = 'row-item';
      div.innerHTML =
        `<span>${esc(c.name)} <span class="badge ${c.kind}">${c.kind}</span>` +
        ` <span class="dim">${c.files} file(s)</span></span>` +
        (c.registered ? '<span class="dim">registered</span>'
                      : `<button class="mini">use</button>`);
      const btn = div.querySelector('button');
      if (btn) btn.onclick = () => {
        $('reg-path').value = c.path;
        if (!$('reg-name').value) $('reg-name').value = c.name;
        $('reg-kind').value = c.kind;
        if (c.kind === 'tokenized') $('reg-raw').checked = true;
      };
      box.appendChild(div);
    }
  } catch (err) { alert(err.message); }
};

$('btn-register').onclick = async () => {
  const body = {
    name: $('reg-name').value.trim(),
    tags: $('reg-tags').value.split(',').map(s => s.trim()).filter(Boolean),
    notes: $('reg-notes').value.trim(),
    kind: $('reg-kind').value || null,
    text_field: $('reg-field').value.trim() || null,
    tok_kind: $('reg-tok-kind').value.trim() || null,
    tok_path: $('reg-tok-path').value.trim() || null,
    special_tokens: $('reg-special').value.trim() || null,
    raw_shards: $('reg-raw').checked,
    recursive: $('reg-recursive').checked,
  };
  if (!body.name) return alert('Name is required');
  try {
    body.path = $('reg-path').value.trim();
    if (!body.path) return alert('Path is required');
    await api('/api/registry', { method: 'POST', body });
    $('register-form').style.display = 'none';
    resetRegisterForm();
    refreshLibrary();
    refreshDatasets();
  } catch (err) { alert(err.message); }
};

/* ---------------- tabs / init --------------------------------------------- */

function showTab(name) {
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tabpane').forEach(p =>
    p.classList.toggle('active', p.id === `tab-${name}`));
  if (name === 'jobs') refreshJobs();
  if (name === 'dedup') refreshClusters();
  if (name === 'sets') refreshSets();
  if (name === 'library') refreshLibrary();
  if (name === 'filters') {
    refreshFilters();
    if (selectedFilterId) renderFilterEvals(selectedFilterId);
  }
  if (name === 'transforms') {
    Promise.all([refreshTransforms(), refreshFilters()])
      .then(() => populateTfmFilterCtx());
    if (selectedTransformId) renderTransformEvals(selectedTransformId);
  }
  if (name === 'export') { refreshSets(); refreshExportPlan(); }
  if (name === 'tokenize') {
    Promise.all([refreshSets(), refreshFilters(), refreshTransforms()])
      .then(() => renderTokenizeTab());
  }
}
document.querySelectorAll('.tab').forEach(t => t.onclick = () => showTab(t.dataset.tab));

api('/api/config').then(c => { state.config = c; }).catch(() => {});
refreshDatasets();
refreshJobs();
refreshLibrary();
refreshFilters();
refreshTransforms();
