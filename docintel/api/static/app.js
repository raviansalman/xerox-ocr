"use strict";
// Document Intelligence web UI. No framework; every value is inserted as text (never as HTML).

const KEY_STORE = "docintel.apiKey";
const state = { key: null, offset: 0, limit: 25, total: 0, poll: null, types: [] };

// Shown when the tenant has not configured example questions (GET/PUT /api/v1/settings). Guidance only: no data.
const GUIDANCE = [
  ["Exact", "Put a phrase in quotes to require it word for word, or type an identifier as written."],
  ["Meaning", "Describe what you are looking for in your own words; documents that say it differently are found too."],
  ["Count", "Start with “How many” and name the kind of document and any conditions."],
  ["Totals", "Ask for the total, average, largest or smallest amount; currencies are kept apart."],
  ["Filters", "Mention dates, years, amounts, governing law, signers or clauses to narrow the results."],
  ["Lookup", "Ask “What is the … of …” or “When does … expire” to get a single value with its source."],
];
const MATCH_LABEL = {
  exact_phrase: "Exact phrase", exact_identifier: "Identifier", exact_term: "Exact word", filename: "File name",
  filename_terms: "File name words", joined_form: "Joined form", partial_identifier: "Partial identifier",
  all_terms: "All words", ocr_tolerant: "OCR-tolerant", fuzzy: "Fuzzy", semantic: "Semantic", structured: "Filters",
};

// ------------------------------------------------------------------------------------------------ helpers

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const $ = (id) => document.getElementById(id);

// Arabic folding as in docintel/text.py: the terms come back normalized ("المكتبه"), the text is as written
// ("المكتبةُ"), so both are compared folded and matches are mapped back to the original characters.
const AR_DROP = /[\u0640\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED\u200C-\u200F]/;
const AR_MAP = { "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ی": "ي", "ک": "ك" };
for (let i = 0; i < 10; i++) { AR_MAP[String.fromCharCode(0x0660 + i)] = String(i); AR_MAP[String.fromCharCode(0x06F0 + i)] = String(i); }

function folded(text) {
  // folded text plus, for each folded character, the index of the original character it came from
  let out = "";
  const at = [];
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (AR_DROP.test(c)) continue;
    const low = (AR_MAP[c] || c).toLowerCase();
    out += low.length === 1 ? low : c;
    at.push(i);
  }
  return { out, at };
}

function textDir(text) {
  // direction of a text from its first letter, so a leading label such as "p. 1" does not decide it
  const m = (text || "").match(/[A-Za-z\u00C0-\u024F\u0590-\u08FF]/);
  return m && m[0] >= "\u0590" ? "rtl" : "ltr";
}

function highlight(text, terms) {
  // Split text around case-insensitive term matches and wrap them in <mark> (built as nodes, not HTML).
  const frag = document.createDocumentFragment();
  const usable = (terms || []).map((t) => folded(t || "").out).filter((t) => t.length > 1).sort((a, b) => b.length - a.length);
  if (!usable.length) { frag.append(text); return frag; }
  // whole words only ("inv" must not light up "Invoice"); a space in a term also matches - / . # _
  const esc = usable.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "[\\s\\-/.#_]+"));
  const re = new RegExp("(?<![\\p{L}\\p{N}])(" + esc.join("|") + ")(?![\\p{L}\\p{N}])", "giu");
  const f = folded(text);
  let last = 0;
  for (const m of f.out.matchAll(re)) {
    const start = f.at[m.index];
    let end = f.at[m.index + m[0].length - 1] + 1;
    while (end < text.length && AR_DROP.test(text[end])) end++;   // keep trailing diacritics inside the mark
    if (start > last) frag.append(text.slice(last, start));
    frag.append(h("mark", {}, text.slice(start, end)));
    last = end;
  }
  if (last < text.length) frag.append(text.slice(last));
  return frag;
}

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (state.key) headers["X-API-Key"] = state.key;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  if (res.status === 401) { openKeyDialog("Your API key was not accepted."); throw new Error("unauthorized"); }
  if (!res.ok) {
    let detail = res.statusText;
    try { const b = await res.json(); detail = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail); } catch (_) { /* ignore */ }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return opts.raw ? res : res.json();
}

function fmtDate(iso) { return iso ? new Date(iso).toLocaleString() : ""; }
function fmtSize(n) { if (!n) return ""; const u = ["B", "KB", "MB", "GB"]; let i = 0; while (n >= 1024 && i < 3) { n /= 1024; i++; } return `${n.toFixed(i ? 1 : 0)} ${u[i]}`; }
function badge(text, cls) { return h("span", { class: `badge ${cls || ""}` }, text); }
function statusBadge(s) { return badge(s, `st-${s}`); }

// ------------------------------------------------------------------------------------------------ session

function openKeyDialog(msg) {
  const err = $("keyError");
  err.hidden = !msg;
  err.textContent = msg || "";
  if (!$("keyDialog").open) $("keyDialog").showModal();
}

async function connect() {
  try {
    const me = await api("/api/v1/me");
    $("tenant").textContent = `${me.tenant_id} · ${me.roles.includes("uploader") ? "editor" : "reader"}`;
    const settings = await api("/api/v1/settings");
    renderExamples(settings.examples);
    const tax = await api("/api/v1/taxonomy");
    state.types = tax.types;
    const sel = $("typeFilter");
    sel.replaceChildren(h("option", { value: "" }, "All types"), ...tax.types.map((t) => h("option", { value: t.type }, t.label)));
    return true;
  } catch (e) {
    return false;
  }
}

async function health() {
  try {
    const r = await fetch("/health/ready");
    $("health").className = "dot " + (r.ok ? "ok" : "bad");
    $("health").title = r.ok ? "All services ready" : "Some services are not ready";
  } catch (_) { $("health").className = "dot bad"; }
}

// ------------------------------------------------------------------------------------------------ ask

function renderExamples(examples) {
  if (examples && examples.length) {
    $("examples").replaceChildren(...examples.map(({ label, q }) =>
      h("button", { class: "chip", type: "button", title: q, dataset: { q }, onclick: () => { $("q").value = q; ask(); } },
        h("span", { class: "chip-label" }, label), q)));
    return;
  }
  $("examples").replaceChildren(h("ul", { class: "guidance", "data-testid": "guidance" },
    ...GUIDANCE.map(([label, text]) => h("li", {}, h("b", {}, label), " ", text))));
}

function renderAnswer(out) {
  const a = out.answer || {};
  const box = $("answer");
  box.hidden = false;
  box.className = `answer kind-${a.kind}`;
  const parts = [];
  if (a.kind === "number") {
    parts.push(h("div", { class: "big", "data-testid": "answer-value" }, a.unit === "%" ? `${a.value}%` : String(a.value)));
  } else if (a.kind === "figure") {
    parts.push(h("div", { class: "big", "data-testid": "answer-value" }, `${a.value} ${a.unit}`));
    parts.push(h("blockquote", { dir: "auto" }, a.snippet));
  } else if (a.kind === "table" && a.rows) {
    const cols = Object.keys(a.rows[0] || {}).filter((c) => c !== "key" || !a.rows[0].label);
    parts.push(h("table", { class: "mini", "data-testid": "answer-table" },
      h("thead", {}, h("tr", {}, ...cols.map((c) => h("th", {}, c)))),
      h("tbody", {}, ...a.rows.map((r) => h("tr", {}, ...cols.map((c) => h("td", {}, typeof r[c] === "number" ? r[c].toLocaleString() : r[c])))))));
  } else if (a.kind === "fields" && a.rows) {
    parts.push(h("div", { class: "big", "data-testid": "answer-value" }, a.rows[0].value));
  }
  parts.push(h("p", { class: "answer-text", dir: "auto", "data-testid": "answer-text" }, a.text || ""));
  if (a.note) parts.push(h("p", { class: "muted" }, a.note));
  if (a.degraded) parts.push(h("p", { class: "warn" }, a.degraded));
  const t = out.timings_ms || {};
  parts.push(h("p", { class: "muted small" }, `${out.total} result${out.total === 1 ? "" : "s"} · ${t.total} ms · intent: ${out.intent}`));
  box.replaceChildren(...parts);
}

function renderResults(out) {
  const terms = out.terms || [];
  const cards = (out.results || []).map((r) => {
    const fields = Object.entries(r.fields || {}).slice(0, 6).map(([k, v]) => h("span", { class: "field" }, h("b", {}, k.replace(/_/g, " ")), " ", v));
    return h("article", { class: "card", "data-testid": "result", dataset: { doc: r.document_id } },
      h("div", { class: "card-head" },
        h("button", { class: "title link", dir: "auto", onclick: () => openDoc(r.document_id), "data-testid": "result-title" }, r.title || r.filename),
        h("span", { class: "muted small" }, r.filename)),
      h("div", { class: "badges" },
        badge(r.doc_type_label || "Unclassified", "type"),
        ...r.match_types.map((m) => badge(MATCH_LABEL[m] || m, `m-${m}`)),
        badge(`confidence ${Math.round(r.confidence * 100)}%`, "conf"),
        r.has_signature ? badge("signature", "sig") : null),
      fields.length ? h("div", { class: "fields" }, ...fields) : null,
      ...r.snippets.map((s) => h("p", { class: "snippet", dir: textDir(s.text) },
        h("span", { class: "page" }, `p. ${s.page}`), " ", highlight(s.text, terms))));
  });
  $("results").replaceChildren(...cards);
}

async function ask(ev) {
  if (ev) ev.preventDefault();
  const q = $("q").value.trim();
  if (!q) return;
  $("results").replaceChildren(h("p", { class: "muted" }, "Searching…"));
  $("answer").hidden = true;
  try {
    const out = await api("/api/v1/query", { method: "POST", json: { q, explain: $("explain").checked, limit: 20 } });
    renderAnswer(out);
    renderResults(out);
    const plan = $("plan");
    plan.hidden = !out.plan;
    plan.textContent = out.plan ? JSON.stringify({ plan: out.plan, timings_ms: out.timings_ms }, null, 2) : "";
  } catch (e) {
    $("results").replaceChildren(h("p", { class: "error", "data-testid": "query-error" }, `Search failed: ${e.message}`));
  }
}

// ------------------------------------------------------------------------------------------------ documents

async function loadStats() {
  try {
    const s = await api("/api/v1/stats");
    const st = s.by_status || {};
    const tile = (label, value, id) => h("div", { class: "tile", "data-testid": id }, h("div", { class: "tile-value" }, Number(value || 0).toLocaleString()), h("div", { class: "tile-label" }, label));
    $("stats").replaceChildren(
      tile("Indexed documents", s.indexed_documents, "stat-indexed"), tile("Pages", s.indexed_pages, "stat-pages"),
      tile("In progress", (st.queued || 0) + (st.processing || 0), "stat-progress"), tile("Failed", st.failed, "stat-failed"));
    return (st.queued || 0) + (st.processing || 0);
  } catch (_) { return 0; }
}

async function loadDocs() {
  const params = new URLSearchParams({ limit: state.limit, offset: state.offset });
  if ($("docFilter").value.trim()) params.set("q", $("docFilter").value.trim());
  if ($("statusFilter").value) params.set("status", $("statusFilter").value);
  if ($("typeFilter").value) params.set("doc_type", $("typeFilter").value);
  const seq = (state.docsSeq = (state.docsSeq || 0) + 1);
  try {
    const out = await api(`/api/v1/documents?${params}`);
    if (seq !== state.docsSeq) return;               // a newer request (filter, delete, page) owns the table now
    state.total = out.total;
    $("docRows").replaceChildren(...out.documents.map((d) => h("tr", { "data-testid": "doc-row", dataset: { doc: d.id, status: d.status } },
      h("td", {}, h("button", { class: "link", dir: "auto", onclick: () => openDoc(d.id) }, d.title && d.title !== d.filename ? d.title : d.filename),
        h("div", { class: "muted small" }, `${d.filename} · ${fmtSize(d.size_bytes)}`)),
      h("td", {}, d.doc_type_label || "—"),
      h("td", {}, statusBadge(d.status), d.error ? h("div", { class: "error small" }, d.error) : null),
      h("td", {}, d.page_count ?? "—"),
      h("td", { class: "small" }, fmtDate(d.created_at)),
      h("td", { class: "actions" },
        h("button", { class: "ghost small", title: "Process again", onclick: () => reprocess(d.id) }, "Reprocess"),
        h("button", { class: "ghost small danger", title: "Delete", onclick: () => removeDoc(d.id, d.filename), "data-testid": "delete" }, "Delete")))));
    const from = out.total ? state.offset + 1 : 0;
    $("pageInfo").textContent = `${from}–${Math.min(state.offset + state.limit, out.total)} of ${out.total}`;
    $("prev").disabled = state.offset === 0;
    $("next").disabled = state.offset + state.limit >= out.total;
  } catch (e) {
    $("docRows").replaceChildren(h("tr", {}, h("td", { colspan: 6, class: "error" }, e.message)));
  }
}

async function refreshDocs() {
  const pending = await loadStats();
  await loadDocs();
  clearTimeout(state.poll);
  if (pending > 0) state.poll = setTimeout(refreshDocs, 2000);
}

async function uploadFiles(files) {
  if (!files.length) return;
  const status = $("uploadStatus");
  const batch = 20;
  let done = 0, dup = 0, rejected = 0;
  for (let i = 0; i < files.length; i += batch) {
    const fd = new FormData();
    for (const f of files.slice(i, i + batch)) fd.append("files", f, f.name);
    status.textContent = `Uploading ${Math.min(i + batch, files.length)} of ${files.length}…`;
    try {
      const out = await api("/api/v1/documents", { method: "POST", body: fd });
      for (const d of out.documents) { if (d.status === "rejected") rejected++; else if (d.duplicate) dup++; else done++; }
    } catch (e) { status.textContent = `Upload failed: ${e.message}`; return; }
  }
  status.textContent = `${done} uploaded, ${dup} already present, ${rejected} rejected. Processing runs in the background.`;
  state.offset = 0;
  refreshDocs();
}

async function removeDoc(id, name) {
  if (!confirm(`Delete “${name}” and all its extracted data?`)) return;
  try { await api(`/api/v1/documents/${id}`, { method: "DELETE" }); refreshDocs(); } catch (e) { alert(e.message); }
}

async function reprocess(id) {
  try { await api(`/api/v1/documents/${id}/reprocess`, { method: "POST" }); refreshDocs(); } catch (e) { alert(e.message); }
}

// ------------------------------------------------------------------------------------------------ document drawer

async function openDoc(id) {
  const drawer = $("drawer");
  drawer.hidden = false;
  $("dTitle").textContent = "Loading…";
  $("dBody").replaceChildren();
  let d;
  try { d = await api(`/api/v1/documents/${id}`); } catch (e) { $("dTitle").textContent = e.message; return; }
  $("dTitle").textContent = d.title || d.filename;
  const kv = (k, v) => (v === null || v === undefined || v === "" ? null : h("div", { class: "kv" }, h("span", {}, k), h("b", {}, String(v))));
  const section = (title, ...body) => h("section", { class: "dsec" }, h("h3", {}, title), ...body);
  const pageView = h("div", { class: "page-view", "data-testid": "page-view" });
  const pageButtons = (d.pages || []).map((p) => h("button", { class: "ghost small", onclick: () => showPage(d.id, p.page_number, pageView) }, `Page ${p.page_number}`));
  $("dBody").replaceChildren(
    h("div", { class: "row" },
      h("button", { class: "primary small", onclick: () => download(d.id, d.filename) }, "Download original")),
    section("Summary",
      kv("File", d.filename), kv("Type", d.doc_type_label ? `${d.doc_type_label} (${Math.round((d.doc_type_confidence || 0) * 100)}%, ${d.doc_type_method})` : null),
      kv("Status", d.status), kv("Kind", d.kind), kv("Pages", d.page_count), kv("Words", d.word_count),
      kv("Signature", d.has_signature ? "signature or signer line found" : "none found"),
      kv("Processed in", d.processing_ms ? `${(d.processing_ms / 1000).toFixed(1)} s` : null), kv("Error", d.error)),
    section("Extracted fields", (d.fields || []).length ? h("table", { class: "mini", "data-testid": "fields-table" },
      h("tbody", {}, ...d.fields.map((f) => h("tr", {}, h("td", {}, f.name.replace(/_/g, " ")), h("td", {}, f.value_text ?? ""),
        h("td", { class: "muted small" }, `p. ${f.page ?? "?"} · ${Math.round(f.confidence * 100)}%`))))) : h("p", { class: "muted" }, "No fields extracted.")),
    section("People, organizations and identifiers", (d.entities || []).length ? h("div", { class: "fields" },
      ...d.entities.map((e) => h("span", { class: "field" }, h("b", {}, e.role ? `${e.type} · ${e.role}` : e.type), " ", e.value))) : h("p", { class: "muted" }, "None found.")),
    section("Clauses", (d.clauses || []).length ? h("div", {},
      ...d.clauses.map((c) => h("details", {}, h("summary", {}, `${c.clause_type.replace(/_/g, " ")}${c.ref ? ` · ${c.ref}` : ""} · p. ${c.page}`), h("p", { dir: "auto" }, c.text)))) : h("p", { class: "muted" }, "None found.")),
    section("Pages", h("div", { class: "row wrap" }, ...pageButtons), pageView));
  if (pageButtons.length) showPage(d.id, 1, pageView);
}

async function showPage(id, n, target) {
  target.replaceChildren(h("p", { class: "muted" }, "Loading page…"));
  try {
    const p = await api(`/api/v1/documents/${id}/pages/${n}`);
    const img = h("img", { class: "page-img", alt: `Page ${n}` });
    const parts = [h("p", { class: "muted small" }, `Page ${n} · ${p.kind}${p.ocr_confidence ? ` · OCR confidence ${Math.round(p.ocr_confidence * 100)}%` : ""}`)];
    try {
      const res = await api(`/api/v1/documents/${id}/pages/${n}/image`, { raw: true });
      img.src = URL.createObjectURL(await res.blob());
      parts.push(img);
    } catch (_) { /* no image for this format */ }
    parts.push(h("pre", { class: "page-text", dir: "auto" }, p.text || "(no text)"));
    target.replaceChildren(...parts);
  } catch (e) { target.replaceChildren(h("p", { class: "error" }, e.message)); }
}

async function download(id, name) {
  const res = await api(`/api/v1/documents/${id}/original`, { raw: true });
  const url = URL.createObjectURL(await res.blob());
  const a = h("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// ------------------------------------------------------------------------------------------------ wiring

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === name));
  if (name === "documents") refreshDocs();
}

async function init() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("askForm").addEventListener("submit", ask);
  $("keyButton").addEventListener("click", () => openKeyDialog());
  $("keyForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    state.key = $("keyInput").value.trim();
    if (await connect()) {
      try { localStorage.setItem(KEY_STORE, state.key); } catch (_) { /* storage unavailable */ }
      $("keyDialog").close();
    } else {
      openKeyDialog("That key was not accepted.");
    }
  });
  $("dClose").addEventListener("click", () => { $("drawer").hidden = true; });
  $("browse").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", (e) => uploadFiles([...e.target.files]));
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); uploadFiles([...e.dataTransfer.files]); });
  $("urlForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = $("urlInput").value.trim();
    if (!url) return;
    try { await api("/api/v1/documents/url", { method: "POST", json: { url } }); $("urlInput").value = ""; refreshDocs(); }
    catch (err) { $("uploadStatus").textContent = `URL rejected: ${err.message}`; }
  });
  let filterTimer;
  $("docFilter").addEventListener("input", () => { clearTimeout(filterTimer); filterTimer = setTimeout(() => { state.offset = 0; loadDocs(); }, 250); });
  $("statusFilter").addEventListener("change", () => { state.offset = 0; loadDocs(); });
  $("typeFilter").addEventListener("change", () => { state.offset = 0; loadDocs(); });
  $("refresh").addEventListener("click", refreshDocs);
  $("prev").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadDocs(); });
  $("next").addEventListener("click", () => { state.offset += state.limit; loadDocs(); });

  health();
  setInterval(health, 15000);
  try { state.key = localStorage.getItem(KEY_STORE); } catch (_) { state.key = null; }
  if (!state.key || !(await connect())) openKeyDialog();
}

init();
