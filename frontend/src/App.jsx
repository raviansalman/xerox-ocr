import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowRight,
  BadgeCheck,
  Check,
  ChevronLeft,
  ChevronRight,
  Circle,
  Clipboard,
  Download,
  Eye,
  FileText,
  HelpCircle,
  KeyRound,
  LoaderCircle,
  Lock,
  RefreshCw,
  RotateCw,
  Search,
  ShieldCheck,
  ThumbsUp,
  Trash2,
  Upload,
  X
} from "lucide-react";
import "./styles.css";

const KEY_STORE = "docintel.apiKey";

const EXAMPLES = [
  ["Find an exact phrase", "Find the phrase \"ninety days written notice\""],
  ["Review renewal terms", "What are the renewal terms across our active contracts?"],
  ["Compare documents", "Compare payment terms in the Acme and Northstar agreements"],
  ["Identify obligations", "What reporting obligations are due this quarter?"],
  ["Summarize a policy", "Summarize our information security policy"],
  ["Trace a decision", "Where was the 2025 pricing adjustment approved?"]
];

const MATCH_LABEL = {
  exact_phrase: "Exact phrase",
  exact_identifier: "Identifier",
  exact_term: "Exact word",
  filename: "File name",
  filename_terms: "File name words",
  joined_form: "Joined form",
  partial_identifier: "Partial identifier",
  all_terms: "All words",
  ocr_tolerant: "OCR-tolerant",
  fuzzy: "Fuzzy",
  semantic: "Semantic",
  structured: "Filters"
};

function fmtSize(n) {
  if (!n) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = n;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(i ? 1 : 0)} ${units[i]}`;
}

function fmtDate(iso) {
  return iso ? new Date(iso).toLocaleString() : "";
}

function fmtNumber(n) {
  return Number(n || 0).toLocaleString();
}

function statusTotal(stats) {
  return Object.values(stats?.by_status || {}).reduce((sum, n) => sum + Number(n || 0), 0);
}

function indexedWorkspaceSub(stats) {
  const total = statusTotal(stats);
  if (!total) return "Indexed and searchable";
  const indexed = Number(stats?.indexed_documents || 0);
  return `${((indexed / total) * 100).toFixed(1)}% of workspace`;
}

function refreshLabel(date) {
  if (!date) return "Live";
  const seconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (seconds < 60) return "Live - refreshed now";
  const minutes = Math.round(seconds / 60);
  return `Live - refreshed ${minutes}m ago`;
}

function resultConfidence(results) {
  const values = (results || []).map((r) => Number(r.confidence)).filter(Number.isFinite);
  if (!values.length) return null;
  return Math.round(Math.max(...values) * 100);
}

function firstSentence(text) {
  return (text || "").split(/(?<=[.!?])\s+/)[0] || "";
}

function highlight(text, terms) {
  const usable = (terms || []).filter((t) => t && t.length > 1).sort((a, b) => b.length - a.length);
  if (!usable.length) return text;
  const esc = usable.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "[\\s\\-/.#_]+"));
  const re = new RegExp("(?<![\\p{L}\\p{N}])(" + esc.join("|") + ")(?![\\p{L}\\p{N}])", "giu");
  const parts = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    parts.push(<mark key={`${m.index}-${m[0]}`}>{m[0]}</mark>);
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function Brand() {
  return (
    <div className="brand">
      <span className="logo" aria-hidden="true"><span /><span /><span /></span>
      <span>Document Intelligence</span>
    </div>
  );
}

function Dot({ status = "ok" }) {
  return <span className={`dot ${status}`} />;
}

function Badge({ children, tone = "" }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

function App() {
  const [apiKey, setApiKey] = useState(() => {
    try { return localStorage.getItem(KEY_STORE) || ""; } catch { return ""; }
  });
  const [keyInput, setKeyInput] = useState(apiKey);
  const [showKey, setShowKey] = useState(false);
  const [keyError, setKeyError] = useState("");
  const [connected, setConnected] = useState(false);
  const [tenant, setTenant] = useState("");
  const [healthReady, setHealthReady] = useState(false);
  const [tab, setTab] = useState("ask");
  const [examples, setExamples] = useState(EXAMPLES);
  const [types, setTypes] = useState([]);
  const [drawerDocId, setDrawerDocId] = useState(null);

  const request = useCallback(async (path, opts = {}) => {
    const headers = { ...(opts.headers || {}) };
    if (apiKey) headers["X-API-Key"] = apiKey;
    const init = { ...opts, headers };
    if (opts.json !== undefined) {
      init.body = JSON.stringify(opts.json);
      init.headers = { ...headers, "Content-Type": "application/json" };
    }
    const res = await fetch(path, init);
    if (res.status === 401) {
      setConnected(false);
      throw new Error("Your API key was not accepted.");
    }
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch {
        /* ignore */
      }
      throw new Error(detail || `HTTP ${res.status}`);
    }
    return opts.raw ? res : res.json();
  }, [apiKey]);

  const connect = useCallback(async (candidate) => {
    const previous = apiKey;
    if (candidate !== undefined) setApiKey(candidate);
    const key = candidate ?? apiKey;
    if (!key) {
      setConnected(false);
      return false;
    }
    try {
      const headers = { "X-API-Key": key };
      const meRes = await fetch("/api/v1/me", { headers });
      if (!meRes.ok) throw new Error("That key was not accepted.");
      const me = await meRes.json();
      const settingsRes = await fetch("/api/v1/settings", { headers });
      const settings = settingsRes.ok ? await settingsRes.json() : { examples: [] };
      const taxRes = await fetch("/api/v1/taxonomy", { headers });
      const tax = taxRes.ok ? await taxRes.json() : { types: [] };
      setTenant(`${me.tenant_id} - ${me.roles?.includes("uploader") ? "editor" : "reader"}`);
      setExamples(settings.examples?.length ? settings.examples.map((e) => [e.label, e.q]) : EXAMPLES);
      setTypes(tax.types || []);
      setConnected(true);
      setKeyError("");
      try { localStorage.setItem(KEY_STORE, key); } catch { /* storage unavailable */ }
      return true;
    } catch (err) {
      if (candidate !== undefined) setApiKey(previous);
      setConnected(false);
      setKeyError(err.message || "That key was not accepted.");
      return false;
    }
  }, [apiKey]);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch("/health/ready");
        setHealthReady(res.ok);
      } catch {
        setHealthReady(false);
      }
    };
    check();
    const timer = setInterval(check, 15000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (apiKey) connect();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const submitKey = async (ev) => {
    ev.preventDefault();
    await connect(keyInput.trim());
  };

  return (
    <>
      {!connected && (
        <ConnectScreen
          keyInput={keyInput}
          setKeyInput={setKeyInput}
          showKey={showKey}
          setShowKey={setShowKey}
          keyError={keyError}
          submitKey={submitKey}
        />
      )}

      {connected && (
        <>
          <Header
            tab={tab}
            setTab={setTab}
            tenant={tenant}
            healthReady={healthReady}
            onDisconnect={() => {
              try { localStorage.removeItem(KEY_STORE); } catch { /* ignore */ }
              setApiKey("");
              setKeyInput("");
              setConnected(false);
            }}
          />
          <main className="app-main">
            {tab === "ask" ? (
              <AskPage api={request} examples={examples} openDoc={setDrawerDocId} />
            ) : (
              <DocumentsPage api={request} types={types} openDoc={setDrawerDocId} />
            )}
          </main>
          {drawerDocId ? <DocumentDrawer api={request} id={drawerDocId} onClose={() => setDrawerDocId(null)} /> : null}
        </>
      )}
    </>
  );
}

function ConnectScreen({ keyInput, setKeyInput, showKey, setShowKey, keyError, submitKey }) {
  return (
    <section className="connect-screen">
      <header className="connect-top">
        <Brand />
        <div className="secure-note"><ShieldCheck size={19} />Enterprise-grade security</div>
      </header>
      <div className="connect-rule" />
      <main className="connect-grid">
        <section className="connect-story">
          <p className="eyebrow"><Dot />Document Intelligence</p>
          <h1>Your documents.<br />Every answer.<br /><span>Backed by evidence.</span></h1>
          <p className="lead">Search every document with secure AI and get precise answers you can trust, each one linked to its original source.</p>
          <PreviewCard />
        </section>

        <section className="connect-card">
          <div className="secure-tile"><ShieldCheck size={25} /></div>
          <p className="eyebrow">Secure Connection</p>
          <h2>Connect to your workspace</h2>
          <p className="muted">Enter your organization's API key to securely access your document intelligence workspace.</p>
          <form className="connect-form" onSubmit={submitKey}>
            <div className="field-row">
              <label htmlFor="keyInput">API key</label>
              <button type="button" className="link subtle">Where do I find this?</button>
            </div>
            <div className={`key-entry ${keyError ? "invalid" : ""}`}>
              <Lock size={19} />
              <input
                id="keyInput"
                type={showKey ? "text" : "password"}
                autoComplete="off"
                placeholder="iz_live_................"
                value={keyInput}
                onChange={(e) => setKeyInput(e.target.value)}
                required
              />
              <button type="button" className="eye-button" aria-label="Show API key" onClick={() => setShowKey((v) => !v)}>
                <Eye size={19} />
              </button>
            </div>
            {keyError && <p className="error">{keyError}</p>}
            <p className="hint">Your key is unique to your tenant and controls workspace access.</p>
            <button type="submit" className="primary connect-submit">Connect to Workspace <ArrowRight size={21} /></button>
          </form>
          <div className="privacy-note"><Lock size={18} /><div><strong>Private by design</strong><small>Your key is encrypted and never stored in this browser.</small></div></div>
          <p className="prototype-tip">Prototype tip: use any 10+ character key. Try "invalid" or "expired" to preview those states.</p>
        </section>
      </main>
      <footer className="connect-footer"><span>Copyright 2025 InvoZone</span><nav><a>Privacy</a><a>Security</a><a>Support</a></nav></footer>
    </section>
  );
}

function PreviewCard() {
  return (
    <div className="preview-card" aria-hidden="true">
      <div className="preview-head">
        <div><span className="preview-dot" /><strong>Ask your documents</strong></div>
        <span className="preview-menu" aria-hidden="true"><i /><i /><i /></span>
        <span className="avatar">AZ</span>
      </div>
      <div className="preview-body">
        <div className="preview-search"><Search size={24} />What are our termination obligations?<kbd>⌘ K</kbd></div>
        <div className="answer-label"><Dot /><strong>Answer</strong><em><Check size={17} />96% confidence</em></div>
        <p>Either party may terminate with <strong>30 days' written notice.</strong> Immediate termination applies in the event of a material breach not remedied within 14 days.</p>
        <div className="source-card">
          <span className="file-chip">PDF</span>
          <div><strong>Master Service Agreement.pdf</strong><small>Page 12 · Section 8.2</small></div>
          <p>8.2 Either party may terminate this Agreement by providing no less than thirty (30) days' prior written notice</p>
        </div>
        <div className="verified-pill"><BadgeCheck size={48} /><div><strong>Evidence verified</strong><small>3 sources matched</small></div></div>
      </div>
    </div>
  );
}

function Header({ tab, setTab, tenant, healthReady, onDisconnect }) {
  return (
    <header className="topbar">
      <Brand />
      <nav className="tabs" role="tablist">
        <button className={`tab ${tab === "ask" ? "active" : ""}`} onClick={() => setTab("ask")}>Ask</button>
        <button className={`tab ${tab === "documents" ? "active" : ""}`} onClick={() => setTab("documents")}>Documents</button>
      </nav>
      <div className="session">
        <span className="status-copy"><Dot status={healthReady ? "ok" : "bad"} />{healthReady ? "All services ready" : "Services need attention"}</span>
        <span className="tenant">{tenant}</span>
        <button className="icon-button" title="Disconnect" aria-label="Disconnect" onClick={onDisconnect}><KeyRound size={20} /></button>
      </div>
    </header>
  );
}

function AskPage({ api, examples, openDoc }) {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState(null);
  const [results, setResults] = useState([]);
  const [stats, setStats] = useState(null);
  const [plan, setPlan] = useState("");
  const [explain, setExplain] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const clearSearch = () => {
    setQ("");
    setAnswer(null);
    setResults([]);
    setPlan("");
    setError("");
  };

  useEffect(() => {
    let active = true;
    api("/api/v1/stats")
      .then((out) => active && setStats(out))
      .catch(() => active && setStats(null));
    return () => { active = false; };
  }, [api]);

  const ask = async (ev, prompt) => {
    if (ev) ev.preventDefault();
    const query = (prompt || q).trim();
    if (!query) return;
    setQ(query);
    setLoading(true);
    setError("");
    setAnswer(null);
    setPlan("");
    try {
      const out = await api("/api/v1/query", { method: "POST", json: { q: query, explain, limit: 20 } });
      setAnswer(out);
      setResults(out.results || []);
      setPlan(out.plan ? JSON.stringify({ plan: out.plan, timings_ms: out.timings_ms }, null, 2) : "");
    } catch (err) {
      setError(`Search failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="ask-panel">
      <div className="ask-hero">
        <p className="eyebrow"><Dot />Intelligent Document Search</p>
        <h1>Find answers across your documents.</h1>
        <p className="lead">Ask questions in plain English and get reliable answers backed by verifiable document evidence.</p>
      </div>
      <form className="ask" onSubmit={ask}>
        <Search size={28} />
        <input value={q} onChange={(e) => setQ(e.target.value)} type="text" placeholder="Ask a question or search your documents" required />
        {(q || answer || error) ? (
          <button className="clear-search" type="button" aria-label="Clear search" title="Clear search" onClick={clearSearch}>
            <X size={20} />
          </button>
        ) : null}
        <button className="primary" type="submit" disabled={loading}>{loading ? <LoaderCircle className="spin" size={21} /> : "Ask"} <ArrowRight size={22} /></button>
      </form>
      <label className="check">
        <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} />
        <span className="check-box" aria-hidden="true"><Check size={14} /></span>
        <span>Show how the question was understood</span>
      </label>
      {error && <p className="error surface-message">{error}</p>}
      {answer ? (
        <>
          <AnswerPanel out={answer} results={results} openDoc={openDoc} />
          {plan ? <pre className="plan">{plan}</pre> : null}
          <ResultCards out={answer} openDoc={openDoc} />
        </>
      ) : (
        <TryAsking examples={examples} stats={stats} onAsk={(prompt) => ask(null, prompt)} />
      )}
      <footer className="app-foot">
        <span>{fmtNumber(stats?.indexed_documents)} indexed documents</span>
        <span>{fmtNumber(stats?.indexed_pages)} indexed pages</span>
        <span>Tenant-isolated search</span>
      </footer>
    </section>
  );
}

function TryAsking({ examples, stats, onAsk }) {
  return (
    <section className="try-section">
      <div className="try-head">
        <div><h2>Try Asking</h2><p>Start with a common question, or write your own above.</p></div>
        <span>{fmtNumber(stats?.indexed_documents)} documents ready</span>
      </div>
      <div className="examples">
        {examples.slice(0, 6).map(([label, prompt], i) => (
          <button className="chip" type="button" key={`${label}-${prompt}`} onClick={() => onAsk(prompt)}>
            <span className="chip-label">{String(i + 1).padStart(2, "0")}</span>
            <span><span className="chip-title">{label}</span><span>{prompt}</span></span>
            <ArrowRight className="chip-arrow" size={20} />
          </button>
        ))}
      </div>
    </section>
  );
}

function AnswerPanel({ out, results, openDoc }) {
  const a = out.answer || {};
  const evidence = results.slice(0, 3);
  const confidence = resultConfidence(results);
  const sourceCount = results.length || out.total || 0;
  const title = a.kind === "number" ? `${a.value}${a.unit === "%" ? "%" : ""}` :
    a.kind === "figure" ? `${a.value} ${a.unit}` :
      a.kind === "fields" && a.rows?.[0]?.value ? a.rows[0].value :
        firstSentence(a.text) || "Evidence-backed answer";

  return (
    <section className="answer">
      <div className="answer-main">
        <p className="answer-kicker"><Dot />Evidence-backed answer<span className="confidence"><Check size={13} />{confidence === null ? "Verified" : `${confidence}% confidence`}</span></p>
        <h2 className="answer-title">{title}</h2>
        {a.kind === "table" && a.rows?.length ? <AnswerTable rows={a.rows} /> : null}
        {a.snippet ? <blockquote>{a.snippet}</blockquote> : null}
        <p className="answer-text">{a.text || ""}</p>
        {a.note ? <p className="muted">{a.note}</p> : null}
        {a.degraded ? <p className="warn">{a.degraded}</p> : null}
        <div className="answer-meta">
          <span>Answer synthesized from {sourceCount} verified source{sourceCount === 1 ? "" : "s"}</span>
          <div className="answer-actions"><button className="ghost"><Clipboard size={16} />Copy</button><button className="ghost"><ThumbsUp size={16} />Helpful</button></div>
        </div>
      </div>
      <aside className="evidence-panel">
        <div className="evidence-head">
          <div><h2>Evidence</h2><p className="muted">{sourceCount} matching source{sourceCount === 1 ? "" : "s"}</p></div>
          <span className="verified-badge">Verified</span>
        </div>
        <div className="evidence-list">
          {evidence.length ? evidence.map((r, i) => {
            const snip = r.snippets?.[0] || {};
            return (
              <article className={`evidence-card ${i === 0 ? "active" : ""}`} key={`${r.document_id}-${i}`}>
                <span className="evidence-num">{String(i + 1).padStart(2, "0")}</span>
                <button className="title link" type="button" onClick={() => openDoc(r.document_id)}>{r.title || r.filename}</button>
                <small>Page {snip.page || "?"} - {r.doc_type_label || "Source"}</small>
                <p>{snip.text ? `"...${snip.text.slice(0, 96)}..."` : "Verified document source."}</p>
              </article>
            );
          }) : <p className="muted">No matching source details returned.</p>}
        </div>
      </aside>
    </section>
  );
}

function ResultCards({ out, openDoc }) {
  const terms = out.terms || [];
  const rows = out.results || [];
  if (!rows.length) return null;
  return (
    <section className="results">
      {rows.map((r) => {
        const fields = Object.entries(r.fields || {}).slice(0, 6);
        return (
          <article className="card" key={r.document_id}>
            <div className="card-head">
              <button className="title link" type="button" onClick={() => openDoc(r.document_id)}>{r.title || r.filename}</button>
              <span className="muted small">{r.filename}</span>
            </div>
            <div className="badges">
              <Badge tone="type">{r.doc_type_label || "Unclassified"}</Badge>
              {(r.match_types || []).map((m) => <Badge key={m} tone={`m-${m}`}>{MATCH_LABEL[m] || m}</Badge>)}
              <Badge tone="conf">confidence {Math.round((r.confidence || 0) * 100)}%</Badge>
              {r.has_signature ? <Badge tone="sig">signature</Badge> : null}
            </div>
            {fields.length ? (
              <div className="fields">{fields.map(([k, v]) => <span className="field" key={k}><b>{k.replace(/_/g, " ")}</b> {String(v)}</span>)}</div>
            ) : null}
            {(r.snippets || []).map((s, i) => (
              <p className="snippet" key={`${s.page}-${i}`}><span className="page">p. {s.page}</span> {highlight(s.text || "", terms)}</p>
            ))}
          </article>
        );
      })}
    </section>
  );
}

function AnswerTable({ rows }) {
  const cols = Object.keys(rows[0] || {}).filter((c) => c !== "key" || !rows[0].label);
  return (
    <table className="mini">
      <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>{rows.map((row, i) => <tr key={i}>{cols.map((c) => <td key={c}>{typeof row[c] === "number" ? row[c].toLocaleString() : row[c]}</td>)}</tr>)}</tbody>
    </table>
  );
}

function DocumentsPage({ api, types, openDoc }) {
  const [stats, setStats] = useState(null);
  const [docs, setDocs] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const [filter, setFilter] = useState("");
  const [status, setStatus] = useState("");
  const [docType, setDocType] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [errorState, setErrorState] = useState(false);
  const [urlVisible, setUrlVisible] = useState(true);
  const fileInput = useRef(null);
  const limit = 25;

  const loadStats = useCallback(async () => {
    try {
      setStats(await api("/api/v1/stats"));
      setLastRefreshed(new Date());
    } catch {
      setStats(null);
    }
  }, [api]);

  const loadDocs = useCallback(async () => {
    const params = new URLSearchParams({ limit, offset });
    if (filter.trim()) params.set("q", filter.trim());
    if (status) params.set("status", status);
    if (docType) params.set("doc_type", docType);
    const out = await api(`/api/v1/documents?${params}`);
    setDocs(out.documents || []);
    setTotal(out.total || 0);
  }, [api, docType, filter, offset, status]);

  useEffect(() => {
    loadStats();
    loadDocs();
  }, [loadStats, loadDocs]);

  useEffect(() => {
    const st = stats?.by_status || {};
    const pending = (st.queued || 0) + (st.processing || 0);
    if (!pending) return undefined;
    const timer = setTimeout(() => {
      loadStats();
      loadDocs();
    }, 2000);
    return () => clearTimeout(timer);
  }, [loadDocs, loadStats, stats]);

  const uploadFiles = async (files) => {
    if (!files.length) return;
    setErrorState(false);
    let done = 0;
    let dup = 0;
    let rejected = 0;
    try {
      for (let i = 0; i < files.length; i += 20) {
        const fd = new FormData();
        files.slice(i, i + 20).forEach((f) => fd.append("files", f, f.name));
        setUploadStatus(`Uploading ${Math.min(i + 20, files.length)} of ${files.length}...`);
        const out = await api("/api/v1/documents", { method: "POST", body: fd });
        out.documents.forEach((d) => {
          if (d.status === "rejected") rejected += 1;
          else if (d.duplicate) dup += 1;
          else done += 1;
        });
      }
      setUploadStatus(`${done} uploaded, ${dup} already present, ${rejected} rejected. Processing runs in the background.`);
      setOffset(0);
      loadStats();
      loadDocs();
    } catch (err) {
      setErrorState(true);
      setUploadStatus(`Upload failed: ${err.message}`);
    }
  };

  const addUrl = async (ev) => {
    ev.preventDefault();
    const form = new FormData(ev.currentTarget);
    const url = String(form.get("url") || "").trim();
    if (!url) return;
    try {
      setErrorState(false);
      await api("/api/v1/documents/url", { method: "POST", json: { url } });
      ev.currentTarget.reset();
      setUploadStatus("URL accepted. Processing runs in the background.");
      loadDocs();
    } catch (err) {
      setErrorState(true);
      setUploadStatus(`URL rejected: ${err.message}`);
    }
  };

  const reprocess = async (id) => {
    try {
      await api(`/api/v1/documents/${id}/reprocess`, { method: "POST" });
      loadStats();
      loadDocs();
    } catch (err) {
      alert(err.message);
    }
  };

  const removeDoc = async (id, name) => {
    if (!confirm(`Delete "${name}" and all its extracted data?`)) return;
    try {
      await api(`/api/v1/documents/${id}`, { method: "DELETE" });
      loadStats();
      loadDocs();
    } catch (err) {
      alert(err.message);
    }
  };

  const st = stats?.by_status || {};
  const progress = (st.queued || 0) + (st.processing || 0);
  const from = total ? offset + 1 : 0;

  return (
    <section className="documents-panel">
      <div className="page-title">
        <div>
          <p className="eyebrow">Connected Knowledge</p>
          <h1>Documents</h1>
          <p className="lead">Upload, process, and inspect every source available to intelligent search.</p>
        </div>
        <span className="live-pill"><Dot />{refreshLabel(lastRefreshed)}</span>
      </div>

      <div className="stats">
        <Stat label="Indexed documents" value={stats?.indexed_documents} sub={indexedWorkspaceSub(stats)} icon={<FileText size={20} />} />
        <Stat label="Pages" value={stats?.indexed_pages} sub="Fully searchable" icon={<Clipboard size={20} />} />
        <Stat label="In progress" value={progress} sub={`${st.processing || 0} processing - ${st.queued || 0} queued`} icon={<Circle size={20} />} />
        <Stat label="Failed" value={st.failed} sub="Needs attention" tone="failed" icon={<HelpCircle size={20} />} />
      </div>

      <div className="upload">
        <div
          className="drop-line"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            uploadFiles([...e.dataTransfer.files]);
          }}
        >
          <div className="doc-icon"><Upload size={18} /></div>
          <div><strong>Drop files here or browse</strong><p>PDF, DOCX, XLSX, PPTX, PNG, JPG, TXT, CSV and EML - Up to 200 MB per file</p></div>
          <button className="ghost" type="button" onClick={() => fileInput.current?.click()}>Browse files</button>
          <input ref={fileInput} type="file" multiple hidden onChange={(e) => uploadFiles([...e.target.files])} />
        </div>
        <div className="url-label">
          <button type="button" className="link subtle" onClick={() => setUrlVisible((v) => !v)}>{urlVisible ? "Hide URL ingest" : "Show URL ingest"}</button>
          <span>Duplicates are detected by file content and reported as already present.</span>
        </div>
        {urlVisible && (
          <form className="url-form" onSubmit={addUrl}>
            <input name="url" type="url" placeholder="https://example.com/document.pdf" />
            <button className="primary small" type="submit">Add URL</button>
          </form>
        )}
        {uploadStatus && <div className={`upload-status ${errorState ? "error-state" : ""}`}>{uploadStatus}{errorState ? <X size={14} /> : null}</div>}
      </div>

      <section className="docs-shell">
        <div className="toolbar">
          <div className="filter-wrap"><Search size={17} /><input value={filter} onChange={(e) => { setOffset(0); setFilter(e.target.value); }} type="search" placeholder="Filter by file name" /></div>
          <select value={status} onChange={(e) => { setOffset(0); setStatus(e.target.value); }}>
            <option value="">All</option><option value="indexed">Indexed</option><option value="processing">Processing</option><option value="queued">Queued</option><option value="failed">Failed</option>
          </select>
          <select value={docType} onChange={(e) => { setOffset(0); setDocType(e.target.value); }}>
            <option value="">All types</option>
            {types.map((t) => <option value={t.type} key={t.type}>{t.label}</option>)}
          </select>
          <button className="ghost" onClick={() => { loadStats(); loadDocs(); }}><RefreshCw size={16} />Refresh</button>
        </div>
        <table className="docs">
          <thead><tr><th>Document</th><th>Type</th><th>Status</th><th>Pages</th><th>Added</th><th>Actions</th></tr></thead>
          <tbody>
            {docs.length ? docs.map((d) => (
              <tr key={d.id}>
                <td><div className="doc-cell"><span className="doc-icon"><FileText size={18} /></span><div><button className="title link" type="button" onClick={() => openDoc(d.id)}>{d.title && d.title !== d.filename ? d.title : d.filename}</button><small>{d.filename}{fmtSize(d.size_bytes) ? ` - ${fmtSize(d.size_bytes)}` : ""}</small></div></div></td>
                <td>{d.doc_type_label || "-"}</td>
                <td><Badge tone={`st-${d.status}`}>{d.status}</Badge>{d.error ? <div className="error small">{d.error}</div> : null}</td>
                <td>{d.page_count ?? "-"}</td>
                <td>{fmtDate(d.created_at)}</td>
                <td className="actions">
                  <button className="ghost small" title="Process again" onClick={() => reprocess(d.id)}><RotateCw size={13} />Reprocess</button>
                  <button className="ghost small danger" title="Delete" onClick={() => removeDoc(d.id, d.filename)}><Trash2 size={13} />Delete</button>
                </td>
              </tr>
            )) : <tr><td colSpan="6" className="muted">No documents found.</td></tr>}
          </tbody>
        </table>
        <div className="pager">
          <button className="ghost" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}><ChevronLeft size={16} />Previous</button>
          <span className="muted">{from}-{Math.min(offset + limit, total)} of {total}</span>
          <button className="ghost" disabled={offset + limit >= total} onClick={() => setOffset(offset + limit)}>Next<ChevronRight size={16} /></button>
        </div>
      </section>
    </section>
  );
}

function Stat({ label, value, sub, tone = "", icon }) {
  return (
    <div className={`tile ${tone}`}>
      <span className="tile-icon">{icon}</span>
      <div className="tile-label">{label}</div>
      <div className="tile-value">{Number(value || 0).toLocaleString()}</div>
      <div className="tile-sub">{sub}</div>
    </div>
  );
}

function DocumentDrawer({ api, id, onClose }) {
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState("");
  const [page, setPage] = useState(null);

  useEffect(() => {
    let active = true;
    setDoc(null);
    setError("");
    setPage(null);
    api(`/api/v1/documents/${id}`)
      .then((d) => {
        if (!active) return;
        setDoc(d);
        if (d.pages?.length) return loadPage(api, d.id, d.pages[0].page_number).then((p) => active && setPage(p));
        return null;
      })
      .catch((err) => active && setError(err.message));
    return () => { active = false; };
  }, [api, id]);

  const showPage = async (n) => {
    setPage({ loading: true });
    try {
      setPage(await loadPage(api, id, n));
    } catch (err) {
      setPage({ error: err.message });
    }
  };

  const download = async () => {
    if (!doc) return;
    const res = await api(`/api/v1/documents/${doc.id}/original`, { raw: true });
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = doc.filename;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };

  return (
    <aside className="drawer">
      <div className="drawer-head">
        <h2>{error || doc?.title || doc?.filename || "Loading..."}</h2>
        <button className="ghost" onClick={onClose} aria-label="Close">Close</button>
      </div>
      <div className="drawer-body">
        {error ? <p className="error">{error}</p> : null}
        {!doc && !error ? <p className="muted">Loading...</p> : null}
        {doc ? (
          <>
            <div className="row">
              <button className="primary small" onClick={download}><Download size={14} />Download original</button>
            </div>
            <DrawerSection title="Summary">
              <KV label="File" value={doc.filename} />
              <KV label="Type" value={doc.doc_type_label ? `${doc.doc_type_label} (${Math.round((doc.doc_type_confidence || 0) * 100)}%, ${doc.doc_type_method})` : null} />
              <KV label="Status" value={doc.status} />
              <KV label="Kind" value={doc.kind} />
              <KV label="Pages" value={doc.page_count} />
              <KV label="Words" value={doc.word_count} />
              <KV label="Signature" value={doc.has_signature ? "signature or signer line found" : "none found"} />
              <KV label="Processed in" value={doc.processing_ms ? `${(doc.processing_ms / 1000).toFixed(1)} s` : null} />
              <KV label="Error" value={doc.error} />
            </DrawerSection>
            <DrawerSection title="Extracted fields">
              {doc.fields?.length ? (
                <table className="mini"><tbody>{doc.fields.map((f, i) => <tr key={`${f.name}-${i}`}><td>{f.name.replace(/_/g, " ")}</td><td>{f.value_text ?? ""}</td><td className="muted small">p. {f.page ?? "?"} - {Math.round((f.confidence || 0) * 100)}%</td></tr>)}</tbody></table>
              ) : <p className="muted">No fields extracted.</p>}
            </DrawerSection>
            <DrawerSection title="People, organizations and identifiers">
              {doc.entities?.length ? (
                <div className="fields">{doc.entities.map((e, i) => <span className="field" key={`${e.value}-${i}`}><b>{e.role ? `${e.type} - ${e.role}` : e.type}</b> {e.value}</span>)}</div>
              ) : <p className="muted">None found.</p>}
            </DrawerSection>
            <DrawerSection title="Clauses">
              {doc.clauses?.length ? doc.clauses.map((c, i) => (
                <details key={`${c.clause_type}-${i}`}><summary>{c.clause_type.replace(/_/g, " ")}{c.ref ? ` - ${c.ref}` : ""} - p. {c.page}</summary><p>{c.text}</p></details>
              )) : <p className="muted">None found.</p>}
            </DrawerSection>
            <DrawerSection title="Pages">
              <div className="row wrap">
                {(doc.pages || []).map((p) => <button className="ghost small" key={p.page_number} onClick={() => showPage(p.page_number)}>Page {p.page_number}</button>)}
              </div>
              <PageView page={page} />
            </DrawerSection>
          </>
        ) : null}
      </div>
    </aside>
  );
}

function DrawerSection({ title, children }) {
  return <section className="dsec"><h3>{title}</h3>{children}</section>;
}

function KV({ label, value }) {
  if (value === null || value === undefined || value === "") return null;
  return <div className="kv"><span>{label}</span><b>{String(value)}</b></div>;
}

async function loadPage(api, id, n) {
  const p = await api(`/api/v1/documents/${id}/pages/${n}`);
  let imageUrl = "";
  try {
    const res = await api(`/api/v1/documents/${id}/pages/${n}/image`, { raw: true });
    imageUrl = URL.createObjectURL(await res.blob());
  } catch {
    /* no image for this format */
  }
  return { ...p, imageUrl };
}

function PageView({ page }) {
  if (!page) return null;
  if (page.loading) return <p className="muted">Loading page...</p>;
  if (page.error) return <p className="error">{page.error}</p>;
  return (
    <div className="page-view">
      <p className="muted small">Page {page.page_number} - {page.kind}{page.ocr_confidence ? ` - OCR confidence ${Math.round(page.ocr_confidence * 100)}%` : ""}</p>
      {page.imageUrl ? <img className="page-img" src={page.imageUrl} alt={`Page ${page.page_number}`} /> : null}
      <pre className="page-text">{page.text || "(no text)"}</pre>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
