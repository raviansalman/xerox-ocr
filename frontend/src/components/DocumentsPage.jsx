import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Circle,
  Clipboard,
  FileText,
  HelpCircle,
  RefreshCw,
  RotateCw,
  Search,
  Trash2,
  Upload,
  X
} from "lucide-react";
import { fmtDate, fmtSize, indexedWorkspaceSub, refreshLabel } from "../utils.jsx";
import { Badge, Dot, Stat } from "./Common.jsx";

export function DocumentsPage({ api, types, openDoc }) {
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
