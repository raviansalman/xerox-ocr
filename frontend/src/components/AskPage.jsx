import { useEffect, useState } from "react";
import { ArrowRight, Check, Clipboard, LoaderCircle, Search, ThumbsUp, X } from "lucide-react";
import { MATCH_LABEL } from "../constants.js";
import { firstSentence, fmtNumber, highlight, resultConfidence } from "../utils.jsx";
import { Badge, Dot } from "./Common.jsx";

export function AskPage({ api, examples, openDoc }) {
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
