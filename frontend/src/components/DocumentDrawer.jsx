import { useEffect, useState } from "react";
import { Download } from "lucide-react";

export function DocumentDrawer({ api, id, onClose }) {
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
