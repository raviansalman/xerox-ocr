export function fmtSize(n) {
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

export function fmtDate(iso) {
  return iso ? new Date(iso).toLocaleString() : "";
}

export function fmtNumber(n) {
  return Number(n || 0).toLocaleString();
}

function statusTotal(stats) {
  return Object.values(stats?.by_status || {}).reduce((sum, n) => sum + Number(n || 0), 0);
}

export function indexedWorkspaceSub(stats) {
  const total = statusTotal(stats);
  if (!total) return "Indexed and searchable";
  const indexed = Number(stats?.indexed_documents || 0);
  return `${((indexed / total) * 100).toFixed(1)}% of workspace`;
}

export function refreshLabel(date) {
  if (!date) return "Live";
  const seconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (seconds < 60) return "Live - refreshed now";
  const minutes = Math.round(seconds / 60);
  return `Live - refreshed ${minutes}m ago`;
}

export function resultConfidence(results) {
  const values = (results || []).map((r) => Number(r.confidence)).filter(Number.isFinite);
  if (!values.length) return null;
  return Math.round(Math.max(...values) * 100);
}

export function firstSentence(text) {
  return (text || "").split(/(?<=[.!?])\s+/)[0] || "";
}

export function highlight(text, terms) {
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
