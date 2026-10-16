export function Brand() {
  return (
    <div className="brand">
      <span className="logo" aria-hidden="true"><span /><span /><span /></span>
      <span>Document Intelligence</span>
    </div>
  );
}

export function Dot({ status = "ok" }) {
  return <span className={`dot ${status}`} />;
}

export function Badge({ children, tone = "" }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

export function Stat({ label, value, sub, tone = "", icon }) {
  return (
    <div className={`tile ${tone}`}>
      <span className="tile-icon">{icon}</span>
      <div className="tile-label">{label}</div>
      <div className="tile-value">{Number(value || 0).toLocaleString()}</div>
      <div className="tile-sub">{sub}</div>
    </div>
  );
}
