import { KeyRound } from "lucide-react";
import { Brand, Dot } from "./Common.jsx";

export function Header({ tab, setTab, tenant, healthReady, onConnection }) {
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
        <button className="icon-button" title="Workspace connection" aria-label="Workspace connection" onClick={onConnection}><KeyRound size={20} /></button>
      </div>
    </header>
  );
}
