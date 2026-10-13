import { ArrowRight, BadgeCheck, Check, Eye, Lock, Search, ShieldCheck } from "lucide-react";
import { Brand, Dot } from "./Common.jsx";

export function ConnectScreen({ keyInput, setKeyInput, showKey, setShowKey, keyError, submitKey }) {
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
