import { useEffect, useRef, useState } from "react";
import { ArrowRight, Eye, EyeOff, LoaderCircle, Lock, X } from "lucide-react";

export function WorkspaceConnection({ apiKey, connected, error, onConnect, onDisconnect, onClose }) {
  const dialog = useRef(null);
  const [keyInput, setKeyInput] = useState(apiKey);
  const [showKey, setShowKey] = useState(false);
  const [pending, setPending] = useState(false);

  useEffect(() => { dialog.current.showModal(); }, []);

  const submit = async (ev) => {
    ev.preventDefault();
    if (pending) return;
    if (!keyInput.trim()) { onClose(); return; }
    setPending(true);
    const accepted = await onConnect(keyInput.trim());
    if (accepted) onClose();
    else setPending(false);
  };

  return (
    <dialog ref={dialog} className="connection-dialog" aria-labelledby="connection-title" onCancel={onClose}>
      <div className="connection-heading">
        <h2 id="connection-title">Workspace connection</h2>
        <button type="button" className="icon-button" aria-label="Close" title="Close" onClick={onClose}><X size={20} /></button>
      </div>
      <form className="connect-form" onSubmit={submit}>
        <label htmlFor="workspaceKey">API key (optional)</label>
        <div className={`key-entry ${error ? "invalid" : ""}`}>
          <Lock size={19} />
          <input id="workspaceKey" type={showKey ? "text" : "password"} autoComplete="off" value={keyInput}
            onChange={(ev) => setKeyInput(ev.target.value)} aria-invalid={!!error} aria-describedby={error ? "connection-error" : undefined} />
          <button type="button" className="eye-button" aria-label={showKey ? "Hide API key" : "Show API key"} onClick={() => setShowKey((value) => !value)}>
            {showKey ? <EyeOff size={19} /> : <Eye size={19} />}
          </button>
        </div>
        {error && <p id="connection-error" className="error" role="alert">{error}</p>}
        <div className="connection-actions">
          {(connected || apiKey) && <button type="button" className="ghost" disabled={pending} onClick={onDisconnect}>Disconnect</button>}
          <button type="submit" className="primary" disabled={pending}>
            {pending ? <LoaderCircle className="spin" size={18} /> : null}
            {keyInput.trim() ? "Connect" : "Continue"}<ArrowRight size={18} />
          </button>
        </div>
      </form>
    </dialog>
  );
}
