import { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { EXAMPLES, KEY_STORE } from "./constants.js";
import { AskPage } from "./components/AskPage.jsx";
import { WorkspaceConnection } from "./components/WorkspaceConnection.jsx";
import { DocumentDrawer } from "./components/DocumentDrawer.jsx";
import { DocumentsPage } from "./components/DocumentsPage.jsx";
import { Header } from "./components/Header.jsx";
import "./styles.css";

function App() {
  const [apiKey, setApiKey] = useState(() => {
    try { return localStorage.getItem(KEY_STORE) || ""; } catch { return ""; }
  });
  const [keyError, setKeyError] = useState("");
  const [connectionOpen, setConnectionOpen] = useState(false);
  const [connected, setConnected] = useState(false);
  const [tenant, setTenant] = useState("Guest");
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
      setTenant("Guest");
      throw new Error(apiKey ? "Your API key was not accepted. Update it using the key button in the header." : "Connect with an API key using the key button in the header to access workspace data.");
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
    const key = candidate ?? apiKey;
    if (!key) {
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
      setApiKey(key);
      setConnected(true);
      setKeyError("");
      try { localStorage.setItem(KEY_STORE, key); } catch { /* storage unavailable */ }
      return true;
    } catch (err) {
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

  const disconnect = () => {
    try { localStorage.removeItem(KEY_STORE); } catch { /* storage unavailable */ }
    setApiKey("");
    setConnected(false);
    setTenant("Guest");
    setExamples(EXAMPLES);
    setTypes([]);
    setDrawerDocId(null);
    setKeyError("");
    setConnectionOpen(false);
  };

  return (
    <>
      <Header
        tab={tab}
        setTab={setTab}
        tenant={tenant}
        healthReady={healthReady}
        onConnection={() => { setKeyError(""); setConnectionOpen(true); }}
      />
      <main className="app-main" key={`${apiKey}:${connected}`}>
        {tab === "ask" ? (
          <AskPage api={request} examples={examples} openDoc={setDrawerDocId} />
        ) : (
          <DocumentsPage api={request} types={types} openDoc={setDrawerDocId} />
        )}
      </main>
      {drawerDocId ? <DocumentDrawer api={request} id={drawerDocId} onClose={() => setDrawerDocId(null)} /> : null}
      {connectionOpen && <WorkspaceConnection
        apiKey={apiKey}
        connected={connected}
        error={keyError}
        onConnect={connect}
        onDisconnect={disconnect}
        onClose={() => setConnectionOpen(false)}
      />}
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
