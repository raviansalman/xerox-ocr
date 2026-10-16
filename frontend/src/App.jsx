import { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { EXAMPLES, KEY_STORE } from "./constants.js";
import { AskPage } from "./components/AskPage.jsx";
import { ConnectScreen } from "./components/ConnectScreen.jsx";
import { DocumentDrawer } from "./components/DocumentDrawer.jsx";
import { DocumentsPage } from "./components/DocumentsPage.jsx";
import { Header } from "./components/Header.jsx";
import "./styles.css";

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

createRoot(document.getElementById("root")).render(<App />);
