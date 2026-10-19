import { useCallback, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { EXAMPLES } from './constants.js';
import { AskPage } from './components/AskPage.jsx';
import { DocumentDrawer } from './components/DocumentDrawer.jsx';
import { DocumentsPage } from './components/DocumentsPage.jsx';
import { Header } from './components/Header.jsx';
import './styles.css';

function App() {
  const [tenant, setTenant] = useState('Workspace');
  const [healthReady, setHealthReady] = useState(false);
  const [tab, setTab] = useState('ask');
  const [examples, setExamples] = useState(EXAMPLES);
  const [types, setTypes] = useState([]);
  const [drawerDocId, setDrawerDocId] = useState(null);

  const request = useCallback(
    async (path, opts = {}) => {
      const headers = { ...(opts.headers || {}) };
      const init = { ...opts, headers };
      if (opts.json !== undefined) {
        init.body = JSON.stringify(opts.json);
        init.headers = { ...headers, 'Content-Type': 'application/json' };
      }
      const res = await fetch(path, init);
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const body = await res.json();
          detail =
            typeof body.detail === 'string'
              ? body.detail
              : JSON.stringify(body.detail);
        } catch {
          /* ignore */
        }
        throw new Error(detail || `HTTP ${res.status}`);
      }
      return opts.raw ? res : res.json();
    },
    [],
  );

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('/health/ready');
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
    let active = true;
    request('/api/v1/me').then((me) => {
      if (active) setTenant(me.tenant_id);
    }).catch(() => {});
    request('/api/v1/settings').then((settings) => {
      if (active && settings.examples?.length) {
        setExamples(settings.examples.map((e) => [e.label, e.q]));
      }
    }).catch(() => {});
    request('/api/v1/taxonomy').then((tax) => {
      if (active) setTypes(tax.types || []);
    }).catch(() => {});
    return () => { active = false; };
  }, [request]);

  return (
    <>
      <Header
        tab={tab}
        setTab={setTab}
        tenant={tenant}
        healthReady={healthReady}
      />
      <main className='app-main'>
        {tab === 'ask' ? (
          <AskPage api={request} examples={examples} openDoc={setDrawerDocId} />
        ) : (
          <DocumentsPage api={request} types={types} openDoc={setDrawerDocId} />
        )}
      </main>
      {drawerDocId ? (
        <DocumentDrawer
          api={request}
          id={drawerDocId}
          onClose={() => setDrawerDocId(null)}
        />
      ) : null}
    </>
  );
}

createRoot(document.getElementById('root')).render(<App />);
