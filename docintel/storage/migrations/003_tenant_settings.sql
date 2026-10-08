-- Per-tenant configuration kept in the database, not in code: enabled domain packs and the example questions the
-- web UI offers. Row-level security as for every tenant table.
CREATE TABLE tenant_settings (
    tenant_id   text PRIMARY KEY,
    packs       text[],                          -- NULL: the deployment default (DOCINTEL_DEFAULT_PACKS)
    examples    jsonb NOT NULL DEFAULT '[]',     -- [{"label": ..., "q": ...}]
    updated_at  timestamptz NOT NULL DEFAULT now(),
    updated_by  text
);
ALTER TABLE tenant_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_settings FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON tenant_settings
    USING (tenant_id = current_setting('docintel.tenant', true))
    WITH CHECK (tenant_id = current_setting('docintel.tenant', true));
