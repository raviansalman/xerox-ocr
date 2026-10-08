-- Lexical index as ordinary rows: one posting per (term, unit).
--
-- Row-level security keeps PostgreSQL from using full-text, trigram and array indexes (their operators are not
-- leakproof), so every lexical lookup would scan all of a tenant's chunks. Text equality and range comparison are
-- leakproof, so a B-tree on (tenant_id, term) is used under the same tenant policy: exact, phrase, identifier,
-- prefix, suffix and stemmed lookups become index lookups, and isolation is unchanged.
--
-- Term prefixes: w: exact token   s: English stem   f: OCR-folded token (only where it differs from the token)
--                i: canonical identifier   r: reversed identifier-like token (suffix search)   n: file-name token
CREATE TABLE unit_terms (
    tenant_id    text NOT NULL,
    term         text COLLATE "C" NOT NULL,
    unit_id      text NOT NULL,
    document_id  uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tf           smallint NOT NULL
);
CREATE INDEX unit_terms_lookup ON unit_terms (tenant_id, term, unit_id) INCLUDE (document_id, tf);
CREATE INDEX unit_terms_document ON unit_terms (document_id);

-- Distinct words per tenant, for typo correction against the tenant's own vocabulary.
CREATE TABLE vocabulary (
    tenant_id  text NOT NULL,
    term       text COLLATE "C" NOT NULL,
    length     smallint NOT NULL,
    PRIMARY KEY (tenant_id, term)
);
CREATE INDEX vocabulary_length ON vocabulary (tenant_id, length);

ALTER TABLE chunks ADD COLUMN n_terms int NOT NULL DEFAULT 0;

-- Entity lookups by normalized value regardless of type (text equality is leakproof, so this is used under RLS).
CREATE INDEX entities_value ON entities (tenant_id, value_norm);

-- These indexes cannot be used under row-level security (see above) and only cost writes.
DROP INDEX IF EXISTS chunks_tsv;
DROP INDEX IF EXISTS chunks_tsv_en;
DROP INDEX IF EXISTS chunks_tsv_fold;
DROP INDEX IF EXISTS chunks_search_trgm;
DROP INDEX IF EXISTS documents_filename_trgm;
DROP INDEX IF EXISTS chunks_idents;
DROP INDEX IF EXISTS entities_norm_trgm;

DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['unit_terms', 'vocabulary'] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
        EXECUTE format($p$CREATE POLICY tenant_isolation ON %I
                       USING (tenant_id = current_setting('docintel.tenant', true))
                       WITH CHECK (tenant_id = current_setting('docintel.tenant', true))$p$, t);
    END LOOP;
END $$;
