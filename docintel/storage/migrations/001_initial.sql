-- Document Intelligence Engine: initial schema.
-- Every tenant table has row-level security bound to the per-transaction setting docintel.tenant.
-- A connection that has not set it sees no rows.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE documents (
    id                   uuid PRIMARY KEY,
    tenant_id            text NOT NULL,
    collection           text,
    filename             text NOT NULL,
    filename_search      text NOT NULL,
    title                text,
    mime_type            text,
    file_ext             text,
    size_bytes           bigint NOT NULL,
    sha256               char(64) NOT NULL,
    storage_key          text NOT NULL,
    source               text NOT NULL DEFAULT 'upload',
    source_url           text,
    status               text NOT NULL,            -- queued | processing | indexed | failed
    error                text,
    kind                 text,                     -- native | scanned | mixed | image | office | text | spreadsheet | presentation | email
    page_count           int,
    word_count           int,
    language             text,
    doc_type             text,
    doc_type_confidence  real,
    doc_type_method      text,
    has_signature        boolean,
    pipeline_version     text,
    processing_ms        int,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now(),
    indexed_at           timestamptz,
    UNIQUE (tenant_id, sha256)
);
CREATE INDEX documents_tenant_status ON documents (tenant_id, status);
CREATE INDEX documents_tenant_type ON documents (tenant_id, doc_type);
CREATE INDEX documents_tenant_created ON documents (tenant_id, created_at DESC);
CREATE INDEX documents_filename_trgm ON documents USING gin (filename_search gin_trgm_ops);

CREATE TABLE pages (
    document_id     uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tenant_id       text NOT NULL,
    page_number     int NOT NULL,
    kind            text NOT NULL,                 -- native | ocr | sheet | slide | section
    width           real,
    height          real,
    ocr_confidence  real,
    text            text NOT NULL,
    words           jsonb,                         -- OCR words: [[text, x0, y0, x1, y1, conf], ...]
    PRIMARY KEY (document_id, page_number)
);

CREATE TABLE chunks (
    id            text PRIMARY KEY,
    document_id   uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tenant_id     text NOT NULL,
    ordinal       int NOT NULL,
    page_start    int NOT NULL,
    page_end      int NOT NULL,
    heading       text,
    text          text NOT NULL,
    search_text   text NOT NULL,
    fold_text     text NOT NULL,
    idents        text[] NOT NULL DEFAULT '{}',
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('simple'::regconfig, search_text)) STORED,
    tsv_en        tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, search_text)) STORED,
    tsv_fold      tsvector GENERATED ALWAYS AS (to_tsvector('simple'::regconfig, fold_text)) STORED
);
CREATE INDEX chunks_document ON chunks (document_id, ordinal);
CREATE INDEX chunks_tenant ON chunks (tenant_id);
CREATE INDEX chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX chunks_tsv_en ON chunks USING gin (tsv_en);
CREATE INDEX chunks_tsv_fold ON chunks USING gin (tsv_fold);
CREATE INDEX chunks_idents ON chunks USING gin (idents);
CREATE INDEX chunks_search_trgm ON chunks USING gin (search_text gin_trgm_ops);

CREATE TABLE fields (
    id           bigserial PRIMARY KEY,
    document_id  uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tenant_id    text NOT NULL,
    name         text NOT NULL,
    value_text   text,
    value_num    numeric,
    value_date   date,
    unit         text,
    page         int,
    snippet      text,
    confidence   real NOT NULL,
    method       text NOT NULL
);
CREATE INDEX fields_lookup ON fields (tenant_id, name);
CREATE INDEX fields_document ON fields (document_id);
CREATE INDEX fields_date ON fields (tenant_id, name, value_date);
CREATE INDEX fields_num ON fields (tenant_id, name, value_num);

CREATE TABLE entities (
    id           bigserial PRIMARY KEY,
    document_id  uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tenant_id    text NOT NULL,
    type         text NOT NULL,                    -- person | organization | email | phone | identifier | jurisdiction
    value        text NOT NULL,
    value_norm   text NOT NULL,
    role         text,
    page         int,
    snippet      text,
    confidence   real NOT NULL,
    method       text NOT NULL
);
CREATE INDEX entities_lookup ON entities (tenant_id, type, value_norm);
CREATE INDEX entities_document ON entities (document_id);
CREATE INDEX entities_norm_trgm ON entities USING gin (value_norm gin_trgm_ops);

CREATE TABLE clauses (
    id           bigserial PRIMARY KEY,
    document_id  uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tenant_id    text NOT NULL,
    clause_type  text NOT NULL,
    ref          text,
    heading      text,
    page         int,
    text         text NOT NULL,
    confidence   real NOT NULL
);
CREATE INDEX clauses_lookup ON clauses (tenant_id, clause_type);
CREATE INDEX clauses_document ON clauses (document_id);

CREATE TABLE audit_events (
    id         bigserial PRIMARY KEY,
    tenant_id  text NOT NULL,
    actor      text NOT NULL,
    action     text NOT NULL,
    target     text,
    detail     jsonb,
    at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX audit_tenant_at ON audit_events (tenant_id, at DESC);

DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['documents', 'pages', 'chunks', 'fields', 'entities', 'clauses', 'audit_events'] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
        EXECUTE format($p$CREATE POLICY tenant_isolation ON %I
                       USING (tenant_id = current_setting('docintel.tenant', true))
                       WITH CHECK (tenant_id = current_setting('docintel.tenant', true))$p$, t);
    END LOOP;
END $$;
