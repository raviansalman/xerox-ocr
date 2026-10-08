-- Canonical document model v2: processing versions, blocks, tables and cells, relations, and character spans for
-- every annotation (offsets into the page's canonical text; see docintel/models.py).

CREATE TABLE document_versions (
    document_id   uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tenant_id     text NOT NULL,
    number        int NOT NULL,
    status        text NOT NULL,                  -- building | current | superseded | failed
    components    jsonb NOT NULL,                 -- pipeline, parser, OCR engine, embedding model, packs
    started_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz,
    error         text,
    PRIMARY KEY (document_id, number)
);
CREATE INDEX document_versions_tenant ON document_versions (tenant_id, status);

ALTER TABLE documents ADD COLUMN current_version int;
ALTER TABLE documents ADD COLUMN components jsonb;

CREATE TABLE blocks (
    document_id   uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tenant_id     text NOT NULL,
    page          int NOT NULL,
    ordinal       int NOT NULL,                   -- reading order within the page
    block_type    text NOT NULL,                  -- heading | paragraph | list_item | table | key_value | ...
    text          text NOT NULL,
    char_start    int NOT NULL,                   -- offset of the block in the page's canonical text
    bbox          real[],
    confidence    real,
    table_ordinal int,
    PRIMARY KEY (document_id, page, ordinal)
);

CREATE TABLE doc_tables (
    document_id   uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tenant_id     text NOT NULL,
    page          int NOT NULL,
    ordinal       int NOT NULL,
    block_ordinal int,
    n_rows        int NOT NULL,
    n_cols        int NOT NULL,
    header_rows   int NOT NULL,
    bbox          real[],
    caption       text,
    PRIMARY KEY (document_id, page, ordinal)
);

CREATE TABLE table_cells (
    document_id   uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tenant_id     text NOT NULL,
    page          int NOT NULL,
    table_ordinal int NOT NULL,
    row_index     int NOT NULL,
    col_index     int NOT NULL,
    text          text NOT NULL,
    header        text,                           -- the column header for data cells
    value_num     numeric,                        -- numeric reading of the cell, when it is a number or amount
    PRIMARY KEY (document_id, page, table_ordinal, row_index, col_index)
);
CREATE INDEX table_cells_header ON table_cells (tenant_id, header);

CREATE TABLE relations (
    id            bigserial PRIMARY KEY,
    document_id   uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tenant_id     text NOT NULL,
    predicate     text NOT NULL,
    subject_type  text NOT NULL,
    subject       text NOT NULL,
    subject_norm  text NOT NULL,
    object_type   text NOT NULL,
    object        text NOT NULL,
    object_norm   text NOT NULL,
    qualifiers    jsonb NOT NULL DEFAULT '{}',
    page          int,
    char_start    int,
    char_end      int,
    snippet       text,
    confidence    real NOT NULL,
    extractor     text NOT NULL
);
CREATE INDEX relations_predicate ON relations (tenant_id, predicate);
CREATE INDEX relations_subject ON relations (tenant_id, subject_norm);
CREATE INDEX relations_object ON relations (tenant_id, object_norm);
CREATE INDEX relations_document ON relations (document_id);

ALTER TABLE fields ADD COLUMN char_start int, ADD COLUMN char_end int, ADD COLUMN block_ordinal int;
ALTER TABLE entities ADD COLUMN char_start int, ADD COLUMN char_end int, ADD COLUMN block_ordinal int;
ALTER TABLE clauses ADD COLUMN char_start int, ADD COLUMN char_end int, ADD COLUMN block_ordinal int;

ALTER TABLE chunks ADD COLUMN unit_type text NOT NULL DEFAULT 'passage';
ALTER TABLE chunks ADD COLUMN char_start int, ADD COLUMN char_end int, ADD COLUMN block_from int, ADD COLUMN block_to int;
ALTER TABLE chunks ADD COLUMN context text NOT NULL DEFAULT '';

DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['document_versions', 'blocks', 'doc_tables', 'table_cells', 'relations'] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
        EXECUTE format($p$CREATE POLICY tenant_isolation ON %I
                       USING (tenant_id = current_setting('docintel.tenant', true))
                       WITH CHECK (tenant_id = current_setting('docintel.tenant', true))$p$, t);
    END LOOP;
END $$;
