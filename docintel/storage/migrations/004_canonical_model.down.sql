ALTER TABLE chunks DROP COLUMN IF EXISTS unit_type, DROP COLUMN IF EXISTS char_start, DROP COLUMN IF EXISTS char_end,
    DROP COLUMN IF EXISTS block_from, DROP COLUMN IF EXISTS block_to, DROP COLUMN IF EXISTS context;
ALTER TABLE clauses DROP COLUMN IF EXISTS char_start, DROP COLUMN IF EXISTS char_end, DROP COLUMN IF EXISTS block_ordinal;
ALTER TABLE entities DROP COLUMN IF EXISTS char_start, DROP COLUMN IF EXISTS char_end, DROP COLUMN IF EXISTS block_ordinal;
ALTER TABLE fields DROP COLUMN IF EXISTS char_start, DROP COLUMN IF EXISTS char_end, DROP COLUMN IF EXISTS block_ordinal;
DROP TABLE IF EXISTS relations, table_cells, doc_tables, blocks, document_versions;
ALTER TABLE documents DROP COLUMN IF EXISTS current_version, DROP COLUMN IF EXISTS components;
