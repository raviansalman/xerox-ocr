-- Arabic support: the stemmed index is built from stem_text (search_text with Arabic words reduced to their light
-- stems by docintel.text.stem_text) instead of search_text. English words are still stemmed by the 'english'
-- configuration. Rows written before this migration have no stem_text and keep using search_text, exactly as before;
-- documents are reprocessed when the pipeline version reports them stale. Pure DDL: migrations run under row-level
-- security, where a data UPDATE would see no rows.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS stem_text text;
ALTER TABLE chunks DROP COLUMN tsv_en;
ALTER TABLE chunks ADD COLUMN tsv_en tsvector
    GENERATED ALWAYS AS (to_tsvector('english'::regconfig, coalesce(stem_text, search_text))) STORED;
