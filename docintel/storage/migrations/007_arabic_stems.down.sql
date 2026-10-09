ALTER TABLE chunks DROP COLUMN tsv_en;
ALTER TABLE chunks ADD COLUMN tsv_en tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, search_text)) STORED;
ALTER TABLE chunks DROP COLUMN IF EXISTS stem_text;
