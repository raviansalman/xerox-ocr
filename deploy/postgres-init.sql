-- Runs once when the PostgreSQL container initializes (docker-entrypoint-initdb.d).
-- The application connects as docintel_app: NOT a superuser and without BYPASSRLS, so row-level security
-- (tenant isolation) is enforced on every query. The password comes from the DOCINTEL_DB_PASSWORD variable.
\set app_password `echo "$DOCINTEL_DB_PASSWORD"`
CREATE ROLE docintel_app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOCREATEDB PASSWORD :'app_password';  -- hardcoded-ok(credential): a psql variable, not a value
-- Byte-order collation: prefix and range lookups on normalized names and terms rely on it.
CREATE DATABASE docintel OWNER docintel_app TEMPLATE template0 ENCODING 'UTF8' LC_COLLATE 'C.UTF-8' LC_CTYPE 'C.UTF-8';
\connect docintel
CREATE EXTENSION IF NOT EXISTS pg_trgm;
