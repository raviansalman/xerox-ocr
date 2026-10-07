# Security architecture (target)

Status: design for review. The **current** boundary (API keys, roles, tenant from identity, Redis/pickle fixes,
SSRF guard, health) is documented and tested in `SECURITY.md`; this document defines the target for the Document
Intelligence Engine and what changes when new stores and AI components are added.

Hard rule: **tenant leakage = 0 is a release gate.** Cross-tenant tests run in CI on every change.

## 1. Assets and threats

| Asset | Threats |
|---|---|
| Document content, page images, extracted fields | Cross-tenant read; over-broad access within a tenant; exfiltration via search, counts, summaries, filenames, "did you mean" |
| Derived knowledge (classifications, counts) | Inference attacks (counting another tenant's documents), aggregation leaks |
| Embeddings | Inversion or bulk export from an exposed vector store |
| Credentials, keys, model weights | Secret leakage in repo, logs, images |
| Processing workers | Malicious files (parser exploits, zip bombs, macros), SSRF via URL ingest |
| LLM components | Prompt injection from document text, data egress to external providers, tool misuse |
| Audit trail | Tampering, gaps |

Evidence that drives the design: restoring the legacy shared `MetadataIndex` produced cross-tenant results through
`/search` (19 case/mode runs, `SEARCH_FORENSICS.md` E2); rows without a tenant fall back to `default_user`
(`HARDCODED_DATA_AUDIT.md` C1); filter values were once injectable (KD-SEC-01, fixed).

## 2. Identity and AuthContext

```
credential (API key now; OIDC/SAML via the customer IdP later)
   ─► verify ─► AuthContext { tenant_id, principal_id, roles, acl_groups, key_id, request_id }   (immutable)
```

* Tenant comes **only** from the verified identity. Service principals that act for several tenants (for example
  a connector) must state the tenant per request and are restricted by an allow-list in their credential.
* Roles: `reader`, `uploader`, `admin`, `service` (existing), plus `auditor` (read audit log) and
  `reviewer` (confirm uncertain extractions).
* Within a tenant: `acl_groups` on documents/collections; a principal sees a document if any group matches or the
  document is tenant-public.
* Keys: hashed at rest (existing), expiry and rotation, per-key rate limits.

## 3. Enforcement at every data access

| Layer | Mechanism |
|---|---|
| Code | Every repository and retriever method takes `AuthContext` as a required argument; there is no overload without it. Static check: no direct database/Milvus client use outside `storage/` and `retrieval/` |
| PostgreSQL | Row-level security on every tenant table: `USING (tenant_id = current_setting('die.tenant') AND (acl_groups && current_setting('die.groups')::text[] OR is_public))`; the setting is applied per transaction by the repository from `AuthContext`; application role without `BYPASSRLS`; a missing setting yields no rows |
| Milvus | `tenant_id` filter (partition key in the target collection) and ACL group filter built from typed, validated values (no string interpolation of untyped input) |
| Object storage | Per-tenant prefix; access only via signed, short-lived URLs issued after a registry authorization check |
| Caches | Keys always start with `tenant_id`; no in-process per-tenant singletons (the E2 failure mode) |
| Response | Final check drops any item whose `tenant_id`/ACL does not match the context and raises a security event |

Aggregations, counts, filenames, classifications, suggestions and RAG context are all derived from these
filtered accesses. No query ever computes over all tenants "and filters later".

## 4. AI-specific controls

* **Planner isolation:** the planner sees the question and the field schema only; it cannot set tenant or ACL
  (`QUERY_PLANNER.md`). Plans are validated against an allow-list.
* **Prompt injection:** document text reaches only the answer generator, framed as untrusted data; the generator
  has no tools and cannot change the plan; outputs are checked for citations that exist in authorized evidence.
* **Egress:** default deployment uses on-prem models only; any external model provider requires an explicit,
  per-tenant configuration and is logged. No document content is sent to third parties by default.
* **Extraction hallucination:** extracted values must have a source span or region; values without one are
  rejected.
* **Model supply chain:** model artefacts pinned by name, revision and sha256 in the model registry; loaded
  offline from the internal registry.

## 5. Ingestion safety

Existing: SSRF guard with public-IP-only targets and redirect re-checks, size caps. Target additions: magic-byte
type detection, archive expansion limits, page/pixel limits, parser time limits, workers without outbound network,
no macro execution, malware scanning hook (optional, customer-provided).

## 6. Secrets and configuration

No secrets in the repository or images; secrets from mounted files or a secret manager; startup refuses to run
with default or empty secrets in production mode; logs redact credentials and document text by default (text
logging only at debug level in non-production).

## 7. Network and data protection

TLS at the edge and between services where they cross hosts; Milvus authentication enabled; Redis password
(existing) and no public exposure; Postgres TLS and least-privilege roles (migrations role vs application role);
encryption at rest for volumes and object storage (customer KMS where available); backups encrypted.

## 8. Audit

Append-only `audit_events`: authentication failures, searches (query hash, plan summary, result counts, not full
text by default), document access, ingest, delete, reprocess, configuration changes, security events from the
final check. Retention per tenant policy; exportable to the customer SIEM.

## 9. Security testing

* Cross-tenant suite (golden tenants, extended): every retriever, every aggregation, RAG, suggestions, filenames,
  delete, reprocess. Gate: zero foreign items.
* RLS tests: queries without the tenant setting return nothing; application role cannot bypass.
* Injection tests: filter values, plan fields, OCR'd text with instructions, filenames with markup.
* Authorization bypass: role escalation, ACL group spoofing, signed URL reuse across tenants.
* Dependency and image scanning in CI.

## 10. Current gaps carried forward (from `SECURITY.md` section 8)

TLS/reverse proxy, Milvus authentication, rate limits and quotas, audit log, OIDC/SSO and key expiry, embedder
authentication, prompt-injection handling, encryption at rest; plus `default_user` fallback (fail closed in S1),
source bind mounts in production (S1), and the repository being public (owner action).
