# Domain packs and the no-hardcoded-data rule

## Generic core

The core understands documents without knowing any business domain:

* structure: pages, blocks, headings, sections, lists, tables, key-value pairs, figures, signatures;
* values: dates and periods, money with currency, quantities with units, percentages, durations, identifiers,
  addresses, e-mails, phones, URLs;
* entities: people, organizations, places, products (generic named-entity recognition, statistical or model-based);
* generic relations: key → value, header → cell, label → field, signer → document, party → document;
* language detection, document-level embeddings and generic document classes (letter, form, table-heavy, report,
  record, correspondence).

A question about any domain gets exact, lexical, fuzzy, semantic, entity and table retrieval from the core alone.

## Packs

A pack adds domain understanding as data plus optional models. It never changes core code.

```
packs/<name>/
  pack.yaml            name, version, description, depends_on, languages
  types.yaml           document types: display name, title patterns, keywords, prototype descriptions, family
  fields.yaml          named fields: value kind, cues and labels, roles, validation (formats, ranges), per-type
  concepts.yaml        concepts with lexical variants and related terms (query expansion and contextual retrieval)
  entities.yaml        domain entity types and gazetteers (formats, not customer lists: e.g. ICD-10 code format)
  relations.yaml       predicates, the patterns or models that extract them, qualifiers
  questions.yaml       intent cues and aggregation measures the planner may use ("spend" → sum of money paid)
  models/              optional: classifiers, extractors, embedding or reranking models, referenced in the model registry
  tests/               the pack's own fixtures and expected extractions (synthetic data only)
```

| Pack | Types (examples) | Fields and relations (examples) |
|---|---|---|
| business (v1's taxonomy) | letter, memo, report, policy, form, press release | dates by role, amounts, signer, parties |
| legal | NDA, service, supply, employment and lease agreements, amendment, power of attorney | parties, effective and expiry dates, governing law, termination and notice, renewal, liability cap, signatures |
| finance | invoice, receipt, quotation, purchase order, statement, payroll | invoice number, totals, tax, due date, vendor, buyer, line items (table rows) |
| supply chain | bill of lading, packing list, delivery note, shipping manifest, certificate of origin | shipment and container numbers, SKU, quantities, ports, ship and delivery dates, carrier |
| medical | discharge summary, lab report, prescription, referral, claim | patient identifiers (format only), encounter dates, diagnoses (code formats), medications, dosages, results with units and ranges |
| HR | offer letter, contract, performance review, leave request, payslip | employee, position, department, start date, salary, leave balance, reporting line |

Loading rules:

* Packs are enabled per tenant (in the database, not in code), and each document records which pack versions
  processed it.
* Enabling or upgrading a pack marks matching documents stale. They are reprocessed in the background
  (CANONICAL_MODEL.md, versioning).
* A tenant may override a pack's lists (for example, extra concept variants) without forking it.
* Sensitive domains (medical, HR) get field-level access tags. A role can be denied a field's value while still
  searching the documents it can read.

## The no-hardcoded-data rule

Production code contains no customer, tenant, user or document identifiers, file names, company names, people,
places, sample questions, IP addresses, credentials, API keys, database URLs, model paths, or assumptions about a
specific customer or corpus.

| Belongs in | Examples |
|---|---|
| Environment or secret manager | database, Milvus, Redis and embedder URLs; passwords; API key hashes |
| Database (tenant configuration) | enabled packs, tenant overrides, example questions for the UI, field access tags |
| Packs (data) | vocabularies, gazetteers of public reference data (countries, currencies, code formats) |
| Model registry | model names, revisions, checksums, thresholds |
| `tests/fixtures/`, `evaluation/`, `demo/` | every sample document, name, identifier and question |

Enforcement:

* A CI check scans production code (`docintel/`, `packs/*/` except `tests/`) for literal patterns: IP addresses,
  URLs with credentials, key-like strings, absolute paths, and a denylist built from the names that appear in
  `tests/fixtures` and `evaluation/`. Any hit fails the build.
* Production settings have no usable defaults for URLs or credentials. A missing required setting fails start-up
  with a clear message. Development defaults live only in a development profile.

### v1 findings to fix first (migration phase 1)

| Finding | Where | Fix |
|---|---|---|
| UI example questions name test data ("Northgate Utilities", "John Smith", "INV-2026-00481", "California", "SAR 400,000") | `docintel/api/static/app.js` (`EXAMPLES`), search box placeholder in `index.html` | Example questions come from tenant configuration (`GET /api/v1/examples`), and the UI uses generic ones when none are configured |
| Settings default to localhost service URLs and a `docintel:docintel` database login | `docintel/config.py` | No defaults in production; a development profile provides local values |
| Docstring examples use a sample tenant name | `docintel/security.py` | Neutral placeholders (`<tenant>`) |
| v1 taxonomy, jurisdictions and extraction cues ship as package resources | `docintel/resources/*.yaml` | Become the `business`, `legal` and `finance` packs. The jurisdictions list stays as public reference data. |
