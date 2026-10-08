# Domain packs

The engine core contains no business vocabulary: it does not know what an invoice or an NDA is. That knowledge comes
from domain packs, YAML folders loaded at start and chosen per tenant.

| Pack | Content |
|---|---|
| `core` (always on) | generic types (letter, memo, report, form, e-mail), date roles (issued, effective, expiry, due, signed), amount roles, generic identifier labels, party and person cues, lookup terms, jurisdictions |
| `business` | press releases, policies |
| `legal` | contracts and agreements (NDA, service, supply, employment, lease, purchase), contract numbers, 13 clause types (termination, governing law, confidentiality, liability, indemnification, payment, term and renewal, warranty ...), 7 concepts and relation patterns (termination with notice, governing law, parties) |
| `finance` | invoices, receipts, quotations, purchase orders, financial statements, payroll; invoice and PO numbers; amount roles; concepts (payment terms, spend, tax) |

Without any pack but `core`, ingestion, exact, lexical, semantic, entity and metadata search all work; what is lost
is type classification for those types, typed fields and clauses, concepts and relations, and the type words the
planner understands ("invoices", "NDAs").

## Choosing packs

* Deployment default: `DOCINTEL_DEFAULT_PACKS=business,legal,finance`.
* Per tenant (admin key): `PUT /api/v1/settings {"packs": ["legal"], "examples": [{"label": "...", "q": "..."}]}`.
  `core` and each pack's dependencies are always added. Unknown packs are refused.
* Changing a tenant's packs marks its documents `stale` (the packs are part of each document version); run
  `docintel reprocess --tenant T --stale` to apply them to existing documents. Questions use the new packs at once.

## Writing a pack

Put a folder in a directory named by `DOCINTEL_PACKS_DIR` (searched before the built-in packs). Pack names are
lowercase letters, digits and underscores.

```
my_pack/
  pack.yaml          name, version, description, depends_on: [core]
  taxonomy.yaml      families, family_terms, types
  extraction.yaml    identifier_fields, date_roles, amount_roles, party_cues, person_cues, clause_types,
                     field_terms, lookup_terms
  concepts.yaml      concepts: name -> variants, optional clause_type and predicate
  relations.yaml     relations: predicate + pattern (a regular expression, or a list of fragments)
```

A document type:

```yaml
types:
  work_order:
    display: Work order
    query_terms: [work order, work orders]          # words the planner maps to this type
    title_patterns: ['\bwork\s+order\b']             # strong evidence when found in the title area
    keywords: {technician: 1, site visit: 1}         # weighted evidence in the body
    description: A request to perform maintenance work at a site.   # used by the embedding fallback
    date_field: issue_date                           # the date that places it in time
```

A concept and a relation:

```yaml
concepts:
  termination:
    variants: [terminate, cancel, cancellation, end the agreement, early termination]
    clause_type: termination        # clauses of this type are evidence
    predicate: may_terminate        # relations with this predicate are evidence
relations:
  - predicate: governed_by
    pattern: '(?P<subject>this agreement)\s+(?:is|shall be)\s+governed by the laws of\s+(?P<object>[^.;\n]+)'
```

Patterns are matched per sentence, case-insensitively (use `(?-i:...)` for parts that must keep their case). The
named group `subject` is required; `object` is optional; other named groups become qualifiers (for example a
notice period). Keep patterns specific: a pack that matches too much produces false positives in every tenant that
enables it. Add a test with representative sentences for every pattern.

Pack files are data, not code: they cannot run anything, and a malformed pack fails at start with the file and the
problem named.
