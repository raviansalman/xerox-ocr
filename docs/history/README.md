# Project history

These documents record how the engine came about. They were written before the `docintel` package existed, against
the previous `ultimate/` service (kept in the first commit of this repository), and they describe that system and
the design that replaced it. They are not a description of the current code: for that, see the
[README](../../README.md), [ARCHITECTURE](../ARCHITECTURE.md), [QUERIES](../QUERIES.md),
[SECURITY](../SECURITY.md) and [OPERATIONS](../OPERATIONS.md).

| Document | Content |
|---|---|
| ASSESSMENT.md | Assessment of the previous service: defects by id (KD-...), scorecard, phased plan |
| FORENSICS.md, SEARCH_FORENSICS.md | Why most of the previous search logic was unreachable; measured search baseline |
| SECURITY.md | Security fixes applied to the previous service |
| HARDCODED_DATA_AUDIT.md | Corpus-specific literals found in the previous service |
| DOCUMENT_INTELLIGENCE_ARCHITECTURE.md and the other design documents | The design the engine was built from. Planned items that were not built (LLM answering, Arabic and multilingual models, a separate model registry service) are described there but not implemented. |
