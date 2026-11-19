# Adding New Documents

When you add a new file to this directory (any format: `.md`, `.txt`, `.html`, `.csv`, `.pdf`), **you must add a matching entry in `_metadata.json`**.

## Why

`_metadata.json` is read by `_load_metadata()` in `rag/rag_core/loaders.py`, which feeds into `load_corpus()` — the function that extracts and indexes every document in this folder. Without a metadata entry, your document will still be extracted but will lack title, department, access control, trust level, and other fields that downstream chunking and retrieval depend on.

## How

Open `_metadata.json` and add a key matching your filename exactly. Example:

```json
"my_new_document.pdf": {
  "title": "Human-Readable Title",
  "department": "support",
  "doc_type": "policy",
  "access": ["public"],
  "version": "1",
  "effective_date": "2026-01-01",
  "status": "active",
  "trust": "trusted",
  "language": "en"
}
```

### Field reference

| Field            | Required | Values / Notes                                                    |
|------------------|----------|-------------------------------------------------------------------|
| `title`          | yes      | Display title used during extraction                              |
| `department`     | yes      | e.g. `support`, `hr`, `engineering`, `finance`, `marketing`       |
| `doc_type`       | yes      | e.g. `policy`, `faq`, `runbook`, `guide`, `table`, `web_page`    |
| `access`         | yes      | List of groups: `["public"]`, `["employees"]`, `["hr"]`           |
| `version`        | yes      | Free-form version string                                          |
| `effective_date` | yes      | `YYYY-MM-DD`                                                      |
| `status`         | yes      | `active` or `archived`                                            |
| `trust`          | yes      | `trusted` (internal) or `untrusted` (user-generated)              |
| `language`       | yes      | ISO 639-1 code (`en`, `hi`, …)                                   |
| `superseded_by`  | no       | Filename of the newer version (for archived docs)                 |
| `contains_pii`   | no       | `true` if the file has personal data                              |

## Supported formats

The loader (`extract_file` in `loaders.py`) handles: `.pdf`, `.html`/`.htm`, `.csv`, `.md`, `.txt`.
