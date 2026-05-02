---
name: pdf-reading
description: Use when the user asks to read, extract, summarize, compare, or analyze PDF attachments, especially scanned, form-heavy, table-heavy, or layout-sensitive PDFs.
---

# PDF Reading

Use this skill when a task depends on the contents of a PDF file.

## Workflow

1. If the current turn already includes the PDF as an attachment/content block,
   answer from the PDF directly. Do not call `read_pdf` first just to prove the
   PDF exists.
2. If you only have a saved PDF path from history, find it in the conversation
   attachment block. Dropped files are
   saved under `.feather/attachments/<session_id>/...` and shown as `[File #n]`.
3. Call `read_pdf` with `mode: "auto"` when you need local text extraction.
4. If the PDF appears scanned, table-heavy, multi-column, form-heavy, or the
   extracted text is obviously incomplete, retry with
   `mode: "opendataloader_hybrid"` when the optional `pdf-hybrid` extra is
   installed or `OPENDATALOADER_PDF_COMMAND` is set.
5. Use `read_file` only for already-extracted text artifacts, not raw PDFs.
6. Cite page numbers only when the extracted text includes reliable page
   markers. Otherwise say that page numbers were not available.

## Tool Usage

Basic extraction:

```json
{
  "path": ".feather/attachments/<session_id>/<attachment-id>-document.pdf",
  "mode": "auto",
  "max_chars": 12000
}
```

Hybrid extraction for complex PDFs:

```json
{
  "path": ".feather/attachments/<session_id>/<attachment-id>-document.pdf",
  "mode": "opendataloader_hybrid",
  "max_chars": 30000
}
```

## Optional OpenDataLoader Hybrid Backend

Feather installs `opendataloader-pdf` and `pypdf` for normal PDF extraction.
For OpenDataLoader hybrid mode on complex PDFs, install the optional extra:

```bash
uv sync --extra pdf-hybrid
```

Hybrid mode may also require Java and the backend services described by
OpenDataLoader. If the Python backend is not available, Feather can use the
environment variable `OPENDATALOADER_PDF_COMMAND` as an override. The command is
a template; Feather replaces `{path}` with the PDF path and `{mode}` with
`hybrid`.

Command-template example:

```bash
export OPENDATALOADER_PDF_COMMAND='opendataloader-pdf --mode {mode} --input {path}'
```

The exact command depends on how OpenDataLoader is installed locally. Do not
invent one if the project docs or local environment do not confirm it.

## Failure Handling

- If `read_pdf` returns little or garbled text, tell the user the extractor may
  need the hybrid backend or OCR dependencies, then answer from the attached PDF
  bytes if they are present in the current model turn.
- If the PDF path is missing, ask the user to drop or attach the PDF again.
- If the file is not a PDF, use the normal file-reading path for text-like
  files or ask for a supported format.
