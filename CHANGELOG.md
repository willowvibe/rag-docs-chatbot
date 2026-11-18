# Changelog

## [1.1.0] — 2026-11-18

The public demo now runs on a free open model, has a redesigned interface, and enforces a
fair rolling limit.

### Added
- OpenRouter provider and `LLM_PROVIDER=openrouter`; the live demo answers with the free
  `inclusionai/ling-3.2-flash:free` model, labelled on the page, with "via <model>" under
  each answer (#11, #12).
- `RAG_FALLBACK_MODELS`: an ordered fallback chain across providers — on the demo,
  `nvidia/nemotron-3.5-super-vl:free` (OpenRouter) and then `glm-5.3-flash` (Ollama Cloud) (#11).
- A "how to use this demo" panel listing the searchable documents (`/api/documents`,
  `corpus/manifest.json`) with example questions (#9).
- `EVAL_MIN_INTERVAL_SECONDS` to pace eval calls under free-model rate limits (#12).

### Changed
- Redesigned demo app and Pages site: answers as document blocks, sources as page slips,
  and a highlighter that marks the cited passage; basic formatting is rendered (#13).
- The per-visitor limit is a rolling 60-minute window instead of clock-hour buckets (#10).
- Re-recorded walkthrough video on the new interface (#13).
- `eval/report.md` re-run with the live demo's model: evidence 87.5%, faithfulness 0.967 (#12).

### Fixed
- A stray "I don't know based on the provided documents." appended to correct, cited
  answers (#13).
- The citation highlight fading before the reader finished (#13).
- Store opened a new database connection per search; later searches over a long-latency
  link drop from ~5.9 s to 0.30 s (#8).

## [1.0.0] — 2026-11-18

First public release: PDF/Markdown/HTML ingestion, local BGE embeddings, Postgres +
pgvector, cited answers with explicit refusals, optional hybrid search and reranking, a
40-question evaluation, Postgres-backed rate limiting, Docker quickstart, CI, a live demo
on Vercel and a narrated walkthrough on GitHub Pages.

[1.1.0]: https://github.com/willowvibe/rag-docs-chatbot/releases/tag/v1.1.0
[1.0.0]: https://github.com/willowvibe/rag-docs-chatbot/releases/tag/v1.0.0
