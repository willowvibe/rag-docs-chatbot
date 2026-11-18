# Retrieval and faithfulness eval

Corpus: 386 chunks across the golden files (nist-sp-800-63-4.pdf: 155; nist-sp-800-63b.pdf: 231) · k = 5 · retrieval: vector · reranker: none · answer model: `inclusionai/ling-3.2-flash:free` · questions: 40

| Metric | Value | Basis |
|---|---|---|
| Evidence hit@5 | **87.5%** | any passage containing the evidence phrase |
| Evidence MRR | **0.713** | any passage containing the evidence phrase |
| Citation contains evidence | **87.5%** | any cited passage containing the evidence phrase |
| Retrieval hit@5 | **90.0%** | (strict: exact gold page) |
| MRR | **0.719** | (strict: exact gold page) |
| Citation points at the gold page | **87.5%** | (strict: exact gold page) |
| Faithfulness (LLM judge, 0–1) | **0.97** | LLM judge |
| Judge replies unparsed | 0 | judge parse status |

Evidence metrics count any retrieved or cited passage that contains the question's evidence phrase; the page-based metrics — Retrieval hit@5 (strict: exact gold page), MRR (strict: exact gold page), and Citation points at the gold page (strict: exact gold page) — only credit the exact gold page.

## Per question

| id | in top-k | rank | gold page cited | faithfulness | evidence in top-k | evidence rank | evidence cited |
|---|---|---|---|---|---|---|---|
| q01 | yes | 2 | yes | 1.00 | yes | 2 | yes |
| q02 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q03 | yes | 1 | yes | 0.33 | yes | 1 | yes |
| q04 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q05 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q06 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q07 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q08 | yes | 1 | yes | 1.00 | yes | 3 | yes |
| q09 | yes | 3 | yes | 1.00 | yes | 4 | yes |
| q10 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q11 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q12 | yes | 2 | yes | 1.00 | yes | 2 | yes |
| q13 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q14 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q15 | yes | 3 | yes | 1.00 | yes | 3 | yes |
| q16 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q17 | yes | 2 | yes | 1.00 | yes | 2 | yes |
| q18 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q19 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q20 | yes | 3 | yes | 1.00 | yes | 3 | yes |
| q21 | yes | 2 | yes | 1.00 | yes | 2 | yes |
| q22 | yes | 2 | yes | 1.00 | yes | 1 | yes |
| q23 | yes | 2 | yes | 1.00 | yes | 1 | yes |
| q24 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q25 | yes | 2 | no | 1.00 | no | - | no |
| q26 | yes | 4 | yes | 1.00 | yes | 4 | yes |
| q27 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q28 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q29 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q30 | yes | 2 | yes | 1.00 | yes | 2 | yes |
| q31 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q32 | no | - | no | 1.00 | no | - | no |
| q33 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q34 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q35 | no | - | no | 1.00 | no | - | no |
| q36 | yes | 2 | yes | 1.00 | yes | 2 | yes |
| q37 | no | - | no | 1.00 | no | - | no |
| q38 | no | - | no | 0.33 | no | - | no |
| q39 | yes | 1 | yes | 1.00 | yes | 1 | yes |
| q40 | yes | 1 | yes | 1.00 | yes | 1 | yes |

Reproduce with `make eval` (needs `OLLAMA_API_KEY`, or `OPENROUTER_API_KEY` with LLM_PROVIDER=openrouter, or `ANTHROPIC_API_KEY` with LLM_PROVIDER=anthropic, plus an ingested corpus).
