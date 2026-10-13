# gapfinder

`gapfinder` is the research pipeline package that gradually replaces the current prompt-wrapper flow.

## Phase 1: data foundation

- OpenAlex retrieval and abstract reconstruction
- DOI + title + embedding deduplication
- PRISMA flow accounting
- deterministic corpus build CLI

```bash
python -m pip install -e .
python -m gapfinder.data.build --topic "graph neural networks" --since 2015 --max 500
```

The build command writes `corpus.jsonl` and `prisma.json` into the current working directory unless `--output-dir` is provided.

The current year-stratified main corpus is `corpus_main_v4.jsonl`, with counts in `prisma_main_v4.json`; v3 is retained for provenance. The v4 build queries each publication year independently (English works, relevance order, cached pages of 100; up to 300 strict phrase candidates per year and a cap of 200 included records per year), and requires a title or abstract match for graph neural network(s), GNN(s), graph convolutional network(s), graph attention network(s), graph representation learning, or message-passing neural network(s). Unqualified “message passing” is not sufficient. OpenAlex search can include full text, so the local title/abstract check is applied before a record is identified. The v4 corpus has 1,951 included records and its 50-record audit is 50/50 under both the prior broad rule and the strict rule; a full-corpus check confirms 1,951/1,951 match the strict patterns. Audit artifacts are `../experiments/annotation/topical_precision_audit_v4.csv` and `.json`. Early years did not reach the 200-record included cap: OpenAlex returned only 233, 282, and 410 total search results in 2015, 2016, and 2017, yielding 40, 44, and 95 strict phrase candidates (38, 40, and 77 included after screening).

```bash
python -m gapfinder.data.build_year_stratified --topic "graph neural networks" --max-per-year 200 --output-dir . --corpus-name corpus_main_v3.jsonl --prisma-name prisma_main_v3.json
```

## Phase 2: survey-recovery evaluation

Install the project and its development/retrieval dependencies, then run:

```bash
python -m pip install -e .[dev,retrieval]
python -m gapfinder.eval.survey_recovery --corpus corpus.jsonl --max-surveys 20 --min-relevant-docs 3
```

The evaluation resolves every selected survey's OpenAlex `referenced_works` using the free works-list endpoint and `filter=ids.openalex:ID1|ID2` batches (up to 100 IDs per request; see [OpenAlex OR filtering](https://help.openalex.org/api/filtering/#addition-or)). Cited records without abstracts are counted and excluded from the expanded retrieval pool. Run artifacts include the expanded pool, source metadata, qualification and PRISMA counts, per-query metric CSVs, bootstrap confidence intervals, and paired Wilcoxon tests.

Survey candidates are selected from the union of Phase-1 records and resolved cited-work metadata, ordered by publication year and OpenAlex ID. Qualification requires an abstract and at least three cited candidates with abstracts in the expanded pool; three is a prespecified floor that avoids evaluating a survey on only one or two recoverable citations. `survey_qualification.csv` names each screened source and its pass/exclusion rationale.

Reported metrics include Recall@10/20/50, Precision@10/20, nDCG@10/20, and MAP over each full ranked pool. Means and 95% percentile bootstrap intervals use 10,000 query-level resamples with a fixed seed. Primary reciprocal-rank fusion is unweighted RRF with `k=60`; weighted RRF is an ablation whose weight is selected leave-one-query-out using training-query nDCG@20. The declared primary metric is nDCG@10; this choice was made after inspecting first-pass results, so it was not pre-registered independently of those results. `paired_tests.csv` retains raw p-values and reports Holm corrections for the five dense-vs-BM25 tests and the full family of ten dense-vs-BM25/dense-vs-RRF tests.

Recall@k is bounded above by `k / |relevant|` for queries with more than `k` relevant works. This corpus has 52–201 relevant documents per qualified survey, so even perfect top-k rankings cannot achieve Recall@10/20/50 above those query-specific ceilings.

Dense retrieval uses exact NumPy cosine search by default. FAISS is optional and is only used when explicitly requested with `DenseRetriever(backend="faiss")` and the optional package is installed.

### Retrieval limitations

- Results cover one topic (graph neural networks) and use a survey-derived candidate pool, not a representative sample of all literature.
- Some cited OpenAlex reference IDs remain unresolved; per-survey rates and a limited singleton recheck are saved separately.
- Ranking and relevance are abstract-only; full-text methods, results, and future-work statements are not assessed here.

**Abstract integrity and rerun:** The initial Phase 2 retrieval evaluation used abstracts reconstructed by an earlier OpenAlex inverted-index routine that kept only the first occurrence of each distinct token, dropping repeated words. Following that finding, the entire candidate pool and embeddings were rebuilt with full token occurrences preserved. The re-evaluated results are stored in `experiments/runs/retrieval_rerun/` (and sensitivity analysis in `experiments/runs/retrieval_unresolved30_rerun/`). On the rebuilt pool, dense retrieval achieved mean nDCG@10 of 0.568 [0.448, 0.687], compared with 0.533 [0.397, 0.667] for RRF and 0.490 [0.350, 0.636] for BM25, confirming early-rank retrieval behavior while RRF led on deeper cutoffs and MAP (0.271 vs. 0.261 dense).
- Survey queries share cited works, so per-query scores are not independent. Bootstrap intervals and paired tests treat queries as sampling units and should be interpreted with that overlap in mind.

## Phase 3: structured extraction preparation

The independent main corpus is `corpus_main_v4.jsonl`; the survey-derived retrieval pool is a separate evaluation artifact. The extraction schema uses `paper_type`, excludes survey/review papers from counts, requires specific named task/method/dataset claims, rejects generic future-work/limitation pointers, and validates every evidence span as an exact abstract substring. Specificity rejections are recorded separately.

Manual annotation instructions are in `../experiments/annotation/ANNOTATION_GUIDE.md`. The original `../experiments/annotation/extraction_sample.csv` and `extraction_sample_2nd.csv` were sampled from corrected v3. The human-annotated 100-paper sheet is present at `../experiments/annotation/extraction_sample_annotated.csv`. When reconciled against the v4 corpus, 18 rows fail the strict topic rule and 6 are outside v4, yielding 76 in-topic v4 papers. Evaluation against this cohort was completed: categorical paper-type F1 is 0.883 (include-in-counts F1 is 0.905); exact matching for free-text Method labels is 0.000 vs. 0.748 under exploratory embedding-cosine semantic matching. Full metrics are in `../experiments/runs/phase3_v4_annotation_metrics.json` and `../experiments/runs/phase3_v4_semantic_annotation_metrics.json`. The v2 audit is retained at `../experiments/annotation/TOPICAL_PRECISION_AUDIT.md`; v3's audit is at `../experiments/annotation/TOPICAL_PRECISION_AUDIT_V3.md`.

Do not interpret annual counts in the year-stratified main corpus as literature volume. Momentum must use concept prevalence shares within each year's papers, with confidence intervals, or separately sourced OpenAlex annual totals. Phase 4 concept-sparsity gap detection was executed on the 1,775 included v4 papers and yielded a negative result: 0 testable fine-grained cells and 0 Benjamini–Hochberg survivors (q ≤ 0.10) across 192 testable coarse pairwise cells. See `../paper/addendum.md` and `../repro/sparsity_coarse/`.

Set `GAPFINDER_MODEL` to exactly one approved model before any request; the Python client and Next.js service do not switch models automatically. The Python client disables SDK retries, honors `Retry-After`, and uses exponential backoff with jitter. Requests with confirmed pre-HTTP transport failures are logged but refunded from the daily and run request budgets; HTTP attempts remain charged against per-model RPM, TPM, and Pacific-time RPD limits in `.cache/gemini_request_budget.json`, partitioned by a one-way API-key fingerprint. Separate keys receive separate local buckets only after an API probe succeeds; Google enforces quotas at the owning project/model level, so keys in the same project still share server quota and a 429 stops the run for the Pacific-day reset. Local or Google-reported daily-cap exhaustion pauses until the next Pacific day and then resumes. Credentials are resolved in order from `GEMINI_API_KEY2`, `GEMINI_API_KEY`, and `GAPFINDER_API_KEY`; the source (never the secret) is recorded in the manifest. Each API attempt and serialized request payload are recorded in `runs/<run_id>/llm_calls.jsonl` and `usage.csv`; a non-reversible credential fingerprint allows per-key request accounting without logging the key. Extraction batches, requeues, successful IDs, and validation failures are separately recorded in `batch_attempts.jsonl`. Successful per-paper outputs are cached and reused on resume, and successful logged responses can seed missing cache entries without another API request. Extraction defaults to batches of eight papers. Invalid evidence spans no longer reject an otherwise schema-valid paper: close spans are repaired to exact raw-abstract substrings, while unsupported spans null only their field; repairs and nulls are logged in `span_adjustments.jsonl` and reported separately from exact spans. Gemini 3 extraction requests use temperature 0 and the minimum configured thinking level (`low`). The prompt instructs the model to copy each evidence span character-for-character as an unmodified abstract substring and includes a correct-copy example; the extraction schema is unchanged. OpenAlex abstract reconstruction must preserve every token occurrence from its inverted index; the earlier code kept only each distinct token's first position and silently dropped repeated words. Evidence validation checks `evidence_span in raw_abstract` directly, without normalization. Revalidation of the saved five-item `gemma-4-31b-it` response against corrected OpenAlex reconstructions found **6/11 non-null spans exact**; that probe is not considered fully span-valid. Install the optional dependencies with `python -m pip install -e .[llm,retrieval]`.

The checked-in `requirements.lock` pins the tested Windows/Python 3.13.2 environment (including development, Gemini, and retrieval dependencies); recreate it with `python -m venv .venv` followed by `.venv\\Scripts\\python.exe -m pip install -r requirements.lock`. FAISS is not included and remains a separate optional extra.

Extraction is an explicit operation; setup and retrieval do not send papers to Gemini. The approved 100-paper comparison used batch size four and the three configured models. Run artifacts and the raw-span audit are under `../experiments/runs/phase3_100_comparison_20260929_113614/`; results are partial because validation rejected some outputs and Gemma had transient HTTP 500/503 and HTTP 401 failures. The single Gemma HTTP 401 occurred on a retry after an HTTP 500 and reported a missing authentication credential; request headers and the raw response body were not captured, so the cause cannot be resolved beyond an isolated authentication/service reliability failure. In the paired eight-paper Gemini 3.1 Flash-Lite thinking ablation, `minimal` used zero thinking tokens versus 269 at `low`, but exact spans were 27/29 versus 29/29; the small sample showed no measurable accuracy benefit from reducing thinking, so extraction remains pinned to `low`. Human annotations for 100 sampled papers are recorded in `../experiments/annotation/extraction_sample_annotated.csv` (76 in v4). The separately approved full-corpus extraction completed for all 1,951 v4 records in `../experiments/runs/phase3_full_corpus_v4_gemini31_low/`. Statistical gap detection via concept sparsity was evaluated on the complete extraction set and documented as a negative result in `../paper/addendum.md`.
