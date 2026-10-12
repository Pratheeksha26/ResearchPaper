# MASTER PROMPT: Turn "Research Gap Explorer" into a Publishable Research Project

Paste everything below the line into your AI coding agent (Claude Code, Cursor, etc.) with the project folder open.
Run it phase by phase. After each phase the agent must stop and report before continuing.

---

## 0. ROLE AND MISSION

You are a senior research engineer and applied-ML researcher. You are working inside an existing Next.js/TypeScript project called `research-gap-explorer`. It currently:

- searches arXiv, CrossRef and Semantic Scholar by keyword (`src/services/paperService.ts`),
- dedupes by normalized title,
- sends abstracts to a Gemini LLM in a single prompt (`src/app/api/analyze/route.ts`) to produce "research gaps" and "extracted tools",
- generates a literature review and a PhD proposal by prompting (`literature-review`, `proposal` routes),
- offers a chat sidecar, BibTeX/RIS export and PDF upload.

**Mission:** Convert this from a prompt-wrapper tool into a rigorous, reproducible research contribution with the thesis:

> *"LLM-only research-gap detection is unreliable. We propose an evidence-grounded pipeline (hybrid retrieval + structured claim extraction + concept-graph statistical sparsity + external novelty verification + uncertainty estimation) and validate it with survey-recovery and temporal hold-out experiments."*

The Next.js app stays as the demo front-end. The research pipeline and experiments live in a new Python package.

## 1. NON-NEGOTIABLE RULES

1. **Never fabricate results.** Do not invent numbers, tables, citations, papers, datasets or human ratings. If an experiment has not been run, output `TODO: not run yet`. Every number in any report must be traceable to a saved output file.
2. **Never read, print, copy or commit `.env.local` or any secret.** Load keys only via environment variables. Add `.env*` to `.gitignore` if missing and warn me if a key appears to have been committed or shared.
3. **Every claim about existing literature must be verifiable.** Do not cite from memory. If you mention a paper, it must come from an API result (OpenAlex/Semantic Scholar/arXiv) with a DOI or URL stored in `related_work/verified_refs.jsonl`. Mark anything unverified as `UNVERIFIED`.
4. **Reproducibility:** fix random seeds, pin library versions, log model name + version + temperature + full prompt + raw response for every LLM call to `runs/<run_id>/llm_calls.jsonl`. No silent model fallback during experiments: if the chosen model fails, the run fails loudly and records why.
5. **Before touching Next.js code,** read the relevant docs in `node_modules/next/dist/docs/` (see `AGENTS.md`); this Next version has breaking changes.
6. **Work in small commits** with clear messages. Do not delete existing features; refactor behind interfaces.
7. **Stop after each phase** and give me: what changed, files touched, how to run it, test results, open questions. Wait for my "continue".
8. If something is ambiguous or blocked (no API key, rate limits, no internet), say so and propose an alternative. Do not guess silently.

## 2. TARGET REPOSITORY LAYOUT

```
ResearchPaper/
  research-gap-explorer/        # existing Next.js app (demo UI)
  gapfinder/                    # NEW python package (research pipeline)
    pyproject.toml
    src/gapfinder/
      config.py                 # env-driven config, model names, seeds
      data/
        openalex.py             # retrieval + citation graph
        semantic_scholar.py
        dedupe.py               # DOI + embedding dedupe
        prisma.py               # screening log + flow counts
      retrieval/
        bm25.py
        dense.py                # SPECTER2 / BGE embeddings + FAISS
        fusion.py               # Reciprocal Rank Fusion
      extraction/
        schema.py               # pydantic claim schema
        extractor.py            # LLM structured extraction
      graph/
        build.py                # concept graph / task-method-dataset tensor
        sparsity.py             # expected-count + Poisson test + BH correction
        link_prediction.py      # Adamic-Adar, node2vec
        momentum.py             # Theil-Sen slope, burst detection
      gaps/
        candidates.py           # statistical candidates
        score.py                # composite GapScore
        explain.py              # LLM names/explains statistical gaps only
        verify.py               # external novelty verification (NLI/LLM judge)
        stability.py            # self-consistency
      baselines/
        llm_only.py             # replica of the current single-prompt method
        keyword.py              # TF-IDF frequency baseline
        topic_only.py           # BERTopic-only baseline
      eval/
        survey_recovery.py
        temporal_holdout.py
        faithfulness.py         # citation grounding + attribution
        human_study.py          # rating sheets + agreement stats
        stats.py                # bootstrap CI, Wilcoxon, Krippendorff alpha
      llm/
        client.py               # single wrapper, logging, pinned models
      api.py                    # FastAPI service for the Next.js app
    tests/
    experiments/
      configs/                  # YAML per experiment
      runs/                     # outputs (gitignored except summaries)
    paper/
      figures/ tables/ main.md refs.bib
    related_work/verified_refs.jsonl
    README.md
```

## 3. PHASES

### PHASE 1: Audit and data foundation

Tasks:
1. Read the whole codebase and write `docs/AUDIT.md`: architecture, data flow, and a list of concrete weaknesses. Specifically check and report on:
   - silent truncation (`substring(0, 80000)`, `15000`, `10000`) in the API routes,
   - placeholder abstracts ("Abstract not indexed…", "…unavailable…") being sent to the LLM,
   - title-only dedupe,
   - the `reverified: true` field being LLM self-assertion with no real check,
   - silent multi-model fallback in `geminiService.ts` (contaminates experiments),
   - regex XML parsing of arXiv (note the `<n>` vs `<name>` pattern in the author regex looks like a bug; verify and fix),
   - `openAlex.ts` being unused.
2. Create the `gapfinder` package skeleton with `pyproject.toml` (pinned versions), `config.py`, and `llm/client.py`.
   - `llm/client.py`: one function `generate(prompt, schema=None, temperature, seed, model)`; model comes from env `GAPFINDER_MODEL`; writes each call to `runs/<run_id>/llm_calls.jsonl`; supports JSON-schema-constrained output; no silent fallback (raise instead, with a `--allow-fallback` flag that records the fallback in the log).
3. Implement `data/openalex.py`: search by topic with date filters; reconstruct abstracts from OpenAlex `abstract_inverted_index`; store DOI, year, venue, concepts, `referenced_works`, `cited_by_count`. Implement polite-pool usage (mailto from env).
4. Implement `data/dedupe.py`: DOI-exact, then normalized-title, then embedding-cosine (threshold configurable, default 0.95).
5. Implement `data/prisma.py`: log counts at each step (identified → deduped → abstract-missing removed → language/date filtered → included) and export a JSON + a simple flow-diagram figure.
6. Add tests for abstract reconstruction, dedupe and PRISMA counts.

Acceptance: `python -m gapfinder.data.build --topic "<topic>" --since 2015 --max 500` produces `corpus.jsonl` + `prisma.json` deterministically; tests pass.

### PHASE 2: Hybrid retrieval

1. `retrieval/bm25.py` (rank_bm25 or Pyserini), `retrieval/dense.py` (sentence-transformers with a scientific embedding model, e.g. SPECTER2 or a BGE model; make model configurable; cache embeddings to disk; FAISS index).
2. `retrieval/fusion.py`: Reciprocal Rank Fusion:
   `RRF(d) = Σ_i 1 / (k + rank_i(d))`, default `k = 60`.
3. Evaluate retrieval on a small labeled set: build queries from ~20 survey-paper titles; treat papers cited by the survey as relevant; report Recall@k and nDCG@k for BM25, dense, and RRF. Save to `experiments/runs/retrieval/`.

Acceptance: a table comparing the three retrievers with numbers produced by code and saved to CSV.

### PHASE 3: Structured claim extraction

1. `extraction/schema.py` (pydantic): per paper fields `task`, `method`, `dataset`, `metric`, `setting`, `stated_limitation`, `stated_future_work`, `evidence_span` (exact text snippet from the abstract supporting each field).
2. `extraction/extractor.py`: LLM extraction with JSON-schema-constrained output, temperature 0, batching, retry with backoff, cache by paper ID + model + prompt hash.
3. Normalize entities: cluster surface forms (e.g., "GNN", "graph neural network") using embeddings + agglomerative clustering; keep a canonical-name map.
4. **Validation:** generate `experiments/annotation/extraction_sample.csv` with 100 randomly sampled papers and blank columns for my manual labels. Write `eval/extraction_eval.py` that computes field-level precision/recall/F1 once I fill it in. Do NOT fill labels yourself.
5. Check that every `evidence_span` is a real substring of the abstract; report the rate.

Acceptance: extraction runs on a full corpus, cache works, span-grounding rate is reported.

### PHASE 4: Concept graph and statistical gap detection (core novelty)

1. `graph/build.py`: build (a) a co-occurrence tensor over (task, method, dataset) counts N(t,m,d), and (b) a concept graph with nodes = canonical concepts and edges = co-occurrence within a paper (weighted), plus citation edges from `referenced_works`.
2. `graph/sparsity.py`:
   - Marginals `p(t), p(m), p(d)`; expected count under independence
     `E[N(t,m,d)] = N · p(t) · p(m) · p(d)`.
   - Lower-tail Poisson test `P(X ≤ N_obs | λ = E)`; a cell is a candidate gap if E ≥ min_expected (default 2) and the p-value is small.
   - Benjamini–Hochberg FDR correction across all tested cells. Also implement pairwise `PMI(m,d) = log[ p(m,d) / (p(m) p(d)) ]`.
3. `graph/link_prediction.py`: Adamic–Adar and node2vec (or a simple GNN via PyTorch Geometric) to score missing edges between concepts.
4. `graph/momentum.py`: Theil–Sen slope of yearly counts per concept; Kleinberg burst detection as an optional extra.
5. `gaps/candidates.py`: merge candidates from sparsity + link prediction; cluster near-duplicates.
6. `gaps/score.py`: composite score
   `GapScore = α·Sparsity + β·Demand + γ·Momentum + δ·Feasibility − λ·Saturation`
   where Sparsity is a normalized `−log10(adjusted p)`, Demand counts explicit `stated_future_work` / `stated_limitation` mentions plus citation velocity, Momentum is the Theil–Sen slope, Feasibility is the availability of a public dataset/tool in the graph, and Saturation is the density of the neighborhood. Weights are read from config, tuned on a validation topic (never the test topics), and the tuning procedure is logged.
7. `gaps/explain.py`: the LLM receives ONLY a statistically flagged gap plus the evidence papers and must produce title, description, research question, and a list of supporting paper IDs from the provided set. Validate that every returned ID exists in the provided set; reject and retry otherwise.

Acceptance: for a sample topic, output a ranked gap list with p-values, scores, evidence paper IDs; unit tests on the sparsity math with hand-computed toy examples.

### PHASE 5: Real verification, stability and faithfulness

1. `gaps/verify.py`: for each candidate gap: (a) generate 3–5 "someone has already solved this" queries; (b) search OpenAlex/Semantic Scholar beyond the working corpus; (c) score each hit with an NLI model (e.g., a DeBERTa-based MNLI model) and/or an LLM judge for "does this paper address the gap?"; (d) `Novelty(g) = 1 − max_p sim(g, p)` and a flag with the top-3 closest papers as evidence. Save all evidence.
2. `gaps/stability.py`: run the explain stage N (default 5) times at temperature > 0, cluster outputs by embedding similarity, define `stability(g)` = fraction of runs containing a member of g's cluster. Report calibration (ECE) of stability vs. expert validity once ratings exist.
3. `eval/faithfulness.py`:
   - citation-grounding rate = fraction of cited paper IDs/titles that exist in the corpus (report hallucinated-citation rate),
   - attribution: NLI entailment of each generated statement by its cited abstract(s).
4. Compare against the ORIGINAL method (`baselines/llm_only.py`, which reproduces the current single-prompt analyze route) on the same corpus.

Acceptance: a report showing hallucinated-citation rate and "already-solved gap" rate for baseline vs. new pipeline, computed from saved outputs.

### PHASE 6: Baselines, ablations and evaluation harness

Implement and run, with fixed seeds, on **at least 5 topics from different fields**:

**Baselines:** (B1) `llm_only` (current method); (B2) TF-IDF/keyword frequency gaps; (B3) topic-model-only gaps (BERTopic/HDBSCAN); (B4) plain "deep research" style single LLM call with retrieval.

**Ablations:** (A1) no hybrid retrieval (BM25 only), (A2) no graph/sparsity stage (LLM explains from clusters), (A3) no external verification, (A4) different LLMs, (A5) corpus size 100/250/500.

**Experiment 1: Survey recovery.**
- Select 20–30 recent survey papers with an explicit "open challenges/future directions" section; find them via OpenAlex and record DOIs.
- Corpus = papers the survey cites that were published before the survey. Hide the survey's own gaps.
- Extract the survey's stated gaps (manually verified, stored in `experiments/gold/survey_gaps.jsonl`; prepare the file with LLM-suggested extractions clearly labeled `needs_human_review`).
- Metric: Recall@k of gold gaps matched by predicted gaps (embedding-cosine threshold + LLM-judge, with the judge validated against a 50-item hand-labeled subset I will provide).

**Experiment 2: Temporal hold-out.**
- Choose cutoff year T; build the corpus from papers ≤ T; generate gaps; then check whether papers from T+1…T+2 address them (using the verify module + manual spot checks).
- Metric: gap fulfillment rate@k, compared across methods.

**Experiment 3: Expert study package.**
- Generate a blinded, shuffled rating sheet (`human_study/ratings.csv`) with columns: item_id, topic, gap text, supporting evidence, then blank 1–5 columns for novelty, validity, actionability. System identity must be hidden and stored in a separate key file.
- Write `eval/human_study.py` computing mean ± bootstrap 95% CI, Wilcoxon signed-rank test between systems, and Krippendorff's α. It must run on my filled-in file. Do NOT generate fake ratings.

Acceptance: `python -m gapfinder.eval.run_all --config experiments/configs/main.yaml` reproduces all tables into `paper/tables/*.csv` and figures into `paper/figures/`, and writes `runs/<run_id>/manifest.json` with git hash, seeds, model versions, and dataset hashes.

### PHASE 7: Integrate into the Next.js demo

1. Add `gapfinder/src/gapfinder/api.py` (FastAPI) exposing `/search`, `/gaps` and `/verify`.
2. In the Next.js app, add a `PIPELINE_MODE` setting (`legacy` | `evidence`). In `evidence` mode, the Dashboard shows for each gap: p-value / GapScore, novelty score, stability, top-3 closest existing papers as evidence, and a "why this is a gap" panel. Keep legacy mode intact.
3. Fix issues found in the Phase 1 audit: arXiv author parsing, silent truncation (chunk or rank instead of cutting), placeholder abstracts filtered, DOI-based dedupe.
4. Add an "Export experiment report" button that downloads the run manifest + gap evidence as JSON.

### PHASE 8: Paper scaffolding

1. Create `paper/main.md` with the structure below. Fill only with things that exist in the repo; leave `TODO` for anything else. Include real numbers ONLY by reading from `paper/tables/*.csv`.
   - Abstract, Introduction, Related Work (from `verified_refs.jsonl` only), Problem Formalization (formal definition of "research gap" and notation), Method (each stage with formulas above), Experimental Setup (topics, PRISMA flow, models, hyperparameters, seeds), Results (baselines, ablations, survey recovery, temporal hold-out, human study), Analysis and Error Cases, Limitations and Ethics (hallucination, citation-count bias, English-only, abstract-only, LLM dependence), Conclusion, Reproducibility Statement.
2. `related_work/`: script that searches OpenAlex/Semantic Scholar for related work on: LLM research-idea generation, research-gap identification, literature-based discovery, scientific claim extraction, AI literature-review tools. Save results with DOIs; produce `refs.bib`. Do not add a paper to related work unless it is in the saved results.
3. `paper/CHECKLIST.md`: venue-style reproducibility checklist (code/data availability, seeds, compute, licenses, human-study ethics/IRB note, limitations).
4. `README.md`: install, run each phase, reproduce all tables, and known limitations.

## 4. QUALITY BAR (apply to every phase)

- Type hints, docstrings for public functions, `ruff` + `mypy` clean, `pytest` passing.
- Deterministic where possible; document any nondeterminism.
- Each module has at least one unit test using a tiny toy dataset committed to `tests/fixtures/`.
- Handle API rate limits with caching and backoff; never hammer public APIs.
- Cost/latency: log tokens and wall-clock time per stage; report them in the results.
- Store large artifacts (embeddings, corpora) in `runs/` and keep them out of git.

## 5. REPORT FORMAT AFTER EACH PHASE

```
PHASE N COMPLETE
1. Summary (3-5 lines)
2. Files created/modified
3. How to run + expected output
4. Tests: passed/failed counts
5. Numbers produced (with file paths where they are saved)
6. Risks, assumptions, and things I must decide or do manually
   (e.g., annotate 100 papers, choose survey papers, recruit raters)
7. Proposed next step
```

Begin with PHASE 1 now. Do not start Phase 2 until I reply "continue".
