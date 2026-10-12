# Phase 1 Audit: Architecture, failure modes, and data-quality issues

## Scope and objective
This audit focuses on the current `research-gap-explorer` Next.js app and the project’s transition toward a reproducible research pipeline. The app’s current behavior is a prompt-wrapper workflow rather than a verifiable evidence pipeline, and this is the core issue to solve before any later phases.

## System overview
The main flow is:

1. User enters a topic in the app.
2. `src/services/paperService.ts` queries external metadata sources (ArXiv, CrossRef, Semantic Scholar).
3. Records are deduplicated by normalized title.
4. The app sends a large paper list plus abstracts to the Gemini analysis route.
5. `src/app/api/analyze/route.ts` synthesizes a prompt and asks the model to produce `gaps` and `extracted_tools`.
6. The result is displayed in the dashboard and can be used for proposal generation, review drafting, and export.

This architecture is convenient for demos, but it is not a traceable scientific pipeline. The tool has no saved run manifest, no verifiable claim registry, and no evidence monitor for gap novelty or citation grounding.

## Concrete Phase 1 red flags

### 1) Silent truncation in API routes
The analysis route builds a single large prompt and then truncates by `substring(0, 80000)`. Similar unbounded truncation patterns appear elsewhere (`15000`, `10000`). The specific routes implicated are:
- `src/app/api/analyze/route.ts`
- `src/app/api/chat/route.ts`
- `src/app/api/literature-review/route.ts`
- `src/app/api/proposal/route.ts`
- `src/app/api/upload/route.ts`

This creates a hidden failure mode in which the model sees only a partial literature snapshot, producing gaps that reflect incomplete data rather than real evidence.

What this means in practice:
- The model may miss crucial late papers or long abstracts.
- A “gap” may be a prompt artifact rather than a real research question.
- The route emits a deterministic-looking result while quietly discarding evidence.

### 2) Placeholder abstracts flowed into the LLM
The app accepts strings such as:
- `Abstract not indexed in CrossRef repository.`
- `Abstract unavailable on Semantic Scholar.`
- `Abstract not indexed in ...`
These pass into the analysis pipeline and become part of the LLM input. This is a data-quality bug because the system is effectively training the model on missing metadata and then inferring a gap from missing information.

### 3) Deduplication is title-only and too weak
The code deduplicates by a normalized title string only. This fails when:
- two sources provide the same paper with slightly different title formatting,
- a DOI is present but title normalization does not catch it,
- one metadata record has a better abstract than another and is discarded by title-only dedupe.

This weak dedupe is a foundational reliability problem because repeated records can distort counts and the downstream prompt.

### 4) `reverified: true` is self-asserted, not checked
The analysis schema includes `reverified: true` and the prompt instructs the model to “re-verify” the gap. There is no real validation step against the corpus, no search beyond the working set, and no external novelty filter. The value is therefore an LLM self-assertion rather than a grounded research verdict.

### 5) Silent multi-model fallback contaminates experiments
`src/services/geminiService.ts` tries multiple model names and silently falls back among them. This is dangerous for research because the same prompt can be answered by different models and the output may not be comparable across runs. A reproducible pipeline cannot silently swap models during an experiment.

### 6) ArXiv XML parsing is brittle and likely contains a real author bug
`parseArxivXml()` uses a regex-based XML parser and reads authors with:
- `<author> ... <name> ... </name>`
The suspected issue is that the arXiv XML may sometimes use `<n>` rather than `<name>`, meaning the app may silently drop author names. That will affect bibliographic metadata quality and any later evidence mapping.

### 7) `openAlex.ts` is present but unused
`src/services/openAlex.ts` exists but is not wired into the live search path. That is a missed opportunity because OpenAlex is a stronger source for citations, concepts, and metadata than the current prompt-wrapper setup. It is also exactly the evidence source this project later needs for DOI-grounded literature verification.

## Specific code findings

### `src/services/paperService.ts`
Key weaknesses:
- `parseArxivXml()` is regex-based and fragile.
- CrossRef and Semantic Scholar records are accepted with placeholder abstracts.
- Deduplication is `normalized title` only.
- Sorting prefers abstract-presence but never validates abstract quality.

### `src/services/geminiService.ts`
Key weaknesses:
- Model list is hard-coded and auto-fallbacks are silent.
- No experiment-specific logging or model/version audit.
- `retriesPerModel` and fallback logic hide model-change effects.

### `src/app/api/analyze/route.ts`
Key weaknesses:
- Builds a giant prompt from the entire corpus.
- Cuts the corpus with `substring(0, 80000)`.
- Accepts unverified placeholder abstracts.
- Does not record evidence and cannot support novelty validation.
- Marks gaps as `reverified: true` without any grounding check.

### `src/services/openAlex.ts`
The file appears to contain OpenAlex-oriented functionality but is not used in the active search path, which means the app is not leveraging the recommended evidence-backed metadata source.

## Project-level implications
The app is currently suitable as a demo, but not as a research system. The project must move to a reproducible architecture with:
- a saved corpus and PRISMA log,
- explicit dedupe (DOI, title, embedding),
- evidence-grounded validation,
- no silent model fallback,
- audit logs for prompts and outputs,
- traceability from a generated gap back to source papers and supporting metadata.

## Phase 1 target state
Phase 1 is the foundation for that transition. It establishes:
1. a clean audit trail,
2. the Python package scaffold,
3. OpenAlex retrieval and abstract reconstruction,
4. deterministic deduplication,
5. PRISMA accounting,
6. tests for the core data-layer claims.

The Next.js demo remains intact, but the research logic is moved behind the new reproducible Python package so the future phases can build on a stable evidence layer.
