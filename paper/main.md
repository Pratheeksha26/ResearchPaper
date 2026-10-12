# Building an Auditable Literature Evidence Pipeline: Corpus Quality, Retrieval, and Structured Extraction for Graph Neural Networks

**Draft for review — 30 September 2026**

## Abstract

Automated research-gap discovery is often presented as a generation problem, although its conclusions depend first on whether the literature corpus is on-topic, whether relevant prior work can be retrieved, and whether extracted claims are both accurate and traceable. We report a staged evaluation of an evidence-oriented pipeline for graph neural network (GNN) literature. We first document a corpus topic-drift failure: year-stratified free-text retrieval admitted papers in which “graph” and “neural network” occurred independently, especially before 2019. We rebuilt the main corpus using qualified GNN phrases as a build-time inclusion rule, yielding 1,951 English, abstract-bearing records dated 2015–2026. In a fixed-seed, 50-record stratified consistency audit—not an independent precision validation—all sampled records matched both the broad and strict topic rules; because the strict rule was also used for inclusion, the audit does not independently estimate topical precision. We next evaluated title-query retrieval against citation-derived relevance sets for 19 graph-learning surveys (2,399 relevant work IDs). Dense retrieval achieved mean nDCG@10 of 0.563 (95% bootstrap interval 0.445–0.682), compared with 0.513 (0.381–0.644) for reciprocal-rank fusion (RRF) and 0.440 (0.300–0.589) for BM25. In a separate 100-paper extraction comparison, strict schema-and-span validity was 90%, 89%, and 86% for Gemini 3.1 Flash-Lite, Gemini 3.5 Flash-Lite, and Gemma 4 31B IT, respectively; all non-null spans in accepted outputs were exact substrings. On 76 human-annotated, in-topic papers, exact string scoring understated free-text agreement: Method F1 increased from 0.000 to 0.748 under an exploratory embedding-cosine match, while categorical paper-type F1 was 0.883. Finally, Gemini 3.1 Flash-Lite produced schema-valid extractions for all 1,951 v4 records. Of 7,218 original non-null evidence spans, 6,858 were already exact, 173 were repaired to exact substrings, and 187 were nulled; all 7,031 retained spans are literal abstract substrings. These results establish corpus, retrieval, extraction, and limited human-agreement evidence, but do not establish the validity of generated research gaps. Concept-graph sparsity testing, external novelty verification, stability analysis, survey-recovery, and temporal-holdout evaluation remain future work.

**Keywords:** research-gap discovery; graph neural networks; literature retrieval; evidence extraction; corpus quality; evaluation

## 1. Introduction

Research-gap discovery systems are usually judged by the plausibility of their generated recommendations. Yet a persuasive gap can still be an artifact of an incomplete or contaminated corpus, retrieval that misses the relevant precedent, or an extractor that produces unsupported claims. A system that cannot trace a claim to a source passage is not made reliable by a fluent explanation.

This paper reports the portions of an evidence-grounded literature pipeline that were implemented and evaluated under a short project deadline. The topic is graph neural networks and related graph-learning methods. The pipeline work comprises: (1) iterative OpenAlex corpus construction and a documented correction for topic drift; (2) citation-grounded retrieval evaluation using survey titles as queries; (3) structured claim extraction with literal evidence spans; and (4) a human-annotated comparison for extraction outputs. A completed corpus-wide extraction provides a scale result. We explicitly do not claim that the system has detected valid research gaps: candidate generation from sparsity, novelty verification, and temporal validation were not run.

The corpus work revealed a substantive methodological failure. Earlier year-stratified searches used broad relevance retrieval and admitted records where generic “graph” and “neural network” terminology appeared without a GNN method. The contamination was concentrated in earlier years, exactly where temporal conclusions would be most sensitive. A later strict build required named GNN-related phrases in the title or reconstructed abstract, excluding unqualified “message passing.” This correction reduced the temptation to treat a stratified sample as a natural time series and motivates a second central design principle: the evidence quality of the corpus must be evaluated separately from downstream model performance.

Our principal contributions are:

1. A transparent account of topic drift and a strict, build-time phrase qualification rule for a 1,951-record corpus.
2. A citation-derived retrieval benchmark comparing BM25, dense retrieval, and RRF, with per-query bootstrap intervals and a high-unresolved-reference sensitivity analysis.
3. A structured extraction reliability study combining a three-model 100-paper comparison, field-level human agreement on 76 in-topic annotations, and a completed 1,951-paper run.
4. A clear boundary between measured pipeline components and unvalidated research-gap claims, with a concrete design for the work still required.

## 2. Methods

### 2.1 Corpus source, scope, and screening

Records were retrieved from the OpenAlex Works API for publication years 2015–2026 and English-language works. Inclusion required an abstract and a title or abstract matching at least one qualified topic phrase: “graph neural network(s),” “GNN(s),” “graph convolutional network(s),” “graph attention network(s),” “graph representation learning,” or “message passing neural network(s).” The rule does **not** accept unqualified “message passing.” OpenAlex’s relevance-ranked phrase search was used to find candidates; the local title/abstract phrase check supplied the required inclusion criterion. Candidate retrieval requested up to 300 works per year and included no more than 200 screened eligible works per year.

The final main corpus contains OpenAlex work records with reconstructed abstracts and metadata. Duplicate removal was staged by DOI and normalized title. The embedding-deduplication stage was recorded as **skipped** for v4 and removed zero records; consequently, the PRISMA counts must not be read as showing that embedding deduplication was applied. Abstract-missing records were removed, followed by the language/date check and yearly inclusion cap.

The build’s abstract reconstruction was also corrected after an earlier implementation retained only the first occurrence of each distinct word in OpenAlex’s inverted index. The corrected reconstruction restores all indexed token occurrences, preserving repeated words and their positions. This is important because corrupted abstracts can damage both retrieval and evidence-span validation. Corpus v4 was built after this correction.

### 2.2 Topic-drift audit

We sampled 50 v4 records with fixed seed 20260928 and proportional year stratification using largest-remainder allocation. Each title and abstract was checked against two rules:

- **Broad audit rule:** graph neural network(s), GNN(s), graph convolutional network(s), graph attention network(s), graph representation learning, or unqualified message passing.
- **Strict inclusion rule:** the same qualified graph-learning phrases, but “message passing” only when qualified by “neural network(s).”

The sample was intended to assess whether the stricter corpus build also satisfies the broader lexical check and the stricter definition. It was not independently labeled by domain experts. Further, since the strict rule is the build-time filter, its full-corpus match rate is a construction invariant, not an independent estimate of topical precision.

### 2.3 Survey-derived retrieval evaluation

The retrieval evaluation uses a citation-expanded candidate set and is separate from the v4 main corpus. The initial Phase 1 pool contained 174 abstract-bearing records. Survey candidates were selected from graph-neural, graph-convolutional, or graph-learning survey titles in the union of this pool and resolved references, ordered by publication year descending and OpenAlex work ID ascending. Survey titles served as search queries. A survey’s OpenAlex `referenced_works` were intersected with the abstract-bearing expanded candidate pool to form that query’s relevance set; the survey itself was excluded from its own ranking. A survey qualified when it had an abstract and at least three cited candidate works with abstracts in the expanded pool.

Of 20 screened survey sources, 19 qualified for the principal evaluation. Reference resolution found 2,353 works, of which 1,991 had abstracts; 641 references remained unresolved. The expanded candidate pool contained 2,072 records, including 1,898 cited records added to Phase 1. The principal metric file reports 19 query surveys and 2,399 relevant query-work IDs across the benchmark.

We compared:

- **BM25**, with \(k_1=1.5\) and \(b=0.75\);
- **Dense retrieval**, using `BAAI/bge-base-en-v1.5` embeddings and exact cosine ranking; and
- **RRF**, combining the two rankings with rank constant \(k=60\).

We report macro-averaged Precision@10/20/50, Recall@10/20/50, nDCG@10/20, and MAP. Confidence intervals are 95% percentile intervals from 10,000 query-level bootstrap resamples. The principal-metric configuration identifies nDCG@10 as primary, but also records that this choice was made after inspection of first-pass results and was not preregistered. A leave-one-query-out weighted-RRF variant was explored as an ablation and is not treated as a headline, independently held-out result.

As a sensitivity analysis, five surveys with more than 30% unresolved references were excluded. This left 14 surveys and 1,861 relevant query-work IDs. The sensitivity analysis is informative about the dependence of ranking scores on uncertain reference resolution; it does not replace or correct the principal run.

### 2.4 Structured extraction and span handling

The extraction schema represents a paper type, an include-in-counts decision, and evidence-bearing values for task, method, dataset, metric, setting, stated limitation, and stated future work. A value must be paired with either a non-null evidence span or a null span; survey/review records are constrained to null task, method, and dataset fields and excluded from counts. Generic task/method/dataset labels are rejected by a specificity stoplist. For an accepted span, validity is literal containment in the source abstract: the span must be a contiguous, unmodified substring.

An initial 100-paper, three-model comparison used batch size four and strict output validation. The primary model decision was Gemini 3.1 Flash-Lite with thinking set to low. The full v4 corpus run used this model, batch size eight, the same schema and extraction instructions, and a persisted per-model/per-credential request limiter. Failed spans no longer invalidate otherwise schema-valid papers: close spans are repaired by snapping to a contiguous substring in the abstract; spans without a sufficiently close match are nulled for that field only. Repaired and nulled values are recorded separately. Repairs establish source-text containment, not that the selected substring semantically supports the model’s claim.

### 2.5 Human-annotated extraction validation

The original 100-paper human sheet contained 18 papers subsequently confirmed off-topic under the strict v4 corpus rule. These labels were not discarded as annotation work, but they were excluded from in-topic model metrics. Of the 82 remaining annotated papers, 76 occur in v4; evaluation was restricted to those 76 IDs. Human paper-type labels were mapped transparently: `survey` to `survey_or_review`, `application_study` to `application`, and two reviewed paper-specific labels to `new_method` or `application`. No `theory` labels remained in the v4 cohort. The Spektral library/tool paper remained an unmapped edge case and was excluded only from paper-type scoring. The literal value `not_stated` was normalized to null for dataset, metric, stated limitation, and stated future work.

Categorical fields were scored by exact match. Free-text fields were scored both by case-folded, whitespace-normalized exact match and by embedding cosine similarity using `BAAI/bge-base-en-v1.5`. The semantic threshold, 0.66549, was chosen from ten fixed-seed exact-match Method disagreements: after sorting their cosine scores, the midpoint of the largest adjacent gap was used. This threshold was then applied uniformly across free-text fields. It is exploratory, calibrated on Method only, and not independently adjudicated. Per-paper bootstrap 95% percentile intervals use 10,000 resamples and seed 20260929. Five cohort IDs had no valid output in the comparison extraction artifact; these were not silently dropped and were treated as empty predictions, counting as false negatives for non-null gold claims.

## 3. Results

### 3.1 Corpus construction and topic drift

The v4 PRISMA flow was: 2,879 identified; 13 DOI duplicates removed; 2,866 after DOI deduplication; 111 normalized-title duplicates removed; 2,755 after title deduplication; embedding deduplication skipped; 536 missing-abstract records removed, leaving 2,219; five records removed by language/date screening, leaving 2,214; and 263 records removed by the 200-record yearly cap, leaving **1,951 included**.

| Publication year | Included | Publication year | Included |
|---:|---:|---:|---:|
| 2015 | 38 | 2021 | 200 |
| 2016 | 40 | 2022 | 200 |
| 2017 | 77 | 2023 | 200 |
| 2018 | 200 | 2024 | 200 |
| 2019 | 200 | 2025 | 200 |
| 2020 | 200 | 2026 | 196 |

The phrase-qualified candidate search could not fill the 200-record target in 2015–2017. Conversely, the cap makes many later years equal by construction. These yearly counts are therefore neither estimates of annual publication volume nor a suitable direct time-series outcome. The 2026 count is also partial-year coverage at the time of retrieval.

In the 50-paper stratified audit, **50/50** matched the broad rule and **50/50** matched the strict rule. Per-year sample sizes and broad/strict matches were: 2015, 1/1; 2016, 1/1; 2017, 2/2; 2018, 6/6; 2019–2026, 5/5 in each year. The early-year strata are especially small, so the sample does not support narrow year-specific precision claims. The 1,951/1,951 full-corpus strict match is expected from the build filter and should not be presented as an independent audit.

The version history matters for interpretation. Earlier broad phrase retrieval admitted unrelated records, with the false positives concentrated in early years; for example, “message passing” can describe a communication protocol rather than a neural-network architecture. A separate v3 cross-reference identified 521 records matching only unqualified “message passing.” The v4 build prevents that phrase alone from satisfying inclusion. Thus, corpus construction itself was an evaluated component and a source of selection bias, not merely preprocessing.

### 3.2 Retrieval effectiveness

Table 2 reports the full set of principal retrieval metrics for the 19-survey benchmark. Values are macro means with 95% query-bootstrap intervals.

| Metric | BM25 | Dense | RRF |
|---|---:|---:|---:|
| Precision@10 | 0.458 [0.321, 0.600] | **0.537 [0.421, 0.658]** | 0.500 [0.379, 0.626] |
| Precision@20 | 0.395 [0.274, 0.524] | 0.434 [0.337, 0.534] | 0.434 [0.332, 0.542] |
| Precision@50 | 0.319 [0.229, 0.412] | 0.375 [0.306, 0.443] | **0.381 [0.294, 0.464]** |
| Recall@10 | 0.039 [0.026, 0.052] | **0.049 [0.036, 0.063]** | 0.045 [0.032, 0.058] |
| Recall@20 | 0.068 [0.045, 0.093] | **0.080 [0.059, 0.102]** | 0.078 [0.057, 0.100] |
| Recall@50 | 0.135 [0.094, 0.178] | 0.165 [0.135, 0.197] | **0.169 [0.127, 0.211]** |
| nDCG@10 | 0.440 [0.300, 0.589] | **0.563 [0.445, 0.682]** | 0.513 [0.381, 0.644] |
| nDCG@20 | 0.401 [0.274, 0.536] | **0.483 [0.383, 0.588]** | 0.462 [0.350, 0.577] |
| MAP | 0.202 [0.152, 0.259] | 0.257 [0.212, 0.303] | **0.262 [0.206, 0.318]** |

Dense retrieval achieved the largest mean nDCG at both cutoffs and the strongest mean precision at 10 and recall at 10 and 20. RRF had slightly higher mean MAP, Precision@50, and Recall@50. The intervals overlap substantially; the table is descriptive and does not establish statistically significant superiority of one system. The relatively low Recall@10–50 also reflects a large citation-derived relevance universe: the benchmark measures recovery of cited works from a finite candidate pool, not topical relevance judgments for every retrieved item.

When excluding the five surveys with more than 30% unresolved references, the sample contained 14 queries and 1,861 relevant IDs. Dense, BM25, and RRF nDCG@10 means were respectively 0.537 [0.401, 0.679], 0.425 [0.260, 0.593], and 0.484 [0.338, 0.633]. Their MAP means were 0.272 [0.216, 0.326], 0.205 [0.145, 0.277], and 0.271 [0.206, 0.338]. Dense retrieval remained strongest by mean nDCG@10; dense and RRF were nearly tied in MAP. The intervals are broad in this smaller sensitivity set.

### 3.3 Three-model, 100-paper extraction comparison

The comparison outputs were partial under strict schema-and-span validation. Gemini 3.1 Flash-Lite produced 90 valid records, Gemini 3.5 Flash-Lite 89, and Gemma 4 31B IT 86. All non-null evidence spans in accepted records were exact abstract substrings; missing records were principally outputs rejected for invalid spans or specificity violations. The comparison should therefore be interpreted as joint model-and-validation yield, not raw language-model accuracy. The Gemma run additionally had repeated HTTP 401 authentication failures on individual attempts and transient server errors.

| Model | Valid / 100 | Valid rate | API calls | Input tokens | Output tokens | Thinking tokens |
|---|---:|---:|---:|---:|---:|---:|
| Gemini 3.1 Flash-Lite | 90 | 90% | 36 | 44,202 | 33,510 | 11,599 |
| Gemini 3.5 Flash-Lite | 89 | 89% | 39 | 45,958 | 32,738 | 0 |
| Gemma 4 31B IT | 86 | 86% | 77 | 45,938 | 31,514 | 0 reported |

The principal extraction model was selected as Gemini 3.1 Flash-Lite with thinking set to low. The comparison alone does not show a decisive quality advantage: validity yields differ by only a few percentage points, and the runs had different retry and server-error behavior. The reasons for pinning a single model were operational reproducibility and stable settings for the full-corpus pass, not a claim that it universally outperforms the alternatives.

### 3.4 Human validation and the exact-match artifact

The human-annotated evaluation covered 76 in-topic v4 papers. Valid model output coverage in the comparison artifact was 71/76; five missing or strictly rejected outputs remained in the denominator and were scored as missing predictions. Exact categorical F1 was **0.883** for paper type (n=75, with Spektral excluded as an unmapped category) and **0.905** for include-in-counts (n=76).

For free text, exact-match scoring was a poor agreement measure: it marked all 76 Method predictions as mismatches despite source-supported paraphrases and acronyms. Under the exploratory semantic match, Method F1 was 0.748. Table 3 retains both measures rather than silently replacing one. Intervals are percentile bootstrap 95% intervals; all rows have n=76 papers. “Insufficient data” for stated limitation reflects only four positive human annotations; its raw estimate is not promoted as a reportable rate.

| Field | Exact F1 | Semantic F1 (95% interval) | Human positives |
|---|---:|---:|---:|
| Task | 0.329 | **0.900** [0.840, 0.952] | 76 |
| Method | 0.000 | **0.748** [0.647, 0.839] | 76 |
| Dataset | 0.039 | **0.725** [0.619, 0.817] | 63 |
| Metric | 0.059 | **0.765** [0.638, 0.868] | 37 |
| Setting | 0.018 | 0.500 [0.377, 0.614] | 76 |
| Stated limitation | 0.091 | Insufficient data (n=4 positive) | 4 |
| Stated future work | 0.000 | **0.909** [0.667, 1.000] | 6 |

Semantic precision and recall for Task were 0.984 and 0.829; Method 0.825 and 0.684; Dataset 0.949 and 0.587; Metric 0.839 and 0.703; Setting 0.778 and 0.368; and Future Work 1.000 and 0.833. For categorical fields, paper-type precision/recall were 0.914/0.853, and include-in-counts precision/recall were 0.969/0.849.

The semantic results are provisional. The threshold (0.66549) was inferred from ten Method disagreements only and applied to all fields; it is not an independently validated semantic boundary. This likely underestimates some matches, especially concise acronyms compared with expanded human descriptions (as in the 2SD-GAT example), but may also accept semantically related yet insufficiently equivalent labels. Setting remains a real area of concern: its semantic recall was 0.368, and inspected disagreements included a human “unsupervised domain adaptation between graphs” versus model “domain adaptation.” We did not recalibrate the global threshold on Setting alone. A larger, independently adjudicated set or a human-checked LLM-judge study is needed before these field scores are treated as definitive.

### 3.5 Full-corpus extraction

The completed full run used Gemini 3.1 Flash-Lite with thinking level low and batch size eight. It produced validated records for **1,951/1,951 papers** (100% completion; no missing or unprocessed papers in the run summary). Eighteen specificity-stoplist rejections were recorded during processing (0.92% of corpus); these are validation/rejection events, not 18 final excluded corpus records.

Of 7,218 original non-null evidence spans, 6,858 were already exact literal substrings, 173 were repaired to literal substrings, and 187 were nulled field-by-field. Thus 7,031 values remained non-null after repair, and **all 7,031/7,031 (100%) were exact substrings** of their supplied abstracts. This is a span-grounding result, not proof that every extracted value is semantically correct or that every chosen span adequately entails its label.

| Predicted paper type | Count |
|---|---:|
| New method | 1,630 |
| Survey or review | 176 |
| Empirical study | 78 |
| Position or other | 32 |
| Application | 20 |
| Benchmark or dataset | 15 |
| **Total** | **1,951** |

The full run used 453 API calls. Usage metadata was present for 407 calls and totaled 705,922 input tokens, 589,297 output tokens, and 213,629 thinking tokens (1,508,848 total reported tokens). Output usage including thinking was 802,926 tokens. No completion claim is made about model quality beyond schema acceptance and substring grounding; the field-level human validation is a separate, smaller evaluation.

## 4. Discussion

### 4.1 Corpus quality is part of the method

The v2-to-v4 progression demonstrates that corpus construction can dominate downstream interpretation. A year-stratified relevance search appeared to provide longitudinal coverage, yet earlier years contained many papers where generic graph or neural-network terminology was unrelated to GNN methods. Treating those records as GNN literature would distort both the apparent historical development and any measured concept frequencies. A strict build-time topic rule reduced that identifiable failure mode. The audit is encouraging but limited: 50 papers, very small early-year strata, and a lexical criterion that was also used in inclusion. Independent expert screening remains necessary for a genuine corpus precision estimate.

Year-stratified counts are not publication volume. The v4 design caps included records at 200 for many years, and earlier years could not reach that cap. Any eventual temporal analysis must compare each concept’s share of included papers within its year, with confidence intervals, or use defensible external per-year totals. Raw counts from this corpus cannot support claims about field growth.

### 4.2 Retrieval helps, but it does not validate gaps

Dense retrieval and RRF showed complementary behavior. Dense retrieval led on early precision, recall, and nDCG, whereas RRF slightly led on MAP and some deeper cutoffs. This supports comparing hybrid retrieval rather than assuming score fusion is automatically superior. However, the evaluation is citation-based and derived from a relatively small set of 19 survey queries. Citation lists are incomplete relevance labels, and unresolved OpenAlex references remain a source of uncertainty. The high-unresolved sensitivity analysis changed both the sample size and relevant universe. In addition, retrieval artifacts were built from a Phase 1 plus citation-expanded candidate pool, not the final v4 corpus; the retrieval numbers are not a v4 end-to-end evaluation. **We have not checked whether the abstracts used for Phase 2 survey queries or candidate-pool construction were affected by the repeated-token reconstruction bug; this remains unconfirmed, and no Phase 2 rebuild or audit has been performed.**

### 4.3 Exact spans and semantic labels answer different questions

Literal substring validation is a useful provenance constraint. It prevents model-generated quotations that cannot be found in the supplied abstract and makes every retained value inspectable. The full-corpus repair policy avoids discarding an otherwise valid paper because of one malformed span, while field-level nulling makes uncertainty explicit. Neither substring containment nor successful JSON-schema validation establishes claim entailment: a copied sentence can be irrelevant to its field, and a repaired span may be only the closest textual match.

The human study makes a complementary point. Exact textual equality between a compact extracted label and a richer human description can be much too strict for methods, datasets, or tasks. In this sample, exact Method F1 was zero while the semantic score was 0.748. Reporting only exact match would suggest total failure; reporting only semantic F1 would hide the subjective threshold and potential false equivalence. Keeping both measures exposes the metric choice. Semantic values remain exploratory because the threshold was calibrated on ten Method examples and transported to every field.

The human cohort is also small and selected. Eighteen labels were removed from the in-topic calculation because the papers were confirmed off-topic, and only 76 of the remaining 82 labels matched v4. The five missing strict-valid outputs count as errors under the chosen scoring rule, but the comparison cache did not supply a valid prediction for them. Field-specific positive support varies widely: only four human labels were positive for stated limitation and six for future work. We therefore report stated limitation as insufficient data in summary comparisons.

### 4.4 What remains untested: gap validity

The completed work evaluates the substrate for a gap-discovery system, not the final scientific claim that a research gap is real. The planned next stage constructs a task–method–dataset co-occurrence tensor and concept graph. A candidate sparsity procedure would compare observed tuple counts with expected counts, apply a Poisson-style test and multiple-testing correction, and merge near-duplicate candidates. An extended attempt at sparsity-based candidate generation is reported in the separate `addendum.md`: at the tested exact and topic-coarsened granularities, no cells yielded statistically defensible candidates. Subsequent stages to verify novelty against external literature, measure output stability across repeated generations, and evaluate survey recovery and temporal holdout remain untested. In particular, a statistically rare combination may be rare because of extraction error, corpus selection, ontology mismatch, or low volume; rarity alone is not evidence of a scientifically important gap.

## 5. Limitations

1. **Lexical topic screening.** The v4 phrase rule is more precise than broad relevance search but can miss relevant work that uses different terminology and can still admit false positives containing qualified phrases incidentally. The audit is small and not expert-adjudicated.
2. **Sampling and temporal claims.** The yearly inclusion cap makes record counts unsuitable as volume estimates; 2026 is partial, and early years are underfilled. No temporal holdout or momentum trend was measured.
3. **Deduplication.** DOI and normalized-title stages removed 13 and 111 records, respectively. Embedding deduplication was skipped in v4, despite the configured pipeline having an embedding stage.
4. **Retrieval benchmark.** Relevance was inferred from survey references, not direct expert judgments. Only 19 principal queries were evaluated; references remained unresolved; the primary metric was selected after first-pass inspection; and the candidate pool was not rebuilt from v4. Whether survey-query or candidate-pool abstracts were affected by the repeated-token reconstruction bug has not been checked; the status is unconfirmed and is an explicit audit TODO.
5. **Extraction validity.** Schema validity and substring evidence do not guarantee that a claim is entailed by its evidence or that field labels are semantically correct. Repair can make spans textually exact without establishing entailment. The specificity stoplist is incomplete by design.
6. **Human evaluation.** The in-topic annotated cohort is 76 papers, with only 71 valid comparison outputs for the matching artifact and uneven positive support by field. Semantic matching uses an exploratory Method-calibrated threshold applied uniformly; the threshold has not been validated independently.
7. **No gap-level outcome evaluation.** Concept sparsity, false-discovery control under realistic extraction noise, novelty checking, stability/calibration, survey recovery, and temporal holdout were not evaluated. Accordingly, the system is not yet validated as a research-gap detector.

## 6. Conclusion

This project established a reproducible, auditable foundation for GNN literature analysis: a strict-filtered 1,951-paper corpus, a citation-derived comparison of sparse and dense retrieval, a structured extraction process with literal evidence spans, and a human comparison that shows why exact and semantic agreement metrics must be reported together. The most consequential engineering finding is methodological: apparently reasonable year-stratified searches can hide severe topic drift, and a contaminated corpus would undermine downstream temporal analysis regardless of model quality.

The evidence supports the extraction system as a corpus-processing component with complete schema-valid coverage and exact retained spans after conservative repair/nulling. It does not yet support claims that generated sparsity candidates are true research gaps. That claim requires the unrun stages: concept-graph construction, corrected-corpus and temporal evaluation, novelty verification, uncertainty/stability measurement, and human assessment of candidate usefulness.

## Reproducibility and artifact map

The following project artifacts contain the underlying machine-readable results:

- v4 PRISMA: `../prisma_main_v4.json` and `../prisma_main_v4.md`
- v4 topic audit: `../experiments/annotation/topical_precision_audit_v4.json` and `.csv`
- Principal retrieval metrics: `../gapfinder/experiments/runs/retrieval/metrics.csv`
- Unresolved-reference sensitivity metrics: `../gapfinder/experiments/runs/retrieval_sensitivity_unresolved30/metrics.csv`
- Retrieval configuration: `../gapfinder/experiments/configs/retrieval.yaml`
- 100-paper model comparison summaries: `../experiments/runs/phase3_100_comparison_20260929_113614/`
- Human-validation metrics: `../experiments/runs/phase3_v4_semantic_annotation_metrics.json`
- Full-corpus extraction summary: `../experiments/runs/phase3_full_corpus_v4_gemini31_low/summary.json`
- Full-corpus manifest and batch log: `../experiments/runs/phase3_full_corpus_v4_gemini31_low/`
- Phase 4 design notes: `../gapfinder/experiments/configs/phase4_design.md` and `../agent_master_prompt.md`
- Sparsity-analysis addendum: `addendum.md`

## References

1. OpenAlex. *Works API and search documentation*. https://developers.openalex.org/api-entities/works
2. Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M. M., & Gatford, M. (1995). Okapi at TREC-3. In *Proceedings of the Third Text REtrieval Conference (TREC-3)*, NIST Special Publication 500-225, 109–126.
3. Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In *Proceedings of SIGIR 2009*, 758–759.
4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP 2019*, 3982–3992.
5. BAAI. *BGE English embedding models: `BAAI/bge-base-en-v1.5` model card*. https://huggingface.co/BAAI/bge-base-en-v1.5
6. Google. *Gemini API documentation: structured outputs and thinking*. https://ai.google.dev/gemini-api/docs
