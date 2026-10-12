# Annotation guide

Annotate only what the paper abstract explicitly states. Do not use the title,
full text, general field knowledge, or model suggestions to fill a field. Use the
existing label columns in `extraction_sample.csv`; put an empty cell for an
unknown or absent value. The source sheet contains paper metadata and abstracts,
not model outputs.

## Paper-level fields

### `paper_type_label`

Choose exactly one schema value:

- `survey_or_review`: synthesizes or reviews prior work rather than reporting a
  new primary method or experiment.
- `new_method`: proposes a new algorithm, model, architecture, or technical
  procedure.
- `benchmark_or_dataset`: primarily introduces or evaluates a benchmark,
  dataset, test suite, or shared task.
- `empirical_study`: primarily reports an empirical analysis or experiment
  without proposing a new method as its main contribution.
- `application`: applies existing techniques to a domain problem as its main
  contribution.
- `position_or_other`: perspective, position, conceptual, or other paper that
  does not fit the categories above.

Positive examples: `survey_or_review`; `new_method`; `benchmark_or_dataset`.
Negative examples: `survey`; `method paper`; `dataset paper`. These are not
schema enum values; choose the closest exact value above.

### `include_in_counts_label`

Use `false` for `survey_or_review`; use `true` for all other paper types. This is
a categorical schema constraint, not a judgment of paper quality.

Positive examples: `true` for a new algorithm; `true` for a dataset paper;
`true` for an empirical study.

Negative examples: `true` for a survey (must be `false`); `false` for a new
method (must be `true`); `false` for an application paper (must be `true`).

## Claim fields

For each of the following fields, the sheet's `*_label` cell should contain the
specific text value or entity supported by the abstract, not an evidence span.
Leave the cell blank when the abstract does not state a usable value. If the
abstract gives only a broad/generic phrase, leave it blank.

### `task_label`

The specific problem or prediction objective addressed.

Positive examples: `node classification`; `link prediction`; `human pose
estimation`.

Negative examples: `machine learning`; `graph task`; `various applications`.

### `method_label`

The named algorithm, model, architecture, or explicitly described technique.

Positive examples: `GraphSAGE`; `Graph Attention Network`; `spatial-temporal
graph convolutional network`.

Negative examples: `novel approach`; `our method`; `Adaptive model`.

### `dataset_label`

The named dataset, benchmark, or explicitly identified data collection.

Positive examples: `Cora`; `ogbn-arxiv`; `KITTI`.

Negative examples: `benchmark data sets`; `various datasets`; `real-world data`.

### `metric_label`

The named evaluation metric or measure, only when the abstract identifies it.

Positive examples: `accuracy`; `AUC`; `F1-score`.

Negative examples: `performance`; `better results`; `evaluation score`.

### `setting_label`

The explicit experimental context, conditions, or domain setting.

Positive examples: `inductive learning`; `zero-shot transfer`; `urban traffic
crash prediction in London`.

Negative examples: `realistic setting`; `challenging environment`; `large-scale`
without a stated scale or context.

### `stated_limitation_label`

The substantive limitation stated in the abstract. Record the limitation itself,
not an introductory pointer to limitations.

Positive examples: `requires a separate model for each graph`; `performance
degrades under heterophily`; `limited by the lack of labeled anomaly data`.

Negative examples: `several limitations`; `four open problems`; `future work is
needed`.

### `stated_future_work_label`

The substantive future direction explicitly described in the abstract. Record
the action or research direction, not a pointer to a list of directions.

Positive examples: `extend the method to dynamic graphs`; `evaluate on
multilingual corpora`; `collect labeled anomaly data for external validation`.

Negative examples: `future research directions`; `more work is needed`; `four
open problems`.

## Nulls and specificity

The extraction schema represents an absent field as both `value=null` and
`evidence_span=null`. On this manual sheet, leave the corresponding label cell
blank. Never infer values from citations, paper title, common practice, or
unstated experimental details. A named entity or concrete task is specific;
generic categories and vague claims are not. Generic task/method/dataset phrases
such as “various datasets,” “benchmark data sets,” and “novel approach” must be
left blank. For limitations and future work, a pointer like “several
limitations” is not substantive content and must be left blank.

## Surveys and reviews

Mark a survey or review as `survey_or_review` and `include_in_counts=false`.
Leave `task_label`, `method_label`, and `dataset_label` blank, even if the
review discusses named tasks, methods, or datasets. The other claim fields may
be annotated only when the abstract explicitly states them and they meet the
same specificity and evidence rules.

## Evidence spans

`evidence_span` is part of every structured extraction claim, rather than a
separate column in the manual sheet. For any non-null extracted field, it must
be copied verbatim as one contiguous substring of that paper's abstract. Case,
punctuation, spacing, and wording must match exactly. Do not paraphrase,
normalize, join separate fragments, or copy text from the title or another
source. If no exact supporting substring exists, the field value and span must
both be null.

Positive examples: abstract contains `node classification` and the span is
exactly `node classification`; abstract contains `AUC of 0.91` and the span is
exactly `AUC of 0.91`; abstract contains `on the Cora dataset` and the span is
exactly `Cora`.

Negative examples: changing `GraphSAGE` to `Graph Sage`; paraphrasing `improves
accuracy` as `better performance`; joining `Cora` and `Citeseer` when those
words occur in separate parts of the abstract.
