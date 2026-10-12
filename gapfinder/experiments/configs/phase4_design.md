# Phase 4 design note: temporal momentum

`corpus_main_v3.jsonl` was assembled with a fixed cap of about 200 candidate
records per publication year, rather than sampling years in proportion to the
size of the literature. Therefore, yearly corpus record counts are not
publication-volume evidence and must not be used as the numerator or denominator
for a temporal trend.

Phase 4 momentum will estimate each concept's annual share among included
papers: concept-positive papers divided by all included papers in that year.
Report a paper-level bootstrap 95% confidence interval for each annual share,
with a fixed seed. If a defensible, topic-specific OpenAlex per-year total is
available, it may be used as a separate volume denominator/offset and must be
reported distinctly from the corpus share. Do not interpret a rise in the
year-stratified corpus's raw count as literature growth.

The helper `gapfinder.graph.momentum.yearly_concept_share` and its test pin the
share denominator and bootstrap behavior. Phase 4 trend estimation and
substantive momentum results remain unrun.
