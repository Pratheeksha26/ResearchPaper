# Addendum: Entity-Level Sparsity Analysis

After the main evaluation, we tested sparsity-based candidate generation on the complete v4 extraction set: 1,775 papers marked `include_in_counts=true` out of 1,951. We built a three-way task–method–dataset tensor, then pairwise task–method, method–dataset, and task–dataset matrices. For each matrix, expected co-occurrence counts were computed under marginal independence; lower-tail Poisson tests were restricted to cells with expected count at least 2, followed by Benjamini–Hochberg (BH) correction. We did not lower this threshold.

At the fine-grained level, entities came from the field-specific canonical map at cosine threshold 0.99. The three-way tensor had 806 observed cells, of which 802 (99.5%) occurred once; none met the minimum expected-count threshold. Pairwise matrices likewise had no testable cells. Their singleton counts were 1,597/1,610 (99.2%) for task–method, 838/844 (99.3%) for method–dataset, and 808/817 (98.9%) for task–dataset. Thus, no fine-grained cells entered BH correction.

We then tested a separate, coarser representation for sparsity only, preserving the fine-grained map for evidence and display. Complete-linkage clustering was explored separately by field at cosine thresholds 0.75 and 0.80, but still yielded 477–826 clusters per field. We also applied field-specific K-means to normalized BGE embeddings, selecting among 30–100 clusters by sampled cosine silhouette. The selected task, method, and dataset cluster counts were 100, 30, and 60, with silhouette scores of 0.123, 0.076, and 0.082, respectively. Manual inspection found incoherent groupings, including clusters that mixed unrelated datasets or bundled distinct methods; these scores and inspections do not support treating the clusters as validated semantic topics.

On the K-means topic map, the three-way tensor contained 741 observed cells, 689 (93.0%) singletons, and no cells meeting the expected-count threshold. Pairwise coarsening made 192 cells testable in total:

| Pairwise matrix | Pair-complete papers | Observed cells | Singleton cells | Testable cells (expected count ≥ 2) | BH survivors, q ≤ 0.05 | BH survivors, q ≤ 0.10 |
|---|---:|---:|---:|---:|---:|---:|
| Task–method | 1,624 | 876 | 530 (60.5%) | 120 | 0 | 0 |
| Method–dataset | 850 | 480 | 294 (61.3%) | 64 | 0 | 0 |
| Task–dataset | 827 | 503 | 343 (68.2%) | 8 | 0 | 0 |
| **Total** | — | — | — | **192** | **0** | **0** |

This is a negative result for sparsity-based candidate generation at the tested granularity, not evidence that the literature contains no research gaps. Two plausible constraints are the long-tailed distribution of entities and co-occurrences in a 1,775-paper analysis set, and the fact that the attempted topic coarsening did not produce reliable semantic groupings. The present analysis cannot isolate their relative contributions. It also provides no basis for proceeding to link prediction or momentum analysis using these same representations.

Future work should test substantially larger literature collections—potentially on the order of 10,000 or more papers, closer to corpus sizes used in classic literature-based discovery—and/or use a topic model validated for semantic coherence before evaluating sparsity-based candidates. These are directions for further study, not guarantees that additional data or a different clustering method will yield defensible candidates.
