# ResearchPaper

Paper and source code for an auditable literature-analysis pipeline for graph neural network research.

## Contents

- `paper/`: manuscript in Markdown and LaTeX, abstract, and sparsity-analysis addendum.
- `gapfinder/`: Python package, tests, experiment configurations, and retrieval evaluation runs.
- `corpus_main_v4.jsonl`: strict phrase-screened 1,951-paper main corpus with PRISMA flow records (`prisma_main_v4.json`/`.md`).
- `experiments/`: human annotation data (`experiments/annotation/extraction_sample_annotated.csv`), topical precision audits, and extraction evaluation metrics.
- `docs/AUDIT.md`: project audit findings.
- `repro/`: post-hoc analysis scripts, coarse/fine sparsity results, and topic clustering artifacts.
- `runs/`: extraction execution logs and model request accounting.

## Reproducibility

Run the Python tests from `gapfinder/`:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

The repository includes the v4 main corpus (`corpus_main_v4.jsonl`), PRISMA flow logs (`prisma_main_v4.json`/`.md`), human annotation sheets (`experiments/annotation/extraction_sample_annotated.csv`), model comparison runs, full-corpus extraction summaries, and post-hoc evaluation artifacts (`repro/`). Ephemeral model caches and private API credentials are not committed. Corpus-wide scripts requiring external model calls use environment variables as documented in `gapfinder/README.md`. No secrets should be committed.

Machine-readable evaluation metrics and run artifacts are preserved under `experiments/runs/` and `gapfinder/experiments/runs/`.
