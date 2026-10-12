# ResearchPaper

Paper and source code for an auditable literature-analysis pipeline for graph neural network research.

## Contents

- `paper/`: manuscript in Markdown and LaTeX, abstract, and sparsity-analysis addendum.
- `gapfinder/`: Python package, tests, experiment configurations, and aggregate evaluation metrics.
- `docs/AUDIT.md`: project audit findings.
- `repro/`: post-hoc analysis script and aggregate score summaries.

## Reproducibility

Run the Python tests from `gapfinder/`:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

The source corpus, full extraction records, human annotation sheets, local credentials, model caches, and generated run logs are not included. Corpus-wide scripts therefore require separately obtained input files; configure API keys locally through environment variables as documented by the relevant commands. No secrets should be committed.

The manuscript's reported aggregate results are retained where practical; the full inputs needed to independently regenerate them are not distributed in this repository.
