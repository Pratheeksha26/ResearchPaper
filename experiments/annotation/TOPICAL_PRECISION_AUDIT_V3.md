# Main-corpus v3 topical precision audit

The v3 main corpus was rebuilt using an English filter and separate publication
year queries for 2015–2026, capped at 200 included search results per year
before deduplication and screening. The OpenAlex works `search` parameter used
quoted phrase alternatives for graph neural networks, GNN, graph convolutional
networks, graph attention networks, message passing, message-passing neural
networks, and graph representation learning. Since OpenAlex search includes
full text as well as title and abstract, each returned record was additionally
required to match one of those phrase patterns in its title or reconstructed
abstract.

The current OpenAlex searching documentation describes quoted phrase search
and confirms that works search covers `title`, `abstract`, and `fulltext`:
https://developers.openalex.org/api/searching. The fielded
`title_and_abstract.search` filter is not the current documented filter name;
field `.search` filters are documented as legacy/deprecated. The build
therefore uses the documented `search` parameter and applies a local
title/abstract inclusion check.

The audit used the same proportional, largest-remainder year stratification
and fixed seed `20260928` as the v2 audit. On-topic classification checks
title plus abstract, case-insensitively, for:

| Pattern | Regular expression |
|---|---|
| Graph neural network(s) | `\bgraph neural networks?\b` |
| GNN(s) | `\bGNNs?\b` |
| Graph convolutional network(s) | `\bgraph convolutional networks?\b` |
| Graph attention network(s) | `\bgraph attention networks?\b` |
| Message passing [neural network(s)] | `\bmessage passing(?: neural networks?)?\b` |
| Graph representation learning | `\bgraph representation learning\b` |

**Audit result: 50/50 on-topic (100%).** For 2015–2018 specifically, the
result is **18/18 (100%)**, exceeding the requested 90% threshold.

| Year | On-topic | Sampled | Fraction |
|---:|---:|---:|---:|
| 2015 | 5 | 5 | 100% |
| 2016 | 5 | 5 | 100% |
| 2017 | 4 | 4 | 100% |
| 2018 | 4 | 4 | 100% |
| 2019 | 5 | 5 | 100% |
| 2020 | 4 | 4 | 100% |
| 2021 | 4 | 4 | 100% |
| 2022 | 4 | 4 | 100% |
| 2023 | 4 | 4 | 100% |
| 2024 | 4 | 4 | 100% |
| 2025 | 4 | 4 | 100% |
| 2026 | 3 | 3 | 100% |

The sampled records and matched patterns are in `topical_precision_audit_v3.csv`;
machine-readable counts and patterns are in `topical_precision_audit_v3.json`.

After the initial v3 build, review of saved extraction evidence found that the
inverted-index reconstruction retained only the first occurrence of each
distinct token. Reconstruction was corrected to restore every indexed token
occurrence, and v3 was regenerated from cached OpenAlex responses. The audit
above was rerun on that corrected corpus; its included count is 1,920.
