# Main-corpus topical precision audit

The audit sampled 50 of 1,867 records from `corpus_main_v2.jsonl`, stratified
proportionally by publication year using seed `20260928` and largest-remainder
allocation. A record counts as on-topic when its title plus abstract matches at
least one of these case-insensitive regular expressions:

| Pattern name | Regex |
|---|---|
| graph neural network | `\bgraph neural networks?\b` |
| graph convolution | `\bgraph convolutions?(?:al)?(?: networks?)?\b` |
| GNN abbreviation | `\bGNNs?\b` |
| graph attention | `\bgraph attention(?: networks?)?\b` |
| message passing | `\bmessage passing\b` |
| graph representation learning | `\bgraph representation learning\b` |
| graph embedding | `\bgraph embeddings?\b` |
| graph-based representation or learning | `\bgraph[- ]based (?:representation|learning)\b` |
| graph-structured data/graphs | `\bgraph[- ]structured (?:data|graphs?)\b` |
| graph neural learning | `\bgraph neural (?:networking|learning)\b` |

Overall: **34/50 on-topic (68%)**, below the 90% threshold.

| Year | On-topic | Sampled | Fraction |
|---:|---:|---:|---:|
| 2015 | 0 | 5 | 0% |
| 2016 | 0 | 4 | 0% |
| 2017 | 2 | 5 | 40% |
| 2018 | 0 | 4 | 0% |
| 2019 | 4 | 4 | 100% |
| 2020 | 4 | 4 | 100% |
| 2021 | 5 | 5 | 100% |
| 2022 | 4 | 4 | 100% |
| 2023 | 4 | 4 | 100% |
| 2024 | 4 | 4 | 100% |
| 2025 | 4 | 4 | 100% |
| 2026 | 3 | 3 | 100% |

Ten sampled off-topic examples:

| Title | Year |
|---|---:|
| 2D image classification for 3D anatomy localization: employing deep convolutional neural networks | 2016 |
| Exploratory graph analysis: A new approach for estimating the number of dimensions in psychological research | 2017 |
| Retrosynthetic Reaction Prediction Using Neural Sequence-to-Sequence Models | 2017 |
| Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks | 2018 |
| Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex | 2016 |
| An Inside Look at Deep Neural Networks Using Graph Signal Processing | 2018 |
| Revisiting Small Batch Training for Deep Neural Networks | 2018 |
| Learning Entity and Relation Embeddings for Knowledge Graph Completion | 2015 |
| Designing Artificial Neural Networks Using Particle Swarm Optimization Algorithms | 2015 |
| Neural Network Matrix Factorization | 2015 |

The current OpenAlex docs support quoted exact-phrase search, e.g.
`search="graph neural networks"`, and say works search includes title, abstract,
and full text. They also describe `title.search`/`abstract.search` fielded
filters as legacy/deprecated but still functional. A more precision-focused
rebuild candidate is an English/year-filtered union of phrase searches for
`"graph neural network"`, `"graph convolutional network"`,
`"graph attention network"`, `"message passing"`, and
`"graph representation learning"`; alternatively query title and abstract
separately with the fielded filters and deduplicate their union. This is only a
proposal: **the corpus and annotation sheet were not changed**. I did not query
the OpenAlex API because this request prohibited API calls outside the approved
LLM pilot, so a fresh phrase-query `meta.count` is **not available**. OpenAlex
documents `meta.count` as the total number of matching works.

The full sampled record list and exact regexes are in
`topical_precision_audit.csv` and `topical_precision_audit.json`.
