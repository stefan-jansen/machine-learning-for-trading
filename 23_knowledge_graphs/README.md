# Chapter 23: Knowledge Graphs for Financial AI

Vector retrieval made evidence-grounded financial question answering practical, but similarity search still operates on
isolated chunks. Questions that depend on ownership chains, supplier networks, contagion paths, or the timing of
relationship changes require explicit structure rather than fuzzy matches across text. Knowledge graphs represent
entities as nodes, relationships as typed edges, and provenance as properties — giving analytical queries something to
traverse, audit, and reuse.

This chapter covers the construction, query, feature-engineering, and evaluation discipline that make financial
knowledge graphs useful rather than novel. It treats graph building as a governance problem (identity, schema,
provenance), introduces Graph RAG as deterministic relational retrieval, turns graph structure into leakage-aware ML
features, connects the statistical-financial-networks literature to the explicit-KG workflow, and closes with a
three-timestamp model for temporal integrity and the engineering choices that keep systems auditable.

## Learning Objectives

- Distinguish financial questions that genuinely require graph structure from those better served by tabular databases
  or vector retrieval
- Design a compact, typed, and auditable financial knowledge graph with stable entity identity, finite relationship
  vocabularies, and edge-level provenance
- Build and validate LLM-assisted extraction pipelines that convert disclosures into replayable graph objects while
  controlling schema, duplication, and temporal consistency
- Explain how Graph RAG differs from vector retrieval and implement safe relational query workflows using constrained
  text-to-query generation and deterministic database execution
- Transform graph structure into leakage-aware machine learning features — topology, crowding, concentration, and
  cross-graph interaction terms
- Evaluate explicit knowledge graphs, statistical financial networks, and learned graph representations pragmatically
  for forecasting, portfolio construction, and risk analysis
- Apply a three-timestamp framework and disclosure-time cutoff rules to prevent temporal leakage in graph queries,
  feature generation, and backtests
- Make sound engineering choices about graph databases, ontology scope, query safety, and schema evolution for
  production-oriented financial systems

## Chapter Sections

| #    | Title                                            | Core Idea                                                                                                  |
|------|--------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| 23.1 | When Relational Structure Unlocks Financial Insight | Graphs earn their overhead when the question is multi-hop, structurally crowded, or temporally evolving   |
| 23.2 | Constructing Financial Knowledge Graphs          | LLM extraction plus identity, schema, and provenance contracts that make the graph replayable and auditable |
| 23.3 | Graph RAG: Deterministic Relational Reasoning    | A five-stage architecture that delegates relational logic to the database and language generation to the LLM |
| 23.4 | From Graphs to Machine Learning Features         | Centrality, crowding, co-ownership, and cross-graph interactions become tabular features for downstream models |
| 23.5 | Financial Networks: From Correlations to Portfolios | How classic MST and correlation-network methods complement explicit KGs, and where GNNs fit in practice |
| 23.6 | Temporal Integrity and Leakage-Safe Evaluation   | The three-timestamp model (event, disclosure, extraction) and a protocol for cutoff-safe features         |
| 23.7 | Building a KG-Ready Pipeline: Engineering Decisions | Engine selection, ontology scope, text-to-Cypher safety, and schema versioning for production             |

## Notebooks

### Graph Construction

| Notebook                                                                   | What It Teaches                                                                                                    |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [`01_sp100_sec_download`](01_sp100_sec_download.ipynb)                           | Loads and inspects 10-K and 8-K filings for S&P 100 companies from SEC EDGAR, verifies coverage and text quality before graph extraction. |
| [`02_supply_chain_kg_construction`](02_supply_chain_kg_construction.ipynb)       | Extracts supplier, customer, and competitor relationships from 10-K filings using Qwen2.5-7B, resolves entities, loads triples to Neo4j, and visualizes the resulting supply-chain graph. |
| [`05_institutional_holdings_kg`](05_institutional_holdings_kg.ipynb)             | Builds an institutional-holdings property graph from EDGAR 13F filings, demonstrates shared-holding and crowding queries, and computes Jaccard co-ownership similarity. |
| [`08_8k_event_extraction`](08_8k_event_extraction.ipynb)                         | Extracts structured event quadruples from SEC 8-K filings with the FinReflectKG critic-corrector reflection loop, then loads the event graph to Neo4j. |

### Retrieval and Evaluation

| Notebook                                                                   | What It Teaches                                                                                                    |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [`03_graph_rag_qa`](03_graph_rag_qa.ipynb)                                       | Implements controlled text-to-Cypher question answering over the 13F holdings graph, enforcing read-only schema validation, dated queries, and row limits on every generated query. |
| [`04_rag_comparison_benchmark`](04_rag_comparison_benchmark.ipynb)               | Benchmarks graph retrieval against embedding-based retrieval on real 13F holdings, measuring support recall and retrieval-token cost across direct-lookup and multi-entity question types. |

### Graph-Derived Features and Networks

| Notebook                                                                   | What It Teaches                                                                                                    |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [`09_knowledge_graph_features`](09_knowledge_graph_features.ipynb)               | Combines supply-chain topology (PageRank, betweenness, HHI) with 13F crowding and cross-graph interactions to produce a tabular feature matrix suitable for gradient-boosted models. |
| [`10_network_portfolio_construction`](10_network_portfolio_construction.ipynb)   | Builds correlation networks and minimum spanning trees from US equities, ranks assets by centrality, constructs network-diversified portfolios, and runs a contagion simulation. |
| [`06_gnn_feature_engineering`](06_gnn_feature_engineering.ipynb)                 | Trains a simplified GAT on a stock correlation network to generate relational embeddings, then compares a tabular-only ridge regression to a hybrid model that concatenates the learned embeddings. |

### Temporal Integrity

| Notebook                                                                   | What It Teaches                                                                                                    |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [`07_dynamic_kg_temporal`](07_dynamic_kg_temporal.ipynb)                         | Builds leakage-safe temporal snapshots from the 8-K event graph by separating event, disclosure, and extraction time, then measures relationship churn and centrality drift while verifying no post-cutoff leakage. |

## Running the Notebooks

```bash
# From the repository root
uv run python 23_knowledge_graphs/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "23_knowledge_graphs"

# Headless (no display)
MPLBACKEND=Agg PLOTLY_RENDERER=json uv run python 23_knowledge_graphs/<notebook>.py
```

A running Neo4j instance is required for `02_supply_chain_kg_construction`, `05_institutional_holdings_kg`,
`08_8k_event_extraction`, `03_graph_rag_qa`, and `07_dynamic_kg_temporal`. Local LLM extraction for the supply-chain and 8-K
pipelines uses Qwen2.5-7B via a vLLM-compatible endpoint.

## Dependencies

**Upstream**

- Chapter 4 (Fundamental and Alternative Data) — 13F institutional-holdings loader
- Chapter 10 (Text Feature Engineering) — NER and filing parsing patterns
- Chapter 22 (RAG for Financial Research) — retrieval-augmented generation foundation

**Downstream**

- Chapter 12 (Advanced Models for Tabular Data) — consumes cross-graph features
- Chapter 14 (Latent Factor Models) — consumes co-ownership and crowding features
- Chapter 24 (Autonomous Agents) — uses structured retrieval, memory, and evidence tracking

**Chapter-specific libraries**

- `neo4j` — graph database driver (Cypher client)
- `networkx` — graph analytics and MST construction
- `torch_geometric` — Graph Attention Network implementation
- `edgartools` — SEC EDGAR filings ingestion
- `sentence-transformers` — embedding baseline for the RAG comparison

## References

- **Abhinav Arun et al.** (2025). [FinReflectKG -- MultiHop: Financial QA Benchmark for Reasoning with Knowledge Graph Evidence](https://doi.org/10.48550/arXiv.2510.02906).
- **Chanyeol Choi et al.** (2025). [FinDER: Financial Dataset for Question Answering and Evaluating Retrieval-Augmented Generation](https://doi.org/10.48550/arXiv.2504.15800).
- **Darren Edge et al.** (2025). [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://doi.org/10.48550/arXiv.2404.16130).
- **Sarah Elhammadi et al.** (2020). [A High Precision Pipeline for Financial Knowledge Graph Construction](https://doi.org/10.18653/v1/2020.coling-main.84). *International Committee on Computational Linguistics*.
- **Robin Greenwood and David Thesmar** (2011). [Stock price fragility](https://doi.org/10.1016/j.jfineco.2011.06.003). *Journal of Financial Economics*.
- **Andrew G. Haldane and Robert M. May** (2011). [Systemic risk in banking ecosystems](https://doi.org/10.1038/nature09659). *Nature*.
- **William L. Hamilton et al.** (2018). [Inductive Representation Learning on Large Graphs](https://doi.org/10.48550/arXiv.1706.02216).
- **Natthawut Kertkeidkachorn et al.** (2023). [FinKG: A Core Financial Knowledge Graph for Financial Analysis](https://doi.org/10.1109/ICSC56153.2023.00020).
- **Thomas N. Kipf and Max Welling** (2017). [Semi-Supervised Classification with Graph Convolutional Networks](https://doi.org/10.48550/arXiv.1609.02907).
- **Gueorgui S. Konstantinov et al.** (2023). [Financial Networks and Portfolio Management](https://doi.org/10.3905/jpm.2023.1.525). *The Journal of Portfolio Management*.
- **Gueorgui S. Konstantinov and Frank J. Fabozzi** (2025). [When Factors Collide: Mapping Causal Spillovers across Global Asset Networks](https://doi.org/10.3905/jpm.2025.1.795). *The Journal of Portfolio Management*.
- **Patrick Lewis et al.** (2021). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://doi.org/10.48550/arXiv.2005.11401).
- **R. N. Mantegna** (1999). [Information and hierarchical structure in financial markets](https://doi.org/10.1016/S0010-4655(99)00302-1). *Computer Physics Communications*.
- **Gautier Marti et al.** (2021). [A Review of Two Decades of Correlations, Hierarchies, Networks and Clustering in Financial Markets](https://doi.org/10.1007/978-3-030-65459-7_10). *Springer International Publishing*.
- **Rui Miao et al.** (2019). [A Dynamic Financial Knowledge Graph Based on Reinforcement Learning and Transfer Learning](https://doi.org/10.1109/BigData47090.2019.9005691).
- **Boci Peng et al.** (2024). [Graph Retrieval-Augmented Generation: A Survey](https://doi.org/10.48550/arXiv.2408.08921).
- **Petar Veličković et al.** (2018). [Graph Attention Networks](https://doi.org/10.48550/arXiv.1710.10903).
- **Mark Weber et al.** (2019). [Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics](https://doi.org/10.48550/arXiv.1908.02591).
- **Samreen Zehra et al.** (2021). [Financial Knowledge Graph Based Financial Report Query System](https://doi.org/10.1109/ACCESS.2021.3077916). *IEEE Access*.
