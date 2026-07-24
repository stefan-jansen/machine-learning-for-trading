# Chapter 22: RAG for Financial Research

The chapter explains why text classification is not enough once the practitioner's task becomes open-ended analysis rather than fixed-label prediction. It positions LLMs as a shift from extracting features to answering analyst-style questions, but immediately frames hallucination as the central obstacle in finance. Readers should care because it sets up the chapter's core claim: generative AI only becomes usable in high-stakes financial settings when it is grounded in verifiable evidence rather than trusted as an oracle.

## Learning Objectives

- Explain why hallucination makes ungrounded LLM use unacceptable in finance and why retrieval-augmented generation is the core architectural response
- Design a financial RAG pipeline from document ingestion through retrieval and grounded generation, including structure-aware parsing, chunking, metadata, embeddings, and citation support
- Compare generic and domain-specific embedding models and evaluate retrieval quality on a target corpus using practical retrieval metrics and latency trade-offs
- Build a retrieval stack that combines semantic search, lexical search, metadata filtering, and re-ranking to improve precision and recall on financial documents
- Use constraint-based prompting, citation checks, and tool-verified computation to make generated answers more faithful, auditable, and numerically reliable
- Diagnose RAG failures by separating retrieval, context, synthesis, computation, and abstention errors, and apply targeted evaluation methods to improve each component
- Distinguish when to use RAG versus fine-tuning for financial applications, and explain how RAG functions as one tool within broader agentic workflows

## Sections

### 22.1 Introduction: The Generative Leap Beyond Feature Extraction

This section explains why text classification is not enough once the practitioner's task becomes open-ended analysis rather than fixed-label prediction. It positions LLMs as a shift from extracting features to answering analyst-style questions, but immediately frames hallucination as the central obstacle in finance. Readers should care because it sets up the chapter's core claim: generative AI only becomes usable in high-stakes financial settings when it is grounded in verifiable evidence rather than trusted as an oracle.

### 22.2 The Solution: Grounding LLMs with Retrieval-Augmented Generation

This section introduces RAG as the architectural answer to hallucination and lays out the index, retrieve, generate pipeline in clear engineering terms. It also distinguishes the appealing simplicity of the baseline design from the much harder production reality in financial documents, where naive pipelines fail quickly. Readers should care because this is the conceptual backbone of the chapter: the model is valuable not because it "knows," but because it can synthesize over retrieved evidence.

### 22.3 Intelligent Document Ingestion

This section argues that RAG quality starts before retrieval, with the way filings and related documents are parsed, chunked, and annotated. It shows why fixed-size chunking breaks tables, headers, temporal context, and citation traceability, then motivates structure-aware parsing, multimodal handling, and rich metadata as non-negotiable for financial corpora. Readers should care because poor ingestion silently corrupts everything downstream: if the semantic units are wrong, neither embeddings nor prompting can recover what was lost.

- [`01_sec_filing_pipeline`](01_sec_filing_pipeline.ipynb) — - Compare two practical approaches to SEC filing ingestion for downstream RAG systems. - Add the metadata needed for point-in-time filtering and citation traceability.

### 22.4 Domain-Specific Embeddings

This section explains why generic embeddings are often inadequate for financial retrieval and why domain-adapted models matter for jargon, entities, regulation, and quantitative concepts. It also adds practical engineering considerations such as benchmark use, corpus-specific evaluation, dimensionality, and storage trade-offs. Readers should care because retrieval quality is not a cosmetic optimization; it determines whether the system can even surface the evidence needed for a trustworthy answer.

- [`02_domain_embeddings_comparison`](02_domain_embeddings_comparison.ipynb) — This notebook compares embedding models for financial document retrieval: Uses synthetic data.

### 22.5 Hybrid Retrieval and Vector Databases

This section shows that semantic search alone is not enough for finance, where exact terms such as tickers, filing types, and codes matter. By combining vector search with lexical ranking and metadata filtering, the chapter presents hybrid retrieval as the practical default rather than an advanced extra. Readers should care because this is where the system becomes robust to the actual query mix analysts use, instead of only performing well on clean semantic paraphrases.

- [`03_hybrid_retrieval`](03_hybrid_retrieval.ipynb) — This notebook demonstrates hybrid retrieval combining: Uses cross_encoder data.

### 22.6 Re-ranking and Constraint-Based Prompting

This section moves from finding candidate evidence to making grounded answers more precise and defensible. It combines cross-encoder re-ranking, context-window discipline, citation-constrained prompting, and tool-verified numeric computation to turn the LLM into a synthesis engine rather than a free-form generator. Readers should care because this is where the chapter makes financial RAG operationally credible: answers must not only sound right, but be supported, calculationally reliable, and audit-friendly.

- [`05_10k_rag_assistant`](05_10k_rag_assistant.ipynb) — This notebook demonstrates a production-grade RAG (Retrieval-Augmented Generation) system for analyzing SEC 10-K filings. Key features: Uses data, documents, parquet_documents data.

### 22.7 Diagnosing RAG Pipeline Bottlenecks

This section treats RAG as a system that must be measured and debugged, not admired through demos. By separating retrieval, context, synthesis, computation, and abstention failures, it gives readers a practical framework for evaluation and iteration, including RAGAs, claim-level checks, and production observability. Readers should care because without this diagnostic lens, teams cannot tell whether a bad answer comes from retrieval, ranking, prompting, or arithmetic, and therefore cannot improve the system systematically.

- [`04_ragas_evaluation`](04_ragas_evaluation.ipynb) — This notebook implements a finance-oriented evaluation harness that...
- [`08_rag_security`](08_rag_security.ipynb) — This notebook demonstrates attack and defense evaluation for document-grounded finance assistants, a critical concern when RAG systems operate on untrusted or adversarial document corpora.

### 22.8 From Theory to Practice: Applications and Strategic Choices

This section anchors the architecture in concrete financial use cases, especially a 10-K due diligence assistant and an ESG analysis comparison. It also gives the clearest strategic boundary in the chapter: fine-tuning is for repeatable label-producing skills, while RAG is for evidence-grounded reasoning over changing documents. Readers should care because this section translates technical design choices into organizational decisions about what kind of AI workflow they are actually building.

- [`05_10k_rag_assistant`](05_10k_rag_assistant.ipynb) — This notebook demonstrates a production-grade RAG (Retrieval-Augmented Generation) system for analyzing SEC 10-K filings. Key features: Uses data, documents, parquet_documents data.
- [`06_esg_rag_vs_finetune`](06_esg_rag_vs_finetune.ipynb) — This notebook compares two approaches to ESG (Environmental, Social, Governance) analysis: Uses finbert_pipeline data.
- [`07_institutional_holdings_graph`](07_institutional_holdings_graph.ipynb) — Build a bipartite institution-stock graph from the 13F holdings artifact and derive co-ownership similarity, institutional momentum, and crowding signals for alpha research.

### 22.9 The Next Frontier: Introduction to Agentic Frameworks

This section positions RAG not as the endpoint, but as one tool inside broader multi-step agent workflows. It introduces the controller, tool, and memory pattern, then shows how grounded document retrieval fits into a larger architecture that may also use code, APIs, and databases. Readers should care because it opens the path from cited question-answering to goal-directed analytical workflows while keeping grounding as a core control mechanism.

## Running the Notebooks

```bash
# From the repository root
uv run python 22_rag_financial_research/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "22_rag_financial_research"
```

## References

- **Guido Baltussen et al.** (2025). [Natural Language Processing for Asset Managers: Turning Text into Alpha](https://doi.org/10.3905/jpm.2025.1.784). *The Journal of Portfolio Management*.
- **Hugo Bowne-Anderson** (2025). [Stop Building AI Agents: Use Smarter LLM Workflows](https://decodingml.substack.com/p/stop-building-ai-agents).
- **Chanyeol Choi et al.** (2025). [FinDER: Financial Dataset for Question Answering and Evaluating Retrieval-Augmented Generation](https://doi.org/10.48550/arXiv.2504.15800).
- **Shahul Es et al.** (2025). [Ragas: Automated Evaluation of Retrieval Augmented Generation](https://doi.org/10.48550/arXiv.2309.15217).
- **Ziang Fang and Jason Moore** (2025). What AI Can (and Can't Yet) Do for Alpha.
- **Manuel Faysse et al.** (2025). [ColPali: Efficient Document Retrieval with Vision Language Models](https://doi.org/10.48550/arXiv.2407.01449).
- **Luyu Gao et al.** (2022). [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://doi.org/10.48550/arXiv.2212.10496).
- **Allen Huang et al.** (2020). [FinBERT—A Deep Learning Approach to Extracting Textual Information](https://doi.org/10.2139/ssrn.3910214). *SSRN Electronic Journal*.
- **Yaxuan Kong et al.** (2024). [Large Language Models for Financial and Investment Management: Models, Opportunities, and Challenges](https://doi.org/10.3905/jpm.2024.1.646). *The Journal of Portfolio Management*.
- **Aditya Kusupati et al.** (2024). [Matryoshka Representation Learning](https://doi.org/10.48550/arXiv.2205.13147).
- **Hoyoung Lee et al.** (2025). [Your AI, Not Your View: The Bias of LLMs in Investment Analysis](https://doi.org/10.48550/arXiv.2507.20957).
- **Patrick Lewis et al.** (2021). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://doi.org/10.48550/arXiv.2005.11401).
- **Nelson F. Liu et al.** (2023). [Lost in the Middle: How Language Models Use Long Contexts](https://doi.org/10.48550/arXiv.2307.03172).
- **Alejandro Lopez-Lira** (2023). [Risk Factors That Matter: Textual Analysis of Risk Disclosures for the Cross-Section of Returns](https://doi.org/10.2139/ssrn.3313663).
- **Alejandro Lopez-Lira and Yuehua Tang** (2025). [Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models](https://doi.org/10.48550/arXiv.2304.07619).
- **Alejandro Lopez-Lira et al.** (2025). [The Memorization Problem: Can We Trust LLMs' Economic Forecasts?](https://doi.org/10.2139/ssrn.5217505).
- **Tim Loughran and Bill Mcdonald** (2011). [When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks](https://doi.org/10.1111/j.1540-6261.2010.01625.x). *The Journal of Finance*.
- **Rodrigo Nogueira and Kyunghyun Cho** (2020). [Passage Re-ranking with BERT](https://doi.org/10.48550/arXiv.1901.04085).
- **Nils Reimers and Iryna Gurevych** (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](http://arxiv.org/abs/1908.10084). *arXiv:1908.10084 [cs]*.
- **Dongyu Ru et al.** (2024). [RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation](https://doi.org/10.48550/arXiv.2408.08067).
- **Preetha Saha et al.** (2025). [Large Language Model Agents for Investment Management: Foundations, Benchmarks, and Research Frontiers](https://doi.org/10.2139/ssrn.5447274).
- **Yixuan Tang and Yi Yang** (2025). [FinMTEB: Finance Massive Text Embedding Benchmark](https://doi.org/10.48550/arXiv.2502.10990).
- **Ashish Vaswani et al.** (2017). [Attention Is All You Need](http://arxiv.org/abs/1706.03762). *arXiv:1706.03762 [cs]*.
- **Jason Wei et al.** (2023). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://doi.org/10.48550/arXiv.2201.11903).
- **Orion Weller et al.** (2025). [On the Theoretical Limitations of Embedding-Based Retrieval](https://doi.org/10.48550/arXiv.2508.21038).
- **Qianqian Xie et al.** (2023). [Pixiu: A large language model, instruction data and evaluation benchmark for finance](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6a386d703b50f1cf1f61ab02a15967bb-Abstract-Datasets_and_Benchmarks.html). *arXiv preprint arXiv:2306.05443*.
- **Qianqian Xie et al.** (2024). [Finben: A holistic financial benchmark for large language models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/adb1d9fa8be4576d28703b396b82ba1b-Abstract-Datasets_and_Benchmarks_Track.html). *Advances in Neural Information Processing Systems*.
- **Shunyu Yao et al.** (2023). [ReAct: Synergizing Reasoning and Acting in Language Models](https://doi.org/10.48550/arXiv.2210.03629).
- **Yangyang Yu et al.** (2024). [FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making](https://doi.org/10.48550/arXiv.2407.06567).
- **Lingyun Zhao et al.** (2020). [A BERT based Sentiment Analysis and Key Entity Detection Approach for Online Financial Texts](http://arxiv.org/abs/2001.05326). *arXiv:2001.05326 [cs]*.
