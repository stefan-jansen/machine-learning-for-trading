# Chapter 24: Autonomous Agents

Quantitative finance increasingly relies on systems that do more than evaluate fixed prediction functions. Earlier chapters focused on ML models that estimate returns and other targets from structured data. This chapter turns to agentic workflows: systems that gather evidence, use tools, maintain state, coordinate intermediate steps, and produce artifacts that can be reviewed, replayed, scored, and audited.

## Learning Objectives

- Explain when agentic workflows add value in quantitative finance and when conventional statistical, rules-based, or batch pipelines remain the better choice.
- Distinguish the roles of ReAct, Tree of Thoughts, and Reflexion, and choose appropriate reasoning budgets for evidence-driven financial tasks.
- Design explicit state and memory schemas that support provenance, checkpointing, replay, schema evolution, and post-outcome evaluation.
- Specify robust tool contracts, structured outputs, source policies, and context-engineering rules for research, forecasting, and workflow agents.
- Compare single-agent, multi-agent, and operator-style architectures, and define a migration path from notebook prototypes to operational services without sacrificing visibility and control.
- Build an evidence-first forecasting workflow with structured output extraction, trace inspection, replayable artifacts, aggregation, calibration, and evaluation.
- Design a bounded research-iteration operator that uses coding, data, registry, and skill-retrieval tools while preserving auditability and human review.
- Define the operational, statistical, and security controls required to make agent outputs decision-grade, including point-in-time integrity, contamination-aware testing, observability, policy gates, sandboxing, and human approval boundaries.

## Sections

### 24.1 From prediction functions to agentic workflows

Large language models change the structure of quantitative work (Korinek, 2025). Traditional pipelines map engineered features to outputs through fixed transformations: a model receives a feature vector and returns a prediction. **Agentic pipelines add an adaptive layer** that can inspect evidence, decide what information is missing, call tools to retrieve it, and update a persistent state before producing an output.

- [`01_react_reasoning`](01_react_reasoning.ipynb) - Replays a captured ReAct session and exposes provider selection, bounded search, and raw message traces. Synthetic flow tests remain explicitly labeled.
- [`02_tool_contracts`](02_tool_contracts.ipynb) - Demonstrates typed tool schemas, validated arguments, provenance, and returned-domain filtering with a labeled synthetic search fixture.
- [`03_state_and_memory`](03_state_and_memory.ipynb) - Demonstrates explicit state, checkpoint round-trips, and coverage, freshness, and consistency gates on labeled teaching fixtures.
- [`04_research_agent`](04_research_agent.ipynb) - Accepts a finite probability only with retained evidence and a cited rationale. Unsupported attempts abstain, and failures retain partial traces.
- [`05_aggregation_math`](05_aggregation_math.ipynb) - Demonstrates aggregation, linear Neyman and log-odds extremization, and held-out Platt fitting on labeled mathematical examples.
- [`08_forecasting_pipeline`](08_forecasting_pipeline.ipynb) - Composes accepted-agent averaging, optional debate, high-confidence cited supervisor updates, and the fixed AIA log-odds correction. A fresh October capture retains actual model and tool calls on an unresolved question.
- [`09_evaluation_and_governance`](09_evaluation_and_governance.ipynb) - Preserves and recovers a historical panel, scores eight of ten questions, and distinguishes proper scores from diagnostics. Demonstrates held-out fitting, coverage, a bounded search policy, and selected injection detection tests.
- [`10_framework_comparison`](10_framework_comparison.ipynb) - Executes native Python and LangGraph over shared stages, including a synthetic abstention test. CrewAI is shown as configuration, with no claimed execution or duration.

### 24.2 Cognitive architectures – How agents reason

Reasoning patterns are foundational to agent design, but they are only one layer of the system. The patterns discussed here build on chain-of-thought prompting (Wei et al., 2023), which showed that externalizing intermediate reasoning steps improves multi-step accuracy without fine-tuning. Li et al. (2025) survey the full spectrum of reasoning architectures in LLMs, from fast pattern matching to deliberate multi-step search. In production, the quality of reasoning also depends on tool contracts, state visibility, and evaluation infrastructure, topics addressed in later sections. This section introduces three key frameworks as building blocks and clarifies where each one earns its complexity.

- [`01_react_reasoning`](01_react_reasoning.ipynb) - Replays a captured ReAct session and exposes provider selection, bounded search, and raw message traces. Synthetic flow tests remain explicitly labeled.

### 24.3 Agent memory – State, persistence, and replay

After selecting a reasoning pattern, the next design task is memory. In agentic finance workflows, memory determines whether the system is testable and auditable. A one-shot prompt can produce a plausible answer, but a durable workflow requires explicit memory across multiple timescales: within a single reasoning step, across a task session, and between sessions over weeks or months.

- [`03_state_and_memory`](03_state_and_memory.ipynb) - Demonstrates explicit state, checkpoint round-trips, and coverage, freshness, and consistency gates on labeled teaching fixtures.

### 24.4 Tool integration – Contracts, controls, and context engineering

With an explicit state defined, tool integration becomes the next control surface. Tools convert language-model reasoning into verifiable operations: retrieving a stock quote, querying a filing index, or computing a rolling statistic. In finance, tool design is often the dominant determinant of agent quality, outweighing prompt engineering and even model selection.

- [`02_tool_contracts`](02_tool_contracts.ipynb) - Demonstrates typed tool schemas, validated arguments, provenance, and returned-domain filtering with a labeled synthetic search fixture.

### 24.5 The engineering stack – Frameworks and migration

Given the tool and state requirements established in the preceding sections, the choice of framework becomes an implementation question. Framework discussion is often presented as a ranking problem: which library posts the highest benchmark score? For agentic finance workflows, this framing is usually unhelpful because the binding constraints are not benchmark accuracy but state visibility, replay capability, and policy enforcement. The better question is which framework best supports the controls required by the target workflow.

- [`10_framework_comparison`](10_framework_comparison.ipynb) - Executes native Python and LangGraph over shared stages, including a synthetic abstention test. CrewAI is shown as configuration, with no claimed execution or duration.

### 24.6 Designing the research agent at the heart of the pipeline

This section operationalizes the design patterns from the preceding sections in a single-agent setting. The workflow remains read-only and evidence-driven. The objective is to produce a probability forecast for an event question using web search and any supplied evidence. Each accepted forecast requires retained evidence and a cited rationale. The system persists enough artifacts for replay and inspection before those same controls are composed into a larger multi-agent pipeline. Raw research probabilities are not calibrated merely because the output schema contains a probability field.

- [`04_research_agent`](04_research_agent.ipynb) - Accepts a finite probability only with retained evidence and a cited rationale. Unsupported attempts abstain, and failures retain partial traces.

### 24.7 Multi-agent forecasting systems

*Section 24.6* established a single-agent research baseline. This section extends that baseline to a multi-agent forecasting architecture designed for probability quality, traceability, and reproducible evaluation. The capstone is a forecasting system, not an autonomous trading engine. Its design draws on the AIA Forecaster and AlphaAgents.

- [`05_aggregation_math`](05_aggregation_math.ipynb) - Demonstrates aggregation, linear Neyman and log-odds extremization, and held-out Platt fitting on labeled mathematical examples.
- [`06_multi_agent_research`](06_multi_agent_research.ipynb) - Replays parallel research agents and compares aggregation sensitivity. Agreement and disagreement do not establish independent information or accuracy.
- [`07_adversarial_debate`](07_adversarial_debate.ipynb) - Replays bull/bear debate, tracks the probability gap, and demonstrates consensus detection. A changed estimate does not establish an improvement in forecasting.
- [`08_forecasting_pipeline`](08_forecasting_pipeline.ipynb) - Composes accepted-agent averaging, optional debate, high-confidence cited supervisor updates, and the fixed AIA log-odds correction. A fresh October capture retains actual model and tool calls on an unresolved question.
- [`09_evaluation_and_governance`](09_evaluation_and_governance.ipynb) - Preserves and recovers a historical panel, scores eight of ten questions, and distinguishes proper scores from diagnostics. Demonstrates held-out fitting, coverage, a bounded search policy, and selected injection detection tests.
- [`10_framework_comparison`](10_framework_comparison.ipynb) - Executes native Python and LangGraph over shared stages, including a synthetic abstention test. CrewAI is shown as configuration, with no claimed execution or duration.

### 24.8 The ML4T research agent

The forecasting workflows in *Sections 24.6* and *24.7* stop when an agent returns a calibrated probability. They are valuable systems, and a live multi-agent forecaster runs on the book’s website, but their deliverables sit closer to event prediction than to the systematic-trading research process this book has developed across its earlier parts. *Chapter 20* ran that process to its first decision point on nine case studies.

- [`11_research_operator`](11_research_operator.ipynb) - Replays two retained operator experiments over case-study artifacts. Live Bash runs on the host unless the caller supplies external isolation.

### 24.9 Preparing for production

A forecasting-agent prototype can look strong in notebooks and still fail in production. Common causes are non-determinism, hidden data leakage, weak observability, and uncontrolled cost. After defining the capstone architecture, the next design challenge is operational robustness: the controls that make forecast quality claims decision-grade. In this chapter, that transition is not hypothetical. The notebook sequence culminates in a production demo that wraps the forecasting pipeline in configuration profiles, prediction-market connectors, persistent storage, evaluation commands, and publishing jobs.

- [`09_evaluation_and_governance`](09_evaluation_and_governance.ipynb) - Preserves and recovers a historical panel, scores eight of ten questions, and distinguishes proper scores from diagnostics. Demonstrates held-out fitting, coverage, a bounded search policy, and selected injection detection tests.

### 24.10 Security and governance

Building on the production controls in *Section 24.9*, this section formalizes security and governance requirements for financial agent systems. Security begins with architecture: the chapter defaults to read-only forecasting and research, thereby reducing the attack surface and simplifying policy enforcement. The production demo preserves that boundary. It pulls markets, searches for evidence, stores forecasts, and publishes artifacts, but it does not place trades or commit capital automatically. The controls described here apply regardless of the chosen framework and should be treated as prerequisites for any deployment that touches financial data or produces outputs consumed by decision-makers.

- [`09_evaluation_and_governance`](09_evaluation_and_governance.ipynb) - Preserves and recovers a historical panel, scores eight of ten questions, and distinguishes proper scores from diagnostics. Demonstrates held-out fitting, coverage, a bounded search policy, and selected injection detection tests.

## Running the Notebooks

Offline defaults replay retained model/tool runs or use labeled mathematical and failure fixtures. Historical records remain available with their provenance limits. Live mode requires a configured model and search provider and fails clearly if an integration is unavailable.

The recovered resolved panel demonstrates scoring and contamination rather than clean forecasting skill. Fresh October captures verify genuine execution on an unresolved question and have no accuracy score. The examples demonstrate public AIA principles without reproducing Bridgewater's proprietary prompts, data, or reported performance.

Set RUN_LIVE only when deliberately requesting paid or local model calls. Inspect the saved question, resolution rule, cutoff, messages, evidence, and status before interpreting a probability. Citation presence enables inspection but does not verify every claim in the rationale.

```bash
# From the repository root
uv run python 24_autonomous_agents/<notebook>.py

# Verify all notebooks with networking and credentials disabled
uv run python scripts/verify_ch24_notebooks.py
```

## References

- **Irene Aldridge et al.** (2025). [Agentic Artificial Intelligence in Finance: A Comprehensive Survey](https://doi.org/10.2139/ssrn.5803628).
- **Rohan Alur et al.** (2025). [AIA Forecaster: Technical Report](https://doi.org/10.48550/arXiv.2511.07678).
- **Andrew Ang et al.** (2026). [The Self Driving Portfolio: Agentic Architecture for Institutional Asset Management](https://doi.org/10.48550/arXiv.2604.02279).
- **Chanyeol Choi et al.** (2025). [FinDER: Financial Dataset for Question Answering and Evaluating Retrieval-Augmented Generation](https://doi.org/10.48550/arXiv.2504.15800).
- **Francesco A. Fabozzi and Marcos López de Prado** (2025). [Implementing AI Foundation Models in Asset Management: A Practical Guide](https://doi.org/10.3905/jpm.2025.1.778). *The Journal of Portfolio Management*.
- **Ziang Fang and Jason Moore** (2025). What AI Can (and Can't Yet) Do for Alpha.
- **Yaxuan Kong et al.** (2024). [Large Language Models for Financial and Investment Management: Models, Opportunities, and Challenges](https://doi.org/10.3905/jpm.2024.1.646). *The Journal of Portfolio Management*.
- **Anton Korinek** (2025). [AI Agents for Economic Research](https://doi.org/10.3386/w34202).
- **Hoyoung Lee et al.** (2025). [Your AI, Not Your View: The Bias of LLMs in Investment Analysis](https://doi.org/10.48550/arXiv.2507.20957).
- **Zhong-Zhi Li et al.** (2025). [From System 1 to System 2: A Survey of Reasoning Large Language Models](https://doi.org/10.48550/arXiv.2502.17419).
- **Alejandro Lopez-Lira** (2025). [Can Large Language Models Trade? Testing Financial Theories with LLM Agents in Market Simulations](https://doi.org/10.48550/arXiv.2504.10789).
- **Alejandro Lopez-Lira et al.** (2025). [The Memorization Problem: Can We Trust LLMs' Economic Forecasts?](https://doi.org/10.2139/ssrn.5217505).
- **OWASP** (2025). [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/).
- **Noah Shinn et al.** (2023). [Reflexion: Language Agents with Verbal Reinforcement Learning](https://doi.org/10.48550/arXiv.2303.11366).
- **Jason Wei et al.** (2023). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://doi.org/10.48550/arXiv.2201.11903).
- **Qianqian Xie et al.** (2024). [Finben: A holistic financial benchmark for large language models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/adb1d9fa8be4576d28703b396b82ba1b-Abstract-Datasets_and_Benchmarks_Track.html). *Advances in Neural Information Processing Systems*.
- **Shunyu Yao et al.** (2023). [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://doi.org/10.48550/arXiv.2305.10601).
- **Shunyu Yao et al.** (2023). [ReAct: Synergizing Reasoning and Acting in Language Models](https://doi.org/10.48550/arXiv.2210.03629).
- **Yangyang Yu et al.** (2024). [FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making](https://doi.org/10.48550/arXiv.2407.06567).
- **Yangyang Yu et al.** (2025). [Finmem: A performance-enhanced llm trading agent with layered memory and character design](https://ieeexplore.ieee.org/abstract/document/11112648/). *IEEE Transactions on Big Data*.
- **Tianjiao Zhao et al.** (2025). [AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions](https://doi.org/10.48550/arXiv.2508.11152).
