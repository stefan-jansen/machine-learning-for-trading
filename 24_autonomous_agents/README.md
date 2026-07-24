# Chapter 24: Autonomous Agents

The chapter explains the chapter's core shift: from fixed prediction functions over prepared datasets to workflows that actively gather, filter, and synthesize messy evidence before producing a forecast. It makes the case that agentic systems are most useful in finance when the bottleneck is evidence acquisition and structured judgment, not when a traditional model already operates on stable, fully structured inputs. The section also defines the chapter's read-only boundary, which keeps the focus on decision support, traceability, and probability forecasts rather than execution.

## Learning Objectives

- Explain when agentic workflows add value in finance and when conventional statistical or rules-based pipelines remain the better choice  
- Distinguish the roles of ReAct, Tree of Thoughts, and Reflexion, and choose appropriate reasoning budgets and compositions for evidence-driven financial tasks  
- Design explicit agent state and memory schemas that support provenance, checkpointing, replay, schema evolution, and post-outcome evaluation  
- Specify robust tool contracts, structured outputs, source policies, and context-engineering rules for read-only research and forecasting agents  
- Compare framework styles and define a migration path from notebook prototypes to operational forecasting services without sacrificing visibility and control  
- Build a single-agent evidence-first research workflow with quality gates, abstention behavior, and replayable artifacts  
- Design and evaluate multi-agent forecasting pipelines using specialist diversity, aggregation, calibration, baselines, and ablation analysis  
- Define the operational, statistical, and security controls required to make financial-agent outputs decision-grade, including point-in-time integrity, contamination-aware testing, observability, policy gates, and human approval boundaries

## Sections

### 24.1 From Prediction Functions to Agentic Workflows

This section explains the chapter's core shift: from fixed prediction functions over prepared datasets to workflows that actively gather, filter, and synthesize messy evidence before producing a forecast. It makes the case that agentic systems are most useful in finance when the bottleneck is evidence acquisition and structured judgment, not when a traditional model already operates on stable, fully structured inputs. The section also defines the chapter's read-only boundary, which keeps the focus on decision support, traceability, and probability forecasts rather than execution.

- [`01_react_reasoning`](01_react_reasoning.ipynb) — This notebook introduces the LLM provider abstraction and the ReAct (Reason + Act) reasoning pattern — the foundational building block for autonomous financial agents.

### 24.2 Cognitive Architectures: How Agents Reason

This section introduces the reasoning patterns that underpin agent workflows and treats them as engineering choices rather than as abstract prompting tricks. ReAct is positioned as the default because it is grounded and auditable, Tree of Thoughts is reserved for branch-heavy decisions, and Reflexion is framed as useful only when explicit memory governance exists. The practical value is not just knowing these patterns, but understanding when each one actually earns its added cost and complexity.

- [`01_react_reasoning`](01_react_reasoning.ipynb) — This notebook introduces the LLM provider abstraction and the ReAct (Reason + Act) reasoning pattern — the foundational building block for autonomous financial agents.

### 24.3 Agent Memory: State, Persistence, and Replay

This section argues that memory design is a model-risk and reproducibility problem, not just a software concern. It distinguishes working, short-term, and long-term memory, then shows why typed state, checkpointing, schema versions, and replay are mandatory if an agent is to be debugged, scored, and audited. For readers building serious systems, this is one of the chapter's most important sections because it turns "conversation history" into explicit, testable artifacts.

- [`03_state_and_memory`](03_state_and_memory.ipynb) — This notebook introduces explicit agent state, quality gates, and checkpoint/replay — the mechanisms that make agents reliable and reproducible. Without these, agents are black boxes that can't be debugged, audited, or compared across runs.

### 24.4 Tool Integration: Contracts, Controls, and Context Engineering

Here the chapter shows that tool quality often matters more than prompt quality. It explains how strong tool contracts, provenance-rich outputs, structured schemas, source policy, and carefully scoped context exposure determine whether an agent produces reliable, inspectable results or fragile narrative guesses. This section is especially valuable because it translates abstract tool use into concrete engineering controls that directly affect forecast quality.

- [`02_tool_contracts`](02_tool_contracts.ipynb) — This notebook teaches typed tool schemas, multi-provider schema translation, provenance enrichment, and domain policy enforcement. Tools are the bridge between agent reasoning and external data — their contracts determine what the agent can do and how trustworthy its outputs are.

### 24.5 The Engineering Stack: Frameworks and Migration

This section reframes framework selection away from hype and toward operational fit. Instead of asking which framework is "best," it asks which one preserves state visibility, replayability, policy enforcement, and debugging quality for the workflow at hand. The section also gives readers a sensible migration path from notebook experiments to a packaged forecasting service, which makes it more practical than most framework-comparison discussions.

- [`10_framework_comparison`](10_framework_comparison.ipynb) — _Runtime: ~14 min._ This optional notebook compares three ways to express the same forecasting pipeline: (a) native Python SDK, (b) CrewAI role-based, and (c) LangGraph state-graph. All three use the same LLMClient from _providers.py for the LLM layer and produce equivalent outputs.

### 24.6 Core Project: The Research Agent

The first capstone applies the chapter's ideas in a constrained, single-agent research workflow. By keeping the task narrow and the quality gates explicit, the section shows how evidence collection, validation, abstention, synthesis, and replay work before the reader has to reason about multi-agent complexity. This makes it the chapter's clearest bridge from design principles to an inspectable implementation.

- [`04_research_agent`](04_research_agent.ipynb) — This notebook is the core project of Chapter 24's first half. It combines the provider abstraction (NB01), search tools (NB02), and state/quality gates (NB03) into a complete `ResearchAgent` that produces calibrated probability forecasts with rich metadata — the building block for the multi-agent pipeline.

### 24.7 Multi-Agent Forecasting Systems

This section expands the single-agent baseline into a probability-forecasting architecture built around specialist diversity, aggregation, optional debate, supervisor review, calibration, and scored outcomes. Its central contribution is to show that a multi-agent system should be judged by the quality of its probabilities and its incremental value over baselines, not by how sophisticated its narratives sound. This is where the chapter most clearly connects agent design to measurable forecast performance.

- [`05_aggregation_math`](05_aggregation_math.ipynb) — When multiple agents produce probability estimates, how do you combine them? Teaches Neyman extremization, weighted aggregation, log-odds calibration, and `find_optimal_d` for tuning on resolved forecasts.
- [`06_multi_agent_research`](06_multi_agent_research.ipynb) — Runs N identical research agents in parallel via `ThreadPoolExecutor`. The AIA paper's key finding: identical agents with temperature diversity naturally produce diverse outputs without role specialization.
- [`07_adversarial_debate`](07_adversarial_debate.ipynb) — Implements structured adversarial debate with real LLM calls — the mechanism that stress-tests agent consensus by forcing explicit bull and bear arguments. The key output is how much probability estimates converge under adversarial pressure.
- [`08_forecasting_pipeline`](08_forecasting_pipeline.ipynb) — _Runtime: ~48 min._ Wires together all components into the complete agent → aggregation → debate → supervisor pipeline (the AIA Forecaster). Builds `SupervisorAgent` and `AIAForecaster` inline. Runs on multiple resolved questions with full trace output.
- [`09_evaluation_and_governance`](09_evaluation_and_governance.ipynb) — _Runtime: ~13 min._ Capstone: scores forecasts with proper scoring rules (Brier, log, ECE), builds calibration curves, runs ablation experiments, tunes calibration via `find_optimal_d`, and implements security controls (Warden pattern, injection defense).

### 24.8 The ML4T Research Agent

This section turns the chapter's second capstone toward the book's own research process. Where §24.6–24.7 stop when an agent returns a calibrated probability, this section builds an **operator**: a thin orchestrator that hands a language model a handful of general-purpose tools — read/write/edit files, run bash, query a SQLite registry, inspect Parquet, and consult a corpus of how-to skills — and lets it execute the "next experiment" each Chapter 20 case study left behind. The task-specific discipline lives in the companion [`ml4t/skills`](https://github.com/ml4t/skills) repository rather than in the prompt, which is what keeps the operator's instructions short while the methodology it can reach grows.

- [`11_research_operator`](11_research_operator.ipynb) — Replays two complete operator runs against captured DeepSeek traces: an ETF ensemble experiment and a US-firm small-cap universe filter. Demonstrates the operator pattern (≈880-line orchestrator, 10 general-purpose tools), on-demand skill consultation, and how the agent recovers a backtest specification from the registry to re-run it under an identical cost-aware configuration.

### 24.9 Preparing for Production

This section explains why promising notebook behavior does not automatically translate into decision-grade systems. It covers observability, repeated-trial testing, point-in-time enforcement, contamination-aware evaluation, cost and latency control, release discipline, and monitoring. The result is a grounded account of what has to be true before any forecast-quality claim becomes credible in production.

- [`08_forecasting_pipeline`](08_forecasting_pipeline.ipynb) — The full agent → aggregation → debate → supervisor pipeline applied to production persistence and replay considerations.
- [`09_evaluation_and_governance`](09_evaluation_and_governance.ipynb) — Evaluation using proper scoring rules, calibration curves, ablation experiments, and replay with frozen tool responses.

### 24.10 Security and Governance

This section formalizes the control layer that protects the entire workflow from adversarial or policy-breaking behavior. It covers prompt injection, retrieval poisoning, least privilege, policy proxies, human approval boundaries, and measurable security testing. Its real importance is that it treats security as a workflow property intertwined with evidence, tools, state, and publication rather than as a narrow infrastructure afterthought.

- [`09_evaluation_and_governance`](09_evaluation_and_governance.ipynb) — Warden proxy pattern for tool-call authorization, prompt injection detection and defense, OWASP Top 10 mapping for LLM applications.

## Running the Notebooks

All notebooks run out of the box in **mock mode** (deterministic, no API calls). For real
LLM-powered forecasting, install one provider SDK and set the corresponding API keys.

### LLM Provider Setup (choose one)

`create_llm_client` auto-selects the first configured provider in the order Anthropic → OpenAI → Google → OpenRouter → Ollama. A single OpenRouter key can serve any model.

| Provider | Install | API Key | Get Key |
|----------|---------|---------|---------|
| **Claude** (recommended) | `uv pip install anthropic httpx` | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| **OpenAI** | `uv pip install openai httpx` | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/) |
| **Google** | `uv pip install httpx` | `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/) |
| **OpenRouter** (any model, one key) | `uv pip install httpx` | `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai/) |
| **Ollama** (free, local) | `uv pip install httpx` | — | [ollama.com](https://ollama.com/) |

For web search (used by the research agent), also set:

| Service | API Key | Get Key |
|---------|---------|---------|
| **Tavily** | `TAVILY_API_KEY` | [tavily.com](https://tavily.com/) |

Add keys to `.env` (see `.env.example`). Without API keys, notebooks auto-detect and fall back to mock mode.

### Execution

```bash
# From the repository root
uv run python 24_autonomous_agents/<notebook>.py

# Test mode (mock providers via Papermill)
uv run pytest tests/test_notebooks.py -v -k "24_autonomous_agents"
```

## References

- **Irene Aldridge et al.** (2025). [Agentic Artificial Intelligence in Finance: A Comprehensive Survey](https://doi.org/10.2139/ssrn.5803628).
- **Rohan Alur et al.** (2025). [AIA Forecaster: Technical Report](https://doi.org/10.48550/arXiv.2511.07678).
- **Chanyeol Choi et al.** (2025). [FinDER: Financial Dataset for Question Answering and Evaluating Retrieval-Augmented Generation](https://doi.org/10.48550/arXiv.2504.15800).
- **Francesco A. Fabozzi and Marcos López de Prado** (2025). [Implementing AI Foundation Models in Asset Management: A Practical Guide](https://doi.org/10.3905/jpm.2025.1.778). *The Journal of Portfolio Management*.
- **Ziang Fang and Jason Moore** (2025). What AI Can (and Can't Yet) Do for Alpha.
- **Yaxuan Kong et al.** (2024). [Large Language Models for Financial and Investment Management: Models, Opportunities, and Challenges](https://doi.org/10.3905/jpm.2024.1.646). *The Journal of Portfolio Management*.
- **Anton Korinek** (2025). [AI Agents for Economic Research](https://doi.org/10.3386/w34202).
- **Hoyoung Lee et al.** (2025). [Your AI, Not Your View: The Bias of LLMs in Investment Analysis](https://doi.org/10.48550/arXiv.2507.20957).
- **Zhong-Zhi Li et al.** (2025). [From System 1 to System 2: A Survey of Reasoning Large Language Models](https://doi.org/10.48550/arXiv.2502.17419).
- **Alejandro Lopez-Lira** (2025). [Can Large Language Models Trade? Testing Financial Theories with LLM Agents in Market Simulations](https://doi.org/10.48550/arXiv.2504.10789).
- **Alejandro Lopez-Lira et al.** (2025). [The Memorization Problem: Can We Trust LLMs' Economic Forecasts?](https://doi.org/10.2139/ssrn.5217505).
- **Noah Shinn et al.** (2023). [Reflexion: Language Agents with Verbal Reinforcement Learning](https://doi.org/10.48550/arXiv.2303.11366).
- **Jason Wei et al.** (2023). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://doi.org/10.48550/arXiv.2201.11903).
- **Qianqian Xie et al.** (2024). [Finben: A holistic financial benchmark for large language models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/adb1d9fa8be4576d28703b396b82ba1b-Abstract-Datasets_and_Benchmarks_Track.html). *Advances in Neural Information Processing Systems*.
- **Shunyu Yao et al.** (2023). [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://doi.org/10.48550/arXiv.2305.10601).
- **Shunyu Yao et al.** (2023). [ReAct: Synergizing Reasoning and Acting in Language Models](https://doi.org/10.48550/arXiv.2210.03629).
- **Yangyang Yu et al.** (2024). [FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making](https://doi.org/10.48550/arXiv.2407.06567).
- **Yangyang Yu et al.** (2025). [Finmem: A performance-enhanced llm trading agent with layered memory and character design](https://ieeexplore.ieee.org/abstract/document/11112648/). *IEEE Transactions on Big Data*.
- **Tianjiao Zhao et al.** (2025). [AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions](https://doi.org/10.48550/arXiv.2508.11152).
