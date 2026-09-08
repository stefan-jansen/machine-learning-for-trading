# Chapter 21: Reinforcement Learning for Execution and Hedging

The chapter explains why reinforcement learning matters in finance only in certain settings. Its core contribution is to separate prediction from control: supervised learning forecasts what may happen, while RL decides what to do next when actions affect future outcomes. The section is strongest when it argues that RL is structurally better matched to execution, market making, and hedging than to broad alpha discovery, because these tasks have clearer objectives, tighter feedback loops, and more defensible reward signals.

## Learning Objectives

- Formulate execution, market making, and derivatives hedging problems as partially observed Markov Decision Processes with economically coherent state, action, reward, and constraint design
- Match value-based and actor-critic RL methods to financial tasks based on action-space structure, sample-efficiency needs, and stability requirements
- Benchmark RL execution policies against TWAP and Almgren-Chriss-style schedules in controlled simulated and crypto-data settings, and interpret apparent gains with appropriate caution
- Compare deep hedging results with delta hedging and Whalley-Wilmott-style benchmarks under transaction costs using P&L distributions and tail-risk metrics
- Distinguish inverse reinforcement learning from behavior cloning and explain what reward inference can and cannot recover from observed trading behavior
- Diagnose the simulation-to-reality risks that govern deployability, including non-stationarity, reward hacking, market impact, partial observability, latency, and benchmark mismatch

## Sections

### 21.1 The Sequential Decision-Making Paradigm in Finance

This section explains why reinforcement learning matters in finance only in certain settings. Its core contribution is to separate prediction from control: supervised learning forecasts what may happen, while RL decides what to do next when actions affect future outcomes. The section is strongest when it argues that RL is structurally better matched to execution, market making, and hedging than to broad alpha discovery, because these tasks have clearer objectives, tighter feedback loops, and more defensible reward signals.

### 21.2 Financial Markets as Markov Decision Processes

This section translates financial trading problems into the language RL needs: state, action, reward, transition, and discounting. It shows that MDP specification is not a technical formality but a modeling decision that embeds market assumptions, constraints, and economic goals. The treatment of partial observability is especially important because it explains why state engineering, recurrent models, and private agent information matter so much in financial applications.

- `rl_environments`
- `rl_calibration`

### 21.3 Core Algorithms: From DQN to Actor-Critic

This section gives readers the practical algorithm map rather than a full RL survey. It explains why discrete-action problems can use value-based methods like DQN, while most realistic financial control tasks require actor-critic methods that handle continuous actions. The section also adds useful depth by introducing risk-sensitive variants and clarifying the practical trade-off between stability, sample efficiency, and action-space flexibility.

- [`algorithms_comparison`](01_algorithms_comparison.ipynb) — Trains DQN, PPO and A2C on one trading environment with the same interaction budget and scores them on the same episodes. The point is that a single average reward cannot tell you whether a policy is trading or has collapsed onto one position, and that the distribution of chosen actions can. The return process is a GARCH(1,1) model fitted to hourly BTCUSDT bars.

### 21.4 Application I: Optimal Trade Execution

This section presents execution as the cleanest institutional RL use case. It frames the problem around implementation shortfall, timing risk, and market impact, then positions RL as a dynamic alternative to fixed schedules like TWAP and Almgren-Chriss when liquidity and volatility vary over time. The key reader takeaway is not that RL wins outright, but that any claim of improvement depends heavily on simulator realism, reward design, and careful benchmarking.

- [`optimal_execution_ppo`](02_optimal_execution_ppo.ipynb) — Trains a PPO agent to pace a liquidation and compares it against TWAP and an Almgren-Chriss schedule on identical simulated paths, with the paired differences and their standard errors. Says which of the simulator's parameters were fitted to market data, which are proxies, and which are the calibration's clamp.
- [`crypto_execution_rl`](04_crypto_execution_rl.ipynb) — Replays recorded hourly perpetual-futures bars instead of a simulated path, so the agent's state can carry the perpetual-spot premium and the hours to the next funding settlement. Builds the panel so every feature a decision uses comes from bars that had closed, and separates training from reported episodes by date.

### 21.5 Application II: Market Making

This section shows why market making is a natural RL problem: quoting decisions must continuously balance spread capture, inventory risk, and adverse selection. The classical Avellaneda-Stoikov benchmark gives readers a principled reference point, while the RL framing shows how adaptive quoting can respond to richer market states without fully specified analytical assumptions. The section works best as an illustration of learned inventory-aware behavior rather than a claim that the learned policy already dominates analytical baselines.

- [`market_making_ppo`](03_market_making_ppo.ipynb) — Trains a PPO agent to choose a quote skew and a spread width against three reservation-price rules that already encode the inventory response, so the comparison asks what learning the response adds to writing it down. Fills arrive with a probability that falls with distance from the mid, which is where the spread-versus-fill-rate trade-off comes from.

### 21.6 Application III: Deep Hedging for Derivatives

This section reframes hedging from exact replication toward friction-aware risk control. It explains why transaction costs, discrete rebalancing, and model misspecification weaken classical delta hedging, and why deep hedging becomes interesting when the objective is explicitly a risk measure of terminal P&L. The comparison with Whalley-Wilmott, tabular Q-learning, and delta hedging is valuable because it teaches readers how to interpret learned hedging results cautiously rather than assuming that neural methods automatically outperform classical benchmarks.

- [`deep_hedging_pfhedge`](05_deep_hedging_pfhedge.ipynb) — Hedges a short call under discrete rebalancing and proportional costs, where exact replication is impossible and the question becomes which P&L distribution to accept. Compares five hedges on identical Heston paths: Black-Scholes delta, Whalley-Wilmott, a `pfhedge` deep hedger, the same idea written from scratch, and a tabular Q-learner.

### 21.7 Inverse Reinforcement Learning: Learning from Observed Behavior

This section broadens the chapter from optimizing known rewards to inferring objectives from observed behavior. It usefully distinguishes inverse RL from behavior cloning and shows why reward inference can sometimes generalize more meaningfully than direct imitation. Its main value for readers is conceptual: it opens a path from imitation to objective discovery while also making clear that identifiability, demonstration quality, and model assumptions sharply limit what can really be inferred.

- [`inverse_reinforcement_learning`](06_inverse_reinforcement_learning.ipynb) — Runs the arrow backwards: given a record of what a trader did, what objective would make those actions sensible? Compares behaviour cloning against two reward-inference methods on demonstrations from a rule whose objective is already known, so the inferred reward can be checked rather than believed.

### 21.8 The Simulation-to-Reality Gap

This is the chapter's governing cautionary section. It argues that non-stationarity, impact reflexivity, latency, poor fill assumptions, and reward hacking are the real barriers to deploying financial RL, not merely choosing the right algorithm. By emphasizing simulator fidelity, offline RL, off-policy evaluation, staged deployment, and governance, the section turns the chapter from a collection of promising applications into a more credible guide to what would have to be true for RL to work in practice.

### Summary

The chapter's message is that reinforcement learning is most credible for sequential control problems with well-defined economic objectives, evaluated under realistic assumptions. Its contribution is the combination of application case studies with disciplined skepticism about benchmarks, simulation design, and deployment risk.

- [`backtest_with_impact`](07_backtest_with_impact.ipynb) — Runs one momentum strategy and one dollar order across real US equity names spanning the liquidity range, under four strengths of a square-root impact model. Measures how much of a paper return the impact charge removes, and how often it is the difference between a profit and a loss, separately for a liquid cohort and a thin one.

## Running the Notebooks

```bash
# From the repository root
uv run python 21_rl_execution_hedging/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "21_rl_execution_hedging"
```

## References

- **Saurabh Arora and Prashant Doshi** (2021). [A survey of inverse reinforcement learning: Challenges, methods and progress](https://doi.org/10.1016/j.artint.2021.103500). *Artificial Intelligence*.
- **H. Buehler et al.** (2019). [Deep hedging](https://doi.org/10.1080/14697688.2019.1571683). *Quantitative Finance*.
- **David Byrd et al.** (2020). [ABIDES: Towards High-Fidelity Multi-Agent Market Simulation](https://doi.org/10.1145/3384441.3395986). *ACM*.
- **Lili Chen et al.** (2021). [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://doi.org/10.48550/arXiv.2106.01345).
- **Matthew Dixon and Igor Halperin** (2020). [G-Learner and GIRL: Goal Based Wealth Management with Reinforcement Learning](https://doi.org/10.48550/arXiv.2002.10990).
- **Tuomas Haarnoja et al.** (2018). [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://doi.org/10.48550/arXiv.1801.01290).
- **Yadh Hafsi and Edoardo Vittori** (2025). [Optimal Execution with Reinforcement Learning](https://doi.org/10.48550/arXiv.2411.06389).
- **Igor Halperin** (2019). [QLBS: Q-Learner in the Black-Scholes(-Merton) Worlds](https://doi.org/10.48550/arXiv.1712.04609).
- **Ben Hambly et al.** (2023). [Recent advances in reinforcement learning in finance](https://doi.org/10.1111/mafi.12382). *Mathematical Finance*.
- **Hado van Hasselt et al.** (2015). [Deep Reinforcement Learning with Double Q-learning](https://doi.org/10.48550/arXiv.1509.06461).
- **Jonathan Ho and Stefano Ermon** (2016). [Generative Adversarial Imitation Learning](https://doi.org/10.48550/arXiv.1606.03476).
- **Petter N. Kolm and Gordon Ritter** (2019). [Modern Perspectives on Reinforcement Learning in Finance](https://doi.org/10.2139/ssrn.3449401).
- **Vijay Konda and John Tsitsiklis** (1999). [Actor-Critic Algorithms](https://papers.nips.cc/paper_files/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). *MIT Press*.
- **Yang Li et al.** (2025). [FlowHFT: Imitation Learning via Flow Matching Policy for Optimal High-Frequency Trading under Diverse Market Conditions](https://doi.org/10.48550/arXiv.2505.05784).
- **Adrian Millea** (2021). [Deep Reinforcement Learning for Trading—A Critical Survey](https://doi.org/10.3390/data6110119). *Data*.
- **Volodymyr Mnih et al.** (2015). [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236). *Nature*.
- **John Schulman et al.** (2017). [Proximal Policy Optimization Algorithms](https://doi.org/10.48550/arXiv.1707.06347).
- **Aaron J. Snoswell et al.** (2020). [Revisiting Maximum Entropy Inverse Reinforcement Learning: New Perspectives and Algorithms](https://doi.org/10.1109/SSCI47803.2020.9308391).
- **Shuo Sun et al.** (2023). [Reinforcement Learning for Quantitative Trading](https://doi.org/10.1145/3582560). *ACM Transactions on Intelligent Systems and Technology*.
- **Richard S Sutton et al.** (2000). [Policy Gradient Methods for Reinforcement Learning with Function Approximation](http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf). *MIT Press*.
- **Ziyu Wang et al.** (2016). [Dueling Network Architectures for Deep Reinforcement Learning](https://doi.org/10.48550/arXiv.1511.06581).
- **Steve Y. Yang et al.** (2015). [Gaussian process-based algorithmic trading strategy identification](https://doi.org/10.1080/14697688.2015.1011684). *Quantitative Finance*.
- **Cong Zheng et al.** (2023). [Option Dynamic Hedging Using Reinforcement Learning](http://arxiv.org/abs/2306.10743).
