# Deep Reinforcement Learning: Building a Trading Agent

Reinforcement Learning (RL) is a computational approach to goal-directed learning performed by an agent that interacts with a typically stochastic environment which the agent has incomplete information about. RL aims to automate how the agent makes decisions to achieve a long-term objective by learning the value of states and actions from a reward signal. The ultimate goal is to derive a policy that encodes behavioral rules and maps states to actions.

This chapter shows how to formulate an RL problem and how to apply various solution methods. It covers model-based and model-free methods, introduces the [OpenAI Gym](https://gym.openai.com/) environment, and combines deep learning with RL to train an agent that navigates a complex environment. Finally, we'll show you how to adapt RL to algorithmic trading by modeling an agent that interacts with the financial market while trying to optimize an objective function. 

#### Table of contents

1. [Key elements of a reinforcement learning system](#key-elements-of-a-reinforcement-learning-system)
    * [The policy: translating states into actions](#the-policy-translating-states-into-actions)
    * [Rewards: learning from actions](#rewards-learning-from-actions)
    * [The value function: optimal decisions for the long run](#the-value-function-optimal-decisions-for-the-long-run)
    * [The environment](#the-environment)
    * [Components of an interactive RL system](#components-of-an-interactive-rl-system)
2. [How to solve RL problems](#how-to-solve-rl-problems)
    * [Code example: dynamic programming – value and policy iteration](#code-example-dynamic-programming--value-and-policy-iteration)
    * [Code example: Q-Learning](#code-example-q-learning)
3. [Deep Reinforcement Learning](#deep-reinforcement-learning)
    * [Value function approximation with neural networks](#value-function-approximation-with-neural-networks)
    * [The Deep Q-learning algorithm and extensions](#the-deep-q-learning-algorithm-and-extensions)
    * [The Open AI Gym – the Lunar Lander environment](#the-open-ai-gym--the-lunar-lander-environment)
    * [Code example: Double Deep Q-Learning using Tensorflow](#code-example-double-deep-q-learning-using-tensorflow)
4. [Code example: deep RL for trading with TensorFlow 2 and OpenAI Gym](#code-example-deep-rl-for-trading-with-tensorflow-2-and-openai-gym)
    * [How to Design an OpenAI trading environment](#how-to-design-an-openai-trading-environment)
    * [How to build a Deep Q-learning agent for the stock market](#how-to-build-a-deep-q-learning-agent-for-the-stock-market)
5. [Resources](#resources)
    * [RL Algorithms](#rl-algorithms)
    * [Investment Applications](#investment-applications)

## Key elements of a reinforcement learning system

RL problems feature several elements that set them apart from the ML settings we have covered so far. The following two sections outline the key features required for defining and solving an RL problem by learning a policy that automates decisions. 
We’ll use the notation and generally follow [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) (Sutton and Barto 2018) and David Silver’s [UCL Courses on RL](https://www.davidsilver.uk/teaching/) that are recommended for further study beyond the brief summary that the scope of this chapter permits.

RL problems aim to optimize an agent's decisions based on an objective function vis-a-vis an environment.

### The policy: translating states into actions
At any point in time, the policy defines the agent’s behavior. It maps any state the agent may encounter to one or several actions. In an environment with a limited number of states and actions, the policy can be a simple lookup table filled in during training. 

### Rewards: learning from actions

The reward signal is a single value that the environment sends to the agent at each time step. The agent’s objective is typically to maximize the total reward received over time. Rewards can also be a stochastic function of the state and the actions. They are typically discounted to facilitate convergence and reflect the time decay of value.
 
### The value function: optimal decisions for the long run
The reward provides immediate feedback on actions. However, solving an RL problem requires decisions that create value in the long run. This is where the value function comes in: it summarizes the utility of states or of actions in a given state in terms of their long-term reward. 
 
### The environment
The environment presents information about its state to the agent, assigns rewards for actions, and transitions the agent to new states subject to probability distributions the agent may or may not know about. 
It may be fully or partially observable, and may also contain other agents. The design of the environment typically requires significant up-front design effort to facilitate goal-oriented learning by the agent during training.

RL problems differ by the complexity of their state and action spaces that can be either discrete or continuous. The latter requires ML to approximate a functional relationship between states, actions, and their value. They also require us to generalize from the subset of states and actions they are experienced by the agent during training.

### Components of an interactive RL system

The components of an RL system typically include:

- Observations by the agent of the state of the environment
- A set of actions that are available to the agent
- A policy that governs the agent's decisions

In addition, the environment emits a reward signal that reflects the new state resulting from the agent's action. At the core, the agent usually learns a value function that shapes its judgment over actions. The agent has an objective function to process the reward signal and translate the value judgments into an optimal policy.

## How to solve RL problems

RL methods aim to learn from experience on how to take actions that achieve a long-term goal. To this end, the agent and the environment interact over a sequence of discrete time steps via the interface of actions, state observations, and rewards that we described in the previous section.

There are numerous approaches to solving RL problems which implies finding rules for the agent's optimal behavior:

- **Dynamic programming** (DP) methods make the often unrealistic assumption of complete knowledge of the environment, but are the conceptual foundation for most other approaches.
- **Monte Carlo** (MC) methods learn about the environment and the costs and benefits of different decisions by sampling entire state-action-reward sequences.
- **Temporal difference** (TD) learning significantly improves sample efficiency by learning from shorter sequences. To this end, it relies on bootstrapping, which is defined as refining its estimates based on its own prior estimates.

Approaches for continuous state and/or action spaces often leverage ML to approximate a value or policy function. Hence, they integrate supervised learning, and in particular, the deep learning methods we discussed in the last several chapters. However, these methods face distinct challenges in the RL context:

- The reward signal does not directly reflect the target concept, such as a labeled sample
- The distribution of the observations depends on the agent's actions and the policy which is itself the subject of the learning process

### Code example: dynamic programming – value and policy iteration

Finite MDPs are a simple yet fundamental framework. This section introduces the trajectories of rewards that the agent aims to optimize, and define the policy and value functions they are used to formulate the optimization problem and the Bellman equations that form the basis for the solution methods.

The notebook [gridworld_dynamic_programming](01_gridworld_dynamic_programming.ipynb) applies Value and Policy Iteration to a toy environment that consists of a 3 x 4 grid.

### Code example: Q-Learning

Q-learning was an early RL breakthrough when it was developed by Chris Watkins for his [PhD thesis]((http://www.cs.rhul.ac.uk/~chrisw/thesis.html)) in 1989 . It introduces incremental dynamic programming to control an MDP without knowing or modeling the transition and reward matrices that we used for value and policy iteration in the previous section. A convergence proof followed three years later by [Watkins and Dayan](http://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html).

Q-learning directly optimizes the action-value function, q, to approximate q*. The learning proceeds off-policy, that is, the algorithm does not need to select actions based on the policy that's implied by the value function alone. However, convergence requires that all state-action pairs continue to be updated throughout the training process. A straightforward way to ensure this is by using an ε-greedy policy.

The Q-learning algorithm keeps improving a state-action value function after random initialization for a given number of episodes. At each time step, it chooses an action based on an ε-greedy policy, and uses a learning rate, α, to update the value function based on the reward  and its current estimate of the value function for the next state.

The notebook [gridworld_q_learning](02_gridworld_q_learning.ipynb) demonstrates how to build a Q-learning agent using the 3 x 4 grid of states from the previous section.

## Deep Reinforcement Learning

This section adapts Q-Learning to continuous states and actions where we cannot use the tabular solution that simply fills an array with state-action values. Instead, we will see how to approximate the optimal state-value function using a neural network to build a deep Q network with various refinements to accelerate convergence. We will then see how we can use the [OpenAI Gym](http://gym.openai.com/docs/) to apply the algorithm to the [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2) environment.

### Value function approximation with neural networks

As in other fields, deep neural networks have become popular for approximating value functions. However, ML faces distinct challenges in the RL context where the data is generated by the interaction of the model with the environment using a (possibly randomized) policy:

- With continuous states, the agent will fail to visit most states and, thus, needs to generalize.
- Supervised learning aims to generalize from a sample of independently and identically distributed samples that are representative and correctly labeled. In the RL context, there is only one sample per time step so that learning needs to occur online.
- Samples can be highly correlated when sequential states are similar and the behavior distribution over states and actions is not stationary, but changes as a result of the agent's learning.

### The Deep Q-learning algorithm and extensions

Deep Q learning estimates the value of the available actions for a given state using a deep neural network. It was introduced by Deep Mind's [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (2013), where RL agents learned to play games solely from pixel input.

The deep Q-learning algorithm approximates the action-value function, q, by learning a set of weights, θ, of a multi-layered Deep Q Network (DQN) that maps states to actions.

Several innovations have improved the accuracy and convergence speed of deep Q-Learning, namely:
- **Experience replay** stores a history of state, action, reward, and next state transitions and randomly samples mini-batches from this experience to update the network weights at each time step before the agent selects an ε-greedy action. It increases sample efficiency, reduces the autocorrelation of samples, and limits the feedback due to the current weights producing training samples that can lead to local minima or divergence.
- **Slowly-changing target network** weakens the feedback loop from the current network parameters on the neural network weight updates. Also invented by by Deep Mind in [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) (2015), it use a slowly-changing target network that has the same architecture as the Q-network, but its weights are only updated periodically. The target network generates the predictions of the next state value used to update the Q-Networks estimate of the current state's value.
- **Double deep Q-learning** addresses the bias of deep Q-Learning to overestimate action values because it purposely samples the highest action value. This bias can negatively affect the learning process and the resulting policy if it does not apply uniformly , as shown by Hado van Hasselt in [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (2015). To decouple the estimation of action values from the selection of actions, Double Deep Q-Learning (DDQN) uses the weights, of one network to select the best action given the next state, and the weights of another network to provide the corresponding action value estimate.

### The Open AI Gym – the Lunar Lander environment

The [OpenAI Gym](https://gym.openai.com/) is a RL platform that provides standardized environments to test and benchmark RL algorithms using Python. It is also possible to extend the platform and register custom environments.

The [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2) (LL) environment requires the agent to control its motion in two dimensions, based on a discrete action space and low-dimensional state observations that include position, orientation, and velocity. At each time step, the environment provides an observation of the new state and a positive or negative reward. Each episode consists of up to 1,000 time steps.

### Code example: Double Deep Q-Learning using Tensorflow

The [lunar_lander_deep_q_learning](03_lunar_lander_deep_q_learning.ipynb) notebook implements a DDQN agent that uses TensorFlow and Open AI Gym's Lunar Lander environment.

## Code example: deep RL for trading with TensorFlow 2 and OpenAI Gym

To train a trading agent, we need to create a market environment that provides price and other information, offers trading-related actions, and keeps track of the portfolio to reward the agent accordingly.

### How to Design an OpenAI trading environment

The OpenAI Gym allows for the design, registration, and utilization of environments that adhere to its architecture, as described in its [documentation](https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym). 
- The [trading_env.py](trading_env.py) file implements an example that illustrates how to create a class that implements the requisite `step()` and `reset()` methods.

The trading environment consists of three classes that interact to facilitate the agent's activities:
 1. The `DataSource` class loads a time series, generates a few features, and provides the latest observation to the agent at each time step. 
 2. `TradingSimulator` tracks the positions, trades and cost, and the performance. It also implements and records the results of a buy-and-hold benchmark strategy. 
 3. `TradingEnvironment` itself orchestrates the process. 
 
### How to build a Deep Q-learning agent for the stock market
 
The notebook [q_learning_for_trading](04_q_learning_for_trading.ipynb) demonstrates how to set up a simple game with a limited set of options, a relatively low-dimensional state, and other parameters that can be easily modified and extended to train the Deep Q-Learning agent used in [lunar_lander_deep_q_learning](03_lunar_lander_deep_q_learning.ipynb).
 
<p align="center">
<img src="https://i.imgur.com/lg0ofbZ.png" width="60%">
</p>


## Resources

- [Reinforcement Learning: An Introduction, 2nd eition](http://incompleteideas.net/book/RLbook2018.pdf), Richard S. Sutton and Andrew G. Barto, 2018
- [University College of London Course on Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html), David Silver, 2015
- [Implementation of Reinforcement Learning Algorithms](https://github.com/dennybritz/reinforcement-learning), Denny Britz
    - This repository provides code, exercises and solutions for popular Reinforcement Learning algorithms. These are meant to serve as a learning tool to complement the theoretical materials from Sutton/Baron and Silver (see above).

### RL Algorithms

- Q Learning
    - [Learning from Delayed Rewards](http://www.cs.rhul.ac.uk/~chrisw/thesis.html), PhD Thesis, Chris Watkins, 1989
    - [Q-Learning](http://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html), Machine Learning, 1992
- Deep Q Networks
    - [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih et al, 2013
    - We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. We apply our method to seven Atari 2600 games from the Arcade Learning Environment, with no adjustment of the architecture or learning algorithm. We find that it outperforms all previous approaches on six of the games and surpasses a human expert on three of them.
- Asynchronous Advantage Actor-Critic (A2C/A3C)
    - [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), Mnih, V. et al. 2016
    - We propose a conceptually simple and lightweight framework for deep reinforcement learning that uses asynchronous gradient descent for optimization of deep neural network controllers. We present asynchronous variants of four standard reinforcement learning algorithms and show that parallel actor-learners have a stabilizing effect on training allowing all four methods to successfully train neural network controllers. The best performing method, an asynchronous variant of actor-critic, surpasses the current state-of-the-art on the Atari domain while training for half the time on a single multi-core CPU instead of a GPU. Furthermore, we show that asynchronous actor-critic succeeds on a wide variety of continuous motor control problems as well as on a new task of navigating random 3D mazes using a visual input.
- Proximal Policy Optimization (PPO)
    - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman et al, 2017
    - We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time.

- Trust Region Policy Optimization (TRPO)
    - [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman et al, 2015
    - We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement. By making several approximations to the theoretically-justified procedure, we develop a practical algorithm, called Trust Region Policy Optimization (TRPO). This algorithm is similar to natural policy gradient methods and is effective for optimizing large nonlinear policies such as neural networks. Our experiments demonstrate its robust performance on a wide variety of tasks: learning simulated robotic swimming, hopping, and walking gaits; and playing Atari games using images of the screen as input. Despite its approximations that deviate from the theory, TRPO tends to give monotonic improvement, with little tuning of hyperparameters.
    
- Deep Deterministic Policy Gradient (DDPG)
    - [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971), Lillicrap et al, 2015
    - We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving. Our algorithm is able to find policies whose performance is competitive with those found by a planning algorithm with full access to the dynamics of the domain and its derivatives. We further demonstrate that for many of the tasks the algorithm can learn policies end-to-end: directly from raw pixel inputs.
- Twin Delayed DDPG (TD3)
    - [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477), Fujimoto et al, 2018
    - In value-based reinforcement learning methods such as deep Q-learning, function approximation errors are known to lead to overestimated value estimates and suboptimal policies. We show that this problem persists in an actor-critic setting and propose novel mechanisms to minimize its effects on both the actor and the critic. Our algorithm builds on Double Q-learning, by taking the minimum value between a pair of critics to limit overestimation. We draw the connection between target networks and overestimation bias, and suggest delaying policy updates to reduce per-update error and further improve performance. We evaluate our method on the suite of OpenAI gym tasks, outperforming the state of the art in every environment tested.
- Soft Actor-Critic (SAC)
    - [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), Haarnoja et al, 2018
    - Model-free deep reinforcement learning (RL) algorithms have been demonstrated on a range of challenging decision making and control tasks. However, these methods typically suffer from two major challenges: very high sample complexity and brittle convergence properties, which necessitate meticulous hyperparameter tuning. Both of these challenges severely limit the applicability of such methods to complex, real-world domains. In this paper, we propose soft actor-critic, an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework. In this framework, the actor aims to maximize expected reward while also maximizing entropy. That is, to succeed at the task while acting as randomly as possible. Prior deep RL methods based on this framework have been formulated as Q-learning methods. By combining off-policy updates with a stable stochastic actor-critic formulation, our method achieves state-of-the-art performance on a range of continuous control benchmark tasks, outperforming prior on-policy and off-policy methods. Furthermore, we demonstrate that, in contrast to other off-policy algorithms, our approach is very stable, achieving very similar performance across different random seeds.
- Categorical 51-Atom DQN (C51)
    - [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), Bellemare, et al 2017
    - In this paper we argue for the fundamental importance of the value distribution: the distribution of the random return received by a reinforcement learning agent. This is in contrast to the common approach to reinforcement learning which models the expectation of this return, or value. Although there is an established body of literature studying the value distribution, thus far it has always been used for a specific purpose such as implementing risk-aware behaviour. We begin with theoretical results in both the policy evaluation and control settings, exposing a significant distributional instability in the latter. We then use the distributional perspective to design a new algorithm which applies Bellman's equation to the learning of approximate value distributions. We evaluate our algorithm using the suite of games from the Arcade Learning Environment. We obtain both state-of-the-art results and anecdotal evidence demonstrating the importance of the value distribution in approximate reinforcement learning. Finally, we combine theoretical and empirical evidence to highlight the ways in which the value distribution impacts learning in the approximate setting.
    
### Investment Applications
- [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/abs/1706.10059), Zhengyao Jiang, Dixing Xu, Jinjun Liang 2017
    - Financial portfolio management is the process of constant redistribution of a fund into different financial products. This paper presents a financial-model-free Reinforcement Learning framework to provide a deep machine learning solution to the portfolio management problem. The framework consists of the Ensemble of Identical Independent Evaluators (EIIE) topology, a Portfolio-Vector Memory (PVM), an Online Stochastic Batch Learning (OSBL) scheme, and a fully exploiting and explicit reward function. This framework is realized in three instants in this work with a Convolutional Neural Network (CNN), a basic Recurrent Neural Network (RNN), and a Long Short-Term Memory (LSTM). They are, along with a number of recently reviewed or published portfolio-selection strategies, examined in three back-test experiments with a trading period of 30 minutes in a cryptocurrency market. Cryptocurrencies are electronic and decentralized alternatives to government-issued money, with Bitcoin as the best-known example of a cryptocurrency. All three instances of the framework monopolize the top three positions in all experiments, outdistancing other compared trading algorithms. Although with a high commission rate of 0.25% in the backtests, the framework is able to achieve at least 4-fold returns in 50 days.
    - [PGPortfolio](https://github.com/ZhengyaoJiang/PGPortfolio); corresponding GitHub repo
- [Financial Trading as a Game: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/1807.02787.pdf), Huang, Chien-Yi, 2018
- [Order placement with Reinforcement Learning](https://github.com/mjuchli/ctc-executioner)
    - CTC-Executioner is a tool that provides an on-demand execution/placement strategy for limit orders on crypto currency markets using Reinforcement Learning techniques. The underlying framework provides functionalities which allow to analyse order book data and derive features thereof. Those findings can then be used in order to dynamically update the decision making process of the execution strategy.
    - The methods being used are based on a research project (master thesis) currently proceeding at TU Delft.
    
- [Q-Trader](https://github.com/edwardhdlu/q-trader)
    - An implementation of Q-learning applied to (short-term) stock trading. The model uses n-day windows of closing prices to determine if the best action to take at a given time is to buy, sell or sit. As a result of the short-term state representation, the model is not very good at making decisions over long-term trends, but is quite good at predicting peaks and troughs.
    