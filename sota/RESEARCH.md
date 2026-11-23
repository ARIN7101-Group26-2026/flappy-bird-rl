# Research notes: what the field does in 2026, and what we took from it

Survey date: 2026-11-21. Every item links to its source. Items are split into what
shaped this implementation and what was considered and set aside, with the reason.

## 1. The task, as others have attacked it

The environment is [`flappy-bird-gymnasium`](https://github.com/markub3327/flappy-bird-gymnasium)
v0.4.0 with the 12-feature observation. Facts below were read from the
[v0.4.0 source](https://github.com/markub3327/flappy-bird-gymnasium/blob/v0.4.0/flappy_bird_gymnasium/envs/flappy_bird_env.py)
and re-checked against the installed package.

| Fact | Consequence for learning |
| --- | --- |
| Reward: +0.1 per step, +1 for a pipe, −1 on death, −0.5 per step above the screen | Dense enough to learn from without shaping |
| A flap sets vertical speed to −9; gravity adds +1 per step | The bird cannot hover in less than a 45 px band; one flap is needed about every 19 steps |
| Gap is 100 px, bird is 24 px tall | 76 px of room for a 45 px oscillation: 31 px of slack, so timing matters |
| Gap top is uniform over {100, 110, …, 170} px; pipe spacing cycles 36/36/41 steps | Pipe positions are deterministic, so 5,000 steps allow **at most 132 pipes** |
| Pipes right of the screen are masked in the observation | The next pipe becomes visible about 49 steps ahead, which is enough |

Publicly reported results we could find (none uses our exact protocol):

| Source | Observation | Method | Reported result |
| --- | --- | --- | --- |
| [Environment author's paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10975254) | LIDAR, 16 frames | Transformer dueling DQN, 1M steps | mean 13,157, max 74,755 pipes with a proximity penalty; mean 380 without |
| [YufanCui/flappy-bird](https://github.com/YufanCui/flappy-bird) | LIDAR, 4 frames | SB3 PPO, 10M steps, trained on a harder variant | mean 1,007 pipes under a 40,000-step cap; 88% of episodes reach the cap |
| [yusuferoglu52](https://github.com/yusuferoglu52/ReinforcementLearning_w_FlappyBird) | 12 features | SB3 PPO, 5M steps | 8 of 10 episodes reach a 200-pipe cap, worst 116 |
| same | 12 features | SB3 DQN, 5M steps | median 41, mean 72 pipes |
| [johnnycode8/dqn_pytorch](https://github.com/johnnycode8/dqn_pytorch) | 12 features | Dueling Double DQN, 24 h | about 22 pipes; "did not learn to fly indefinitely" |
| [akshaygulabrao/flappybyrd](https://github.com/akshaygulabrao/flappybyrd) | 12 features | Double + Dueling DQN, 1.3M steps | mean 20.8 pipes |

Two things stand out. Replay-buffer DQN does poorly on the feature observation, and we
found no learned agent on this environment that is shown to keep flying indefinitely.

Failure modes that others documented, and that guided our design:

- **Random exploration flaps are fatal.** A uniformly random action flaps half the time
  and pins the bird to the ceiling. Teams report having to cut the exploration floor
  ([FlapAI](https://arxiv.org/abs/2003.09579),
  [StefanVlad0](https://github.com/StefanVlad0/flappy-bird-rl)).
- **A constant learning rate stalls or collapses**; linear decay to zero fixed it for
  PPO ([YufanCui, round 3](https://github.com/YufanCui/flappy-bird/tree/main/project/results/improve_r3_20261118)).
- **Edge-hugging.** Crashes concentrate on the lower lip of the upper pipe; training on
  a harder variant of the game is one reported remedy (YufanCui).

## 2. Algorithms: the 2023–2026 landscape

**Value-based.**
[Rainbow](https://arxiv.org/abs/1710.02298) (2018) →
[BBF](https://arxiv.org/abs/2305.19452) (ICML 2023; sample efficiency via a large network
and a high replay ratio) →
[BTR](https://arxiv.org/abs/2411.03820) (ICML 2025; Rainbow with an Impala network, IQN,
Munchausen and vectorised environments) →
[PQN](https://arxiv.org/abs/2407.04811) (ICLR 2025 spotlight). PQN removes both the replay
buffer and the target network: stability comes from LayerNorm in the Q-network, data from
many parallel environments, and targets from λ-returns over short rollouts. Follow-ups
([Hadamax](https://arxiv.org/abs/2505.15345), and several 2026 preprints) change the
encoder for pixel Atari and do not alter the algorithm for vector observations.

**Policy gradient.** [PPO](https://arxiv.org/abs/1707.06347) remains the default in
[CleanRL](https://github.com/vwxyzjn/cleanrl) and
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). The speed frontier is
[PufferLib](https://puffer.ai/docs.html), which reports tens of millions of steps per
second on a GPU, but as of version 5.0 it has no CPU training path.

**Model-based and search.** [DreamerV3](https://arxiv.org/abs/2301.04104) (Nature 2025),
[TD-MPC2](https://arxiv.org/abs/2310.16828) (continuous control only),
[EfficientZero V2](https://arxiv.org/abs/2403.00564) (ICML 2024),
[MuZero](https://arxiv.org/abs/1911.08265).

**Architecture and plasticity.** LayerNorm keeps value networks trainable
([Lyle et al., 2023](https://arxiv.org/abs/2303.01486)); the same idea underlies
[SimBa](https://arxiv.org/abs/2410.09754), [BRO](https://arxiv.org/abs/2405.16158) and
[CrossQ](https://arxiv.org/abs/1902.05605) in continuous control.

### Why PQN for this project

| Requirement here | PQN | Replay DQN / Rainbow / BTR | PPO | Dreamer / EfficientZero |
| --- | --- | --- | --- | --- |
| Simulator is cheap (140k steps/s on one core) | Built for many parallel envs | Built for scarce data | Good | Built for scarce data |
| CPU only | Small MLP, no buffer | Workable | Workable | Heavy |
| Deterministic greedy play at test time | Yes, argmax | Yes | Needs a mode action | Yes, with search |
| Not already planned in this repository | New | DQN is planned | PPO is planned | New, but heavy |
| Small enough to implement and audit from scratch | A few hundred lines | Larger | Similar | Much larger |

PQN is the newest method in this list that matches the setting: data is nearly free,
compute is modest, and the action space is tiny.

### What we took from the reference implementations, and what we changed

Read from the official [purejaxql](https://github.com/mttga/purejaxql) code and the
[CleanRL port](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn.py).

| Element | Reference PQN | This implementation |
| --- | --- | --- |
| Network | Dense → LayerNorm → ReLU, linear head | Same |
| Loss | 0.5 × squared error to the λ-return | Same |
| Optimiser | RAdam, gradient-norm clip 10, linear learning-rate decay | Same |
| λ-returns | Computed once per rollout | Recomputed at each epoch with the current network |
| Episode ends | Termination and truncation merged, so no bootstrap at a time limit | Bootstrap at truncation, stop only at death ([Pardo et al., 2018](https://arxiv.org/abs/1712.00378)) |
| Exploration | Uniform random action with probability ε | Random action flaps with probability 0.1, the rate that scored highest in this repository's Random study |

## 3. Evaluation practice

- [Agarwal et al., 2021](https://arxiv.org/abs/2108.13264) ("Deep RL at the Edge of the
  Statistical Precipice"): report the interquartile mean with bootstrap confidence
  intervals rather than a mean or a best run.
- [Patterson et al., 2024](https://arxiv.org/abs/2304.01315): three to five runs is
  risky; a time limit hides variance, and treating a cutoff as a death is incorrect.
- [Jordan et al., 2024](https://arxiv.org/abs/2406.16241): bootstrap intervals are
  unreliable with few runs.
- We found no RL-specific guidance for scores cut off by a time limit. Statistically an
  episode stopped at the limit is a right-censored lifetime, so we use the tools built
  for that: Kaplan–Meier curves, and exact binomial bounds on the failure rate.
- None of these can separate two agents that both reach the time limit in every sampled
  episode. For that we added an exhaustive enumeration of the policy's reachable states,
  described in the [report](README.md#3-evaluation-three-levels-of-evidence).

## 4. Not verified

Venue claims for Hadamax and the 2026 PQN follow-ups, and release dates of PufferLib 4.0
and 5.0, were not checked against primary sources. The public Flappy Bird results in
section 1 are quoted as their authors report them; we did not re-run them.
