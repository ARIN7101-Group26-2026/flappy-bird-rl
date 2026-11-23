# Game Recordings

Recordings are organized by algorithm. Each video is accompanied by an episode CSV, runtime metadata JSON, and PNG preview.

| Algorithm | Directory | Contents |
| --- | --- | --- |
| Random | [random/](random/) | Uniform random policy and probability comparison |
| DQN | [dqn/](dqn/) | DQN policy recordings |
| PPO | [ppo/](ppo/) | PPO policy recordings |
| PQN | [pqn/](pqn/) | Full 5,000-step episode of the best trained agent |

## Random Policy

| Recording | Policy | Episodes | Duration |
| --- | --- | ---: | ---: |
| [Uniform random policy](random/random_demo.mp4) | Flap probability 0.5 | 3 | 5.1 s |
| [Probability comparison](random/random_comparison.mp4) | Flap probabilities 0.5, 0.2, 0.1, 0.05 | 12 | 19.13 s |

[![Random policy comparison](random/random_comparison.png)](random/random_comparison.mp4)

## PQN Agent

| Recording | Policy | Episodes | Duration |
| --- | --- | ---: | ---: |
| [Best PQN agent](pqn/pqn_demo.mp4) | Training seed 0, greedy; 5,000 steps, 132 pipes | 1 | 166.7 s |

[![PQN agent](pqn/pqn_demo.png)](pqn/pqn_demo.mp4)

The agent, its training, and its evaluation are described in [sota/README.md](../sota/README.md). A results comparison in Chinese is in [docs/pqn_effect_comparison.pdf](../docs/pqn_effect_comparison.pdf).

## Reproduce

Run from the repository root:

```bash
flappy-rl demo
flappy-rl demo --flap-probs 0.5 0.2 0.1 0.05 --output videos/random/random_comparison.mp4
python -m sota.record --checkpoint sota/models/pqn_champion.npz --output videos/pqn/pqn_demo.mp4
```

Random recordings use environment seeds 42–44 and include all consecutive episodes; the PQN recording uses seed 42. See each video's JSON and CSV for its complete configuration and measured episode metrics.
