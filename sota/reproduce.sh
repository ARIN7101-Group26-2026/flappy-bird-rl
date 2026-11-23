#!/usr/bin/env bash
# The full pipeline behind sota/results, sota/models and sota/videos.
# Run from the repository root after: python -m pip install -e '.[dev,sota]'
# About an hour on a 32-thread CPU: 52 training runs, then verification and evaluation.
set -euo pipefail

# 1. Train: 20 seeds of the full method, 4 seeds of each of 8 ablations (one process per run).
python -m sota.sweep --group main --variants sota/configs/main.json --seeds $(seq 0 19)
python -m sota.sweep --group ablation --variants sota/configs/ablation.json --seeds 0 1 2 3
python -m sota.analyze collect --group main --runs sota/runs/main
python -m sota.analyze collect --group ablation --runs sota/runs/ablation

# 2. Keep the 20 main agents in the repository. The agent of a run is its last checkpoint.
for run in sota/runs/main/full_s*; do
  mkdir -p "sota/models/$(basename "$run")"
  cp "$run/final.npz" "sota/models/$(basename "$run")/final.npz"
done

# 3. Exhaustive verification, with exact survival probabilities over a 5,000-step episode.
python -m sota.verify --checkpoints sota/models/full_s*/final.npz --exact-horizon 5000 \
  --output sota/results/verification/main.json
python -m sota.verify --checkpoints sota/runs/ablation/*/final.npz --exact-horizon 5000 \
  --output sota/results/verification/ablation.json

# 4. Champion: the main agent with the lowest exact death probability. Verify it over a
#    longer horizon as well (40,000 steps = 1,061 pipes).
python -m sota.analyze champion --destination sota/models/pqn_champion.npz
python -m sota.verify --checkpoints sota/models/pqn_champion.npz --exact-horizon 40000 \
  --output sota/results/verification/champion.json

# 5. The framework's standard protocol: validation seeds 1000-1099, test seeds 10000-10099.
python -m sota.evaluate protocol --checkpoints sota/models/full_s*/final.npz \
  --output-dir sota/results/protocol
python -m sota.evaluate protocol --checkpoints sota/runs/ablation/*/final.npz --splits test \
  --output-dir sota/results/protocol_ablation

# 6. Sampling beyond the protocol, champion only, on seeds used nowhere else.
python -m sota.evaluate stress --checkpoints sota/models/pqn_champion.npz \
  --max-steps 5000 --episodes 10000 --seed-start 100000 \
  --output sota/results/large_sample/large_sample.csv
python -m sota.evaluate stress --checkpoints sota/models/pqn_champion.npz \
  --max-steps 1000000 --episodes 32 --seed-start 20000 \
  --output sota/results/endurance/endurance.csv

# 7. One failure in detail, the recording, and the report.
python -m sota.anatomy --failing "${ANATOMY_FAILING:-sota/runs/ablation/no_restarts_s2/final.npz}" \
  --reference sota/models/pqn_champion.npz --seed 10056
python -m sota.record --checkpoint sota/models/pqn_champion.npz \
  --output sota/videos/pqn_demo.mp4 --episodes 1 --seed-start 42
python -m sota.analyze report
