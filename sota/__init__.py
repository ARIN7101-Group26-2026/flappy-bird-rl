"""Standalone high-score track: a PQN agent trained on the shared Flappy Bird environment.

This package lives outside ``src/flappy_bird_rl`` on purpose. It reuses the shared
environment factory and episode runner so that its numbers are directly comparable
with every other agent in the repository, but it does not change the framework.
"""

import os

# The networks here are tiny, so BLAS/OpenMP thread pools only add contention when many
# seeds train side by side. One thread per process, unless the caller says otherwise.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
