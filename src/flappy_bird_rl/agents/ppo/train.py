"""Implementation location for PPO training.

Read algorithm settings from ``configs/ppo.json`` and create environments
through ``flappy_bird_rl.env.make_env``. Store checkpoints in ``models/ppo/``,
logs in ``logs/ppo/``, evaluation results in ``results/ppo/``, and recordings
in ``videos/ppo/``. Record the actual sampled environment step count.
"""
