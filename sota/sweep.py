"""Run several PQN variants x training seeds in parallel, one single-threaded process each.

    python -m sota.sweep --group ablation --variants sota/configs/ablation.json --seeds 0 1 2

``--variants`` is a JSON object mapping a variant name to config overrides, applied on
top of ``--config``. Every run writes to ``sota/runs/<group>/<variant>_s<seed>/``.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from flappy_bird_rl.config import load_env_config


def _run(job: tuple) -> dict:
    base, env_path, overrides, name, seed, run_dir = job
    from sota.pqn import load_pqn_config, train

    config = load_pqn_config(base, **overrides)
    summary = train(config, seed, run_dir, load_env_config(env_path), quiet=True)
    return {"variant": name, **summary}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parallel PQN sweep")
    parser.add_argument("--group", required=True)
    parser.add_argument("--config", type=Path, default=Path("sota/configs/pqn.json"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.json"))
    parser.add_argument("--variants", type=Path, help="JSON: {name: {key: value}}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--root", type=Path, default=Path("sota/runs"))
    args = parser.parse_args(argv)

    variants = json.loads(args.variants.read_text(encoding="utf-8")) if args.variants else {}
    variants = variants or {"pqn": {}}
    base = args.config if args.config.exists() else None
    jobs = []
    for name, overrides in variants.items():
        overrides = dict(overrides)
        if args.total_steps is not None:
            overrides["total_steps"] = args.total_steps
        for seed in args.seeds:
            run_dir = args.root / args.group / f"{name}_s{seed}"
            if (run_dir / "summary.json").exists():
                continue
            jobs.append((base, args.env_config, overrides, name, seed, run_dir))

    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run, job) for job in jobs]
        for future in as_completed(futures):
            s = future.result()
            final = s["final_eval"]
            print(
                f"{s['variant']:<24} seed={s['train_seed']:<3} "
                f"validation pipes={final['eval_mean_pipes']:7.2f} "
                f"survival={final['eval_survival']:.2f} | {s['seconds']:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
