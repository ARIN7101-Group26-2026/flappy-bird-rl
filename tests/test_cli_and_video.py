import csv
import json

import imageio.v2 as imageio
import pytest

from flappy_bird_rl.cli import main
from flappy_bird_rl.config import EnvConfig, RandomConfig
from flappy_bird_rl.demo import record_random
from flappy_bird_rl.evaluate import evaluate_random


def test_cli_outputs_plots_and_switches_to_separate_test_seeds(tmp_path):
    main(["random", "--episodes", "2", "--split", "test", "--output-dir", str(tmp_path)])
    with (tmp_path / "random_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {int(row["eval_seed"]) for row in rows} == {10000, 10001}
    for filename in ("mean_pipes.png", "score_distribution.png"):
        assert (tmp_path / "figures" / filename).stat().st_size > 1000


@pytest.mark.parametrize(
    "args",
    [
        ["--episodes", "0"],
        ["--flap-probs", "nan"],
        ["--seed-start", "-1"],
        ["--flap-probs", "0.5", "0.5"],
    ],
)
def test_invalid_cli_exits_before_creating_outputs(tmp_path, args):
    output = tmp_path / "bad"
    with pytest.raises(SystemExit) as exc:
        main(["random", *args, "--output-dir", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_video_decodes_and_matches_headless_trajectories(tmp_path):
    config = RandomConfig(episodes=2, flap_probs=(0.5, 0.1), seed_start=42, split="demo")
    env_config = EnvConfig(max_episode_steps=3)
    output = tmp_path / "demo.mp4"
    results = record_random(output, config, env_config)
    assert results == evaluate_random(config, env_config)
    reader = imageio.get_reader(str(output))
    try:
        assert reader.count_frames() == sum(row.episode_steps + 1 for row in results)
        assert reader.get_data(0).shape == (576, 288, 3)
        assert reader.get_meta_data()["fps"] == 30
    finally:
        reader.close()
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["experiment"]["frames"] == 16
    assert output.with_suffix(".csv").exists()
    assert output.with_suffix(".png").exists()
