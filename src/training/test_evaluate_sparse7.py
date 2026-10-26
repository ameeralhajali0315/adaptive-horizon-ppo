import torch
import numpy as np
import pytest
from src.models.actor_critic import Actor
from src.training.evaluate_sparse7 import evaluate_checkpoint

from src.training.evaluate_sparse7 import summarize, write_csv


def make_episode(success, steps, episode=1, seed=10000):
    return {
        "algorithm": "fixed",
        "training_seed": 42,
        "episode": episode,
        "evaluation_seed": seed,
        "success": int(success),
        "reward": float(success),
        "steps": steps,
        "successful_steps": steps if success else "",
    }


def test_summarize_calculates_success_rate_and_means():
    episodes = [
        make_episode(True, 5, episode=1, seed=10000),
        make_episode(False, 40, episode=2, seed=10001),
        make_episode(True, 7, episode=3, seed=10002),
        make_episode(False, 40, episode=4, seed=10003),
    ]

    result = summarize("fixed", 42, episodes)

    assert result["algorithm"] == "fixed"
    assert result["training_seed"] == 42
    assert result["episodes"] == 4
    assert result["success_rate"] == pytest.approx(0.5)
    assert result["mean_reward"] == pytest.approx(0.5)
    assert result["mean_steps"] == pytest.approx(23.0)
    assert result["mean_successful_steps"] == pytest.approx(6.0)


def test_summarize_handles_no_successful_episodes():
    episodes = [
        make_episode(False, 40, episode=1, seed=10000),
        make_episode(False, 40, episode=2, seed=10001),
    ]

    result = summarize("fixed", 42, episodes)

    assert result["success_rate"] == 0.0
    assert result["mean_successful_steps"] == ""


def test_summarize_handles_all_successful_episodes():
    episodes = [
        make_episode(True, 5, episode=1, seed=10000),
        make_episode(True, 7, episode=2, seed=10001),
    ]

    result = summarize("adaptive", 42, episodes)

    assert result["success_rate"] == 1.0
    assert result["mean_reward"] == 1.0
    assert result["mean_steps"] == pytest.approx(6.0)
    assert result["mean_successful_steps"] == pytest.approx(6.0)


def test_write_csv_writes_header_and_rows(tmp_path):
    episodes = [
        make_episode(True, 5, episode=1, seed=10000),
        make_episode(False, 40, episode=2, seed=10001),
    ]
    path = tmp_path / "episodes.csv"

    write_csv(path, episodes)

    content = path.read_text(encoding="utf-8").splitlines()

    assert len(content) == 3
    assert content[0].startswith("algorithm,training_seed,episode")
    assert "fixed" in content[1]
    assert "10001" in content[2]


def test_write_csv_rejects_empty_results(tmp_path):
    path = tmp_path / "empty.csv"

    with pytest.raises(ValueError, match="No results to write"):
        write_csv(path, [])


def make_temporary_checkpoint(tmp_path):
    torch.manual_seed(42)
    actor = Actor(observation_dim=4, action_dim=4)
    checkpoint_path = tmp_path / "test_actor.pt"

    torch.save({"actor": actor.state_dict()}, checkpoint_path)

    return checkpoint_path


def test_evaluate_checkpoint_runs_expected_episode_count(tmp_path, monkeypatch):
    import src.training.evaluate_sparse7 as evaluator

    monkeypatch.setattr(evaluator, "EVALUATION_EPISODES", 3)

    checkpoint_path = make_temporary_checkpoint(tmp_path)
    episodes = evaluate_checkpoint(checkpoint_path, training_seed=42)

    assert len(episodes) == 3
    assert [row["episode"] for row in episodes] == [1, 2, 3]
    assert [row["evaluation_seed"] for row in episodes] == [10000, 10001, 10002]
    assert all(row["training_seed"] == 42 for row in episodes)
    assert all(row["success"] in (0, 1) for row in episodes)
    assert all(1 <= row["steps"] <= 40 for row in episodes)


def test_evaluate_checkpoint_is_reproducible(tmp_path, monkeypatch):
    import src.training.evaluate_sparse7 as evaluator

    monkeypatch.setattr(evaluator, "EVALUATION_EPISODES", 3)

    checkpoint_path = make_temporary_checkpoint(tmp_path)

    first = evaluate_checkpoint(checkpoint_path, training_seed=42)
    second = evaluate_checkpoint(checkpoint_path, training_seed=42)

    comparable_fields = [
        "episode",
        "evaluation_seed",
        "success",
        "reward",
        "steps",
        "successful_steps",
    ]

    for first_row, second_row in zip(first, second):
        for field in comparable_fields:
            assert first_row[field] == second_row[field]

