from __future__ import annotations

import json
from typing import Dict, Any

import numpy as np
from stable_baselines3 import PPO

from jaipur_rl.envs.jaipur_env import JaipurEnv


def run_episode(env: JaipurEnv, model: PPO | None = None) -> Dict[str, Any]:
    obs, _ = env.reset()
    done = False

    total_reward = 0
    steps = 0

    while not done:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # fallback: random valid action
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        steps += 1
        done = terminated or truncated

    return {
        "reward": float(total_reward),
        "steps": steps,
    }


def evaluate_model(
    model_path: str,
    n_episodes: int = 50,
) -> Dict[str, Any]:
    env = JaipurEnv()
    model = PPO.load(model_path)

    results = [run_episode(env, model) for _ in range(n_episodes)]

    rewards = [r["reward"] for r in results]
    steps = [r["steps"] for r in results]

    return {
        "episodes": n_episodes,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
    }


def evaluate_random_baseline(
    n_episodes: int = 50,
) -> Dict[str, Any]:
    env = JaipurEnv()

    results = [run_episode(env, None) for _ in range(n_episodes)]

    rewards = [r["reward"] for r in results]
    steps = [r["steps"] for r in results]

    return {
        "episodes": n_episodes,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
    }


def save_results(results: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)