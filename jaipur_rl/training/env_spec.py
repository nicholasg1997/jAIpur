from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np

from jaipur_rl.envs.jaipur_env import JaipurEnv


def _space_size(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    if isinstance(space, gym.spaces.MultiBinary):
        return int(np.prod(space.shape))
    if isinstance(space, gym.spaces.MultiDiscrete):
        return int(np.prod(space.nvec))
    raise TypeError(f"Unsupported space type: {type(space)}")


def flattened_observation_size(env: JaipurEnv | None = None) -> int:
    env = env or JaipurEnv()
    total = 0
    for _, subspace in env.observation_space.spaces.items():
        total += _space_size(subspace)
    return total


def action_branch_summary(env: JaipurEnv | None = None) -> Dict[str, int]:
    env = env or JaipurEnv()
    action_space = env.action_space

    if not isinstance(action_space, gym.spaces.Dict):
        raise TypeError("Expected JaipurEnv.action_space to be gym.spaces.Dict")

    summary: Dict[str, int] = {}
    for name, subspace in action_space.spaces.items():
        if isinstance(subspace, gym.spaces.Discrete):
            summary[name] = int(subspace.n)
        elif isinstance(subspace, gym.spaces.MultiBinary):
            summary[name] = int(2 ** np.prod(subspace.shape))
        else:
            raise TypeError(f"Unsupported action subspace type for '{name}': {type(subspace)}")

    return summary


def total_action_configurations(env: JaipurEnv | None = None) -> int:
    summary = action_branch_summary(env)
    total = 1
    for count in summary.values():
        total *= count
    return total


def build_env_spec(env: JaipurEnv | None = None) -> Dict[str, Any]:
    env = env or JaipurEnv()

    obs_summary: Dict[str, int] = {}
    for name, subspace in env.observation_space.spaces.items():
        obs_summary[name] = _space_size(subspace)

    action_summary = action_branch_summary(env)

    return {
        "observation_features_total": flattened_observation_size(env),
        "observation_features_by_key": obs_summary,
        "action_branches": action_summary,
        "action_configurations_upper_bound": total_action_configurations(env),
    }