# Copied from stable baselines, slightly modified.
import os
import sys
import time
from typing import Any, Callable, Dict, Optional, Type, Union
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env import VecNormalize


def make_vec_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls=None,  # Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    wrap_in_vecnormalise=False,
    vec_normalise_kwargs={},
):
    # This basically just makes a vec env from a classname or env ID.
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(vectorise_index=rank + 1, **env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        time.sleep(0.1)
        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = SubprocVecEnv
    ans = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)

    if wrap_in_vecnormalise:
        print("Loading a VecNormalize wrapper")
        ans = VecNormalize(ans, **vec_normalise_kwargs)
    return ans
