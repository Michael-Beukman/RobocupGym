import copy
import multiprocessing as mp
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


class MyAsyncSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns: List[Callable[[], Env]], start_method: str | None = None, percentage_to_skip=0.0):
        super().__init__(env_fns, start_method)
        self.skip_threshold = int(self.num_envs * percentage_to_skip)
        self.percentage_to_skip = percentage_to_skip
        self.minimum_success_envs = int(self.num_envs * (1 - self.percentage_to_skip))
        self.skipped_envs = [False] * self.num_envs
        self.observations = np.zeros((self.num_envs, self.observation_space.shape[-1]))
        # print(f"Will always have {self.minimum_success_envs} envs working | Will skip {self.skip_threshold} envs")

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action, skipped in zip(self.remotes, actions, self.skipped_envs):
            if not skipped:
                remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        self.skipped_envs = [True] * self.num_envs
        successes = []
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        truncateds = np.zeros(self.num_envs, dtype=np.bool_)

        infos = [{} for _ in range(self.num_envs)]
        while True:
            for i, pipe in enumerate(self.remotes):
                if not self.skipped_envs[i]:
                    continue
                if pipe.poll():
                    result = pipe.recv()
                    result = result[:-1]
                    success = True
                    self.skipped_envs[i] = False
                    self.observations[i] = result[0]  # we succeeded so use this.
                    # print("TEST", len(result))
                else:
                    result, success = (None, 0.0, False, {}), True
                    self.skipped_envs[i] = True
                successes.append(success)

                if success:
                    obs, rew, done, info = result

                    rewards[i] = rew
                    dones[i] = done
                    # truncateds[i] = truncated
                    # infos = self._add_info(infos, info, i)
                    infos[i] = info
                    # observations[i] = obs

            if sum(self.skipped_envs) <= self.skip_threshold:
                break

        # print("skipped", sum(self.skipped_envs))
        # results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        return (
            copy.deepcopy(self.observations),
            rewards,
            dones,
            infos,
        )
