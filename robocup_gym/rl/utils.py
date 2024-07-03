import glob
from natsort import natsorted
from stable_baselines3.common.callbacks import BaseCallback
import os
from typing import Type
import numpy as np

from stable_baselines3 import PPO
from robocup_gym.rl.envs.base_env import BaseEnv


class MyCheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``
    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        verbose: int = 0,
        counter_start=0,
        env=None,
        is_vecnorm_env=False,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.counter_start = counter_start
        self.env = env
        self.is_vecnorm_env = is_vecnorm_env

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            nnn = self.num_timesteps
            # if self.n_calls > 50_000: nnn = 'end_ones'
            print(f"Saving {nnn}")
            path = os.path.join(self.save_path, f"{self.name_prefix}_{str(nnn).zfill(10)}_steps")
            self.model.save(path)
            if self.is_vecnorm_env:
                self.env.save(os.path.join(self.save_path, f"env_{self.name_prefix}_{str(nnn).zfill(10)}_steps"))
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


def get_latest_checkpoint(name):
    final = f"save_models/{name}/final_model.zip"
    if os.path.exists(final):
        return final
    return natsorted(glob.glob(f"save_models/{name}/*.zip"))[-1]


def evaluate_agent(
    ENV_CLASS: Type[BaseEnv],
    model_path: str,
    AGENT_CLASS: Type[PPO] = PPO,
    env_kwargs={},
    total_eps=100,
    verbose=True,
    targets=None,
    return_env=False,
    vectorise_index=500,
):

    model = AGENT_CLASS.load(model_path)
    env = ENV_CLASS(vectorise_index=vectorise_index, **env_kwargs)
    # env.target = {"desired_angle": 0, "desired_distance": 10}
    rews = []
    for ep in range(total_eps):
        done = False
        if targets is not None:
            env.target = targets[ep % len(targets)]
        state, _ = env.reset()
        total_rew = 0
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, rew, term, trunc, info = env.step(action)
            done = term or trunc
            total_rew += rew
        if verbose:
            print(f"\t{total_rew}")
        rews.append(total_rew)

    if verbose:
        print("\tMEAN Rew", np.mean(rews))
    if return_env:
        return np.mean(rews), np.std(rews), env
    return np.mean(rews), np.std(rews)
