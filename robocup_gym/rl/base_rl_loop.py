import copy
import glob
import os
import shutil
import socket
import sys
from typing import Type

import numpy as np
import stable_baselines3
import torch
from robocup_gym.rl.envs.base_env import BaseEnv
from robocup_gym.infra.utils import get_date
from robocup_gym.rl.utils import MyCheckpointCallback
from robocup_gym.infra.wandb_setup import init_wandb
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from wandb.integration.sb3 import WandbCallback

from robocup_gym.rl.envs.utils.vec_envs import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def merge_keys(dict_keep, dict_default):
    dict_keep = copy.deepcopy(dict_keep)
    for k, v in dict_default.items():
        if k not in dict_keep:
            dict_keep[k] = v
    return dict_keep


def run_experiment(
    SAVE_DIR: str,
    ENV_CLASS: Type[BaseEnv],
    AGENT_CLASS: Type[PPO] = PPO,
    env_kwargs={},
    agent_kwargs={},
    n_steps=1_000_000,
    n_env_procs=1,
    init_checkpoint: str = None,
    force_device: torch.device = None,
    wandb_project_name: str = "Robocup_Kick_2024",
    vec_env_class=SubprocVecEnv,
    start_idx: int = 0,
    make_vec_env_kwargs={},
    wandb_kwargs={},
    eval_callback_env: Type[BaseEnv] = None,
):
    """This is the general RL loop. It takes in the environment and agent classes, and then runs the experiment.
        In particular, if models already exist in the `save_models/{SAVE_DIR}_{HOSTNAME}` directory, it will load the latest one and continue training from there. Otherwise it will create an agent using `AGENT_CLASS` or load it from `init_checkpoint`.

    Args:
        SAVE_DIR (str): This is the directory where the checkpoints will be saved, specifically `save_models/{SAVE_DIR}_{HOSTNAME}`
        ENV_CLASS (Type[BaseEnv]): This is a class representing the environment. Should be a subclass of `BaseEnv`
        AGENT_CLASS (Type[PPO], optional): A classname for the stable baselines 3 agent. Defaults to PPO.
        env_kwargs (dict, optional): These are the kwargs to pass to the `ENV_CLASS` when constructing it. Defaults to {}.
        agent_kwargs (dict, optional): The kwargs to pass to the `AGENT_CLASS` when constructing it. Defaults to {}.
        n_steps (_type_, optional): How long should we train for. Defaults to 1_000_000.
        n_env_procs (int, optional): How many environments must we run in parallel. Defaults to 1.
        init_checkpoint (str, optional): If given, loads the initial model from this checkpoint; otherwise starts it by calling `ENV_CLASS`. Defaults to None.
        force_device (torch.device, optional): If given, forces the model to run on this device. Defaults to None.
        wandb_project_name (str, optional): The name of the wandb project. Defaults to "Robocup_Kick_2023".
        vec_env_class (SubprocVecEnv, optional): The class to use for the vectorized environment. Defaults to SubprocVecEnv.
        start_idx (int, optional): The starting index for the vectorized environment. Defaults to 0.
        make_vec_env_kwargs (dict): Kwargs to pass to the `make_vec_env` function. Defaults to {}.
        wandb_kwargs (dict): Kwargs to pass to the `init_wandb` function. Defaults to {}.
    """

    if force_device is None:
        DEVICE = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        DEVICE = force_device

    HOST = socket.gethostname()
    if "mscluster" in HOST:
        HOST = "mscluster"

    name_prefix = "checkpoint"
    wandb_run = init_wandb(
        exp_name=f"{SAVE_DIR}_{HOST}",
        config={
            "env_kwargs": env_kwargs,
            "env_class": ENV_CLASS.__name__,
            "agent_class": AGENT_CLASS.__name__,
            "agent_kwargs": agent_kwargs,
            "host": HOST,
            "init_checkpoint": init_checkpoint,
        },
        project_name=wandb_project_name,
        **wandb_kwargs,
    )

    date = SAVE_DIR + "_" + get_date()

    env = make_vec_env(
        env_id=ENV_CLASS,
        n_envs=n_env_procs,
        env_kwargs=env_kwargs | {"logger": wandb_run, "date": date},
        vec_env_cls=vec_env_class,
        start_index=start_idx,
        **make_vec_env_kwargs,
    )

    checkpoint_path = os.path.join("save_models", f"{SAVE_DIR}_{HOST}")
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_callback = MyCheckpointCallback(
        save_freq=5_000,
        save_path=checkpoint_path,
        name_prefix=name_prefix,
        counter_start=0,
        env=env,
        is_vecnorm_env=make_vec_env_kwargs.get("wrap_in_vecnormalise", False),
    )

    if os.path.exists(checkpoint_path) and len(glob.glob(os.path.join(checkpoint_path, "*.zip"))) > 0:
        path = sorted(
            [g for g in glob.glob(os.path.join(checkpoint_path, "*_*.zip")) if os.stat(g).st_size > 0],
            key=os.path.getctime,
        )[-1]
        print(f"Loading checkpoint from {path} ")
        model = AGENT_CLASS.load(path, env=env, device=DEVICE)
    elif init_checkpoint is not None:
        model = AGENT_CLASS.load(init_checkpoint, env=env, device=DEVICE)
    else:
        model = AGENT_CLASS("MlpPolicy", env, verbose=2, device=DEVICE, **agent_kwargs)
        model.set_env(env)
    new_logger = configure(os.path.join("tmp", "path", SAVE_DIR), ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    callbacks = ([] if wandb_run is None else [WandbCallback()]) + [checkpoint_callback]
    if eval_callback_env is not None:
        callbacks.append(
            EvalCallback(
                eval_callback_env(),
                best_model_save_path=None,
                log_path=os.path.join("tmp", "path_logs", SAVE_DIR),
                eval_freq=500,
                deterministic=True,
                render=False,
                n_eval_episodes=5,
            )
        )

    model.learn(n_steps, callback=CallbackList(callbacks), reset_num_timesteps=False)
    model.save(os.path.join(checkpoint_path, "final_model"))
    if make_vec_env_kwargs.get("wrap_in_vecnormalise", False):
        env.save(os.path.join(checkpoint_path, "final_vec_env"))
    if wandb_run is not None:
        wandb_run.finish()
