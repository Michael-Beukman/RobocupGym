import os
import time
import datetime
import numpy as np
from typing import List

import torch

from robocup_gym.rl.envs.configs.env_config import NormalisationMode


DIVIDER = "=" * 20


def path(*paths: List[str], mk: bool = True) -> str:
    """Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name

    Returns:
        str:
    """
    dir = os.path.join(*paths)
    if mk:
        if "." in (splits := dir.split(os.sep))[-1]:
            # The final one is a file
            os.makedirs(os.path.join(*splits[:-1]), exist_ok=True)
        else:
            os.makedirs(dir, exist_ok=True)
    return dir


def killall(sleep=1):
    """
    This kills rcssserver3d processes.
    """
    os.system("killall -9 rcssserver3d")
    time.sleep(sleep)


def do_all_seeding(seed: int):
    """Seeds numpy and torch"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_normalisation(norm_mode: str) -> NormalisationMode:
    """norm_mode should be one of:
        "min_max_analytic"
        "min_max_analytic_informed"
        "min_max_empirical"
        "mean_std_empirical"
        "none"
    Args:
        norm_mode (str):

    Returns:
        NormalisationMode:
    """
    return {
        "min_max_analytic": NormalisationMode.MIN_MAX_ANALYTIC,
        "min_max_analytic_informed": NormalisationMode.MIN_MAX_ANALYTIC_INFORMED,
        "min_max_empirical": NormalisationMode.MIN_MAX_EMPIRICAL,
        "mean_std_empirical": NormalisationMode.MEAN_STD_EMPIRICAL,
        "none": NormalisationMode.NONE,
    }[norm_mode]


def get_date() -> str:
    """
    Returns the current date in a nice YYYY-MM-DD_H_m_s format
    Returns:
        str
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_mat(li):
    return np.array(list(map(float, li))).reshape(4, 4).T
