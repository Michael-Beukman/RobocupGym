from dataclasses import asdict, dataclass, field
import enum
import pathlib
from re import L

import numpy as np

from robocup_gym.config.types import Vec3
from robocup_gym.rl.envs.configs.data_values import MAX_VELOCITY, OBSERVATIONS_MIN_MAX


class ActionMode(enum.StrEnum):
    VELOCITIES = "VELOCITIES"  # actions correspond directly to joint velocities.
    DESIRED_ANGLES = "DESIRED_ANGLES"  # actions correspond to desired angles
    DESIRED_ANGLES_MAX_SPEED = (
        "DESIRED_ANGLES_MAX_SPEED"  # actions correspond to desired angles and a max speed for every joint.
    )


class NormalisationMode(enum.StrEnum):
    MIN_MAX_ANALYTIC = "MIN_MAX_ANALYTIC"
    MIN_MAX_ANALYTIC_INFORMED = "MIN_MAX_ANALYTIC_INFORMED"
    MIN_MAX_EMPIRICAL = "MIN_MAX_EMPIRICAL"
    MEAN_STD_EMPIRICAL = "MEAN_STD_EMPIRICAL"
    NONE = "NONE"


@dataclass
class Options:
    frame_stacking: int
    normalise_mode: NormalisationMode
    max_number_of_timesteps: int = 30  # This is when the episode terminates
    obs_clip: float = np.inf


@dataclass
class EnvConfig:
    """Represents an environment configuration, detailing the observation space, action space, etc."""

    # The observations names, corresponding to the strings in the dict that the native agent code sends to Python.
    observation_names: list[str]

    # These are the action names, corresponding to the effector names
    action_names: list[str]

    action_mode: ActionMode

    options: Options

    # Where the joints should start. This should have size of 22 -- one for each joint
    joint_start_pos: dict[str, float]
    # The noise to apply to the above position before every episode. Also size 22.
    joint_pos_noise: dict[str, float]

    observation_noise: tuple[float, float] = (0.0, 0.0)  # mean/std of gaussian noise to add to observations
    action_noise: tuple[float, float] = (0.0, 0.0)  # mean/std of gaussian noise to add to actions

    player_start_pos: Vec3 = (-3.2, 0.03 + 0.0258778, 0.375)
    ball_start_pos: Vec3 = (-3.0, 0.0, 0.044)

    # The noise to apply to the ball position before every episode. This is a radius in the X and Y directions.
    ball_spawn_noise: float = 0.0

    randomise_target: bool = False
    keep_direction_constant: bool = False

    override_normalisations: dict[str, tuple[float, float]] = field(default_factory=dict)  # {}

    def to_json(self) -> dict:
        """This converts the environment config to a JSON serialisable dict.

        Returns:
            dict: The JSON serialisable dict
        """

        return asdict(self)


def get_actions_low_high_from_envconfig(env_config: EnvConfig) -> tuple[np.ndarray, np.ndarray]:
    """This returns the action low and high values from the environment config.

    Args:
        env_config (EnvConfig): The environment config to use

    Returns:
        tuple[np.ndarray, np.ndarray]: The action low and high values
    """
    n_actions = len(env_config.action_names)
    max_val = np.pi  # For desired angles.
    if env_config.action_mode == ActionMode.VELOCITIES:
        max_val = MAX_VELOCITY

    if env_config.action_mode == ActionMode.DESIRED_ANGLES_MAX_SPEED:
        n_actions *= 2  # If we have both desired angles and max speeds, then we have twice as many actions.
    low = np.ones(n_actions, dtype=np.float32) * -max_val
    high = np.ones(n_actions, dtype=np.float32) * max_val

    if env_config.action_mode == ActionMode.DESIRED_ANGLES_MAX_SPEED:
        low[1::2] = 0.0  # make min speed = 0
        high[1::2] = MAX_VELOCITY  # make max speed = MAX_VELOCITY
    return low, high
