import copy
import numpy as np
from robocup_gym.rl.envs.configs.data_names import (
    ACTION_ARMS_LEGS,
    OBSERVATION_ALL_BALL_DIST_ANGLE,
    OBSERVATION_ALL_BALL_DIST_ANGLE_NO_TARGET,
    OBSERVATION_JOINTS_SENSORS_FOOT_MINIMAL_TIMESTEPS_VELOCITIES_CORRECT_BALL_ORIENTATION,
    OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS,
    OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES,
    OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES_CORRECT_BALL_ORIENTATION,
    OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES_NO_BALL,
)
from robocup_gym.rl.envs.configs.data_values import (
    SMALL_JOINT_VARIANCE_5_DEGREES,
    STANDING_JOINT_ANGLES,
    ZERO_JOINT_ANGLES,
)
from robocup_gym.rl.envs.configs.env_config import ActionMode, EnvConfig, NormalisationMode, Options
from robocup_gym.infra.utils import get_normalisation


# This is our default environment configuration. It uses all joints (their velocities too) -- except the head ones -- the ball pos and velocity, foot resistance, accelerometer, gyro and the timestep value.
DefaultConfig = EnvConfig(
    observation_names=OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS,
    action_names=ACTION_ARMS_LEGS,
    action_mode=ActionMode.DESIRED_ANGLES,
    options=Options(frame_stacking=4, normalise_mode=NormalisationMode.MIN_MAX_ANALYTIC),
    joint_start_pos=STANDING_JOINT_ANGLES,
    joint_pos_noise=ZERO_JOINT_ANGLES,
)


# Default Conf, with more velocities
DefaultConfigVelocities = EnvConfig(
    observation_names=OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES,
    action_names=ACTION_ARMS_LEGS,
    action_mode=ActionMode.DESIRED_ANGLES,
    options=Options(frame_stacking=4, normalise_mode=NormalisationMode.MIN_MAX_ANALYTIC),
    joint_start_pos=STANDING_JOINT_ANGLES,
    joint_pos_noise=ZERO_JOINT_ANGLES,
)

# Default Conf, with more velocities
GoodConfigVelocities = EnvConfig(
    observation_names=OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES,
    action_names=ACTION_ARMS_LEGS,
    action_mode=ActionMode.VELOCITIES,
    options=Options(frame_stacking=1, normalise_mode=NormalisationMode.MIN_MAX_ANALYTIC, max_number_of_timesteps=20),
    joint_start_pos=STANDING_JOINT_ANGLES,
    joint_pos_noise=ZERO_JOINT_ANGLES,
)

GoodConfigVelocitiesNoBall = EnvConfig(
    observation_names=OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES_NO_BALL,
    action_names=ACTION_ARMS_LEGS,
    action_mode=ActionMode.VELOCITIES,
    options=Options(frame_stacking=1, normalise_mode=NormalisationMode.MIN_MAX_ANALYTIC, max_number_of_timesteps=20),
    joint_start_pos=STANDING_JOINT_ANGLES,
    joint_pos_noise=ZERO_JOINT_ANGLES,
)


# Default Conf, with more velocities, and noisy values
NoisyConfigVelocities = EnvConfig(
    observation_names=OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES,
    action_names=ACTION_ARMS_LEGS,
    action_mode=ActionMode.VELOCITIES,
    options=Options(frame_stacking=1, normalise_mode=NormalisationMode.MIN_MAX_ANALYTIC, max_number_of_timesteps=20),
    joint_start_pos=STANDING_JOINT_ANGLES,
    joint_pos_noise=SMALL_JOINT_VARIANCE_5_DEGREES,
    ball_spawn_noise=0.05,
)

GoodConfigVelocitiesOrientationBall = EnvConfig(
    observation_names=OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES_CORRECT_BALL_ORIENTATION,
    action_names=ACTION_ARMS_LEGS,
    action_mode=ActionMode.VELOCITIES,
    options=Options(frame_stacking=1, normalise_mode=NormalisationMode.MIN_MAX_ANALYTIC, max_number_of_timesteps=20),
    joint_start_pos=STANDING_JOINT_ANGLES,
    joint_pos_noise=ZERO_JOINT_ANGLES,
)

GoodConfigVelocitiesTargets = EnvConfig(
    observation_names=OBSERVATION_ALL_BALL_DIST_ANGLE,
    action_names=ACTION_ARMS_LEGS,
    action_mode=ActionMode.VELOCITIES,
    options=Options(
        frame_stacking=1, normalise_mode=NormalisationMode.MIN_MAX_ANALYTIC, max_number_of_timesteps=20, obs_clip=np.inf
    ),
    joint_start_pos=STANDING_JOINT_ANGLES,
    joint_pos_noise=ZERO_JOINT_ANGLES,
)

GoodConfigVelocities = EnvConfig(
    observation_names=OBSERVATION_ALL_BALL_DIST_ANGLE_NO_TARGET,
    action_names=ACTION_ARMS_LEGS,
    action_mode=ActionMode.VELOCITIES,
    options=Options(
        frame_stacking=1, normalise_mode=NormalisationMode.MIN_MAX_ANALYTIC, max_number_of_timesteps=20, obs_clip=np.inf
    ),
    joint_start_pos=STANDING_JOINT_ANGLES,
    joint_pos_noise=ZERO_JOINT_ANGLES,
)

GoodConfigVelocitiesMinimal = EnvConfig(
    observation_names=OBSERVATION_JOINTS_SENSORS_FOOT_MINIMAL_TIMESTEPS_VELOCITIES_CORRECT_BALL_ORIENTATION,
    action_names=ACTION_ARMS_LEGS,
    action_mode=ActionMode.VELOCITIES,
    options=Options(
        frame_stacking=1, normalise_mode=NormalisationMode.MIN_MAX_ANALYTIC, max_number_of_timesteps=20, obs_clip=np.inf
    ),
    joint_start_pos=STANDING_JOINT_ANGLES,
    joint_pos_noise=ZERO_JOINT_ANGLES,
)


def create_good_minimal_config(
    timesteps: int, clip_value: float, norm_mode: str, noise_a=0.0, noise_o=0.0
) -> EnvConfig:
    conf = copy.deepcopy(GoodConfigVelocitiesMinimal)
    conf.action_noise = (0.0, noise_a)
    conf.observation_noise = (0.0, noise_o)
    conf.options.max_number_of_timesteps = timesteps
    conf.options.obs_clip = clip_value if clip_value is not None else np.inf

    conf.options.normalise_mode = get_normalisation(norm_mode)
    return conf
