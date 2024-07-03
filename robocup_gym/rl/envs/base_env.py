import json
import os
import time
from typing import Any

import gymnasium as gym
import numpy as np
from robocup_gym.config.types import ActionType, InfoType, MaybeInfoType, ObservationType, SeedType, Vec3
from robocup_gym.config.vars import EXPORT_ENV_DIRECTORY, ROBOCUP_GYM_ROOT, LOG_DIR, PYTHON_NATIVE_AGENT_PORT_START
from robocup_gym.rl.envs.configs.data_values import MAX_VELOCITY
from robocup_gym.rl.envs.configs.env_config import ActionMode, EnvConfig, get_actions_low_high_from_envconfig
from robocup_gym.infra.processes.rcss_server import RCSSServer
from robocup_gym.infra.direct_server_connection import DirectServerConnection


class BaseEnv(gym.Env):
    """
    This is the base environment responsible for connecting to the server. It follows the [Gymnasium](gymnasium.farama.org) interface, in that it provides `env.step(action)` and `env.reset()` methods. These methods perform all the necessary communication with the server, to actually perform the action and get the new state. Generally they do not need to be overridden.

    This class should be subclassed to implement the `_get_reward` and `_is_terminated` methods.
    By default, the episode terminates when the agent falls down. The reward can be arbitrary.



    Args:
        env_config (EnvConfig): The main configuration object for the environment.
        vectorise_index (int): Two `BaseEnvs` cannot both run simultaneously with the same vectorised_index, as they both use the same ports. This allows one to circumvent this by using different ports. Additionally, when running a vectorised environment, this index increments by one for each environment.
        wait_steps (int, optional): This is the number of steps we should wait for after the episode has ended but before we call self._get_reward() for the final timestep. Defaults to 0.
        sleep_time_after_proc_starts (int, optional): How many seconds to sleep after starting processes. Defaults to 2.
        wait_steps_after_pid (int, optional): This waits for this many steps after doing PID to stabilise. Defaults to 20.
        logger: A wandb logger
        make_action_scale_1 (bool, optional): Whether to scale the actions to be between -1 and 1. Defaults to True.
        make_joints_zero_after_start (bool, optional): Whether to make the joints zero after starting. Defaults to True.
        div_reward_by (float, optional): This divides the reward by this number. Defaults to 1.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        vectorise_index: int,
        wait_steps: int = 0,
        sleep_time_after_proc_starts: int = 2,
        agent_type: int = 0,
        init_target_distance: float = 20.0,
        logger: Any = None,
        make_action_scale_1: bool = True,
        make_joints_zero_after_start=True,
        wait_steps_after_pid: int = 20,
        div_reward_by: float = 1,
        date: str = "None",
    ):

        # Assign Variables
        self.wait_steps_after_done = wait_steps
        self.step_counter = 0
        self.vectorise_index = vectorise_index
        self.env_config = env_config
        self.make_action_scale_1 = make_action_scale_1
        self.div_reward_by = div_reward_by

        self._setup_obs_action_spaces()

        # These manage the NativeAgent and RCSSServer processes.
        self.rcssserver_conn = RCSSServer(vectorise_index=vectorise_index)

        # Start the server, and then the agent
        self.rcssserver_conn.start_process()

        time.sleep(sleep_time_after_proc_starts)

        # Now open the server connection that connects to the server
        self.python_agent_conn = DirectServerConnection(env_config, vectorise_index=vectorise_index)
        self.python_agent_conn.send_init_messages(agent_type=agent_type)

        self.latest_ball_pos = env_config.ball_start_pos
        self.target = {
            "desired_distance": init_target_distance,
            "desired_angle": 0,
        }

        self.logger = logger
        self.to_log_at_end = {}
        self.obs_scales = {k: (0, 0) for k in env_config.observation_names}
        self.init_obs = {k: (0) for k in env_config.observation_names}
        self.make_joints_zero_after_start = make_joints_zero_after_start
        self.wait_steps_after_pid = wait_steps_after_pid
        self.reset_count = 0

    # ======================================= Standard API Methods ======================================= #

    def step(self, action: ActionType) -> tuple[ObservationType, float, bool, bool, dict]:
        """Performs an environment step. This takes in the action from the agent and then returns the new state, whether we are done, and the info dict.

        Args:
            action (ActionType):

        Returns:
            tuple[ObservationType, float, bool, bool, dict]: state, reward, terminated, truncated, info
        """

        if self.make_action_scale_1:
            # Rescale the actions because the agent outputs actions between -1 and 1.
            action = action * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2

        if self.env_config.action_noise[1] != 0.0:
            action = action + np.random.normal(
                loc=self.env_config.action_noise[0], scale=self.env_config.action_noise[1], size=action.shape
            )

        self.python_agent_conn.send_array_message(action)
        # Receive the new state from the server
        self.python_agent_conn.recv_message()
        old_state = self._get_state().copy()
        self.step_counter += 1

        # Are we done
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        done = terminated or truncated

        for idx, k in enumerate(self.env_config.observation_names):
            self.obs_scales[k] = (
                min(self.obs_scales[k][0], old_state[idx]),
                max(self.obs_scales[k][1], old_state[idx]),
            )

        if done:
            # We already have the state, now just wait
            self.has_completed = True

            for i in range(self.wait_steps_after_done):
                # Send a zero velocity action so that the agent's joints come to a stop.
                self.python_agent_conn.delay_one_step(full_joints=self.make_joints_zero_after_start and i == 0)

            self._get_state()
            # Update the state as the one after the last rl action step, not after the long waiting time.
            self.state = old_state

        # But make the reward the one after the long waiting time, e.g. after the ball has come to a stop.
        reward = self._get_reward(action)
        if done:
            self._log_positions()
        info = {}
        return old_state, reward / self.div_reward_by, terminated, truncated, info

    def reset(self, seed: SeedType = None, options: MaybeInfoType = None) -> tuple[ObservationType, InfoType]:
        """This resets the environment and returns a state.

        Returns:
            np.ndarray:
        """
        self.to_log_at_end = {}
        self.has_completed = False
        self.python_agent_conn.reset_all(
            player_loc=self.env_config.player_start_pos,
            ball_loc=self._get_noisy_ball_pos(),
        )

        self.step_counter = 0

        self.python_agent_conn.send_reset_message(desired_ball_angle_dist=self._get_new_target_pos())
        self.python_agent_conn.recv_message()

        for _ in range(self.wait_steps_after_pid):
            self.python_agent_conn.delay_one_step()

        out = self._get_state(), {}

        self.init_obs = {k: out[0][idx] for idx, k in enumerate(self.env_config.observation_names)}
        self.reset_count += 1
        return out

    # ======================================= Helper Methods ======================================= #

    def _setup_obs_action_spaces(self):
        # This creates the correct observation and action spaces.
        n_obs = len(self.env_config.observation_names) * self.env_config.options.frame_stacking
        low = np.zeros(n_obs, dtype=np.float32)
        high = np.ones(n_obs, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high)

        action_low, action_high = get_actions_low_high_from_envconfig(self.env_config)

        self.action_low = action_low
        self.action_high = action_high

        if self.make_action_scale_1:
            ones = np.ones_like(action_low)
            self.action_space = gym.spaces.Box(low=-ones, high=ones)
        else:
            self.action_space = gym.spaces.Box(low=action_low, high=action_high)

    def _get_state(self) -> ObservationType:
        # This just sets the state.
        self.state = self.python_agent_conn.latest_rl_state
        if self.env_config.observation_noise[1] != 0.0:
            self.state += np.random.normal(
                loc=self.env_config.observation_noise[0],
                scale=self.env_config.observation_noise[1],
                size=self.state.shape,
            )
        return self.state

    def _is_truncated(self) -> bool:
        """This returns True if the episode has reached the maximum number of timesteps."""
        return self.step_counter >= self.env_config.options.max_number_of_timesteps

    def _get_noisy_ball_pos(self) -> Vec3:
        """This returns the noisy ball position. This is used to spawn the ball in a random position.

        Returns:
            Vec3: The new ball pos
        """
        og_pos = self.env_config.ball_start_pos
        if self.env_config.ball_spawn_noise == 0.0:
            self.latest_ball_pos = og_pos
            return og_pos

        random_angle = np.random.rand() * 2 * np.pi
        a = np.cos(random_angle)
        b = np.sin(random_angle)
        r = self.env_config.ball_spawn_noise

        self.latest_ball_pos = (og_pos[0] + a * r, og_pos[1] + b * r, og_pos[2])
        return self.latest_ball_pos

    def _get_new_target_pos(self) -> dict[str, str]:
        if self.env_config.randomise_target:
            self.target = {
                "desired_distance": np.random.uniform(3, 10),
                "desired_angle": np.random.uniform(-45, 45) if not self.env_config.keep_direction_constant else 0,
            }
        return self.target

    # ======================================= Logging Methods ======================================= #

    def _log(self, dic: dict[str, Any]):
        if self.logger is not None and self.vectorise_index == 1:
            self.logger.log(dic)

    def _log_positions(self):
        self._log(
            {
                "ballpos": {
                    "x": self.python_agent_conn.player.real_ball_pos[0],
                    "y": self.python_agent_conn.player.real_ball_pos[1],
                    "z": self.python_agent_conn.player.real_ball_pos[2],
                },
                "ballvel": {
                    "x": self.python_agent_conn.player.ball_velocity[0],
                    "y": self.python_agent_conn.player.ball_velocity[1],
                    "z": self.python_agent_conn.player.ball_velocity[2],
                },
                "ballacc": {
                    "x": self.python_agent_conn.player.ball_acceleration[0],
                    "y": self.python_agent_conn.player.ball_acceleration[1],
                    "z": self.python_agent_conn.player.ball_acceleration[2],
                },
                "obs_scales": {
                    "min": {k: v[0] for k, v in self.obs_scales.items()},
                    "max": {k: v[1] for k, v in self.obs_scales.items()},
                    "init": {k: v for k, v in self.init_obs.items()},
                },
            }
            | self.to_log_at_end
        )
        pass

    # ======================================= Methods to be overridden ======================================= #

    def _get_reward(self, action: ActionType) -> float:
        """This should return the reward for the current action. Can use `self.has_completed` to determine if we are done (i.e. at the end of the episode)"""
        raise NotImplementedError()

    def _is_terminated(self) -> bool:
        """This returns True if the environment has reached a terminal state. By default, if the player has fallen

        Returns:
            bool:
        """
        return self.python_agent_conn.player.is_fallen
