import copy
import socket
from typing import Any, Dict, Tuple

import numpy as np
from robocup_gym.config.vars import (
    DEFAULT_HOST,
    MAX_LENGTH_TO_RECEIVE,
    RCSSSERVER_AGENT_PORT_START,
    RCSSSERVER_MONITOR_PORT_START,
)
from robocup_gym.config.types import Vec3
from robocup_gym.rl.envs.configs.data_names import ACTION_ARMS_LEGS, ACTION_HEAD, ALL_JOINT_OBSERVATIONS
from robocup_gym.rl.envs.configs.data_values import (
    EFFECTORS_TO_JOINT_NAMES,
    JOINT_NAMES_TO_EFFECTORS,
    OBSERVATIONS_MEAN_STD_EMPIRICAL,
    OBSERVATIONS_MIN_MAX,
    OBSERVATIONS_MIN_MAX_EMPIRICAL,
    OBSERVATIONS_MIN_MAX_INFORMED,
)
from robocup_gym.rl.envs.configs.env_config import ActionMode, EnvConfig, NormalisationMode
from robocup_gym.infra.monitor import Monitor
from robocup_gym.infra.player import Player
from robocup_gym.infra.network import get_socket

Socket = socket.socket


class DirectServerConnection:
    """
    This class connects directly to the server and sends messages to it/parses the returns
    """

    def __init__(
        self,
        env_config: EnvConfig,
        host: str = DEFAULT_HOST,
        vectorise_index: int = 0,
    ) -> None:
        """
        Args:
            env_config (EnvConfig): The environment configuration.
            host (str, optional): The host to connect to. Defaults to DEFAULT_HOST, which is 127.0.0.1.
            vectorise_index (int, optional): The index to use for creating a server connection. Defaults to 0.
        """
        self.host = host
        self.monitor_port = RCSSSERVER_MONITOR_PORT_START + vectorise_index
        self.agent_comm_port = RCSSSERVER_AGENT_PORT_START + vectorise_index

        self.env_config = env_config
        self.socket = get_socket(host, self.agent_comm_port)
        self.monitor = Monitor(self.host, self.monitor_port)
        self.vectorise_index = vectorise_index
        self.latest_rl_state = None
        self.latest_received_data = None
        self.player: Player = Player(unum=1)
        self.target_joint_array, self.target_joint_dict = self._setup_targets()

        self.ball_target = {
            "desired_distance": 20,
            "desired_angle": 0,
        }
        self.time = 0
        self.doing_pid = False

    def reset_all(self, player_loc: Vec3, ball_loc: Vec3):
        """Resets the player's joints, its position and the ball pos / server time.

        Args:
            player_loc (Vec3):
            ball_loc (Vec3):
        """
        self._reset_agent(player_loc, ball_loc)
        self._reset_server(ball_loc)

        # Two actions to make sure the ball position has registered.
        for i in range(2):
            self.delay_one_step()

    def delay_one_step(self, full_joints=False):
        """Moves the server one step forward by sending a no-op passthrough message"""
        if full_joints:
            t = ACTION_ARMS_LEGS + ACTION_HEAD
            mess = "".join([f"({s} 0)" for s in t])
            self.send_message(f"{mess}(syn)")
        else:
            self.send_message("(he1 0)(syn)")
        self.recv_message()

    def recv_message(self) -> Dict[str, Any]:
        server_msg = self.socket.recv(MAX_LENGTH_TO_RECEIVE).decode("unicode_escape")
        parsed_state = self._parse_server_message(server_msg)
        data = {
            "server_message": server_msg,
            "observation": self._get_observation(parsed_state),
            "parsed_state": parsed_state,
        }
        self.latest_rl_state = data["observation"]
        self.latest_received_data = data
        return data

    def send_message(self, message: str):
        message = message + "(syn)"
        length = int.to_bytes(len(message), 4, "big")
        fullmessage = length + message.encode()
        self.socket.sendall(fullmessage)

    def send_init_messages(
        self, beam_coords: Tuple[float, float, float] = (0.0, 0.0, 0.0), beam: bool = True, agent_type: int = 0
    ):
        """Sends the initialisation messages to the server.

        Args:
            beam_coords (Tuple[float, float, float], optional): The beam coordinates, (x, y, rotation). Defaults to (0.0, 0.0, 0.0).
        """
        messages = [
            f"(scene rsg/agent/nao/nao_hetero.rsg {agent_type})(syn)",
            "(init (unum 1)(teamname wits_test))(syn)",
            "(syn)",
            "(beam {} {} {})(syn)".format(*beam_coords),
        ]
        if not beam:
            messages = messages[:-1]
        for message in messages:
            self.send_message(message)
            self.recv_message()

    def _get_observation(self, parsed_state) -> Dict[str, Any]:
        data = parsed_state["data"]

        def min_max_norm(val, min, max):
            return (val - min) / (max - min)

        def mean_std_norm(val, mean, std):
            return (val - mean) / std

        main_dict, operation = {
            NormalisationMode.MEAN_STD_EMPIRICAL: (OBSERVATIONS_MEAN_STD_EMPIRICAL, mean_std_norm),
            NormalisationMode.MIN_MAX_EMPIRICAL: (OBSERVATIONS_MIN_MAX_EMPIRICAL, min_max_norm),
            NormalisationMode.MIN_MAX_ANALYTIC: (OBSERVATIONS_MIN_MAX, min_max_norm),
            NormalisationMode.MIN_MAX_ANALYTIC_INFORMED: (OBSERVATIONS_MIN_MAX_INFORMED, min_max_norm),
            NormalisationMode.NONE: (OBSERVATIONS_MIN_MAX, min_max_norm),
        }[self.env_config.options.normalise_mode]

        def v(k):
            mi, ma = main_dict[k]
            return operation(data[k], mi, ma)

        if self.env_config.options.normalise_mode == NormalisationMode.NONE:
            arr = np.array([data[k] for k in self.env_config.observation_names])
        else:
            arr = np.array([v(k) for k in self.env_config.observation_names])

        arr = np.clip(arr, -self.env_config.options.obs_clip, self.env_config.options.obs_clip)
        return arr

    # ======================================= Helpers/Private ======================================= #

    def _reset_agent(self, beam_coords: Vec3, ball_loc: Vec3):
        """Resets the agent's joints to the starting position and moves it to the starting position

        Args:
            beam_coords (Vec3):
        """
        is_pid_successful = self._do_pid()
        self.monitor.send_message("(agent (unum 1)(team Left) (move {} {} {} 270.0))".format(*beam_coords))
        is_pid_successful2 = None
        if not is_pid_successful:
            # do PID again while standing.
            is_pid_successful2 = self._do_pid()
        # And teleport again
        self.monitor.send_message("(agent (unum 1)(team Left) (move {} {} {} 270.0))".format(*beam_coords))
        self.monitor.set_play_mode(self.monitor.PLAY_ON)
        self.delay_one_step(full_joints=True)  # Make the joints zero afterwards

        if not is_pid_successful and not is_pid_successful2:
            print("Both PIDs were bad", self.latest_received_data["parsed_state"]["extra"]["pid_joint_error"])

        self.player.reset()
        return

    def send_passthrough_message(self, message: str):
        self.send_message(message)

    def send_reset_message(self, desired_ball_angle_dist: dict[str, str] = {}):
        self.ball_target = copy.deepcopy(desired_ball_angle_dist)
        self.send_message("(he1 0)")

    def send_array_message(self, array: np.ndarray):
        if self.env_config.action_mode != ActionMode.VELOCITIES:
            player_joint_vector = np.array(
                [
                    self.player.current_dict_state["joints"][EFFECTORS_TO_JOINT_NAMES[k]]
                    for k in self.env_config.action_names
                ]
            )
        if self.env_config.action_mode == ActionMode.DESIRED_ANGLES_MAX_SPEED:
            actions = array[0::2]
            velocities = actions - player_joint_vector * np.pi / 180
            maxs = array[1::2]
            velocities = np.clip(velocities, -maxs, maxs)
        elif self.env_config.action_mode == ActionMode.DESIRED_ANGLES:
            velocities = array - player_joint_vector * np.pi / 180
        elif self.env_config.action_mode == ActionMode.VELOCITIES:
            velocities = array
        else:
            raise ValueError(f"Action mode {self.env_config.action_mode} not supported")

        message = ""
        for i, j in enumerate(self.env_config.action_names):
            message += f"({j} {velocities[i]})"
        return self.send_message(message)

    def send_pid_message(self):
        # Make the PID message
        state = self.latest_received_data["parsed_state"]
        pid_message = ""
        for j in ALL_JOINT_OBSERVATIONS:
            target = self.target_joint_dict[j]
            curr = state["data"][j]
            delta = (target - curr) * 0.1
            eff = JOINT_NAMES_TO_EFFECTORS[j]
            pid_message += f"({eff} {delta})"
        self.send_message(pid_message)

    def _reset_server(self, ball_loc: Vec3):
        """This resets the server, moving the ball to the given location and resetting the time.

        Args:
            ball_loc (Vec3):
        """
        self.monitor.beam_ball(ball_loc)
        self.monitor.reset_time()

    def _do_pid(self) -> bool:
        """Returns whether the PID was successful"""
        nsteps = 0
        max_steps = 2_000
        self.doing_pid = True
        prev = None
        curr = self.player.joint_vector
        allclose_count = 0
        THRESHOLD = 0.01
        while True:
            self.send_pid_message()
            self.recv_message()
            curr = self.player.joint_vector
            nsteps += 1
            if self.latest_received_data["parsed_state"]["extra"]["pid_joint_error"] < THRESHOLD or nsteps > max_steps:
                break
            if prev is not None:
                if np.allclose(prev, curr):
                    allclose_count += 1
                    if allclose_count >= 100:
                        break
                else:
                    allclose_count = 0
            prev = curr
        self.doing_pid = False

        success = self.latest_received_data["parsed_state"]["extra"]["pid_joint_error"] < THRESHOLD
        return success

    def _parse_server_message(self, server_msg: str) -> Dict[str, Any]:
        self.player.update_player_stats(server_msg)
        flat = self.player.get_state_dict()
        flat["desired_rl_kick_distance"] = self.ball_target["desired_distance"]
        flat["desired_rl_kick_angle_degrees"] = self.ball_target["desired_angle"]

        state = {
            "data": flat,
            "extra": {"pid_joint_error": ((self.target_joint_array - self.player.joint_vector) ** 2).sum()},
        }

        return state

    def _setup_targets(self):
        arr = np.array([self.env_config.joint_start_pos[k] for k in ALL_JOINT_OBSERVATIONS])
        dict = {k: self.env_config.joint_start_pos[k] for k in ALL_JOINT_OBSERVATIONS}
        return arr, dict
