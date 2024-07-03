import math
import numpy as np
from robocup_gym.config.types import InfoType, MaybeInfoType, ObservationType, SeedType
from robocup_gym.rl.envs.base_env import BaseEnv
import numpy as np

from robocup_gym.rl.envs.configs.env_config import EnvConfig


class KickStraightVelY(BaseEnv):
    def __init__(
        self,
        env_config: EnvConfig,
        vectorise_index: int,
        wait_steps: int = 20,
        coefficients=[1, 1, 1],
        square_y=False,
        penalise_y_vel_too=False,
        abs_x=True,
        div_reward_by: float = 1,
        **kwargs
    ):
        super().__init__(env_config, vectorise_index, wait_steps, **kwargs)
        self.start_ball_pos_np = np.array(self.env_config.ball_start_pos)
        self.coefficients = coefficients
        self.square_y = square_y
        self.penalise_y_vel_too = penalise_y_vel_too
        self.abs_x = abs_x
        self.div_reward_by = div_reward_by

    def _get_reward(self, action: np.ndarray) -> float:
        if self.has_completed:
            y_pos = self.python_agent_conn.player.real_ball_pos[1]
            CURRENT_BALL_POS = np.array(self.python_agent_conn.player.real_ball_pos)

            dist = CURRENT_BALL_POS[0] - self.latest_ball_pos[0]  # X Dist
            if self.abs_x:
                dist = abs(dist)
            xv, yv, zv = self.python_agent_conn.player.ball_velocity
            vel = (xv**2 + zv**2) ** 0.5  # Also only in X and Z direction

            a, b, c = self.coefficients

            if self.penalise_y_vel_too:
                y_penalty = abs(y_pos) + abs(yv)
            else:
                y_penalty = abs(y_pos)

            pos_reward = a * dist + b * vel
            penalty_y = c * y_penalty ** (2 if self.square_y else 1)
            return pos_reward - penalty_y
        return 0
