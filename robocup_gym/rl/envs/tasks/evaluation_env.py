import math
import numpy as np
from robocup_gym.rl.envs.base_env import BaseEnv
import numpy as np

from robocup_gym.rl.envs.configs.env_config import EnvConfig


class EnvKickEvaluation(BaseEnv):
    def __init__(self, env_config: EnvConfig, vectorise_index: int, wait_steps: int = 50, do_only_x=False, **kwargs):
        super().__init__(env_config, vectorise_index, wait_steps, **kwargs)
        self.start_ball_pos_np = np.array(self.env_config.ball_start_pos)
        self.do_only_x = do_only_x
        self.positions = []

    def _get_reward(self, action: np.ndarray) -> float:
        if self.has_completed:
            CURRENT_BALL_POS = np.array(self.python_agent_conn.player.real_ball_pos)
            self.positions.append(CURRENT_BALL_POS)
            if self.do_only_x:
                return CURRENT_BALL_POS[0] - self.start_ball_pos_np[0]
            return np.linalg.norm(CURRENT_BALL_POS - self.start_ball_pos_np)
        return 0

    def get_x_y_z_pos(self):
        pos = np.array(self.positions)

        xs = pos[:, 0] - self.start_ball_pos_np[0]
        ys = pos[:, 1] - self.start_ball_pos_np[1]
        zs = pos[:, 2] - self.start_ball_pos_np[2]

        # return mean and standard for each:
        return (np.mean(xs), np.std(xs)), (np.mean(ys), np.std(ys)), (np.mean(zs), np.std(zs))
