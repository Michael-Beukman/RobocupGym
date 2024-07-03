import numpy as np
from robocup_gym.rl.envs.base_env import BaseEnv
import numpy as np


class EnvSimpleKick(BaseEnv):
    """A very simple environment where the reward incentivises the agent to kick the ball."""

    def _get_reward(self, action: np.ndarray) -> float:
        if self.has_completed:
            CURRENT_BALL_POS = np.array(self.python_agent_conn.player.real_ball_pos)
            start_ball_pos = np.array(self.env_config.ball_start_pos)
            return np.linalg.norm(CURRENT_BALL_POS - start_ball_pos)
        return 0
