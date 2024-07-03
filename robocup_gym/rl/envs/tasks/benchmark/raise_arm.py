import numpy as np
from robocup_gym.rl.envs.base_env import BaseEnv


class EnvRaiseArm(BaseEnv):
    """A very simple environment where the reward incentivises the agent to raise its arm."""

    def _get_reward(self, action: np.ndarray) -> float:
        return -self.python_agent_conn.player.rightShoulderPitch
