import numpy as np
from robocup_gym.rl.envs.base_env import BaseEnv


class EnvArmUp(BaseEnv):
    """A simple environment where the agent should move its arm up. Mostly provided as a sanity check to see if algorithms work."""

    def _get_reward(self, action: np.ndarray) -> float:
        return -self.python_agent_conn.player.rightShoulderPitch / 180 / 40
