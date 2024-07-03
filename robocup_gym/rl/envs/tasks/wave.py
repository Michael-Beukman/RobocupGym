import numpy as np
from robocup_gym.rl.envs.base_env import BaseEnv


class EnvArmStraight(BaseEnv):
    def _get_reward(self, action: np.ndarray) -> float:
        X = -np.abs(self.python_agent_conn.player.rightShoulderPitch) - np.abs(
            self.python_agent_conn.player.leftShoulderPitch
        )
        if self.python_agent_conn.player.is_fallen:
            return (X - 1000) / 10_000
        return X / 10_000

    def _is_terminated(self) -> bool:
        return False
