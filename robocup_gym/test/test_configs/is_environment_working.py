from robocup_gym.rl.envs.base_env import BaseEnv
from robocup_gym.rl.envs.configs.default import DefaultConfig, GoodConfigVelocitiesMinimal
from robocup_gym.rl.envs.configs.env_config import EnvConfig
from robocup_gym.infra.processes.rcss_server import RCSSServer
import time
import os

from robocup_gym.infra.utils import DIVIDER, killall


X = BaseEnv(GoodConfigVelocitiesMinimal, 1)
state = X.reset()


time.sleep(2)
killall()
print(DIVIDER)
print("[Success] The environment seems to work")
