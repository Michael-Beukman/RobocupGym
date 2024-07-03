from robocup_gym.rl.envs.base_env import BaseEnv
from robocup_gym.rl.envs.configs.env_config import EnvConfig
from robocup_gym.infra.processes.rcss_server import RCSSServer
import time
import subprocess
from robocup_gym.infra.utils import DIVIDER, killall

killall()

server = RCSSServer()
server.start_process()

time.sleep(4)
proc1 = subprocess.Popen(["ps", "-aux"], stdout=subprocess.PIPE)
proc2 = subprocess.Popen(["grep", "rcssserver3d"], stdin=proc1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
proc3 = subprocess.Popen(["grep", "-v", "grep"], stdin=proc2.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

proc1.stdout.close()  # Allow proc1 to receive a SIGPIPE if proc2 exits.
proc2.stdout.close()  # Allow proc1 to receive a SIGPIPE if proc2 exits.
out, err = proc3.communicate()
print("Received stdout: {0}".format(out.decode()))
print("Received stderr: {0}".format(err.decode()))
print(DIVIDER)
if len(out) == 0 or len(err) != 0:
    print("[Error], RCSSServer Failed to Start")
else:
    print("[Success], RCSSServer Started Successfully")
