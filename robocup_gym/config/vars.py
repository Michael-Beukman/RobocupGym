import os

DIR = os.path.dirname(os.path.abspath(__file__))
ROBOCUP_GYM_ROOT = os.path.abspath(os.path.join(DIR, "..", ".."))

# These are the ports we use. Generally, ports above 10_000 are rarely used by other programs.
RCSSSERVER_AGENT_PORT_START = 10_000
RCSSSERVER_MONITOR_PORT_START = 20_000
PYTHON_NATIVE_AGENT_PORT_START = 30_000

DEFAULT_HOST = "127.0.0.1"
MAX_LENGTH_TO_RECEIVE = 2**20

RCSSSERVER_SCRIPT_LOCATION = os.path.join(ROBOCUP_GYM_ROOT, "robocup_gym", "scripts", "rcssserver_singularity.sh")
BASH_SCRIPT_LOCATION = os.path.join(ROBOCUP_GYM_ROOT, "robocup_gym", "scripts", "bash_singularity.sh")


# Uncomment these if you do not use singularity
# RCSSSERVER_SCRIPT_LOCATION = os.path.join(KUDU_GYM_ROOT, "scripts", "rcssserver.sh")


EXPORT_ENV_DIRECTORY = os.path.join(ROBOCUP_GYM_ROOT, "robocup_gym", "rl_env_configs")


CONDA_ENV_NAME = "kudu_gym"
SLURM_DIR = os.path.join("artifacts", "slurms")
LOG_DIR = os.path.join("artifacts", "logs")
