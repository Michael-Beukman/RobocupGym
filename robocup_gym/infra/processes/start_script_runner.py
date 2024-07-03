import os
from robocup_gym.config.vars import (
    BASH_SCRIPT_LOCATION,
    ROBOCUP_GYM_ROOT,
    RCSSSERVER_AGENT_PORT_START,
    RCSSSERVER_MONITOR_PORT_START,
    RCSSSERVER_SCRIPT_LOCATION,
)
from robocup_gym.infra.processes.bash_process_starter import BashScriptRunner


class StartScriptRunner(BashScriptRunner):
    """
    This runs the start.sh files
    """

    def __init__(self, start_script_path: str, arguments: str = "", vectorise_index: int = 0) -> None:

        server_agent_port = RCSSSERVER_AGENT_PORT_START + vectorise_index
        args = f"{start_script_path} {arguments} --port {server_agent_port}"

        super().__init__(BASH_SCRIPT_LOCATION, args, pipe_to_devnull=True)
