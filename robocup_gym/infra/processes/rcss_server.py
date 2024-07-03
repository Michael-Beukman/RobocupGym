import os
from robocup_gym.config.vars import (
    ROBOCUP_GYM_ROOT,
    LOG_DIR,
    RCSSSERVER_AGENT_PORT_START,
    RCSSSERVER_MONITOR_PORT_START,
    RCSSSERVER_SCRIPT_LOCATION,
)
from robocup_gym.infra.processes.bash_process_starter import BashScriptRunner
from robocup_gym.infra.utils import get_date


class RCSSServer(BashScriptRunner):
    """
    This starts up the rcssserver process.
    """

    def __init__(self, arguments: str = "", vectorise_index: int = 0) -> None:

        server_agent_port = RCSSSERVER_AGENT_PORT_START + vectorise_index
        server_monitor_port = RCSSSERVER_MONITOR_PORT_START + vectorise_index
        args = f"{arguments} --agent-port {server_agent_port} --server-port {server_monitor_port}"
        super().__init__(RCSSSERVER_SCRIPT_LOCATION, args, pipe_to_devnull=True)
        # , logfile_name=os.path.join(LOG_DIR, f"simspark",f"simspark_{get_date()}_{server_agent_port}.log"))
        # , logfile_name=os.path.join(LOG_DIR, f"simspark",f"simspark_{get_date()}_{server_agent_port}.log"))
