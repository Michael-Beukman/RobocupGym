import subprocess
from robocup_gym.config.vars import RCSSSERVER_AGENT_PORT_START
from robocup_gym.infra.processes.bash_process_starter import BashScriptRunner

from robocup_gym.infra.processes.process_starter import ProcessStarter


class NativeAgent(BashScriptRunner):
    """
    This is the class that manages starting up the native agent code (e.g. in C++)
    """

    SERVER_PORT_ARG_NAME = "--port "

    def __init__(self, path_to_bash_script: str, arguments: str, vectorise_index: int, **kwargs) -> None:
        args = arguments + " " + self.SERVER_PORT_ARG_NAME + str(RCSSSERVER_AGENT_PORT_START + vectorise_index)
        super().__init__(path_to_bash_script, args, **kwargs)
