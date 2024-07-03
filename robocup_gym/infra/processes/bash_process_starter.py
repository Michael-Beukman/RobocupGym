import subprocess

from robocup_gym.infra.processes.process_starter import ProcessStarter


class BashScriptRunner(ProcessStarter):
    """
    This specifically runs a bash script
    """

    def __init__(self, path_to_bash_script: str, arguments: str, **kwargs) -> None:
        super().__init__("bash", path_to_bash_script + " " + arguments, **kwargs)
