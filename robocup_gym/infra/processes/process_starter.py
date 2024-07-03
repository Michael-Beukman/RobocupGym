import pathlib
import subprocess


class ProcessStarter:
    """
    This is a general class that starts a process in the background and keeps track of it.
    """

    def __init__(self, binary: str, arguments: str, pipe_to_devnull=False, logfile_name: str | None = None) -> None:
        self.binary = binary
        self.arguments = arguments
        self.curr_process: subprocess.Popen = None
        self.pipe_to_devnull = pipe_to_devnull
        self.logfile_name = logfile_name

        if self.logfile_name is not None:
            pathlib.Path(self.logfile_name).parent.mkdir(parents=True, exist_ok=True)
            self.logfile = open(self.logfile_name, "w+")
        else:
            self.logfile = None

    def start_process(self):
        """This method starts the binary in a separate process"""
        args = [self.binary] + self.arguments.split(" ")

        kwargs = {}
        if self.pipe_to_devnull:
            kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif self.logfile is not None:
            kwargs = dict(stdout=self.logfile, stderr=self.logfile)
        self.curr_process = subprocess.Popen(args, **kwargs)

    def __del__(self):
        if self.curr_process is not None:
            self.curr_process.send_signal(9)

        if self.logfile is not None:
            self.logfile.close()
