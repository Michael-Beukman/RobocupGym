import select

from robocup_gym.config.vars import DEFAULT_HOST
from robocup_gym.infra.network import get_socket
from robocup_gym.config.types import Vec3


class Monitor:
    """This is a class that connects to the monitor port of the server. It can perform actions like beaming the agent."""

    PLAY_ON = "PlayOn"
    BEFORE_KICK_OFF = "BeforeKickOff"
    KICK_OFF_RIGHT = "KickOff_Right"
    KICK_OFF_LEFT = "KickOff_Left"

    def __init__(self, host: str = DEFAULT_HOST, port: int = 3200):
        self.socket = get_socket(host, port)

    # ======================================= API Methods ======================================= #

    def send_message(self, message: str):
        """Sends a message to the server. This prepends the message length to the message before sending.

        Args:
            message (str): _description_
        """
        length = int.to_bytes(len(message), 4, "big")
        fullmessage = length + message.encode()
        self.socket.sendall(fullmessage)

    def set_play_mode(self, playmode: str):
        self.send_message("(playMode %s)" % (playmode))

    def beam_player(self, location: Vec3):
        self.send_message("(agent (unum 1) (team Left) (pos {} {} {}))".format(*location))

    def beam_ball(self, location: Vec3):
        self.send_message("(ball (pos {} {} {}) (vel 0 0 0))".format(*location))

    def reset_time(self):
        self.send_message("(time 0)")
