import socket


def get_socket(host: str, port: int) -> socket.socket:
    """Returns a connected socket.

    Args:
        host (str):
        port (int):

    Returns:
        socket.socket:
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # print("Getting socket", host, port)
    s.connect((host, port))
    return s
