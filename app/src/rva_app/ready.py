"""Telling the launcher that started this process that it is ready.

Like systemd's `sd_notify`, through a pipe rather than a socket: the launcher passes the pipe's
write end, and its number in `RVA_READY_FD`; the process writes to it once ready, and closes it.
"""

import os
from typing import Final

READY_FD: Final = "RVA_READY_FD"
"""The environment variable with the pipe's file descriptor, set by the launcher."""
READY: Final = b"ready\n"


def notify_ready() -> None:
    """Tell the launcher this process is ready, if a launcher started it; else do nothing.

    Run by hand, a process has nobody to tell. Only the first call writes.
    """
    fd = os.environ.pop(READY_FD, None)
    if fd is None:
        return
    os.write(int(fd), READY)
    os.close(int(fd))
