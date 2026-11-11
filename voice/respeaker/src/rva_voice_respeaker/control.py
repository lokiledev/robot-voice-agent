"""The XVF3800's control protocol, as Seeed's `xvf_host.py` speaks it.

Each read is a vendor request to the device on USB endpoint 0: the command's id with bit 7 set
as `wValue`, its resource's id as `wIndex`. The reply starts with a status byte, then the values.
"""

import struct
import time
from typing import Final, NamedTuple

import usb.core
import usb.util

VENDOR_ID: Final = 0x2886
PRODUCT_ID: Final = 0x001A
UDEV_RULE: Final = "config/udev/70-respeaker-xvf3800.rules"

_READ: Final = usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE
_MAX_ATTEMPTS: Final = 100
_TIMEOUT_MS: Final = 1000


class FloatCommand(NamedTuple):
    """A read-only command whose reply is `count` little-endian 32-bit floats."""

    resource: int
    command: int
    count: int


BEAM_AZIMUTHS: Final = FloatCommand(resource=33, command=75, count=4)
"""`AEC_AZIMUTH_VALUES`: the azimuth of each beam in radians: the two focused beams, the
free-running one that scans, and the one selected automatically."""
BEAM_SPEECH_ENERGIES: Final = FloatCommand(resource=33, command=80, count=4)
"""`AEC_SPENERGY_VALUES`: the speech energy of each beam, in the same order; above 0 is speech."""


class ReSpeakerUnavailableError(Exception):
    """The array is not on USB, or this user may not talk to it."""


class XvfControl:
    """Reads the XVF3800's parameters over USB. Each call blocks: run it in a thread."""

    def __init__(self) -> None:
        """Find the array on USB and check it answers.

        Raises:
            ReSpeakerUnavailableError: If it is not plugged in, or this user may not talk to it.
        """
        device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        if device is None:
            raise ReSpeakerUnavailableError(
                f"No reSpeaker XVF3800 ({VENDOR_ID:04x}:{PRODUCT_ID:04x}) on USB"
            )
        self._device: Final = device
        try:
            self.read(BEAM_AZIMUTHS)
        except usb.core.USBError as error:
            self.close()
            raise ReSpeakerUnavailableError(
                f"Cannot talk to the reSpeaker XVF3800 over USB ({error}); without root, install"
                f" {UDEV_RULE} as its first lines say"
            ) from error

    def close(self) -> None:
        usb.util.dispose_resources(self._device)

    def read(self, command: FloatCommand) -> tuple[float, ...]:
        length = 1 + 4 * command.count
        for _ in range(_MAX_ATTEMPTS):
            reply = self._device.ctrl_transfer(
                _READ, 0, 0x80 | command.command, command.resource, length, _TIMEOUT_MS
            )
            values = parse_floats(bytes(reply), command.count)
            if values is not None:
                return values
            time.sleep(0.001)
        raise ReSpeakerUnavailableError(f"The reSpeaker kept {command} waiting")


def parse_floats(reply: bytes, count: int) -> tuple[float, ...] | None:
    """The values of a read's `reply`; `None` when the chip asks to be asked again.

    Raises:
        ReSpeakerUnavailableError: On any other status.
    """
    match reply[0]:
        case 0:
            return tuple(
                float(value) for value in struct.unpack(f"<{count}f", reply[1 : 1 + 4 * count])
            )
        case 64:
            # The chip has not answered yet.
            return None
        case status:
            raise ReSpeakerUnavailableError(f"The reSpeaker answered with status {status}")
