"""The panorama around the simulated world: an HDRI from Poly Haven, which lights it too."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Final, Literal

import httpx
from loguru import logger

from rva_pydantic import RVABaseModel

_API: Final = "https://api.polyhaven.com/files"


class HdriConfig(RVABaseModel):
    """An HDR panorama from Poly Haven, CC0, downloaded on first use."""

    name: str
    """Poly Haven's id of the HDRI, e.g. `artist_workshop`."""
    resolution: Literal["1k", "2k", "4k"]
    """Width of the panorama, in thousands of pixels."""
    brightness: float
    """Multiplies the light it gives."""
    rotation_deg: float
    """About the vertical, counterclockwise seen from above."""
    directory: Path
    """Where HDRIs are kept, outside the source tree; `~` is expanded."""


def install_hdri(config: HdriConfig, client: httpx.Client) -> Path:
    """The HDRI as a Radiance `.hdr` file, downloaded if missing.

    The download is checked against the MD5 Poly Haven lists for it.

    Raises:
        RuntimeError: If the download does not match Poly Haven's checksum.
        httpx.HTTPError: If a download fails, e.g. for an HDRI that does not exist.
    """
    directory = config.directory.expanduser()
    path = directory / f"{config.name}_{config.resolution}.hdr"
    if path.exists():
        return path
    files = client.get(f"{_API}/{config.name}").raise_for_status().json()
    file = files["hdri"][config.resolution]["hdr"]
    logger.info("Downloading the {} HDRI from Poly Haven", config.name)
    content = client.get(file["url"]).raise_for_status().content
    if hashlib.md5(content).hexdigest() != file["md5"]:
        raise RuntimeError(f"{path.name} does not match Poly Haven's checksum")
    directory.mkdir(parents=True, exist_ok=True)
    # Written aside, then moved in place: an interrupted download leaves no half file.
    descriptor, written = tempfile.mkstemp(dir=directory)
    with os.fdopen(descriptor, "wb") as output:
        output.write(content)
    os.replace(written, path)
    return path
