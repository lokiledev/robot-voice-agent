import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Final

import httpx
from loguru import logger

from rva_robot_sim.config import ReachyMiniDescriptionConfig

_PYPI: Final = "https://pypi.org/pypi/reachy-mini"
_MJCF: Final = PurePosixPath("reachy_mini/descriptions/reachy_mini/mjcf")
"""Where the wheel keeps the MJCF and its meshes."""
MJCF_FILE: Final = "reachy_mini.xml"


def install_reachy_mini(config: ReachyMiniDescriptionConfig, client: httpx.Client) -> Path:
    """The MJCF of Reachy Mini, unpacked from its wheel on PyPI if missing.

    Only the MJCF and its meshes are kept: the package itself, Pollen's SDK and daemon, is not
    installed.

    Raises:
        RuntimeError: If the release has no wheel, or the download does not match PyPI's digest.
        httpx.HTTPError: If the download fails.
    """
    directory = config.directory.expanduser() / config.version
    mjcf = directory / MJCF_FILE
    if mjcf.exists():
        return mjcf
    release = client.get(f"{_PYPI}/{config.version}/json").raise_for_status().json()
    wheels = [file for file in release["urls"] if file["filename"].endswith("-none-any.whl")]
    if not wheels:
        raise RuntimeError(f"reachy-mini {config.version} has no pure Python wheel on PyPI")
    wheel = wheels[0]
    logger.info("Downloading the Reachy Mini description from {}", wheel["filename"])
    content = client.get(wheel["url"]).raise_for_status().content
    if hashlib.sha256(content).hexdigest() != wheel["digests"]["sha256"]:
        raise RuntimeError(f"{wheel['filename']} does not match PyPI's digest")
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Unpacked aside, then moved in place: an interrupted download leaves no half description.
    unpacked = Path(tempfile.mkdtemp(dir=directory.parent))
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or not path.is_relative_to(_MJCF):
                continue
            if ".." in path.parts:
                raise RuntimeError(f"{wheel['filename']} has a path out of its tree: {path}")
            target = unpacked / path.relative_to(_MJCF)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    if not (unpacked / MJCF_FILE).exists():
        raise RuntimeError(f"{wheel['filename']} has no {_MJCF / MJCF_FILE}")
    os.replace(unpacked, directory)
    return mjcf
