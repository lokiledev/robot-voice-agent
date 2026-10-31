import hashlib
import io
import os
import platform
import tarfile
import tempfile
from pathlib import Path
from typing import Final

import httpx
from loguru import logger

from rva_pydantic import RVABaseModel

_RELEASES: Final = "https://github.com/bluenviron/mediamtx/releases/download"
_ARCHITECTURES: Final = {"x86_64": "amd64", "aarch64": "arm64"}
"""MediaMTX's name for each machine `platform.machine()` reports."""

MEDIA_SERVER_ENVIRONMENT: Final = {"MTX_RTMP": "no", "MTX_HLS": "no", "MTX_SRT": "no"}
"""Only MediaMTX's RTSP and WebRTC servers are needed; the others would only take ports."""


class MediaServerConfig(RVABaseModel):
    """MediaMTX, which takes the camera stream over RTSP and serves it over WebRTC."""

    version: str
    """Release downloaded from GitHub on first use, e.g. `v1.21.2`."""
    directory: Path
    """Where releases are kept, outside the source tree; `~` is expanded."""


def install_media_server(config: MediaServerConfig, client: httpx.Client) -> Path:
    """MediaMTX's executable, downloaded and checked against the release's checksums if missing.

    Raises:
        RuntimeError: If the machine has no MediaMTX release, or the download is corrupt.
        httpx.HTTPError: If the download fails.
    """
    directory = media_server_directory(config)
    executable = directory / "mediamtx"
    if executable.exists():
        return executable
    archive_name = f"mediamtx_{config.version}_linux_{_architecture()}.tar.gz"
    logger.info("Downloading MediaMTX {} to {}", config.version, directory)
    base = f"{_RELEASES}/{config.version}"
    checksums = client.get(f"{base}/checksums.sha256").raise_for_status().text
    archive = client.get(f"{base}/{archive_name}").raise_for_status().content
    if hashlib.sha256(archive).hexdigest() != _checksum(checksums, archive_name):
        raise RuntimeError(f"{archive_name} does not match the release's checksum")
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Unpacked aside, then moved in place: an interrupted download leaves no half release.
    unpacked = Path(tempfile.mkdtemp(dir=directory.parent))
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(unpacked, filter="data")
    os.replace(unpacked, directory)
    return executable


def media_server_directory(config: MediaServerConfig) -> Path:
    """Where the release of `config` is unpacked."""
    return config.directory.expanduser() / config.version


def media_server_command(executable: Path) -> list[str]:
    """MediaMTX with the configuration its release ships; see `MEDIA_SERVER_ENVIRONMENT`."""
    return [str(executable), str(executable.parent / "mediamtx.yml")]


def _architecture() -> str:
    machine = platform.machine()
    if platform.system() != "Linux" or machine not in _ARCHITECTURES:
        raise RuntimeError(f"No MediaMTX release for {platform.system()} on {machine}")
    return _ARCHITECTURES[machine]


def _checksum(checksums: str, file_name: str) -> str:
    # sha256sum's format: the digest, a space, then `*` before the name of a binary file.
    for line in checksums.splitlines():
        digest, _, name = line.partition(" ")
        if name.removeprefix("*").strip() == file_name:
            return digest
    raise RuntimeError(f"The release's checksums do not list {file_name}")
