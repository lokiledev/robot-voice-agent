import hashlib
import io
import tarfile
from pathlib import Path

import httpx
import pytest

from rva_ui.media import MediaServerConfig, install_media_server, media_server_command

CONFIG_VERSION = "v1.0.0"


def _release() -> bytes:
    """A release archive like MediaMTX's: the executable and its configuration."""
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name, content in [("mediamtx", b"#!/bin/sh\n"), ("mediamtx.yml", b"rtsp: yes\n")]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(content))
    return archive.getvalue()


def _github(archive: bytes, digest: str) -> httpx.Client:
    """A client of a fake GitHub serving one release, whatever the machine's architecture."""

    def respond(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "checksums.sha256":
            lines = [
                f"{digest} *mediamtx_{CONFIG_VERSION}_linux_{arch}.tar.gz"
                for arch in ("amd64", "arm64")
            ]
            return httpx.Response(200, text="\n".join(lines) + "\n")
        if name.startswith(f"mediamtx_{CONFIG_VERSION}_linux_"):
            return httpx.Response(200, content=archive)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(respond))


def _offline(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"Unexpected download of {request.url}")


def test_downloads_checks_and_unpacks_a_release_once(tmp_path: Path) -> None:
    archive = _release()
    config = MediaServerConfig(version=CONFIG_VERSION, directory=tmp_path)

    with _github(archive, hashlib.sha256(archive).hexdigest()) as client:
        executable = install_media_server(config, client)
    with httpx.Client(transport=httpx.MockTransport(_offline)) as client:
        again = install_media_server(config, client)

    assert executable == tmp_path / CONFIG_VERSION / "mediamtx"
    assert executable.read_bytes() == b"#!/bin/sh\n"
    assert again == executable
    assert media_server_command(executable) == [
        str(executable),
        str(tmp_path / CONFIG_VERSION / "mediamtx.yml"),
    ]


def test_a_corrupt_download_is_refused_and_leaves_nothing(tmp_path: Path) -> None:
    config = MediaServerConfig(version=CONFIG_VERSION, directory=tmp_path)

    with (
        _github(_release(), "0" * 64) as client,
        pytest.raises(RuntimeError, match="does not match the release's checksum"),
    ):
        install_media_server(config, client)

    assert not (tmp_path / CONFIG_VERSION).exists()
