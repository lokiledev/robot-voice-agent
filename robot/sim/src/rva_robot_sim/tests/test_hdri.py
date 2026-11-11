import hashlib
from pathlib import Path

import httpx
import pytest

from rva_robot_sim.hdri import HdriConfig, install_hdri

HDR = b"#?RADIANCE\n"


def _polyhaven(digest: str) -> httpx.Client:
    """A fake Poly Haven with one HDRI, in two resolutions."""

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files/artist_workshop":
            files = {
                resolution: {"hdr": {"url": f"https://dl/{resolution}.hdr", "md5": digest}}
                for resolution in ("1k", "2k")
            }
            return httpx.Response(200, json={"hdri": files})
        if request.url.path == "/2k.hdr":
            return httpx.Response(200, content=HDR)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(respond))


def _config(directory: Path) -> HdriConfig:
    return HdriConfig(
        name="artist_workshop",
        resolution="2k",
        brightness=1.0,
        rotation_deg=0.0,
        directory=directory,
    )


def test_downloads_a_checked_hdri_once(tmp_path: Path) -> None:
    with _polyhaven(hashlib.md5(HDR).hexdigest()) as client:
        path = install_hdri(_config(tmp_path), client)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))) as client:
        again = install_hdri(_config(tmp_path), client)

    assert path == again == tmp_path / "artist_workshop_2k.hdr"
    assert path.read_bytes() == HDR


def test_a_corrupt_download_is_refused_and_leaves_nothing(tmp_path: Path) -> None:
    with (
        _polyhaven("0" * 32) as client,
        pytest.raises(RuntimeError, match="does not match Poly Haven's checksum"),
    ):
        install_hdri(_config(tmp_path), client)

    assert list(tmp_path.iterdir()) == []
