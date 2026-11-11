import hashlib
import io
import zipfile
from pathlib import Path

import httpx
import pytest

from rva_robot_sim.config import ReachyMiniDescriptionConfig
from rva_robot_sim.description import install_reachy_mini

VERSION = "1.14.0"
WHEEL = f"reachy_mini-{VERSION}-py3-none-any.whl"
MJCF = "reachy_mini/descriptions/reachy_mini/mjcf"


def _wheel() -> bytes:
    """A wheel like reachy-mini's: the SDK, and the MJCF with its meshes."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as wheel:
        wheel.writestr("reachy_mini/__init__.py", "")
        wheel.writestr(f"{MJCF}/reachy_mini.xml", "<mujoco/>")
        wheel.writestr(f"{MJCF}/assets/antenna.stl", b"solid antenna")
    return archive.getvalue()


def _pypi(wheel: bytes, digest: str) -> httpx.Client:
    """A client of a fake PyPI serving one release of reachy-mini."""

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/pypi/reachy-mini/{VERSION}/json":
            files = [
                {"filename": f"reachy_mini-{VERSION}.tar.gz", "url": "https://files/sdist"},
                {
                    "filename": WHEEL,
                    "url": f"https://files/{WHEEL}",
                    "digests": {"sha256": digest},
                },
            ]
            return httpx.Response(200, json={"urls": files})
        if request.url.path == f"/{WHEEL}":
            return httpx.Response(200, content=wheel)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(respond))


def _offline(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"Unexpected download of {request.url}")


def test_unpacks_only_the_mjcf_of_a_checked_wheel_once(tmp_path: Path) -> None:
    wheel = _wheel()
    config = ReachyMiniDescriptionConfig(version=VERSION, directory=tmp_path)

    with _pypi(wheel, hashlib.sha256(wheel).hexdigest()) as client:
        mjcf = install_reachy_mini(config, client)
    with httpx.Client(transport=httpx.MockTransport(_offline)) as client:
        again = install_reachy_mini(config, client)

    assert mjcf == again == tmp_path / VERSION / "reachy_mini.xml"
    assert mjcf.read_text() == "<mujoco/>"
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()) == [
        Path(VERSION) / "assets" / "antenna.stl",
        Path(VERSION) / "reachy_mini.xml",
    ]


def test_a_corrupt_download_is_refused_and_leaves_nothing(tmp_path: Path) -> None:
    config = ReachyMiniDescriptionConfig(version=VERSION, directory=tmp_path)

    with (
        _pypi(_wheel(), "0" * 64) as client,
        pytest.raises(RuntimeError, match="does not match PyPI's digest"),
    ):
        install_reachy_mini(config, client)

    assert list(tmp_path.iterdir()) == []
