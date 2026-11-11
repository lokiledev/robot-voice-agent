import json
import stat
import sys
from pathlib import Path

import httpx
import pytest

from rva_robot_sim.people import AvatarsConfig, PlacedPerson, install_avatar, yaw_deg

COMMIT = "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
RAW = f"/microsoft/Microsoft-Rocketbox/{COMMIT}/Assets/Avatars/Adults/Female_Adult_01"
TEXTURES = (
    "/repos/microsoft/Microsoft-Rocketbox/contents/Assets/Avatars/Adults/Female_Adult_01/Textures"
)


@pytest.mark.parametrize(
    ("position", "turned_deg", "yaw"),
    [
        # Baked avatars face -y: in front of the robot, facing back along -x, is -90 degrees.
        ((1.0, 0.0), 0.0, 270.0),
        ((0.0, 1.0), 0.0, 0.0),
        ((1.0, 0.0), 45.0, 315.0),
    ],
)
def test_people_face_the_robot_unless_turned(
    position: tuple[float, float], turned_deg: float, yaw: float
) -> None:
    person = PlacedPerson(avatar="Adults/Female_Adult_01", position=position, turned_deg=turned_deg)

    assert yaw_deg(person) % 360 == pytest.approx(yaw)


def _github() -> httpx.Client:
    """A fake GitHub with one avatar: its FBX, and textures of which only the colours are needed."""

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"{RAW}/Export/Female_Adult_01.fbx":
            return httpx.Response(200, content=b"fbx")
        if request.url.path == TEXTURES and request.url.params["ref"] == COMMIT:
            names = ["f001_body_color.tga", "f001_body_normal.tga", "f001_opacity_color.tga"]
            return httpx.Response(200, text=json.dumps([{"name": name} for name in names]))
        if request.url.path.startswith(f"{RAW}/Textures/"):
            return httpx.Response(200, content=request.url.path.rsplit("/", 1)[-1].encode())
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(respond))


def _blender(tmp_path: Path, script: str) -> Path:
    """A stand-in for Blender, running `script` with the FBX, the textures and the GLB's paths."""
    path = tmp_path / "blender"
    path.write_text(f"#!{sys.executable}\nimport sys\nfbx, textures, glb = sys.argv[-3:]\n{script}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_bakes_the_fbx_with_its_colour_textures_once(tmp_path: Path) -> None:
    blender = _blender(
        tmp_path,
        "import os\n"
        "given = open(fbx).read() + ' ' + ' '.join(sorted(os.listdir(textures)))\n"
        "open(glb, 'w').write(given)",
    )
    config = AvatarsConfig(commit=COMMIT, directory=tmp_path / "avatars", blender=blender)

    with _github() as client:
        glb = install_avatar(config, "Adults/Female_Adult_01", client)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))) as client:
        again = install_avatar(config, "Adults/Female_Adult_01", client)

    assert glb == again == tmp_path / "avatars" / COMMIT / "Female_Adult_01.glb"
    assert glb.read_text() == "fbx Female_Adult_01.fbx f001_body_color.tga f001_opacity_color.tga"
    assert [path.name for path in glb.parent.iterdir()] == ["Female_Adult_01.glb"]


def test_a_failed_bake_says_why_and_leaves_nothing(tmp_path: Path) -> None:
    blender = _blender(tmp_path, "print('Error: no armature'); sys.exit(1)")
    config = AvatarsConfig(commit=COMMIT, directory=tmp_path / "avatars", blender=blender)

    with (
        _github() as client,
        pytest.raises(RuntimeError, match="(?s)could not bake Female_Adult_01.fbx.*no armature"),
    ):
        install_avatar(config, "Adults/Female_Adult_01", client)

    assert list((tmp_path / "avatars" / COMMIT).iterdir()) == []
