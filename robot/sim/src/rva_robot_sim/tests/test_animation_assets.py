import json
import stat
import sys
from pathlib import Path

import httpx
import pytest

from rva_robot_sim.animation_assets import ClipName, install_clip, install_skin
from rva_robot_sim.people import AvatarsConfig

COMMIT = "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
RAW = f"/microsoft/Microsoft-Rocketbox/{COMMIT}/Assets"
AVATAR = f"{RAW}/Avatars/Adults/Female_Adult_01"
TEXTURES = (
    "/repos/microsoft/Microsoft-Rocketbox/contents/Assets/Avatars/Adults/Female_Adult_01/Textures"
)


def _github() -> httpx.Client:
    """A fake GitHub with one avatar and one clip.

    Of the avatar, only the face FBX and the colour textures are needed.
    """

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"{AVATAR}/Export/Female_Adult_01_facial.fbx":
            return httpx.Response(200, content=b"face fbx")
        if request.url.path == TEXTURES and request.url.params["ref"] == COMMIT:
            names = ["f001_body_color.tga", "f001_body_normal.tga", "f001_opacity_color.tga"]
            return httpx.Response(200, text=json.dumps([{"name": name} for name in names]))
        if request.url.path.startswith(f"{AVATAR}/Textures/"):
            return httpx.Response(200, content=request.url.path.rsplit("/", 1)[-1].encode())
        clip = f"{RAW}/Animations/all_animations_max_motextr_xyz/f_turn_left_90.max.fbx"
        if request.url.path == clip:
            return httpx.Response(200, content=b"clip fbx")
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(respond))


def _offline() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))


def _blender(tmp_path: Path, script: str) -> Path:
    """A stand-in for Blender, running `script` with the arguments given after `--`."""
    path = tmp_path / "blender"
    path.write_text(
        f"#!{sys.executable}\nimport sys\nargs = sys.argv[sys.argv.index('--') + 1 :]\n{script}"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_bakes_the_face_fbx_with_its_colour_textures_once(tmp_path: Path) -> None:
    blender = _blender(
        tmp_path,
        "import os\n"
        "fbx, textures, glb, npz, shapes = args\n"
        "given = open(fbx).read() + ' ' + ' '.join(sorted(os.listdir(textures)))\n"
        "open(glb, 'w').write(given)\n"
        "open(npz, 'w').write(shapes)",
    )
    config = AvatarsConfig(commit=COMMIT, directory=tmp_path / "avatars", blender=blender)

    with _github() as client:
        glb, skin = install_skin(config, "Adults/Female_Adult_01", client)
    with _offline() as client:
        again = install_skin(config, "Adults/Female_Adult_01", client)

    skins = tmp_path / "avatars" / COMMIT / "skins"
    assert (glb, skin) == again == (skins / "Female_Adult_01.glb", skins / "Female_Adult_01.npz")
    assert glb.read_text() == (
        "face fbx Female_Adult_01_facial.fbx f001_body_color.tga f001_opacity_color.tga"
    )
    assert skin.read_text().startswith("AA_VI_00_Sil,AA_VI_01_PP,")
    assert sorted(path.name for path in skins.iterdir()) == [glb.name, skin.name]


def test_a_failed_bake_says_why_and_leaves_nothing(tmp_path: Path) -> None:
    blender = _blender(tmp_path, "print('Error: no armature'); sys.exit(1)")
    config = AvatarsConfig(commit=COMMIT, directory=tmp_path / "avatars", blender=blender)

    with (
        _github() as client,
        pytest.raises(RuntimeError, match="(?s)could not bake .*_facial.fbx.*no armature"),
    ):
        install_skin(config, "Adults/Female_Adult_01", client)

    assert list((tmp_path / "avatars" / COMMIT / "skins").iterdir()) == []


def test_bakes_a_clip_from_its_set_once(tmp_path: Path) -> None:
    blender = _blender(tmp_path, "fbx, npz = args\nopen(npz, 'w').write(open(fbx).read())")
    config = AvatarsConfig(commit=COMMIT, directory=tmp_path / "avatars", blender=blender)
    clip = ClipName.parse("xyz/f_turn_left_90")

    with _github() as client:
        baked = install_clip(config, clip, client)
    with _offline() as client:
        again = install_clip(config, clip, client)

    assert baked == again == tmp_path / "avatars" / COMMIT / "clips" / "xyz" / "f_turn_left_90.npz"
    assert baked.read_text() == "clip fbx"


@pytest.mark.parametrize("text", ["static/f_gestic_talk_neutral_01", "xyz/m_turn_left_90"])
def test_clips_are_named_by_their_set(text: str) -> None:
    clip = ClipName.parse(text)

    assert f"{clip.motion}/{clip.name}" == text


@pytest.mark.parametrize("text", ["f_idle_neutral_01", "walk/f_walk_neutral_01", "static/"])
def test_clip_names_need_a_set(text: str) -> None:
    with pytest.raises(ValueError, match="is not static"):
        ClipName.parse(text)
