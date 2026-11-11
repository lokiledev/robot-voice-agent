"""People standing in the simulated world: Microsoft's Rocketbox avatars, baked to static meshes."""

import math
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Final

import httpx
from loguru import logger

from rva_pydantic import RVABaseModel

_REPOSITORY: Final = "microsoft/Microsoft-Rocketbox"
_AVATARS: Final = PurePosixPath("Assets/Avatars")
_BAKE_TIMEOUT_S: Final = 300
_BAKE: Final = """
import os, sys
import bpy

fbx, textures, out = sys.argv[sys.argv.index("--") + 1 :]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)
for item in bpy.data.objects:
    item.select_set(False)
# The FBX names its textures by the absolute paths of the machine it was made on.
for image in bpy.data.images:
    path = os.path.join(textures, os.path.basename(image.filepath.replace("\\\\", "/")))
    if os.path.exists(path):
        image.filepath = path
        image.reload()
# Only the most detailed of the levels of detail, without its skeleton: a statue in its rest pose.
for item in list(bpy.data.objects):
    if item.type != "MESH" or "hipoly" not in item.name.lower():
        continue
    for modifier in list(item.modifiers):
        item.modifiers.remove(modifier)
    world = item.matrix_world.copy()
    item.parent = None
    item.matrix_world = world
    item.vertex_groups.clear()
    item.select_set(True)
    bpy.context.view_layer.objects.active = item
for item in list(bpy.data.objects):
    if not item.select_get():
        bpy.data.objects.remove(item, do_unlink=True)
# In meters, rather than the centimeters the skeleton scaled the meshes from.
bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
bpy.ops.export_scene.gltf(
    filepath=out, export_format="GLB", export_animations=False, export_image_format="JPEG"
)
"""
"""Run by Blender: the FBX's highest level of detail, textured and in meters, as a static GLB."""


class AvatarsConfig(RVABaseModel):
    """Microsoft's Rocketbox avatars, MIT-licensed, downloaded from GitHub on first use."""

    commit: str
    """Commit of github.com/microsoft/Microsoft-Rocketbox the avatars are taken from."""
    directory: Path
    """Where the baked avatars are kept, outside the source tree; `~` is expanded."""
    blender: Path
    """Blender's executable, which bakes each avatar's FBX into a GLB once."""


class PlacedPerson(RVABaseModel):
    avatar: str
    """A Rocketbox avatar, by its directory under `Assets/Avatars`, e.g. `Adults/Female_Adult_01`."""
    position: tuple[float, float]
    """Where they stand on the floor, in meters, in the robot's frame: it faces +x, +y is its
    left."""
    turned_deg: float
    """How far they are turned from facing the robot, in degrees, counterclockwise seen from
    above; 0 faces it."""


def yaw_deg(position: tuple[float, float], turned_deg: float) -> float:
    """The rotation about the vertical of a baked avatar, which faces -y, placed as given.

    Args:
        position: Where they stand, as `PlacedPerson.position`.
        turned_deg: How far they are turned from facing the robot, as `PlacedPerson.turned_deg`.
    """
    x, y = position
    towards_robot = math.degrees(math.atan2(-y, -x))
    return towards_robot + 90.0 + turned_deg


def install_avatar(config: AvatarsConfig, avatar: str, client: httpx.Client) -> Path:
    """The avatar as a static GLB, downloaded and baked with Blender if missing.

    Only the FBX and its colour and opacity textures are downloaded, about 45 MB; the GLB keeps
    them as JPEG, about 8 MB.

    Raises:
        RuntimeError: If Blender fails to bake it.
        httpx.HTTPError: If a download fails, e.g. for an avatar that does not exist.
    """
    name = PurePosixPath(avatar).name
    directory = config.directory.expanduser() / config.commit
    glb = directory / f"{name}.glb"
    if glb.exists():
        return glb
    logger.info("Downloading the {} avatar and baking it with Blender", avatar)
    root = _AVATARS / avatar
    raw = f"https://raw.githubusercontent.com/{_REPOSITORY}/{config.commit}"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as work:
        source = Path(work)
        fbx = source / f"{name}.fbx"
        fbx.write_bytes(_get(client, f"{raw}/{root}/Export/{name}.fbx"))
        listing = client.get(
            f"https://api.github.com/repos/{_REPOSITORY}/contents/{root}/Textures",
            params={"ref": config.commit},
        )
        for texture in listing.raise_for_status().json():
            if texture["name"].endswith("_color.tga"):
                path = source / texture["name"]
                path.write_bytes(_get(client, f"{raw}/{root}/Textures/{texture['name']}"))
        baked = source / glb.name
        _bake(config.blender, fbx, baked)
        # Moved in place once whole: an interrupted bake leaves no half avatar.
        os.replace(baked, glb)
    return glb


def _get(client: httpx.Client, url: str) -> bytes:
    return client.get(url).raise_for_status().content


def _bake(blender: Path, fbx: Path, glb: Path) -> None:
    command = [str(blender), "--background", "--factory-startup", "--python-expr", _BAKE]
    done = subprocess.run(
        [*command, "--", str(fbx), str(fbx.parent), str(glb)],
        capture_output=True,
        text=True,
        timeout=_BAKE_TIMEOUT_S,
        check=False,
    )
    if done.returncode != 0 or not glb.exists():
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-20:])
        raise RuntimeError(f"Blender could not bake {fbx.name}:\n{tail}")
