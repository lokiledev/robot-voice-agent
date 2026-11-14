"""Rocketbox avatars that can be animated, and Rocketbox clips to animate them, baked by Blender."""

import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Final, Literal

import httpx
from loguru import logger

from rva_pydantic import RVABaseModel
from rva_robot_sim.people import AvatarsConfig

_REPOSITORY: Final = "microsoft/Microsoft-Rocketbox"
_AVATARS: Final = PurePosixPath("Assets/Avatars")
_ANIMATIONS: Final = PurePosixPath("Assets/Animations")
_BAKE_TIMEOUT_S: Final = 300
SHAPES: Final = (
    "AA_VI_00_Sil",
    "AA_VI_01_PP",
    "AA_VI_02_FF",
    "AA_VI_03_TH",
    "AA_VI_04_DD",
    "AA_VI_05_KK",
    "AA_VI_06_CH",
    "AA_VI_07_SS",
    "AA_VI_08_nn",
    "AA_VI_09_RR",
    "AA_VI_10_aa",
    "AA_VI_11_E",
    "AA_VI_12_I",
    "AA_VI_13_O",
    "AA_VI_14_U",
    "AK_09_EyeBlinkLeft",
    "AK_10_EyeBlinkRight",
    "AK_25_JawOpen",
)
"""Face shapes kept with each skin: the 15 visemes of Oculus Lipsync, ARKit's blinks and jaw."""

_BAKE_SKIN: Final = """
import os, sys
import bpy
import numpy as np

fbx, textures, glb, npz, shapes = sys.argv[sys.argv.index("--") + 1 :]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)
# The FBX names its textures by the absolute paths of the machine it was made on.
for image in bpy.data.images:
    path = os.path.join(textures, os.path.basename(image.filepath.replace("\\\\", "/")))
    if os.path.exists(path):
        image.filepath = path
        image.reload()
armature = next(item for item in bpy.data.objects if item.type == "ARMATURE")
mesh = next(o for o in bpy.data.objects if o.type == "MESH" and "hipoly" in o.name.lower())


def rigid(matrix):
    location, rotation, _ = matrix.decompose()
    out = np.eye(4)
    out[:3, :3] = np.array(rotation.to_matrix())
    out[:3, 3] = location
    return out


# Parents before children, as forward kinematics walks them.
ordered = []
stack = [bone for bone in armature.data.bones if bone.parent is None][::-1]
while stack:
    bone = stack.pop()
    ordered.append(bone)
    stack.extend(list(bone.children)[::-1])
names = [bone.name for bone in ordered]
to_world = mesh.matrix_world.copy()
linear = np.array(to_world.to_3x3())
basis = np.array([vertex.co for vertex in mesh.data.vertices])
group_bone = {g.index: names.index(g.name) for g in mesh.vertex_groups if g.name in names}
bone_index = np.zeros((len(basis), 4), dtype=np.int32)
bone_weight = np.zeros((len(basis), 4))
for vertex in mesh.data.vertices:
    weights = sorted(
        ((g.weight, group_bone[g.group]) for g in vertex.groups if g.group in group_bone),
        reverse=True,
    )[:4]
    total = sum(weight for weight, _ in weights)
    assert total > 0, f"vertex {vertex.index} has no bone"
    for slot, (weight, bone) in enumerate(weights):
        bone_index[vertex.index, slot] = bone
        bone_weight[vertex.index, slot] = weight / total
keys = mesh.data.shape_keys.key_blocks
shape_names = shapes.split(",")
np.savez(
    npz,
    bone_names=np.array(names),
    bone_parents=np.array([names.index(b.parent.name) if b.parent else -1 for b in ordered]),
    bone_rest=np.stack([rigid(armature.matrix_world @ bone.matrix_local) for bone in ordered]),
    positions=basis @ linear.T + np.array(to_world.translation),
    bone_index=bone_index,
    bone_weight=bone_weight,
    shape_names=np.array(shape_names),
    shape_deltas=np.stack(
        [(np.array([p.co for p in keys[name].data]) - basis) @ linear.T for name in shape_names]
    ),
)
# The mesh alone, in its rest pose and in meters, without its shapes: the sim moves its vertices.
bpy.context.view_layer.objects.active = mesh
bpy.ops.object.shape_key_remove(all=True, apply_mix=False)
for modifier in list(mesh.modifiers):
    mesh.modifiers.remove(modifier)
mesh.parent = None
mesh.matrix_world = to_world
mesh.vertex_groups.clear()
for item in list(bpy.data.objects):
    if item != mesh:
        bpy.data.objects.remove(item, do_unlink=True)
mesh.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.export_scene.gltf(
    filepath=glb,
    export_format="GLB",
    export_animations=False,
    export_skins=False,
    export_morph=False,
    export_image_format="JPEG",
)
"""
"""Run by Blender: the face FBX's mesh as a static GLB, and its skin and face shapes as arrays."""

_BAKE_CLIP: Final = """
import sys
import bpy
import numpy as np

fbx, npz = sys.argv[sys.argv.index("--") + 1 :]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)
armature = next(item for item in bpy.data.objects if item.type == "ARMATURE")
first, last = (round(frame) for frame in armature.animation_data.action.frame_range)
scene = bpy.context.scene
bones = list(armature.pose.bones)
root = next(bone for bone in bones if bone.parent is None)
rotations = np.zeros((last - first + 1, len(bones), 4))
root_position = np.zeros((last - first + 1, 3))
for row, frame in enumerate(range(first, last + 1)):
    scene.frame_set(frame)
    for column, bone in enumerate(bones):
        if bone.parent:
            rotations[row, column] = (bone.parent.matrix.inverted() @ bone.matrix).to_quaternion()
        else:
            # The armature is the Biped's Bip01, which carries the motion over the floor.
            location, rotation, _ = (armature.matrix_world @ bone.matrix).decompose()
            rotations[row, column] = rotation
            root_position[row] = location
location, _, _ = (armature.matrix_world @ root.bone.matrix_local).decompose()
np.savez(
    npz,
    bone_names=np.array([bone.name for bone in bones]),
    rotations=rotations,
    root_position=root_position,
    rest_root_height=location.z,
    fps=scene.render.fps / scene.render.fps_base,
)
"""
"""Run by Blender: each bone's rotation in its parent, frame by frame; the root's in the world."""

type RootMotion = Literal["static", "xy", "xyz"]


class ClipName(RVABaseModel):
    """One of Rocketbox's animations."""

    motion: RootMotion
    """How its root moves: `static` in place (idling, talking); `xy` over the floor (walks, runs);
    `xyz` in space too (turns, sitting down)."""
    name: str
    """The FBX's name without `.max.fbx`, e.g. `f_idle_neutral_01`; `f_` for women, `m_` for
    men."""

    @classmethod
    def parse(cls, text: str) -> "ClipName":
        """From `motion/name`, e.g. `static/f_idle_neutral_01`.

        Raises:
            ValueError: If `text` is not of that form.
        """
        motion, _, name = text.partition("/")
        match motion:
            case "static" | "xy" | "xyz" if name:
                return cls(motion=motion, name=name)
            case _:
                raise ValueError(f"{text!r} is not static/<name>, xy/<name> or xyz/<name>")


def install_skin(config: AvatarsConfig, avatar: str, client: httpx.Client) -> tuple[Path, Path]:
    """The avatar's face FBX as a static GLB and its skin, downloaded and baked if missing.

    Args:
        config: Where the avatars come from and are kept.
        avatar: Its directory under `Assets/Avatars`, e.g. `Adults/Female_Adult_01`.
        client: To download it.

    Returns:
        The GLB, which Genesis loads, and the skin's arrays, which `skinning.load_skin` reads.

    Raises:
        RuntimeError: If Blender fails to bake it.
        httpx.HTTPError: If a download fails, e.g. for an avatar that does not exist.
    """
    name = PurePosixPath(avatar).name
    directory = config.directory.expanduser() / config.commit / "skins"
    glb, npz = directory / f"{name}.glb", directory / f"{name}.npz"
    if glb.exists() and npz.exists():
        return glb, npz
    logger.info("Downloading the {} avatar's face FBX and baking its skin with Blender", avatar)
    root = _AVATARS / avatar
    raw = f"https://raw.githubusercontent.com/{_REPOSITORY}/{config.commit}"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as work:
        source = Path(work)
        fbx = source / f"{name}_facial.fbx"
        fbx.write_bytes(_get(client, f"{raw}/{root}/Export/{fbx.name}"))
        listing = client.get(
            f"https://api.github.com/repos/{_REPOSITORY}/contents/{root}/Textures",
            params={"ref": config.commit},
        )
        for texture in listing.raise_for_status().json():
            if texture["name"].endswith("_color.tga"):
                path = source / texture["name"]
                path.write_bytes(_get(client, f"{raw}/{root}/Textures/{texture['name']}"))
        baked_glb, baked_npz = source / glb.name, source / npz.name
        _bake(config.blender, _BAKE_SKIN, (fbx, source, baked_glb, baked_npz, ",".join(SHAPES)))
        # Moved in place once whole: an interrupted bake leaves no half avatar.
        os.replace(baked_npz, npz)
        os.replace(baked_glb, glb)
    return glb, npz


def install_clip(config: AvatarsConfig, clip: ClipName, client: httpx.Client) -> Path:
    """The clip's bone rotations, which `skinning.load_clip` reads, downloaded and baked if missing.

    Raises:
        RuntimeError: If Blender fails to bake it.
        httpx.HTTPError: If the download fails, e.g. for a clip that does not exist.
    """
    directory = config.directory.expanduser() / config.commit / "clips" / clip.motion
    npz = directory / f"{clip.name}.npz"
    if npz.exists():
        return npz
    logger.info("Downloading the {}/{} clip and baking it with Blender", clip.motion, clip.name)
    url = (
        f"https://raw.githubusercontent.com/{_REPOSITORY}/{config.commit}/{_ANIMATIONS}"
        f"/all_animations_max_motextr_{clip.motion}/{clip.name}.max.fbx"
    )
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as work:
        fbx = Path(work) / f"{clip.name}.fbx"
        fbx.write_bytes(_get(client, url))
        baked = Path(work) / npz.name
        _bake(config.blender, _BAKE_CLIP, (fbx, baked))
        os.replace(baked, npz)
    return npz


def _get(client: httpx.Client, url: str) -> bytes:
    return client.get(url).raise_for_status().content


def _bake(blender: Path, script: str, arguments: tuple[Path | str, ...]) -> None:
    command = [str(blender), "--background", "--factory-startup", "--python-exit-code", "1"]
    done = subprocess.run(
        [*command, "--python-expr", script, "--", *(str(argument) for argument in arguments)],
        capture_output=True,
        text=True,
        timeout=_BAKE_TIMEOUT_S,
        check=False,
    )
    if done.returncode != 0:
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-20:])
        raise RuntimeError(f"Blender could not bake {arguments[0]}:\n{tail}")
