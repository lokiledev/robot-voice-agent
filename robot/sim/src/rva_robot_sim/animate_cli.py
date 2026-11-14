"""An experiment: one Rocketbox avatar playing Rocketbox clips in Genesis, filmed to two videos."""

import logging
import math
import time
from pathlib import Path
from typing import Annotated, Final, Literal

import cv2
import cyclopts
import genesis as gs
import gs_nyx.nyx_py_renderer as npr
import httpx
import numpy as np
import numpy.typing as npt
from genesis.constants import backend
from genesis.utils.misc import tensor_to_array
from genesis.vis.camera import Camera
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
from loguru import logger

from rva_pydantic import RVABaseModel, load_yaml
from rva_robot_sim import nyx_skinned
from rva_robot_sim.animation_assets import ClipName, install_clip, install_skin
from rva_robot_sim.config import NyxRenderer, SimConfig
from rva_robot_sim.hdri import install_hdri
from rva_robot_sim.scene import environment_map
from rva_robot_sim.skinning import (
    Floats,
    Skin,
    deform,
    displacement,
    load_clip,
    load_skin,
    placement,
    pose,
    retarget,
    sample,
)

_FPS: Final = 30
"""Frames a second, of the clips, the videos and the simulation's steps."""
_STILL: Final = frozenset({"Bip01 REye", "Bip01 LEye", "Bip01 MJaw"})
"""Bones of the face the clips would move from their own rest pose, which is not the avatar's: the
eyes and the jaw keep the avatar's, and the face shapes move them."""
_BLINK_PERIOD_S: Final = 3.7
_BLINK_S: Final = 0.16
_SYLLABLE_HZ: Final = 4.5
_LIGHTS: Final = (
    # Key: warm, high, in front on the camera's left. The avatar faces -y at rest, the camera.
    gs.options.vis.DirectionalLight(dir=(1.5, 2.0, -2.5), color=(1.0, 0.92, 0.82), intensity=4.0),
    # Fill: cool and soft, low on the other side, so the shadowed half of the face still reads.
    gs.options.vis.DirectionalLight(dir=(-2.0, 2.0, -1.2), color=(0.75, 0.85, 1.0), intensity=1.6),
    # Rim: from behind and above, outlining the hair and shoulders against the backdrop.
    gs.options.vis.DirectionalLight(dir=(-0.5, -3.0, -2.5), color=(1.0, 1.0, 1.0), intensity=3.0),
)
_AMBIENT: Final = (0.22, 0.22, 0.25)
_BACKDROP: Final = (0.58, 0.62, 0.68)
"""A light grey-blue, like a studio's paper backdrop."""
_NYX_LIGHTS: Final = (
    # The rasterizer's three, in lux, over the sim's panorama, which lights the rest.
    {"type": "directional", "dir": (1.5, 2.0, -2.5), "color": (1.0, 0.92, 0.82), "intensity": 3.0},
    {
        "type": "directional",
        "dir": (-2.0, 2.0, -1.2),
        "color": (0.75, 0.85, 1.0),
        "intensity": 1.0,
        "shadow": False,
    },
    {
        "type": "directional",
        "dir": (-0.5, -3.0, -2.5),
        "color": (1.0, 1.0, 1.0),
        "intensity": 2.0,
        "shadow": False,
    },
)


class _View(RVABaseModel):
    res: tuple[int, int]
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fov: float
    """Vertical, in degrees."""


_BODY: Final = _View(res=(640, 720), pos=(0.0, -3.2, 1.1), lookat=(0.0, 0.0, 0.9), fov=40.0)
"""From 3 m in front: the avatar faces -y at rest."""
_FACE: Final = _View(res=(480, 480), pos=(0.0, -0.6, 1.6), lookat=(0.0, 0.0, 1.6), fov=30.0)
"""A close-up, which follows the face."""

type RendererName = Literal["rasterizer", "nyx"]

app = cyclopts.App(
    name="animate",
    help="Play Rocketbox clips on a Rocketbox avatar in Genesis, filmed by the rasterizer or Nyx.",
)


@app.default
def main(
    *,
    profile: Annotated[Path, cyclopts.Parameter(help="Robot profile directory.")] = Path(
        "config/reachy_mini_sim"
    ),
    avatar: str = "Adults/Female_Adult_01",
    clips: tuple[str, ...] = (
        "static/f_idle_neutral_01",
        "static/f_gestic_talk_neutral_01",
        "xyz/f_turn_left_90",
        "static/f_idle_look_around_01",
    ),
    seconds_each: float = 6.0,
    renderer: RendererName = "rasterizer",
    output: Path = Path("animation"),
    viewer: bool = False,
) -> None:
    """Play the clips one after the other, the mouth moving in the talking ones, blinking.

    Writes `<output>_body.mp4`, from 3 m in front, and `<output>_face.mp4`, a close-up that follows
    the face. The avatar and the clips are downloaded and baked with Blender on first use.

    Args:
        profile: Robot profile directory, whose `sim.yaml` says where avatars come from.
        avatar: A Rocketbox avatar, by its directory under `Assets/Avatars`.
        clips: Rocketbox clips, as `static/<name>`, `xy/<name>` or `xyz/<name>`.
        seconds_each: At most, of each clip.
        renderer: `nyx` path traces with `sim.yaml`'s Nyx settings and panorama, through a patch of
            the plugin (`nyx_skinned`); the rasterizer is flat but needs no ray-tracing GPU.
        output: Prefix of the videos' paths.
        viewer: Open Genesis' viewer too.
    """
    config = load_yaml(SimConfig, profile / "sim.yaml")
    names = [ClipName.parse(clip) for clip in clips]
    nyx = _nyx_settings(config, profile) if renderer == "nyx" else None
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        glb, skin_path = install_skin(config.avatars, avatar, client)
        clip_paths = [install_clip(config.avatars, name, client) for name in names]
        hdri = install_hdri(nyx.environment, client) if nyx is not None else None
    skin = load_skin(skin_path)
    motions = [retarget(skin, load_clip(path), _STILL) for path in clip_paths]

    gs.init(backend=backend.cpu, logging_level="warning")
    logging.getLogger("genesis").addFilter(_without_get_trimesh)
    scene = gs.Scene(
        show_viewer=viewer,
        sim_options=gs.options.SimOptions(dt=1 / _FPS),
        vis_options=gs.options.VisOptions(
            background_color=_BACKDROP, ambient_light=_AMBIENT, lights=list(_LIGHTS)
        ),
    )
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(color=(0.42, 0.4, 0.37)))
    person = scene.add_entity(
        gs.morphs.Mesh(file=str(glb), enable_custom_vverts=True, fixed=True, collision=False)
    )
    if nyx is None:
        film: _RasterizerFilm | _NyxFilm = _RasterizerFilm(scene, output)
    else:
        assert hdri is not None, "installed for Nyx"
        nyx_skinned.install()
        film = _NyxFilm(scene, output, nyx, hdri)
    scene.build()
    order = _vertex_order(skin, tensor_to_array(person.get_vverts()))

    head = skin.bone_names.index("Bip01 Head")
    weights = np.zeros(len(skin.shape_names))
    blinks = [skin.shape_index("AK_09_EyeBlinkLeft"), skin.shape_index("AK_10_EyeBlinkRight")]
    mouth = skin.shape_index("AA_VI_10_aa")
    film.start()
    anchor = placement(0.0, 0.0, 0.0)
    elapsed, costs = 0.0, []
    for name, motion in zip(names, motions, strict=True):
        length = min(motion.duration, seconds_each)
        logger.info("Playing {}/{} for {:.1f} s", name.motion, name.name, length)
        for frame in range(round(length * _FPS)):
            started = time.perf_counter()
            rotations, root = sample(motion, frame / _FPS)
            bones = pose(skin, rotations, root, anchor)
            weights[blinks] = _blink(elapsed)
            weights[mouth] = _syllables(elapsed) if "_talk" in name.name else 0.0
            person.set_vverts(deform(skin, bones, weights)[order])
            costs.append(time.perf_counter() - started)
            film.aim_close_up(*_close_up(bones[head], anchor))
            scene.step()
            film.capture()
            elapsed += 1 / _FPS
        anchor = anchor @ displacement(skin, motion, length)
    film.stop()
    logger.info(
        "Posed and skinned in {:.2f} ms a frame on average, {:.2f} at most",
        1000 * float(np.mean(costs)),
        1000 * float(np.max(costs)),
    )


class _RasterizerFilm:
    """Genesis' cameras, which record as the scene steps."""

    def __init__(self, scene: gs.Scene, output: Path) -> None:
        self._output: Final = output
        self._body: Final = _rasterizer_camera(scene, _BODY)
        self._face: Final = _rasterizer_camera(scene, _FACE)

    def start(self) -> None:
        self._body.start_recording(save_to_filename=f"{self._output}_body.mp4", fps=_FPS)
        self._face.start_recording(save_to_filename=f"{self._output}_face.mp4", fps=_FPS)

    def aim_close_up(self, pos: Floats, lookat: Floats) -> None:
        self._face.set_pose(pos=pos, lookat=lookat)

    def capture(self) -> None:
        """Nothing: the cameras rendered as the scene stepped."""

    def stop(self) -> None:
        self._body.stop_recording()
        self._face.stop_recording()


class _NyxFilm:
    """Nyx's cameras, read after each step and written by OpenCV."""

    def __init__(self, scene: gs.Scene, output: Path, nyx: NyxRenderer, hdri: Path) -> None:
        self._output: Final = output
        sky = environment_map(nyx, hdri)
        self._cameras: Final = [
            scene.add_sensor(
                NyxCameraOptions(
                    res=view.res,
                    pos=view.pos,
                    lookat=view.lookat,
                    fov=view.fov,
                    spp=nyx.samples_per_pixel,
                    render_mode=npr.ERenderMode.FastPathTracer,
                    env_maps=(sky,),
                    lights=list(_NYX_LIGHTS),
                )
            )
            for view in (_BODY, _FACE)
        ]
        self._writers: list[cv2.VideoWriter] = []

    def start(self) -> None:
        codec = cv2.VideoWriter.fourcc(*"mp4v")
        self._writers[:] = [
            cv2.VideoWriter(f"{self._output}_{name}.mp4", codec, _FPS, view.res)
            for name, view in (("body", _BODY), ("face", _FACE))
        ]

    def aim_close_up(self, pos: Floats, lookat: Floats) -> None:
        self._cameras[1].update_camera_pose(pos=tuple(pos), lookat=tuple(lookat))

    def capture(self) -> None:
        for camera, writer in zip(self._cameras, self._writers, strict=True):
            rgb = tensor_to_array(camera.read().rgb)
            writer.write(np.ascontiguousarray(rgb[..., ::-1]))

    def stop(self) -> None:
        for writer in self._writers:
            writer.release()


def _rasterizer_camera(scene: gs.Scene, view: _View) -> Camera:
    camera = scene.add_camera(res=view.res, pos=view.pos, lookat=view.lookat, fov=view.fov)
    assert isinstance(camera, Camera), "Genesis' rasterizer camera"
    return camera


def _nyx_settings(config: SimConfig, profile: Path) -> NyxRenderer:
    match config.camera_renderer:
        case NyxRenderer() as nyx:
            return nyx
        case _:
            raise SystemExit(f"{profile / 'sim.yaml'} renders with the rasterizer, not Nyx")


def _vertex_order(skin: Skin, loaded: Floats) -> npt.NDArray[np.intp]:
    """For each vertex Genesis loaded, the skin's vertex at the same place.

    Genesis splits vertices where the texture's seams are, and may reorder them: copies of a vertex
    share its rest position, and so its bones and weights.

    Raises:
        ValueError: If a loaded vertex is not one of the skin's.
    """
    nearest = np.empty(len(loaded), dtype=np.intp)
    distance = np.empty(len(loaded))
    for start in range(0, len(loaded), 1024):
        chunk = loaded[start : start + 1024]
        squared = ((chunk[:, None, :] - skin.positions[None, :, :]) ** 2).sum(axis=-1)
        nearest[start : start + 1024] = squared.argmin(axis=1)
        distance[start : start + 1024] = np.sqrt(squared.min(axis=1))
    logger.info(
        "Genesis loaded {} vertices of the skin's {}, at most {:.2g} m from them",
        len(loaded),
        len(skin.positions),
        distance.max(),
    )
    if distance.max() > 1e-4:
        raise ValueError(f"A vertex Genesis loaded is {distance.max():.3g} m from the skin's")
    return nearest


def _blink(elapsed: float) -> float:
    phase = elapsed % _BLINK_PERIOD_S
    return max(0.0, 1.0 - abs(phase - _BLINK_S / 2) / (_BLINK_S / 2))


def _syllables(elapsed: float) -> float:
    """The mouth's opening while talking: syllables, louder and softer over a phrase."""
    syllable = math.pow(max(0.0, math.sin(2 * math.pi * _SYLLABLE_HZ * elapsed)), 0.7)
    phrase = 0.6 + 0.4 * math.sin(2 * math.pi * 0.37 * elapsed)
    return 0.8 * syllable * phrase


def _close_up(head: Floats, anchor: Floats) -> tuple[Floats, Floats]:
    """Where the close-up looks from and at: the face, from 0.6 m in front of the avatar."""
    forward = anchor[:3, :3] @ np.array((0.0, -1.0, 0.0))
    target = head[:3, 3] + np.array((0.0, 0.0, 0.07))
    return target + 0.6 * forward, target


def _without_get_trimesh(record: logging.LogRecord) -> bool:
    return "'get_trimesh' is deprecated" not in record.getMessage()
