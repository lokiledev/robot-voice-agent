import math
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from rva_app.config import RuntimeConfig, load_admins
from rva_app.doa import RobotDoaConfig
from rva_interaction.bearings import speech_bearing
from rva_pydantic import load_yaml
from rva_robot_sim.config import SimConfig
from rva_schemas.frames import StaticTransforms
from rva_schemas.profile import Admins
from rva_schemas.sound import HeardDirection
from rva_schemas.vision import CameraIntrinsics
from rva_vision_faces.config import KeptFaces, RobotSource, SessionFaces

_REPOSITORY = Path(__file__).parents[4]


def test_admins_are_read_from_their_file(tmp_path: Path) -> None:
    path = tmp_path / "admins.yaml"
    path.write_text("user_ids:\n  - 00000000-0000-0000-0000-000000000001\n")

    assert load_admins(path) == Admins(user_ids=frozenset({UUID(int=1)}))


def test_without_an_admins_file_there_are_no_admins(tmp_path: Path) -> None:
    assert load_admins(tmp_path / "admins.yaml") == Admins(user_ids=frozenset())


def test_the_admins_template_lists_nobody() -> None:
    template = _REPOSITORY / "config" / "admins.example.yaml"

    assert load_admins(template) == Admins(user_ids=frozenset())


def test_the_default_profile_keeps_its_data_outside_the_source_tree() -> None:
    config = RuntimeConfig.load(_REPOSITORY / "config" / "default")

    assert config.vision.database == KeptFaces(
        path=Path("~/.local/share/robot-voice-agent/faces.sqlite3")
    )
    assert [
        config.admins,
        config.media_server.directory,
        config.observability.captures_dir,
        config.tts.model_path,
        config.tts.voices_path,
    ] == [
        Path("~/.config/robot-voice-agent/admins.yaml"),
        Path("~/.local/share/robot-voice-agent/mediamtx"),
        Path("~/Captures/robot-voice-agent"),
        Path("~/.cache/robot-voice-agent/kokoro/kokoro-v1.0.onnx"),
        Path("~/.cache/robot-voice-agent/kokoro/voices-v1.0.bin"),
    ]


def test_the_camera_intrinsics_are_found_next_to_the_profile_and_marked_assumed() -> None:
    profile = _REPOSITORY / "config" / "default"
    config = RuntimeConfig.load(profile)

    assert config.vision.camera.intrinsics == profile / "cameras" / "c920.yaml"
    intrinsics = load_yaml(CameraIntrinsics, config.vision.camera.intrinsics)
    assert not intrinsics.calibrated
    assert (intrinsics.width, intrinsics.height, intrinsics.fx) == (1920, 1080, 1360.0)


def test_the_reachy_mini_profile_watches_the_simulated_head_camera() -> None:
    profile = _REPOSITORY / "config" / "reachy_mini_sim"
    config = RuntimeConfig.load(profile)
    sim = load_yaml(SimConfig, profile / "sim.yaml")

    assert config.vision.camera.source == RobotSource()
    assert config.robot is not None
    assert config.robot.namespace == sim.namespace
    assert config.vision.camera.intrinsics is not None
    intrinsics = load_yaml(CameraIntrinsics, config.vision.camera.intrinsics)
    # The MJCF's 1280x720 at an 80 degree vertical field of view.
    assert (intrinsics.width, intrinsics.height) == (config.vision.camera.width, 720)
    assert intrinsics.fy == pytest.approx(360 / math.tan(math.radians(40)), abs=0.01)


def test_a_camera_of_the_robot_needs_a_robot() -> None:
    config = RuntimeConfig.load(_REPOSITORY / "config" / "reachy_mini_sim")

    with pytest.raises(ValueError, match="there is no robot"):
        RuntimeConfig.model_validate({**config.model_dump(), "robot": None})


def test_a_head_controller_needs_a_robot() -> None:
    reachy = RuntimeConfig.load(_REPOSITORY / "config" / "reachy_mini_sim")
    default = RuntimeConfig.load(_REPOSITORY / "config" / "default")
    assert reachy.head_controller is not None

    with pytest.raises(ValueError, match="no robot for it to drive"):
        RuntimeConfig.model_validate(
            {**default.model_dump(), "head_controller": reachy.head_controller.model_dump()}
        )


def test_the_simulated_people_meet_the_robot_anew_each_session() -> None:
    default = RuntimeConfig.load(_REPOSITORY / "config" / "default")
    reachy = RuntimeConfig.load(_REPOSITORY / "config" / "reachy_mini_sim")

    assert reachy.vision.database == SessionFaces()
    assert reachy.admins == Path("~/.config/robot-voice-agent/reachy_mini_sim/admins.yaml")
    assert reachy.admins != default.admins


def test_the_respeaker_needs_a_transform_to_the_camera() -> None:
    config = RuntimeConfig.load(_REPOSITORY / "config" / "default")

    with pytest.raises(ValidationError, match="No transform between"):
        RuntimeConfig.model_validate({**config.model_dump(), "transforms": ()})


@pytest.mark.parametrize(("azimuth_deg", "bearing_deg"), [(0.0, 0.0), (30.0, -30.0), (-50.0, 50.0)])
def test_the_simulated_microphones_hear_from_where_the_camera_sees(
    azimuth_deg: float, bearing_deg: float
) -> None:
    config = RuntimeConfig.load(_REPOSITORY / "config" / "reachy_mini_sim")
    assert isinstance(config.doa, RobotDoaConfig)
    # The array's azimuths grow to the left, a camera's bearings to the right.
    heard = HeardDirection(
        time=0.0, frame=config.doa.frame, speech_azimuth=math.radians(azimuth_deg)
    )

    bearing = speech_bearing(heard, config.vision.camera.frame, StaticTransforms(config.transforms))

    # Within the profile's rounding of a quarter turn.
    assert bearing == pytest.approx(bearing_deg, abs=1e-4)


def test_the_robots_microphones_need_a_robot() -> None:
    config = RuntimeConfig.load(_REPOSITORY / "config" / "reachy_mini_sim")
    # The default profile's camera, which needs no robot either.
    webcam = RuntimeConfig.load(_REPOSITORY / "config" / "default").vision

    with pytest.raises(ValidationError, match="doa is the robot's, but there is no robot"):
        RuntimeConfig.model_validate(
            {
                **config.model_dump(),
                "robot": None,
                "vision": webcam.model_dump(),
                "head_controller": None,
            }
        )


def test_directions_of_arrival_need_the_active_speaker() -> None:
    config = RuntimeConfig.load(_REPOSITORY / "config" / "reachy_mini_sim")

    with pytest.raises(ValidationError, match="only help the active speaker"):
        RuntimeConfig.model_validate({**config.model_dump(), "active_speaker": None})


@pytest.mark.parametrize(
    ("azimuth_deg", "bearing_deg"),
    [
        # Measured with `lune doa`: someone talking straight in front of the camera.
        (176.0, 0.0),
        # The board's azimuths grow to its left, which faces back: to the camera's left.
        (206.0, -30.0),
        (146.0, 30.0),
    ],
)
def test_the_respeaker_hears_ahead_of_the_camera_behind_its_own_zero(
    azimuth_deg: float, bearing_deg: float
) -> None:
    config = RuntimeConfig.load(_REPOSITORY / "config" / "default")
    assert config.doa is not None
    heard = HeardDirection(
        time=0.0, frame=config.doa.frame, speech_azimuth=math.radians(azimuth_deg)
    )

    bearing = speech_bearing(heard, config.vision.camera.frame, StaticTransforms(config.transforms))

    # The board is pitched 4 degrees off flat: a bearing turns a little differently from the
    # azimuth.
    assert bearing == pytest.approx(bearing_deg, abs=0.5)
