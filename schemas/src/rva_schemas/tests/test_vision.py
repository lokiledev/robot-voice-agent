from rva_schemas.vision import CameraIntrinsics, Point3D

INTRINSICS = CameraIntrinsics(
    calibrated=False, width=1920, height=1080, fx=1000.0, fy=1000.0, cx=960.0, cy=540.0
)


def test_a_point_on_the_axis_projects_to_the_principal_point() -> None:
    assert INTRINSICS.project(Point3D(x=0.0, y=0.0, z=2.0)) == (960.0, 540.0)


def test_a_point_to_the_right_and_below_projects_right_and_down() -> None:
    assert INTRINSICS.project(Point3D(x=0.5, y=0.25, z=2.0)) == (1210.0, 665.0)


def test_a_point_behind_the_camera_has_no_pixel() -> None:
    assert INTRINSICS.project(Point3D(x=0.0, y=0.0, z=-1.0)) is None
