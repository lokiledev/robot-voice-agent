import numpy as np

from rva_robot_schemas.camera import CameraFrame, decode_frame, encode_frame, encoded_size


def test_a_frame_reads_back_as_written_and_owns_its_pixels() -> None:
    # Two rows of three BGR pixels: blue, green, red, then black, grey, white.
    image = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[0, 0, 0], [128, 128, 128], [255, 255, 255]]],
        np.uint8,
    )
    buffer = memoryview(bytearray(64))

    written = encode_frame(CameraFrame(timestamp=3.25, image=image), buffer)
    frame = decode_frame(buffer)
    buffer[:] = bytes(64)

    assert written == encoded_size(3, 2) == 16 + 18
    assert frame.timestamp == 3.25
    np.testing.assert_array_equal(frame.image, image)


def test_the_header_holds_the_timestamp_then_height_and_width() -> None:
    buffer = memoryview(bytearray(encoded_size(1, 1)))

    encode_frame(CameraFrame(timestamp=1.0, image=np.full((1, 1, 3), 7, np.uint8)), buffer)

    assert bytes(buffer) == (
        b"\x00\x00\x00\x00\x00\x00\xf0\x3f"  # 1.0, a little-endian double
        b"\x01\x00\x00\x00\x01\x00\x00\x00"  # height 1, width 1
        b"\x07\x07\x07"
    )
