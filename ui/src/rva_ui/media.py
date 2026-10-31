from rva_pydantic import RVABaseModel


class MediaServerConfig(RVABaseModel):
    """How to run MediaMTX, which takes the camera stream over RTSP and serves it over WebRTC."""

    image: str
    """Docker image, e.g. `bluenviron/mediamtx:1.19.1`."""


def media_server_command(config: MediaServerConfig) -> list[str]:
    """MediaMTX in Docker on the host's network, with only its RTSP and WebRTC servers."""
    return [
        "docker", "run", "--rm", "--network=host", "--name", "rva-mediamtx",
        "--env", "MTX_RTMP=no",
        "--env", "MTX_HLS=no",
        "--env", "MTX_SRT=no",
        config.image,
    ]  # fmt: skip
