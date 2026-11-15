"""The room around the simulated robot and what its people carry: models, downloaded on first use."""

import hashlib
import os
import tempfile
from pathlib import Path

import httpx
from loguru import logger

from rva_pydantic import RVABaseModel
from rva_robot_sim.scenes import Interior, Prop


class InteriorsConfig(RVABaseModel):
    directory: Path
    """Where the scenes' rooms and props are kept, outside the source tree; `~` is expanded."""


def install_model(model: Interior | Prop, config: InteriorsConfig, client: httpx.Client) -> Path:
    """A room's or a prop's model as a GLB file, downloaded if missing.

    Raises:
        RuntimeError: If the download does not match the checksum the scene gives for it.
        httpx.HTTPError: If the download fails.
    """
    directory = config.directory.expanduser()
    path = directory / f"{model.name}.glb"
    if path.exists():
        return path
    logger.info("Downloading the {} model from {}", model.name, model.url)
    content = client.get(model.url).raise_for_status().content
    if hashlib.sha256(content).hexdigest() != model.sha256:
        raise RuntimeError(f"{path.name} does not match the scene's checksum")
    directory.mkdir(parents=True, exist_ok=True)
    # Written aside, then moved in place: an interrupted download leaves no half file.
    descriptor, written = tempfile.mkstemp(dir=directory)
    with os.fdopen(descriptor, "wb") as output:
        output.write(content)
    os.replace(written, path)
    return path
