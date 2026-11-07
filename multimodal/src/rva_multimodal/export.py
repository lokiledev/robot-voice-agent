"""Download Light-ASD's checkpoint and convert it to ONNX: `python -m rva_multimodal.export <out>`.

Run once per machine by `open_light_asd`; needs the `export` extra, for PyTorch.
"""

import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from loguru import logger

from rva_multimodal.light_asd import CHECKPOINT_SHA256, CHECKPOINT_URL
from rva_multimodal.light_asd_network import LightAsdNetwork


def export(out: Path) -> None:
    """Write the ONNX model to `out`, checked against PyTorch.

    Raises:
        ValueError: If the downloaded checkpoint is not the pinned one.
    """
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "light_asd.model"
        logger.info("Downloading {}", CHECKPOINT_URL)
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError(f"Light-ASD checkpoint has SHA-256 {digest}, not {CHECKPOINT_SHA256}")
        network = LightAsdNetwork()
        network.load_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=True))
    network.eval()
    frames = 25
    audio = torch.randn(1, 4 * frames, 13)
    faces = torch.randint(0, 256, (1, frames, 112, 112)).float()
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial")
    torch.onnx.export(
        network,
        (audio, faces),
        partial,
        input_names=["audio", "faces"],
        output_names=["scores"],
        dynamic_axes={
            "audio": {1: "audio_frames"},
            "faces": {1: "frames"},
            "scores": {0: "frames"},
        },
        opset_version=17,
        dynamo=False,
    )
    # A shorter clip than the export's, to check the time axis is dynamic.
    audio, faces = audio[:, :52], faces[:, :13]
    with torch.no_grad():
        expected = network(audio, faces).numpy()
    session = onnxruntime.InferenceSession(str(partial), providers=["CPUExecutionProvider"])
    (scores,) = session.run(None, {"audio": audio.numpy(), "faces": faces.numpy()})
    np.testing.assert_allclose(np.asarray(scores), expected, atol=1e-4)
    partial.rename(out)
    logger.info("Light-ASD converted to {}", out)


if __name__ == "__main__":
    export(Path(sys.argv[1]))
