from typing import Literal

import onnx
import onnxruntime
from huggingface_hub import hf_hub_download

from rva_vision_faces.config import ModelFile


def load_session(
    model: ModelFile, device: Literal["cuda", "cpu"], *, dynamic_image_size: bool
) -> onnxruntime.InferenceSession:
    """Download a model if needed and open it on `device`.

    Args:
        model: The ONNX file.
        device: Where it runs; CUDA failing to load is an error, not a fallback to the CPU.
        dynamic_image_size: Let the image input take any height and width. The YuNet export fixes
            them to 640x640 although the network is fully convolutional.
    """
    proto = onnx.load(hf_hub_download(model.repo, model.file))
    if dynamic_image_size:
        _make_image_size_dynamic(proto)
    match device:
        case "cuda":
            # Loads the CUDA and cuDNN libraries installed from PyPI by onnxruntime-gpu's extras.
            onnxruntime.preload_dlls()
            providers = ["CUDAExecutionProvider"]
        case "cpu":
            providers = ["CPUExecutionProvider"]
    # The opencv_zoo exports warn about initializers listed as inputs.
    onnxruntime.set_default_logger_severity(3)  # ty: ignore[possibly-missing-attribute]  # Imported in a try block of onnxruntime/__init__.py.
    session = onnxruntime.InferenceSession(proto.SerializeToString(), providers=providers)
    if session.get_providers()[0] != providers[0]:
        raise RuntimeError(f"{model.file} could not run on {device}: {session.get_providers()}")
    return session


def _make_image_size_dynamic(proto: onnx.ModelProto) -> None:
    dims = proto.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_param = "height"
    dims[3].dim_param = "width"
    for output in proto.graph.output:
        output.type.tensor_type.shape.dim[1].dim_param = "anchors"
