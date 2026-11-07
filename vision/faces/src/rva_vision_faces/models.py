from typing import Final, assert_never

import onnx
import onnxruntime
import tensorrt_libs  # noqa: F401  # Loads TensorRT 10's libraries, which ONNX Runtime opens by name.
from huggingface_hub import hf_hub_download

from rva_vision_faces.config import Cpu, Cuda, Device, ModelFile, TensorRt

_MAX_BATCH: Final = 8
"""Most faces or people a TensorRT engine is built for; a larger batch rebuilds it, once, in about
half a second."""


def load_session(
    model: ModelFile,
    device: Device,
    *,
    dynamic_image_size: bool,
    batch_item: tuple[int, ...] | None,
) -> onnxruntime.InferenceSession:
    """Download a model if needed and open it on `device`.

    Args:
        model: The ONNX file.
        device: Where it runs; a GPU provider failing to load is an error, not a fallback.
        dynamic_image_size: Let the image input take any height and width. The YuNet export fixes
            them to 640x640 although the network is fully convolutional. Such a model runs on CUDA
            rather than TensorRT, which takes YuNet twice as long on the whole frame.
        batch_item: The shape of one item of the input's batch, for a model given one item per
            face or person; `None` for a fixed input.

    Raises:
        RuntimeError: If the model could not run on `device`.
    """
    proto = onnx.load(hf_hub_download(model.repo, model.file))
    if dynamic_image_size:
        _make_image_size_dynamic(proto)
    match device:
        case TensorRt() if not dynamic_image_size:
            onnxruntime.preload_dlls()
            providers = [
                ("TensorrtExecutionProvider", _tensorrt_options(device, proto, batch_item)),
                ("CUDAExecutionProvider", {}),
            ]
        case TensorRt() | Cuda():
            # Loads the CUDA and cuDNN libraries installed from PyPI by onnxruntime-gpu's extras.
            onnxruntime.preload_dlls()
            providers = [("CUDAExecutionProvider", {})]
        case Cpu():
            providers = [("CPUExecutionProvider", {})]
        case _:
            assert_never(device)
    # The opencv_zoo exports warn about initializers listed as inputs.
    onnxruntime.set_default_logger_severity(3)  # ty: ignore[possibly-missing-attribute]  # Imported in a try block of onnxruntime/__init__.py.
    session = onnxruntime.InferenceSession(proto.SerializeToString(), providers=providers)
    # ONNX Runtime retries on the next provider when one fails, and only says so on stderr.
    if session.get_providers()[0] != providers[0][0]:
        raise RuntimeError(f"{model.file} could not run on {device}: {session.get_providers()}")
    return session


def _tensorrt_options(
    device: TensorRt, proto: onnx.ModelProto, batch_item: tuple[int, ...] | None
) -> dict[str, object]:
    engine_cache = device.engine_cache.expanduser()
    # ONNX Runtime does not make it, and without it falls back to CUDA.
    engine_cache.mkdir(parents=True, exist_ok=True)
    options: dict[str, object] = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(engine_cache),
        "trt_timing_cache_enable": True,
    }
    if batch_item is None:
        return options
    name = proto.graph.input[0].name
    item = "x".join(str(size) for size in batch_item)
    return options | {
        "trt_profile_min_shapes": f"{name}:1x{item}",
        "trt_profile_opt_shapes": f"{name}:1x{item}",
        "trt_profile_max_shapes": f"{name}:{_MAX_BATCH}x{item}",
    }


def _make_image_size_dynamic(proto: onnx.ModelProto) -> None:
    dims = proto.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_param = "height"
    dims[3].dim_param = "width"
    for output in proto.graph.output:
        output.type.tensor_type.shape.dim[1].dim_param = "anchors"
