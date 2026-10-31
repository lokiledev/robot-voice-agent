# Third-party notices

This repository contains no third-party code or model weights. Python packages are installed by uv
from PyPI, models are downloaded on first use (mostly from the Hugging Face Hub), and the external
programs below are installed separately. They are listed here to credit their authors and to state
the terms they come under.

## Models

| Model | Used for | Authors | Downloaded from | License |
|---|---|---|---|---|
| Qwen3.5-35B-A3B, Q4_K_M GGUF | Dialogue and tool calls | Qwen team, Alibaba Cloud; GGUF by Unsloth | `unsloth/Qwen3.5-35B-A3B-GGUF` | Apache-2.0 |
| Parakeet TDT 0.6B v2 | Speech recognition | NVIDIA; ONNX export by Ilya Stupakov (onnx-asr) | `istupakov/parakeet-tdt-0.6b-v2-onnx` | CC-BY-4.0 |
| Silero VAD v6.2 | Voice activity detection | Silero Team; ONNX by Ilya Stupakov | `istupakov/silero-vad-onnx` | MIT |
| smart-turn v3 | End of turn detection | Pipecat (Daily) | `pipecat-ai/smart-turn-v3` | BSD-2-Clause |
| Kokoro-82M v1.0 | Speech synthesis | hexgrad; ONNX export and voices by thewh1teagle (kokoro-onnx) | kokoro-onnx GitHub releases | Apache-2.0 |
| YuNet 2023mar | Face detection | Shiqi Yu, Wei Wu et al., OpenCV Zoo | `opencv/face_detection_yunet` | MIT |
| SFace 2021dec | Face recognition | Yaoyao Zhong, Weihong Deng et al., OpenCV Zoo | `opencv/face_recognition_sface` | Apache-2.0 |

Parakeet TDT 0.6B v2 by NVIDIA is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); it is used unchanged, through the ONNX
export of onnx-asr.

`voice/audio/src/rva_voice_audio/turn.py` computes Whisper's log-mel input features (OpenAI
Whisper, MIT) in its own numpy code, written to match Hugging Face's `WhisperFeatureExtractor`, as
smart-turn expects.

## Programs run beside the agent

| Program | Used for | License |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` | Serving the LLM | MIT |
| [MediaMTX](https://github.com/bluenviron/mediamtx), Docker image `bluenviron/mediamtx` | Camera stream to the session page | MIT |
| [FFmpeg](https://ffmpeg.org) | Encoding the camera stream; run as a separate process | LGPL-2.1-or-later or GPL-2.0-or-later, depending on the build |
| [PipeWire](https://pipewire.org) echo-cancel module | Echo cancellation | MIT |

## Python packages

Packages the agent runs with, by license (`uv export --no-dev` lists them with their versions):

- **MIT:** annotated-doc, annotated-types, anyio, attrs, dlinfo, docstring_parser, espeakng-loader
  (see below), fastapi, filelock, h11, jiter, kokoro-onnx, loguru, markdown-it-py, mdurl, onnx-asr,
  onnxruntime-gpu, pydantic, pydantic_core, PyYAML, rich, rich-rst, sounddevice, truststore,
  typing-inspection
- **MIT OR Apache-2.0:** rerun-sdk, sniffio
- **MIT-0:** cffi
- **MIT-CMU:** pillow
- **BSD-2-Clause:** Pygments
- **BSD-3-Clause:** click, cloudpickle, fsspec, httpcore, httpcore2, httpx, httpx2, idna, Jinja2,
  joblib, MarkupSafe, protobuf, psutil, pycparser, starlette, uvicorn, websockets
- **BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0:** numpy
- **Apache-2.0:** cyclopts, flatbuffers, hf-xet, huggingface_hub, ml_dtypes, onnx, openai,
  opencv-python, opentelemetry-api, opentelemetry-sdk, opentelemetry-semantic-conventions, pyarrow
- **Apache-2.0 OR BSD-2-Clause:** packaging
- **PSF-2.0:** typing_extensions
- **MPL-2.0:** certifi; **MPL-2.0 AND MIT:** tqdm. Used unmodified.
- **GPL-3.0-or-later:** phonemizer, and the espeak-ng library that espeakng-loader bundles.
- **NVIDIA proprietary (CUDA EULA):** nvidia-cublas, nvidia-cuda-nvrtc, nvidia-cuda-runtime,
  nvidia-cudnn-cu13, nvidia-cufft, nvidia-curand, nvidia-nvjitlink, installed with onnxruntime-gpu.

Development tools only (tests, linting, hooks): MIT for cfgv, deptry, executing, identify,
iniconfig, inline-snapshot, platformdirs, pluggy, pre-commit, pytest, pytest-timeout,
python-discovery, ruff, tomli, ty and virtualenv; Apache-2.0 for asttokens, pytest-asyncio,
requirements-parser and sortedcontainers; BSD-3-Clause for nodeenv; PSF-2.0 for distlib; MPL-2.0
for hypothesis.

### GPL components of speech synthesis

kokoro-onnx turns text into phonemes with phonemizer and espeak-ng, both GPL-3.0-or-later, which
run in the agent's process. They are not part of this repository and are installed from PyPI like
any other dependency. Anyone who distributes an installed copy of the agent, for example a
container image, distributes them too and has to follow the GPL for them.
