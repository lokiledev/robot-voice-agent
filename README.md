# robot-voice-agent

Local voice agent for a humanoid robot: listens to the user, reasons with a local LLM, talks back, and sends short-horizon natural-language tasks to the robot's VLA policy.

Everything runs on this machine (RTX 4090, 64 GB RAM). Nothing calls a cloud API.

- Design and measurements: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Conventions and package layout: [AGENTS.md](AGENTS.md)
- Robot profiles: [config/](config/)

## Setup

```bash
uv sync
uv run pytest
```

The LLM runs in llama.cpp's `llama-server`, built with CUDA:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/.local/src/llama.cpp
cmake -S ~/.local/src/llama.cpp -B ~/.local/src/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ~/.local/src/llama.cpp/build --target llama-server -j 24
hf download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q4_K_M.gguf
```

CUDA 13.0 does not compile against glibc 2.43, which declares `rsqrt` differently. On such a system, build with a private copy of the toolkit whose `crt/math_functions.h` gives `rsqrt` and `rsqrtf` glibc's `noexcept`, and pass `-DCMAKE_CUDA_COMPILER=<copy>/bin/nvcc -DCUDAToolkit_ROOT=<copy>` to the first `cmake`. This machine has one at `~/.local/src/cuda-13.0-glibc243`.

Voice models (about 2.9 GB): Parakeet and smart-turn come from the Hugging Face cache, and Kokoro from the kokoro-onnx release:

```bash
hf download istupakov/parakeet-tdt-0.6b-v2-onnx --exclude "*.int8.onnx"
hf download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx
mkdir -p ~/.cache/robot-voice-agent/kokoro && cd ~/.cache/robot-voice-agent/kokoro
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

The GPU code calls `onnxruntime.preload_dlls()` so the CUDA 13 libraries from the wheels win over a system cuDNN built for CUDA 12.

Tests that need model weights are skipped by default; run them with `uv run pytest -m download`.

With speakers, the robot needs echo cancellation, or it hears itself and keeps interrupting its own speech. Install PipeWire's WebRTC echo canceller once:

```bash
mkdir -p ~/.config/pipewire/pipewire.conf.d && cp config/pipewire/60-rva-echo-cancel.conf ~/.config/pipewire/pipewire.conf.d/
systemctl --user restart pipewire pipewire-pulse wireplumber
```

It adds the nodes `rva_ec_source` and `rva_ec_sink`, which `config/default/runtime.yaml` uses. Set `audio.input_device` and `audio.output_device` to `null` to use the system defaults instead, for example with headphones.

## Trying it

Three terminals:

```bash
uv run rva serve-llm
```

```bash
uv run rva-vla-mock --interactive
```

```bash
uv run rva run
```

Talk to the robot (or use `uv run rva text` to type instead). Only operators can make Lune act, once the camera recognizes them. Make yourself an admin: name your face once with `uv run rva see` (press `n`), find your user id with `uv run rva people`, and add it to `config/default/admins.yaml`. Admins and other operators can then authorize people by voice ("Lune, authorize Bob"). In the mock's terminal, type `s` or `f <reason>` to end each task. The session is recorded under `~/Captures/robot-voice-agent/`, with every tracked face and, with `--video`, the camera frames, so it replays in rerun: open `session.rrd` with `uv run rerun`, or pass `--viewer` to watch it live. `uv run rva run --show --viewer --video` shows everything at once.

To present a session, add the web page: start MediaMTX in a fourth terminal (it needs Docker, and `ffmpeg` on the path):

```bash
uv run rva serve-video
```

then run `uv run rva run --ui` and open <http://127.0.0.1:8700>. It shows the camera with the tracked faces, the conversation with every tool call, and which actions are running; you can also type to Lune there. `--no-camera` needs no MediaMTX.

Each part also runs alone: `rva listen` prints what the microphone hears, `rva say` speaks what you type, `rva llm` chats with the model without the agent, `rva prompt` prints the system prompt, and the mock robot runs VLA tasks, walking and animations with random durations, failures and timeouts, logging every call (`rva-vla-mock --help`).
