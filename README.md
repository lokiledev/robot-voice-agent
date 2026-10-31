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

## Trying it

Three terminals:

```bash
uv run rva serve-llm
```

```bash
uv run rva-vla-mock --interactive
```

```bash
uv run rva text
```

Type to talk to the robot; in the mock's terminal, type `s` or `f <reason>` to end each task. The session is recorded under `~/Captures/robot-voice-agent/`; open `session.rrd` with `uv run rerun`, or pass `--viewer` to watch it live.

Each part also runs alone: `rva prompt` prints the system prompt, `rva llm` chats with the model without the agent, and the mock VLA can run with random outcomes (`rva-vla-mock --help`).
