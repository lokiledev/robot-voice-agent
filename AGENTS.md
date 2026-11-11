# robot-voice-agent

Local voice agent for a humanoid robot. See `docs/ARCHITECTURE.md` for the design and the reasons behind it. `docs/ROADMAP.md` lists what is missing or weak, and what was decided about it.

## Commands

```bash
uv sync                                     # install the whole workspace
uv run ruff check --fix . && uv run ruff format .
uv run ty check
uv run pytest                               # all packages
uv run pytest <path-to-test>::<name> -xvs   # a single test
```

## Package layout

The repository is one uv workspace. Each package lives in its domain directory with code in `src/<module>` and tests in `src/<module>/tests`.

| Directory | Package | Role |
|---|---|---|
| `common/pydantic` | `rva-pydantic` | `RVABaseModel` and YAML loading |
| `common/observability` | `rva-observability` | loguru, OpenTelemetry and rerun setup; implements `Recorder` |
| `vla/schemas` | `rva-vla-schemas` | VLA task data model and the `VlaClient` interface |
| `vla/client` | `rva-vla-client` | HTTP + WebSocket `VlaClient` |
| `vla/mock` | `rva-vla-mock` | Mock robot server: VLA and body |
| `body/schemas` | `rva-body-schemas` | Body motion, gaze data model and the `BodyClient` interface |
| `body/client` | `rva-body-client` | HTTP + WebSocket `BodyClient` |
| `robot/schemas` | `rva-robot-schemas` | Joints and camera data model, wire format and kickmsg topic names |
| `robot/client` | `rva-robot-client` | asyncio kickmsg client of the joints and head camera |
| `robot/kinematics` | `rva-robot-kinematics` | Reachy Mini's head kinematics: a head pose to the body's yaw and the six Stewart motors' angles, with the geometry read from the MJCF |
| `robot/sim` | `rva-robot-sim` | Reachy Mini simulated in Genesis (CPU physics, Nyx camera) with Rocketbox people, serving the robot's topics |
| `schemas` | `rva-schemas` | Shared data model and interfaces (`abc/`) for agent, voice and LLM |
| `agent` | `rva-agent` | Dialogue loop, tools, VLA task supervision |
| `llm` | `rva-llm` | `LanguageModel` over an OpenAI-compatible server |
| `voice/audio` | `rva-voice-audio` | Audio I/O, VAD, end of turn, barge-in; implements `UserInput` and `SpeechOutput` |
| `voice/asr` | `rva-voice-asr` | `SpeechRecognizer` with Parakeet |
| `voice/tts` | `rva-voice-tts` | `SpeechSynthesizer` with Kokoro |
| `vision/faces` | `rva-vision-faces` | Camera, face detection, tracking and recognition, face features, skeletons, depth, face memory; implements `Vision` |
| `vision/viewer` | `rva-vision-viewer` | Shows the camera as it is processed: the overlay, a Dear PyGui debug window and the RTSP stream; implements `Recorder` |
| `attention` | `rva-attention` | Visual attention: scores the faces and hands in view and picks the one to attend to, as a 3D point |
| `interaction` | `rva-interaction` | Where perception meets the dialogue: who pays attention to the robot, which face said each utterance and whether to the robot; implements `Engagement` and `UserInput` |
| `multimodal` | `rva-multimodal` | Where camera and microphone meet: which face speaks, with Light-ASD; implements `SpeakerDetector` |
| `ui` | `rva-ui` | Session web page: conversation, actions and camera, with typed input; implements `Recorder` and `UserInput` |
| `app` | `rva-app` | CLI and composition root; the only package that imports implementations |

## Package boundaries

- Data types and interfaces go in a `*-schemas` package; implementations depend on schemas, never on each other. `rva-agent` depends only on `rva-schemas` and `rva-vla-schemas`, never on `rva-llm`, the voice packages or `rva-vla-client`.
- Interfaces are `typing.Protocol`s in an `abc/` subpackage. Add one only where there are two implementations or a test fake needs it.
- Each implementation owns its config model; `rva-app` aggregates them into the runtime config.
- `rva-vla-*` packages never import from the agent side: the VLA API is a service boundary.
- Keep heavy dependencies (onnxruntime, model runtimes) in the narrowest package, and keep transitive dependencies out of other packages' imports.
- One `ruff.toml`, one `ty.toml`, one `uv.lock` for the whole workspace. No per-package overrides.
- Dependencies are declared in `pyproject.toml` with lower bounds.

## Trying parts by hand

Every component gets a command to run it alone (see the table in the README), wired in `rva-app`. Add one with each new component, so it can be checked by ear or by eye before it joins the loop.

## Code style

- `cyclopts` for CLIs.
- No re-exports in `__init__.py`.
- Top-level absolute imports only.
- asyncio throughout; no blocking calls in async code. Run CPU- or GPU-bound model calls with `asyncio.to_thread`.

## Design principles

- Remove dead code and unused arguments. Add a variant when a second one actually exists.
- Keep functions and classes small, with one responsibility. Prefer composition; subclass only to implement an interface.
- Make invalid states unrepresentable: discriminated unions with `Literal` tags, not several nullable fields. `None`, not sentinel values.
- No unnecessary indirection: no getters, registries or wrappers around a single implementation.
- Keep the public API small; prefix private names with `_`.

## Pydantic

- Inherit from `rva_pydantic.RVABaseModel` (frozen, `extra="forbid"`), not `pydantic.BaseModel`.
- Discriminated unions with `Literal` tags for polymorphic types. The tag field is the one field that may have a default.
- No defaults on config fields or function arguments unless strongly justified.
- Specific types: `Path` for paths, tuples for immutable sequences, no untyped `dict` across module boundaries.
- Document fields with attribute docstrings placed after the field.

## Logging and telemetry

- `loguru` only, never `print` or stdlib `logging`. Use `{}` formatting: `logger.info("Starting {}", name)`.
- `logger.exception()` in `except` blocks.
- Measure durations with OpenTelemetry spans (`opentelemetry-api`); only `rva-observability` and `rva-app` touch the SDK or rerun.

## Testing

- pytest. Tests sit next to the code in `src/<module>/tests`, files start with `test_`.
- Tests are documentation: inline literal inputs and expected outputs.
- No mocking libraries. Write a small fake that implements the Protocol, or redesign.
- Hermetic: no network, no model downloads, no audio devices, unless marked `cuda`, `download` or `audio`.
- Serde tests hardcode the serialized form and check deserialization.

## Static checking

- Prefer designs ty can check over runtime asserts.
- `match`/`case` for type dispatch, ending with `case _: assert_never(x)`.
- Suppressions name the rule and say why: `# ty: ignore[<rule>]  # <reason>`. Never a bare ignore.

## Comments and docstrings

- No docstrings that restate the name or signature. Docstrings state the contract; comments explain why.
- Google convention (`Args:`, `Returns:`, `Raises:`) when a docstring has sections.

## Errors

- Catch specific exceptions, never bare `except Exception`.
- `assert` for invariants, `raise` for expected errors.
- Fail loudly: no silent fallbacks or swallowed exceptions.

## Commits

- Imperative subject with no prefix, e.g. `Add Kokoro speech synthesizer`. The body explains motivation and design choices.
