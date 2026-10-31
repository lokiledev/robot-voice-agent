# Architecture

Lune is the high-level loop of a humanoid robot: she holds the conversation, keeps track of who is
around, and drives the body through a few long-running actions. A VLA policy and the robot's
controllers do the moving; Lune decides what to ask them. Three ideas shape the design: every
model runs locally, voice and vision feed one agent, and who may make the robot act is decided by
recognizing faces, in code.

This document zooms in step by step, following the [C4 model](https://c4model.com): the system and
what it talks to, the processes it runs as, the parts of the main process, and the code.

## 1. System context

```mermaid
flowchart LR
    human(["Person near the robot<br/>[person]"])
    presenter(["Presenter<br/>[person]"])
    lune["Lune<br/>[software system]<br/>holds the conversation, recognizes people,<br/>decides what the body does"]
    body["Robot body<br/>[external system]<br/>VLA policy, walking, gestures, gaze"]

    human -- "speaks (audio in),<br/>is seen (video in)" --> lune
    lune -- "answers out loud (audio out)" --> human
    presenter -- "watches the session,<br/>types messages" --> lune
    lune -- "starts and stops actions" --> body
    body -- "reports how each one ends" --> lune

    classDef person fill:#08427b,stroke:#052e56,color:#fff
    classDef system fill:#1168bd,stroke:#0b4884,color:#fff
    classDef external fill:#8a8a8a,stroke:#6b6b6b,color:#fff
    class human,presenter person
    class lune system
    class body external
```

Everything runs on one PC with one GPU (an RTX 4090 with 24 GB here); nothing calls a cloud API.
The robot body is reached through a small HTTP and WebSocket API, served for now by a mock.

## 2. Containers

```mermaid
flowchart LR
    human(["Person near the robot<br/>[person]"])
    subgraph pc ["Lune, on one PC"]
        agent["Agent process<br/>[Python, asyncio]<br/>voice, vision, dialogue, tools,<br/>session page server"]
        llm["LLM server<br/>[llama.cpp]<br/>Qwen3.5-35B-A3B"]
        mtx["MediaMTX<br/>[Docker]<br/>RTSP in, WebRTC out"]
        page["Session page<br/>[HTML, JavaScript, in a browser]"]
        data[("Local data<br/>[YAML, SQLite, rerun]<br/>profile, admins, faces,<br/>recordings")]
    end
    body["Robot API<br/>[external; a FastAPI mock here]<br/>VLA tasks, motions, gaze"]
    presenter(["Presenter<br/>[person]"])

    human <-- "audio in and out,<br/>video in" --> agent
    agent <-- "chat, streamed<br/>[HTTP]" --> llm
    agent <-- "start, cancel; status events<br/>[HTTP, WebSocket]" --> body
    agent <--> data
    agent -- "camera with overlay<br/>[RTSP]" --> mtx
    mtx -- "[WebRTC]" --> page
    agent <-- "session entries, typed lines<br/>[WebSocket]" --> page
    page <-- "watches, types" --> presenter

    classDef person fill:#08427b,stroke:#052e56,color:#fff
    classDef container fill:#438dd5,stroke:#2e6295,color:#fff
    classDef external fill:#8a8a8a,stroke:#6b6b6b,color:#fff
    class human,presenter person
    class agent,llm,mtx,page,data container
    class body external
```

| Container | Started with | Notes |
|---|---|---|
| Agent process | `rva run` (voice) or `rva text` (typing) | One asyncio loop; the camera has a thread of its own |
| LLM server | `rva serve-llm` | OpenAI-compatible API; another model is a config change |
| Robot API | `rva-vla-mock` | Replace with the real robot's API |
| MediaMTX | `rva serve-video` | Only for the camera on the session page |
| Session page | `--ui`, then <http://127.0.0.1:8700> | Served by the agent process |
| Local data | paths in `runtime.yaml` | Profile, admins, face database, recordings; all but the profile outside the source tree |

The profile is the robot's story (`character.yaml`, `environment.yaml`, `capabilities.yaml`,
rendered into the system prompt) plus the machine (`runtime.yaml`: models, devices, thresholds,
paths). `capabilities.yaml` matters most: the LLM should phrase VLA tasks the way the policy was
trained.

## 3. Inside the agent process

```mermaid
flowchart LR
    mic([Microphone]) --> vin
    cam([Camera]) --> vision
    subgraph process ["Agent process"]
        vin["Voice input<br/>voice activity, end of turn, ASR"]
        vision["Vision<br/>detect, track, recognize faces"]
        ui["Session page server"]
        loop["Agent loop<br/>inbox, history, LLM turns"]
        tools["Tools<br/>body, tasks, people, permissions"]
        vout["Voice output<br/>sentences to speech"]
        rec["Recorder<br/>every part records to it"]
    end
    vin -- "utterances" --> loop
    vision -- "people coming and going" --> loop
    ui -- "typed lines" --> loop
    loop -- "sentences" --> vout
    loop -- "tool calls" --> tools
    tools -- "how actions end" --> loop
    loop <--> llm([LLM server])
    tools <--> robot([Robot API])
    vout --> spk([Speaker])
    vision -- "annotated frames" --> mtx([MediaMTX])
    rec -. "live entries" .-> ui
    rec -. "rerun file" .-> disk([Recordings])

    classDef component fill:#85bbf0,stroke:#5d82a8,color:#000
    class vin,vision,loop,tools,vout,ui,rec component
```

The agent loop depends only on interfaces (`typing.Protocol`s): where utterances come from, where
sentences go, the LLM, the robot, the camera, the recorder. The same loop runs with the voice
stack, with a terminal, or with the fakes the tests use.

### Voice

```mermaid
flowchart TB
    subgraph hearing ["Hearing"]
        direction LR
        mic([Microphone]) --> aec["Echo canceller<br/>[PipeWire, WebRTC AEC]"]
        aec --> vad["Voice activity<br/>[Silero VAD]"]
        vad --> turn["End of turn<br/>[smart-turn v3]"]
        turn --> asr["Speech recognition<br/>[Parakeet TDT 0.6B]"]
        asr --> echo["Echo filter<br/>drops her own words"]
    end
    loop([Agent loop])
    subgraph speaking ["Speaking"]
        direction LR
        split["Sentence splitter"] --> tts["Speech synthesis<br/>[Kokoro-82M]"]
        tts --> spk([Speaker])
    end
    hearing -- "utterances" --> loop
    loop -- "LLM text, streamed" --> speaking
    hearing -. "speech while she speaks:<br/>barge-in" .-> speaking
```

- **End of turn.** About 200 ms into a silence, smart-turn judges from the audio whether the
  sentence is finished; if not, Lune keeps listening, up to 1.5 s of silence.
- **Recognition on whole utterances.** The turn detector already says where an utterance ends,
  so a streaming ASR model is not needed: Parakeet takes about 40 ms per utterance.
- **Speaking early.** LLM tokens are cut into sentences, and each goes to Kokoro as soon as it is
  complete while the previous one plays.
- **Barge-in.** Speech while Lune talks stops the playback and the LLM stream, and the history
  keeps only what was heard.
- **Not interrupting herself.** With speakers, three layers keep Lune from hearing her own voice:
  PipeWire subtracts what she plays from the microphone, a barge-in needs clearer and longer
  speech while she talks, and a transcript repeating what she said in the last 2 s is dropped.

### Vision

```mermaid
flowchart LR
    cam([Camera]) --> det["Face detection<br/>[YuNet, GPU]"]
    det --> emb["Face embedding<br/>[SFace, GPU]"]
    det & emb --> trk["Tracker<br/>box overlap, then<br/>re-identification"]
    trk --> gal["Gallery<br/>cosine match"]
    db[("Face database")] <--> gal
    trk -- "appeared, left" --> loop([Agent loop])
    trk -- "boxes, names,<br/>last sentence" --> ff["ffmpeg<br/>H.264 over RTSP"]
    ff --> mtx([MediaMTX])
```

- **Per frame**, 1920x1080 at up to 30 fps: YuNet finds faces in about 6 ms; faces too small to
  be close are ignored. Each track gets an SFace embedding when it starts, then every second.
- **Tracking.** A face keeps its track number across frames by box overlap, and across a quick
  move or occlusion by embedding. It is announced after 0.5 s in view and said to have left after
  3 s out of view.
- **Recognition.** A track is someone when its embeddings match that person's closely enough and
  clearly better than anyone else's. The largest face is taken as the person in front of the
  robot, and as whoever is speaking.
- **Memory.** The database is pseudonymous: people are user ids, names sit in a table of their
  own, and embeddings, operators and admins refer to ids only. No image is stored.

### Agent loop and tools

```mermaid
flowchart LR
    utt([Utterances]) --> inbox
    faces([People coming and going]) --> inbox
    ends([Actions ending]) --> inbox
    inbox[["Inbox<br/>one turn at a time"]] --> fixed{"Stop word or<br/>mute phrase?"}
    fixed -- "yes" --> code["Handled in code:<br/>stop the body, mute, unmute"]
    fixed -- "no" --> turn["LLM turn<br/>tagged with who speaks<br/>and what the body is doing"]
    turn <--> llm([LLM server])
    turn -- "sentences" --> speech([Voice output])
    turn -- "tool calls" --> tools["Tools<br/>permissions checked in code"]
    tools --> sup["Supervisors<br/>one body action at a time"]
    sup <--> robot([Robot API])
    sup -. "how it ended,<br/>in the first person" .-> ends
```

**Every body action is a long-running tool call.** It returns as soon as the action starts; the
action can be asked about and cancelled, and its end comes back through the inbox for the LLM to
react to. Lune keeps talking meanwhile. Actions that hold the body exclude each other, and a
second one is refused rather than queued:

| Part | Tools | Runs | Who may |
|---|---|---|---|
| Body | `start_task` (VLA), `go_to_location`, `go_to_human`, `play_animation` | One at a time | Operators for tasks and walking; anyone for gestures |
| Body control | `stay`, `cancel_task`, `get_task_status` | Instant | Anyone |
| Head | `look_at_user`, `stop_look_at_user` | Alongside the body | Anyone |
| People | `who_is_here`, `remember_person`, `rename_person`, `forget_person`, `list_operators` | Instant | Anyone |
| Permissions | `authorize_person`, `revoke_person` | Instant | Operators |

- **Operators** are the admins, listed in a file of the machine, and the people an operator
  authorized by voice. They are only obeyed while the camera recognizes them as the face in
  front. The tools check this themselves, so a slip of the LLM cannot bypass it. Without a camera
  nobody can be recognized, so anyone may make the robot act.
- **What must always work is handled in code.** "Stop" stops the body, and "stop talking" mutes
  Lune until an operator lets her talk again, without asking the LLM: as a tool, muting failed
  in a real session.
- **Guidance goes where the model reads it.** The LLM follows the text of a turn far more than
  rules in its system prompt. So each user turn says who is speaking and what the body is busy
  with; a person arriving in front of the robot comes with "greet them"; and an action's end is
  told as Lune's own ("You could not pick up the red cup: the grasp slipped. Say it as your own
  failure."). Replaying a real session, that last hint took replies like "the task failed" from
  8 in 10 to none.
- **Initiative.** Lune looks at people and gestures on her own, but starts tasks and walks only
  when asked.
- **History.** One session's messages, trimmed from the oldest when the context fills; the
  system prompt always stays.

### Session page

```mermaid
flowchart LR
    parts([Agent components]) -- "records" --> srv["Session page server<br/>[FastAPI, in the agent process]"]
    srv <-- "entries out, typed lines in<br/>[WebSocket]" --> page["Page<br/>[HTML, CSS, JavaScript]"]
    mtx([MediaMTX]) -- "camera<br/>[WebRTC]" --> page
    srv -- "typed lines" --> inbox([Agent inbox])
```

- **The agent is the only reporter.** The page is one more recorder: it gets the same entries as
  the rerun recording, and a page opened mid-session first gets the whole session so far. The
  agent follows the robot's own status streams, so what the page shows running is the robot's
  real state.
- **What it shows.** The camera with boxes, names and the last sentence under the speaker's
  face; the conversation, sentence by sentence as Lune speaks, with each tool call and how
  actions end; and the actions board, where whatever holds the body glows and the actions it
  excludes are dimmed.
- **Typed lines** join the microphone's and are attributed to the face in front, like speech.
- **Plain HTML served by the agent**, rather than a Streamlit app: entries are pushed as they
  happen, with no polling and nothing to build.

### Robot API

The mock serves what a real robot would: VLA tasks, and the body's motions and gaze. Every call
is logged in its terminal.

| Endpoint | Purpose |
|---|---|
| `POST /tasks`, `POST /tasks/{id}/cancel`, `GET /tasks/{id}` | Start a VLA task from an instruction, cancel it, read it; 409 while the body is busy |
| `WS /events` | Task status changes: running, succeeded, failed (with a cause and a reason), cancelled |
| `POST /body/motions`, `POST /body/motions/{id}/cancel`, `POST /body/stay` | Walk to a place or a person, play an animation, stop |
| `PUT`, `DELETE /body/gaze`; `GET /body/state` | Follow a person with the gaze, or stop; what the body is doing |
| `WS /body/events` | Motion status changes |

Tasks take 5 s to 2 min, walking 5 to 30 s, animations 2 to 8 s, and some fail or time out at
random. With `--interactive`, typing `s`, `f <reason>` or `t` ends the running action instead.

## A turn, end to end

```mermaid
sequenceDiagram
    autonumber
    actor U as Person
    participant V as Voice
    participant A as Agent loop
    participant L as LLM server
    participant S as Supervisor
    participant R as Robot API

    U->>V: "Could you put the cup on the shelf?"
    V->>A: utterance, once the turn has ended
    A->>L: turn, tagged [Alice, track 3]
    L-->>A: "Sure." and start_task("put the cup on the shelf")
    A->>V: "Sure."
    A->>S: start the task
    S->>R: POST /tasks
    Note over U,R: The body is busy, the conversation is not
    U->>V: "What's your name?"
    V->>A: utterance
    A->>L: turn, tagged (Your body is busy: put the cup on the shelf.)
    L-->>A: "I'm Lune."
    A->>V: "I'm Lune."
    R-->>S: succeeded
    S->>A: "You managed to put the cup on the shelf."
    A->>L: turn with the event
    L-->>A: "The cup is on the shelf."
    A->>V: "The cup is on the shelf."
```

## 4. Code

The repository is one uv workspace. Each domain has a schemas package with its data types and its
interfaces, and implementations depend on the schemas, never on each other. Only `rva-app`
imports implementations: it is the composition root.

```mermaid
flowchart BT
    PYD["rva-pydantic"]
    VLAS["rva-vla-schemas"]
    BODYS["rva-body-schemas"]
    SCH["rva-schemas"]
    VLAC["rva-vla-client"]
    BODYC["rva-body-client"]
    MOCK["rva-vla-mock"]
    AGENT["rva-agent"]
    LLM["rva-llm"]
    AUD["rva-voice-audio"]
    ASR["rva-voice-asr"]
    TTS["rva-voice-tts"]
    VIS["rva-vision-faces"]
    UI["rva-ui"]
    OBS["rva-observability"]
    APP["rva-app"]
    VLAS & BODYS --> PYD
    SCH --> VLAS & BODYS
    VLAC --> VLAS
    BODYC --> BODYS
    MOCK --> VLAS & BODYS
    AGENT & LLM & AUD & ASR & TTS & VIS & UI & OBS --> SCH
    APP --> AGENT & LLM & AUD & ASR & TTS & VIS & UI & OBS & VLAC & BODYC
```

| Package | Contents |
|---|---|
| `rva-schemas` | Shared data types (dialogue, profile, faces, records) and the agent's interfaces |
| `rva-vla-schemas`, `rva-body-schemas` | The robot API's data types and client interfaces |
| `rva-agent` | Agent loop, history, system prompt, tools, supervisors, permissions |
| `rva-llm` | The LLM client over the OpenAI API, and llama-server's launch settings |
| `rva-voice-audio`, `-asr`, `-tts` | Microphone and speaker, VAD, end of turn, barge-in; Parakeet; Kokoro |
| `rva-vision-faces` | Camera, detection, tracking, recognition, face database, camera stream |
| `rva-ui` | Session page server and page; MediaMTX's launch settings |
| `rva-observability` | Logging, OpenTelemetry spans and the rerun recorder |
| `rva-vla-client`, `rva-body-client` | HTTP and WebSocket clients of the robot API |
| `rva-vla-mock` | The mock robot |
| `rva-app` | Command line, runtime configuration, wiring |

[AGENTS.md](../AGENTS.md) lists the conventions the code follows.

## Models

| Task | Model | Runs on |
|---|---|---|
| Voice activity | Silero VAD v6.2 | CPU |
| End of turn | smart-turn v3 | CPU |
| Speech recognition | Parakeet TDT 0.6B v2, English | GPU, about 2 GB |
| Dialogue and tools | Qwen3.5-35B-A3B, Q4_K_M | GPU, about 16 GB; some expert layers on the CPU |
| Speech synthesis | Kokoro-82M, voice `af_heart` | GPU, about 1 GB |
| Face detection and embeddings | YuNet, SFace | GPU, under 0.5 GB |

**Why a cascade.** Speech-to-speech models such as Kyutai's Moshi reason over audio: they have no
text history to edit, no tool calls and no system prompt of your own, which an agent driving a
robot needs.

**Why this LLM.** Qwen3.5-35B-A3B is a mixture of experts with about 3B parameters active per
token, so it answers much faster than a dense model of similar quality. It does not fit beside
the voice models at Q4_K_M, so llama.cpp keeps some expert layers in system RAM, run by the CPU.
Thinking is off: the turns are short and latency matters more.

## Observability

Each session gets a directory under `~/Captures/robot-voice-agent/`, with a rerun recording and
the log. The recording holds the dialogue, every tool call, the actions as they start and end,
people coming and going, and the boxes of every tracked face, with `--video` the camera frames
under them. Durations are OpenTelemetry spans, drawn as plots in the same recording: each turn,
the LLM call with its time to the first text, the speech with its time to the first audio, ASR,
each tool, and the vision steps.

## Measured

On the RTX 4090 and a Ryzen 9 7950X3D:

- From the end of the user's speech to Lune's first audio: 0.72 to 0.86 s.
- LLM: about 120 tokens/s; first token in 110 to 175 ms, first sentence in 180 to 330 ms.
- Parakeet about 40 ms per utterance, Kokoro about 100 ms per sentence, smart-turn about 35 ms.
- GPU memory for the whole stack: 21 of 24 GB.
- Vision: about 6 ms per frame for detection, so it follows the camera at 24 to 30 fps.
- Through the speakers with echo cancellation, Lune finished a five-sentence paragraph without
  interrupting herself; without it, she cut herself off in the first sentence.
