"use strict";

// The tools the LLM is offered, by the part of the robot they use. A tool the agent offers that is
// missing here shows up under "Other"; one listed here that it does not offer is left out.
const LANES = [
  {
    id: "body",
    title: "Body",
    rule: "One action at a time: each takes the whole body.",
    tools: ["start_task", "go_to_location", "go_to_human", "play_animation"],
    controls: ["stay", "cancel_task", "get_task_status"],
  },
  {
    id: "head",
    title: "Head",
    rule: "Alongside the body.",
    tools: ["look_at_user", "stop_look_at_user"],
  },
  { id: "voice", title: "Voice", rule: "Alongside everything: talks while the body works." },
  {
    id: "people",
    title: "People",
    rule: "Instant: memory and permissions.",
    tools: [
      "who_is_here",
      "remember_person",
      "rename_person",
      "forget_person",
      "list_operators",
      "authorize_person",
      "revoke_person",
    ],
  },
];

const LABELS = {
  start_task: "VLA task",
  go_to_location: "Walk to a place",
  go_to_human: "Walk to a person",
  play_animation: "Gesture",
  stay: "Stop moving",
  cancel_task: "Cancel task",
  get_task_status: "Task status",
  look_at_user: "Follow with gaze",
  stop_look_at_user: "Stop following",
  who_is_here: "Who is here",
  remember_person: "Remember",
  rename_person: "Rename",
  forget_person: "Forget",
  list_operators: "Operators",
  authorize_person: "Authorize",
  revoke_person: "Revoke",
};

// Only for an operator in front of the robot.
const OPERATORS_ONLY = new Set([
  "start_task",
  "go_to_location",
  "go_to_human",
  "authorize_person",
  "revoke_person",
]);

const MOTION_TOOLS = {
  go_to_location: "go_to_location",
  go_to_human: "go_to_human",
  animation: "play_animation",
};

// Older entries are the replay of the session so far, which should not flash the tools.
const FRESH_MS = 3000;
const FLASH_MS = 1600;
const RECONNECT_MS = 1000;

const state = {
  socket: null,
  videoPage: undefined,
  name: "Robot",
  tools: new Map(), // name -> description
  people: new Map(), // user id -> name
  body: null, // {id, tool, what, since}: the action holding the body
  gaze: null, // name of the person followed with the gaze
  muted: false,
  speaking: null, // the robot's message being spoken
  vad: 0,
};

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function clock(date) {
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  });
}

function elapsed(since) {
  const seconds = Math.max(0, Math.floor((Date.now() - since) / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

// Connection

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/session`);
  state.socket = socket;
  socket.addEventListener("open", () => setOnline(true));
  socket.addEventListener("message", (event) => receive(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    setOnline(false);
    setTimeout(connect, RECONNECT_MS);
  });
}

function setOnline(online) {
  $("link").classList.toggle("online", online);
  $("link-text").textContent = online ? "Live" : "Reconnecting";
  $("message").disabled = !online;
  $("send").disabled = !online;
}

function receive(message) {
  if (message.kind === "hello") {
    reset(message.video_page);
    return;
  }
  const at = new Date(message.at);
  apply(message.record, at, Date.now() - at.getTime() < FRESH_MS);
}

function reset(videoPage) {
  // The server replays the whole session after its hello.
  Object.assign(state, {
    tools: new Map(),
    people: new Map(),
    body: null,
    gaze: null,
    muted: false,
    speaking: null,
  });
  $("log").replaceChildren();
  $("lanes").replaceChildren(el("p", "waiting", "Waiting for the agent…"));
  showVideo(videoPage);
  renderStatus();
}

function showVideo(videoPage) {
  // Kept playing across reconnections.
  if (videoPage === state.videoPage) return;
  state.videoPage = videoPage;
  const video = $("video");
  video.querySelector("iframe")?.remove();
  $("video-off").hidden = videoPage !== null;
  if (videoPage === null) return;
  const url = new URL(videoPage);
  for (const [key, value] of Object.entries({
    controls: "false",
    muted: "true",
    autoplay: "true",
    playsinline: "true",
  })) {
    url.searchParams.set(key, value);
  }
  const frame = el("iframe");
  frame.src = url.href;
  frame.allow = "autoplay";
  frame.title = "Camera with the tracked faces";
  video.append(frame);
}

function apply(record, at, fresh) {
  switch (record.kind) {
    case "agent_started":
      return onStarted(record, at);
    case "user_line":
      return onUserLine(record, at);
    case "robot_sentence":
      return onSentence(record, at);
    case "robot_line":
      return onRobotLine(record, at);
    case "tool":
      return onTool(record, at, fresh);
    case "task":
      return onTask(record.task, at, fresh);
    case "motion":
      return onMotion(record.motion, at, fresh);
    case "gaze":
      return onGaze(record.looking_at);
    case "mute":
      return onMute(record.muted, at);
    case "face":
      return onFace(record.event, at);
    case "measurement":
      if (record.name === "vad") {
        state.vad = Math.max(state.vad, record.value);
        const hearing = $("voice-hearing");
        if (hearing !== null) hearing.hidden = false;
      }
      return;
  }
}

// Actions board

function onStarted(record, at) {
  state.name = record.name;
  state.tools = new Map(record.tools.map((tool) => [tool.name, tool.description]));
  $("robot-name").textContent = record.name;
  document.title = `${record.name} session`;
  $("message").placeholder = `Type to ${record.name}…`;
  buildLanes();
  addEvent(at, "", `${record.name} is up, with ${record.tools.length} tools`);
}

function buildLanes() {
  const lanes = $("lanes");
  lanes.replaceChildren();
  const placed = new Set();
  for (const lane of LANES) {
    if (lane.id === "voice") {
      lanes.append(voiceLane(lane));
      continue;
    }
    const tools = lane.tools.filter((name) => state.tools.has(name));
    const controls = (lane.controls ?? []).filter((name) => state.tools.has(name));
    for (const name of [...tools, ...controls]) placed.add(name);
    // Without a camera, nothing about people.
    if (tools.length + controls.length > 0) lanes.append(laneElement(lane, tools, controls));
  }
  const others = [...state.tools.keys()].filter((name) => !placed.has(name));
  if (others.length > 0) {
    lanes.append(laneElement({ id: "other", title: "Other", rule: "" }, others, []));
  }
  renderLanes();
}

function laneHead(lane) {
  const head = el("div", "lane-head");
  head.append(el("div", "lane-title", lane.title), el("div", "lane-rule", lane.rule));
  return head;
}

function laneElement(lane, tools, controls) {
  const row = el("div", "lane");
  const head = laneHead(lane);
  if (lane.id === "body") {
    const status = el("div", "lane-state", "Free");
    status.id = "body-state";
    head.append(status);
  }
  const list = el("div", "lane-tools");
  for (const name of tools) list.append(toolChip(name, lane.id, false));
  if (controls.length > 0) {
    list.append(el("div", "lane-sep"));
    for (const name of controls) list.append(toolChip(name, lane.id, true));
  }
  row.append(head, list);
  return row;
}

function toolChip(name, lane, control) {
  const chip = el("div", control ? "tool control" : "tool");
  chip.dataset.tool = name;
  chip.dataset.lane = lane;
  chip.title = state.tools.get(name);
  const top = el("div", "tool-top", LABELS[name] ?? name);
  if (OPERATORS_ONLY.has(name)) top.append(el("span", "badge", "operator"));
  chip.append(top, el("code", "tool-name", name), el("div", "tool-detail"));
  return chip;
}

function voiceLane(lane) {
  const row = el("div", "lane");
  const list = el("div", "lane-tools");
  list.append(
    stateChip("speaking", "Speaking"),
    stateChip("hearing", "Hearing"),
    stateChip("mute", "Mute"),
  );
  const hearing = list.querySelector("#voice-hearing");
  hearing.querySelector(".tool-top").after(meter());
  // Shown once the microphone reports: there is none in text mode.
  hearing.hidden = true;
  row.append(laneHead(lane), list);
  return row;
}

function stateChip(id, label) {
  const chip = el("div", "tool state-chip");
  chip.id = `voice-${id}`;
  chip.append(el("div", "tool-top", label), el("div", "tool-detail"));
  return chip;
}

function meter() {
  const meter = el("span", "meter");
  meter.append(el("span"));
  return meter;
}

function renderLanes() {
  for (const chip of document.querySelectorAll(".tool[data-tool]")) {
    const name = chip.dataset.tool;
    const detail = chip.querySelector(".tool-detail");
    let active = false;
    let locked = false;
    let text = "";
    if (chip.dataset.lane === "body" && !chip.classList.contains("control")) {
      active = state.body?.tool === name;
      locked = state.body !== null && !active;
      if (active) text = state.body.what;
      else if (locked) text = "waits for the body";
    } else if (name === "look_at_user" && state.gaze !== null) {
      active = true;
      text = `looking at ${state.gaze}`;
    }
    chip.classList.toggle("active", active);
    chip.classList.toggle("locked", locked);
    detail.textContent = text;
    if (active && chip.dataset.lane === "body") {
      detail.append(el("span", "timer", ` · ${elapsed(state.body.since)}`));
    }
  }
  const status = $("body-state");
  if (status !== null) {
    status.textContent = state.body === null ? "Free" : `Busy · ${elapsed(state.body.since)}`;
    status.classList.toggle("busy", state.body !== null);
  }
  renderVoice();
}

function renderVoice() {
  const speaking = $("voice-speaking");
  if (speaking === null) return;
  speaking.classList.toggle("on", state.speaking !== null);
  speaking.querySelector(".tool-detail").textContent = state.speaking ? "talking" : "quiet";
  const mute = $("voice-mute");
  mute.classList.toggle("warn", state.muted);
  mute.querySelector(".tool-detail").textContent = state.muted
    ? "muted until an operator says “you can talk”"
    : "anyone can say “stop talking”";
}

function renderStatus() {
  $("muted-pill").hidden = !state.muted;
  renderVoice();
}

function flash(name, className) {
  const chip = document.querySelector(`.tool[data-tool="${CSS.escape(name)}"]`);
  if (chip === null) return;
  chip.classList.remove("called-ok", "called-error");
  void chip.offsetWidth; // Restarts the animation.
  chip.classList.add(className);
  setTimeout(() => chip.classList.remove(className), FLASH_MS);
}

let shownVad = 0;

function animateMeter() {
  shownVad = Math.max(state.vad, shownVad * 0.9);
  state.vad *= 0.8;
  const chip = $("voice-hearing");
  if (chip !== null) {
    chip.querySelector(".meter span").style.width = `${Math.round(shownVad * 100)}%`;
    const hearing = shownVad > 0.5;
    if (chip.classList.contains("hearing") !== hearing || !chip.dataset.drawn) {
      chip.classList.toggle("hearing", hearing);
      chip.querySelector(".tool-detail").textContent = hearing ? "someone is talking" : "listening";
      chip.dataset.drawn = "yes";
    }
  }
  requestAnimationFrame(animateMeter);
}

// Conversation

function keepingBottom(change) {
  const log = $("log");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
  change();
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function addItem(item) {
  keepingBottom(() => $("log").append(item));
}

function addEvent(at, tone, text) {
  const item = el("li", `event ${tone}`.trim(), text);
  item.append(el("time", null, clock(at)));
  addItem(item);
}

function onUserLine(record, at) {
  const item = el("li", "msg user");
  const who = el("div", "who");
  who.append(...speaker(record.speaker), el("time", null, clock(at)));
  item.append(who, el("div", "bubble", record.text));
  addItem(item);
}

function speaker(speaker) {
  if (speaker === null) return [el("span", "name", "User")];
  if (speaker.person === null) return [el("span", "name stranger", `Unknown #${speaker.track_id}`)];
  state.people.set(speaker.person.user_id, speaker.person.name);
  const parts = [el("span", "name", speaker.person.name)];
  if (speaker.operator) parts.push(el("span", "badge", "operator"));
  return parts;
}

function robotMessage(at) {
  const item = el("li", "msg robot");
  const who = el("div", "who");
  who.append(el("span", "name", state.name), el("time", null, clock(at)));
  item.append(who, el("div", "bubble"));
  addItem(item);
  return item;
}

function onSentence(record, at) {
  if (state.speaking === null) {
    state.speaking = robotMessage(at);
    state.speaking.classList.add("speaking");
  }
  const bubble = state.speaking.querySelector(".bubble");
  keepingBottom(() => {
    bubble.textContent = bubble.textContent ? `${bubble.textContent} ${record.text}` : record.text;
  });
  renderVoice();
}

function onRobotLine(record, at) {
  const item = state.speaking ?? robotMessage(at);
  state.speaking = null;
  item.classList.remove("speaking");
  // What was heard, which is less than what was started if the user talked over it.
  if (record.text) item.querySelector(".bubble").textContent = record.text;
  if (record.interrupted) {
    item.classList.add("interrupted");
    item.querySelector(".who").append(el("span", "cut", "cut off"));
  }
  renderVoice();
}

function onTool(record, at, fresh) {
  const result = summarize(record.result);
  const item = el("li", result.error ? "call error" : "call");
  const head = el("div", "call-head");
  head.append(
    el("code", null, record.name),
    el("span", "args", describeArgs(record.arguments)),
    el("time", null, clock(at)),
  );
  const short = `${result.error ? "✗" : "→"} ${result.text}`;
  const outcome = el("div", "call-result", short);
  outcome.title = "Click for what the LLM was told";
  outcome.addEventListener("click", () => {
    const open = outcome.classList.toggle("open");
    outcome.textContent = open ? record.result : short;
  });
  item.append(head, outcome);
  addItem(item);
  if (fresh) flash(record.name, result.error ? "called-error" : "called-ok");
}

function describeArgs(json) {
  let args;
  try {
    args = JSON.parse(json || "{}");
  } catch {
    return json;
  }
  if (args === null || typeof args !== "object") return json;
  return Object.entries(args)
    .map(([key, value]) => {
      if (key === "user_id") return `${key}: ${personName(value)}`;
      return `${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`;
    })
    .join(", ");
}

function summarize(result) {
  let value;
  try {
    value = JSON.parse(result);
  } catch {
    return { error: false, text: result };
  }
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return { error: false, text: JSON.stringify(value) };
  }
  if ("error" in value) return { error: true, text: String(value.error) };
  // `next` tells the LLM what to do next: shown only in the whole result.
  const parts = Object.entries(value)
    .filter(([key]) => key !== "next")
    .map(([key, field]) => {
      const text = typeof field === "string" ? field : JSON.stringify(field);
      return key === "status" ? text : `${key}: ${text}`;
    });
  return { error: false, text: parts.join(" · ") };
}

function personName(userId) {
  return state.people.get(userId) ?? `user ${String(userId).slice(0, 8)}`;
}

// Long-running actions

function onTask(task, at, fresh) {
  if (task.status.state === "running") {
    state.body = { id: task.id, tool: "start_task", what: task.instruction, since: at.getTime() };
    renderLanes();
    return;
  }
  ended(task.id, "start_task", task.instruction, task.status, at, fresh);
}

function onMotion(motion, at, fresh) {
  const tool = MOTION_TOOLS[motion.action.kind];
  const what = describeAction(motion.action);
  if (motion.status.state === "running") {
    state.body = { id: motion.id, tool, what, since: at.getTime() };
    renderLanes();
    return;
  }
  ended(motion.id, tool, what, motion.status, at, fresh);
}

function describeAction(action) {
  switch (action.kind) {
    case "go_to_location":
      return `walk to the ${action.location}`;
    case "go_to_human":
      return `walk to ${personName(action.user_id)}`;
    case "animation":
      return action.animation.replaceAll("_", " ");
  }
  return action.kind;
}

function ended(id, tool, what, status, at, fresh) {
  if (state.body?.id === id) state.body = null;
  renderLanes();
  const [tone, text] = outcome(status, what);
  addEvent(at, tone, text);
  if (fresh && tone !== "") flash(tool, tone === "failure" ? "called-error" : "called-ok");
}

function outcome(status, what) {
  switch (status.state) {
    case "succeeded":
      return ["success", `✓ Done: ${what}`];
    case "failed":
      if (status.cause === "timeout") return ["failure", `⏱ Gave up, too long: ${what}`];
      return ["failure", `✗ Could not ${what}: ${status.reason}`];
    case "cancelled":
      return ["", `■ Stopped: ${what}`];
  }
  return ["", `${status.state}: ${what}`];
}

function onGaze(person) {
  if (person !== null) state.people.set(person.user_id, person.name);
  state.gaze = person === null ? null : person.name;
  renderLanes();
}

function onMute(muted, at) {
  state.muted = muted;
  addEvent(
    at,
    "attention",
    muted ? "Muted: says nothing until an operator lets it talk" : "Can talk again",
  );
  renderStatus();
}

function onFace(event, at) {
  if (event.kind === "appeared") {
    const person = event.face.person;
    if (person !== null) state.people.set(person.user_id, person.name);
    const who = person === null ? `Someone unknown (#${event.face.track_id})` : person.name;
    addEvent(at, "", `${who} came into view`);
    return;
  }
  const who = event.person === null ? `Unknown #${event.track_id}` : event.person.name;
  addEvent(at, "", `${who} left`);
}

// Typing to the robot

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $("message");
  const text = input.value.trim();
  if (text === "" || state.socket?.readyState !== WebSocket.OPEN) return;
  state.socket.send(JSON.stringify({ text }));
  input.value = "";
});

setInterval(() => {
  if (state.body !== null) renderLanes();
}, 1000);
requestAnimationFrame(animateMeter);
connect();
