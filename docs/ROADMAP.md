# Roadmap

What is known to be missing or weak, and what was decided about it. See `ARCHITECTURE.md` for
what exists and why.

## Interaction

- **Whether an utterance is said to the robot** is the weakest judgement. With several people
  it rests on how much of the utterance its speaker looked at the robot, and a glance is enough
  for someone engaged: on recorded group conversations the verdict flips. Either tune the rule
  on recordings (weigh the end of the utterance, ask for more than a glance), or label a few
  minutes and fit a small model.
- **A learned model for engagement and the addressee.** The scores are recorded each frame to
  learn from later. Labels can come from what happened next (who went on to talk to the robot),
  or from public data: Ego4D has "looking at me" and "talking to me" labels, UE-HRI has
  engagement with a robot; their licences and formats are not checked yet.
- **Tuning with several people on the real robot**: the share of an utterance a face must be
  seen speaking, the shortest piece a turn is cut into, the gaze thresholds, and barge-in by
  engaged people only while the robot itself speaks. Nothing was tuned beyond one person and
  two videos.
- **Turns cut in the middle of a sentence**, where someone throws in a second of speech, or at
  the 20 s cap. A margin between the surest speaker and the next was considered.
- **Engagement flaps** in a seated group that glances at the camera now and then.
- **Speaking keeps someone engaged whoever they speak to.** It should only when nobody else is
  in view, like the rule that trusts a lone speaker.
- **A reply stops at the end of its sentence** when its listener leaves, which is long for a
  long sentence. It could stop at once when they have left the view, and still finish the
  sentence when they only look away. Left as it is for now.
- **Someone unknown who leaves the view and comes back** is a newcomer: a reply cut short for
  them is not picked up, and they are greeted again.
- **Other formulas for the attention's score.** It is a weighted mean of the features times the
  habituation, so a feature can only weigh so much against the rest, and each new one dilutes
  the others: the Reachy profile weighs speaking and a raised hand 3 each, of 10.5, and the
  margin and minimum score were scaled down with the sum. Habituation also wears the speaking
  bonus down: a speaker attended long scores less than a fresh face would speaking. A speech
  onset could spike like novelty does, or speaking could hold a floor while it lasts; to
  explore.
- **Someone the head does not look at reads further from eye contact** in the simulation, so
  engages slowly, if at all. It is not the perspective: the gaze's crop is now normalized to a
  virtual camera looking at the face, which takes the image's skew out, but changed the readings
  of a `newcomer` session by about 1 degree. Off the image's centre (over 30 degrees), the avatars
  look away a quarter of the time, against 4 % centred: their clips turn their heads while the
  robot attends to someone else. Without those moments, Cleo still reads 3 to 8 degrees further,
  mostly in pitch, about 10 degrees higher, with her face higher in the image; rotating the
  camera alone does not move the pitch. To compare with the head controller's recording: the
  head's pitch and roll, and how far its turns move the camera.
- **The head does not favour the person the robot talks with**: the attention scores them like
  anyone else.
- **Permissions look at the face in front**, not at the face seen saying the request.
- **A plan going on can take up what a stranger asked meanwhile.** The next step of an
  operator's plan may start with nobody in front. The LLM chooses that step, with all it heard
  in mind: a request it declined to someone else during the plan is not kept from it by code.
- **Voices out of view are nobody's.** The reSpeaker tells where they come from, but nothing
  turns to them yet: speech from out of view could be a candidate of the attention, so that the
  head turns to look, and the utterance could carry its direction.
- **The sound's direction is not measured against faces yet.** The 15 degrees a face may be from
  the speech, and the transform from the array to the camera, are guesses: check them with
  `lune doa` and with recordings of several people. The array's offset from the camera is left
  out, which is wrong by a few degrees for someone close and off to the side.
- **Steering the array's beams** at the people engaged, which the chip's fixed-beam mode can do
  for two: each beam's speech energy would then tell which of them speaks, and the second USB
  channel could carry the second beam, a stream per person when they talk over each other. It
  waits until the directions heard prove right; the beams' gating must stay off, or someone out
  of view could no longer interrupt.

## Robot body

- **The head follows the attention's target late**, by a quarter to a third of a second in the
  simulation (`npc walk`): the camera pipeline's latency, then the motors'. Each target is placed
  from where the head was when its frame was taken, so the lag does not make it oscillate, but
  nothing anticipates: the target's velocity in the base frame, over the last frames, would let
  the controller aim where it will be. Raw targets, sent each frame, also jitter with the face
  mesh; a filter would go with the prediction.
- **A restarted head controller** replaces its `camera_pose` topic, and the perception stops
  hearing from it until it restarts too: the head's turns then look like motion again.
- **Tracks can still swap in a very fast head swing.** With the camera's turn taken out, the
  predicted image shift matched the faces' within 10 to 30 pixels at a swing of 100 pixels a
  frame. In one run, though, faces swapped tracks as the image slid about 130 pixels in a frame,
  as if the turn had not been applied. The pose is the one nearest the frame's time within 50
  ms, which at that speed can be some 65 pixels behind: interpolating between the poses either
  side of the frame's time, and logging a pose too old, would tell.
- **A frame's time is taken once it is read**, not when it was taken: the pipeline reads the
  latest frame, up to a frame period old, so a moving head places its targets a little off. The
  robot stamps its frames; `FrameSource.read` could pass the stamp on.
- **The head's yaw strays** by a few degrees in the simulation when the platform turns it near
  Pollen's 65° from the body: the motors' small errors weigh more there. The profile keeps 30°.
- **The camera's pose is computed, not measured**: the controller has it by forward kinematics and
  the MJCF's `camera_optical` site. The simulation knows the real one and could publish the head
  pose as ground truth to check it. The intrinsics are in the profile, from the MJCF's field of
  view.
- **The simulation in four layers**, to be kept apart as it grows:
  - *Describing a scene*: which robots, assets and HDRI to load, at which pose, at the start.
    Today `rva_robot_sim.scenes` for the table and the people, written as functions, and
    `sim.yaml` (`SimConfig`) for the Reachy Mini, the renderer and the rates.
  - *Controlling the robots*: bridging their sensors and actuators with outside controllers. Today
    `RobotBus` and `serve`, over the robot's kickmsg topics.
  - *Controlling the NPCs*: today `rva_robot_sim.npc`, which places people through the
    `person_pose` mailbox of the simulation's `sim` node, by name, and its `npc` CLI. Only the
    scenario, inside the simulation, tells them to play a clip or to talk (`steps.Stage`);
    nothing from outside does, and nothing tells them to look somewhere.
  - *Executing a scenario*: what the people do and say, to test the interactive loops
    (attention, visual servoing, engagement, the dialogue) by hand and in integration tests. A
    py_trees behavior tree of steps (`rva_robot_sim.scenarios`, `rva_robot_sim.steps`), ticked
    by the simulation's loop, picked by name by `sim.yaml`'s `scenario`, `lune --scenario` or
    `lune-sim --scenario`; one person may be live, someone at the microphone speaking for them.
    The people idle until `lune` has everything up, the agent last, and starts the scenario
    through the sim's `scenario_start` mailbox (or `npc start`). Steps wait on the robot's
    speech, which the agent publishes on the `robot_speech` topic. Not yet:
    - **The waits' times come from four runs of `newcomer`**: 2 s for the robot to start an
      answer, 4 s after a name, 5 s for a greeting after a turn, a story cut short 5 s in.
      The last run was clean; `greeting` was not run since it waits on the robot.
    - **The people must keep facing the robot while it talks to them**, or it stops: a glance
      away of 0.9 s cut a story. Cleo plays a listening clip that keeps her head within 10
      degrees, Ben his steadiest idle clip, 24 at most; the men's listening clips turn it 40.
      To settle on the agent's side: how long a look away ends an engagement.
    - **A line of two sentences is heard as two utterances**, the turn ending after 0.2 s of
      silence: the robot starts answering the first and the second cuts it, a second lost.
      Ben's was split in two lines with a wait; Mia's two still do it.
    - **A view of a run as a timeline**: a lane of speech and one of clips per person, the
      robot's speech, and each wait with how it ended. A tool of its own, not the session
      page; the steps' starts and ends are only logged today, and would be published for it.
    - **Waits on more than speech**: on an engagement or the robot's gaze, e.g. Cleo giving her
      name once the robot looks at her. A wait for the robot's speech is met by anything it
      says, to anyone.
    - **Checks of what the robot did**, for integration tests: a wait that fails already fails
      the scenario unless wrapped with `anyway`.
    - Walking to a given point and looking somewhere.
- **The simulated people walk only where their clips take them.** In `greeting` they idle, talk
  with gestures and listen, blinking, skinned in numpy and shown by Nyx through a patch
  ([AVATAR_ANIMATION.md](AVATAR_ANIMATION.md)). Mia says her lines at their times, gesturing,
  synthesized once by Kokoro and played into the `rva_sim_room` PipeWire sink the agent listens
  to, her mouth opening as loud as the line is: one viseme, `AA_VI_10_aa`, not the shapes of the
  sounds yet. In `greeting_as_mia` or `greeting_as_ben` the real microphone speaks for one of
  them instead, the mouth following its level, with no gesture. In `newcomer` Mia and Ben turn
  to the robot with turn clips, and Cleo walks in with a walk's start, a cycle and its stop, then
  a turn, played as one step: she ends where Rocketbox's clips take her, which the scene was
  placed for by hand. `npc walk` still slides one sideways without walk clips, and nobody turns
  only their head or eyes to look at the robot. Clips start and end without blending, so the
  pose jumps when a gesture cuts in or ends.
- **Onboarding in the simulation.** The Reachy profile makes a new face database for each
  session, so the avatars are strangers who meet Milo as new users would, every run, and the
  first person named becomes an operator. The profile sets `active_speaker: null`, which hands each
  utterance to the face in front. In `greeting` Mia's mouth now follows her lines, so
  Light-ASD could see her speak: **try putting it back in `config/reachy_mini_sim/runtime.yaml`**.
  Someone talking through the real microphone does so as a live person in the scene, whose mouth
  moves with their voice, so Light-ASD sees them speak too; how well it does on a mouth that only
  opens and shuts is untested.
- **Reachy's gestures go to the mock's body API**, not to the simulated head, which follows the
  attention alone: a body client over kickmsg is to replace it, and to decide with the attention
  who has the head.
- **Only Reachy's head follows the attention.** The agent no longer turns a body to someone
  with a tool, and the attention's targets go to a head controller alone: a humanoid looks at
  nobody until its body takes them too.
- **The head controller does not notice a robot that stopped** until its commands pile up.
- **The head looks up 7 to 10 degrees more than it is sent** while it looks up at people: its
  weight pulls it back, and P-only servos hold it within their dead band. Faces sit a little low
  in the image. Correcting the aim by the measured error, or servos identified on a robot, would
  tell whether the real one does the same.
- **The servos' gearboxes are guessed.** Their inertia and friction are Pollen's estimate for the
  XC330 ("would need to re-identify"), which leaves each simulated servo a dead band of about
  3 degrees and the head a few degrees off its goal; their speed and torque are ROBOTIS'. A step
  response measured on a Reachy Mini would settle them.
- **The simulated direction of arrival is exact.** The simulated microphones hear the loudest
  person speaking from their head, with no error, delay or echoes, where the reSpeaker is off
  by several degrees and 60 ms late. Light-ASD is back on in the Reachy profile with it. In one
  run of `newcomer` it flagged two or three faces in most frames of a line, listeners included;
  with the direction, the face picked was the talker in 17 and 18 of the 19 frames of two lines.
  More lines, and the reSpeaker's error, are not measured yet.
- **The attention swings between two people every few seconds.** In `greeting` it goes from Mia
  to Ben and back every 2 to 7 s, as habituation wears the one attended down; the head now
  turns smoothly, but often. With nobody in view the head drifts back ahead; exploring would be
  better.
- **A restarted simulation** replaces its topics: clients of the old one stop hearing from it and
  raise once silent too long, rather than reopening the new ones. Restart them with it.
- **The real Reachy Mini** needs a bridge from Pollen's daemon to the same topics, so that the
  controller and the clients do not change.

## Sound

- **Which of the reSpeaker's channels to recognize.** Lune reads the first, the selected beam
  after all the chip's processing. The second carries the beam's speech recognition output,
  without the noise and residual echo suppression and the gain control, whose artifacts can
  cost a recognizer words, but with the robot's own voice left in it (0.045 against 0.0003 RMS
  during a chirp). Both could be read: the voice activity and barge-in on the first, which is
  free of echo, and Parakeet on the second over the same stretch, while the robot is silent.
  Record both while people talk, with and without the robot speaking, and compare the
  transcripts' errors and the false barge-ins.
- **Links from the robot's description.** Transforms are given in the profile, between two
  links at a time, and fixed. Read from the URDF and the joints, chained through the tree, the
  array and the camera could sit on links that move apart, the head turning above the body.

- **The last second of the microphone** is missing from a recording: the frames are joined into
  clips of a second, and the one still filling when the session ends is not written.
- **A live waveform instead of the level.** The debug window and rerun show how loud the
  sound was, a point per 32 ms, on a scale from 0 to 1 that leaves quiet speech low. What is
  wanted is the waveform itself, scrolling live.

## Tools

- **Turns that are dropped** as not said to the robot are only logged: they are neither on the
  session page nor in the recording with their scores.
- **The session page does not follow a replayed video**; only the debug window does.
- **`lune rerun` opens the latest session only.**
- **The debug window's checkboxes all start ticked.**

## Beyond the simulation

Left from a review of what only holds in the `newcomer` scenario: its exact timestamps, steady
30 frames a second, microphone without echo, and Qwen's habits. Found by reading the code, not
by a failure seen, unless said.

### Dialogue

- **A turn ends when the LLM calls a tool without a word**, which fits Qwen writing its words and
  its calls in one message: asked again, it made up an event and said it. A model that calls
  first and speaks after would gesture and say nothing. To measure against the LLM before
  changing: one more round only when a person spoke and nothing was said.
- **A line said while the robot spoke is told "that reply already answered it"**, written for
  "Someone's coming" as the robot greets the newcomer. A real question asked meanwhile is told
  the same. It also compares the wall clock to the recognizer's times, so it fires more with a
  slower recognizer. To measure: stating only the fact answered the same again 5 times in 10.
- **Someone else's name is taken for the speaker's.** Asked their name, "Ben said you tell good
  stories." was remembered as Ben every time, by the reply and by the check: the name was said,
  so it passes.
- **Stage directions are taken out by their asterisks**, with emphasis and "2 * 3" too, and not
  "(waves)" or "[nods]", which other models write. With the greeting naming an animation the
  robot has, the log tells whether any are still written.
- **English is in the code**: the lines said without the LLM ("Stopping.", "Okay.", "I can talk
  again."), Parakeet's English model, Kokoro's `en-us`, and a sentence splitter that knows no
  Chinese or Japanese punctuation, so such a reply is said only once whole.
- **A stop word is the whole utterance or nothing**: "please stop" goes to the LLM. It also
  waits for the turn under way to end, the name check included, before the body stops.
- **The history drops a message per message** once full, so the start of what the LLM server
  has read changes every turn and it reads everything again: drop them by blocks.
- **A reply cut at `max_tokens` in the middle of a call** comes back as invalid arguments, the
  reason unread; and no request to the LLM times out.
- **A call repeated in a turn ends it without a word**, right after the robot said it was
  doing it.

### Voice

- **Barge-in is only armed when a turn starts.** A cough that opens a turn just before the
  robot speaks leaves nobody able to interrupt that reply. It was also only tried in the
  simulation, whose microphone hears nothing of the robot's voice: 0.8 and 400 ms are unproven
  against an echo. And the robot counts as silent once its samples are handed to the sound
  card, not once they are heard: late by 100 to 200 ms over Bluetooth or HDMI. A timeline of what
  is played, with the output's latency, would serve all three.
- **A one-word answer the robot just said is dropped as its echo** ("You." after "you"): text
  cannot tell them apart, the timeline could.
- **Speech is timed when it is processed**, not when it was captured, and the voice activity
  and end-of-turn models run in the event loop: on a slower machine the frames queue up and the
  times drift from the camera's and the array's, which are matched within 0.2 s.
- **16 kHz, 512 samples and Kokoro's 24 kHz are opened as they are** on the device: Thor needs
  an ALSA plug outside the repository for it. Resample in the microphone and take the rate from
  the synthesizer.

### Perception

- **A recognized track is never checked again.** One swap as two people cross, or one wrong
  match, keeps the wrong name for the track's life, denies the right person theirs, and
  `remember` then stores it. Embed recognized tracks now and then, and drop a match that keeps
  disagreeing.
- **`min_face_height` is both how far to care and how small the models can read.** The Reachy
  profile's 0.06 lets in faces of 43 pixels, with the thresholds of recognition, gaze and lips
  unchanged: fine on rendered faces, unlikely on a real camera's. Separate a range, from the
  intrinsics, from a smallest size per model.
- **A new track dies on one missed detection** before it is announced: 15 in a row at 30 frames
  a second, which a real face far away or in profile may never give. Loosening it makes ghosts:
  to judge on real footage.
- **Overlap, then distance, then embedding, each greedy.** After a long stall the distance gate
  is wide, and a face that came during it can take the track of one that left. One cost, solved
  once, on bearings rather than pixels.
- **The camera's pose is only known for the robot's own camera with a head controller**, at the
  frame's time as the simulation stamps it. A webcam on a moving head gets no compensation, and
  no warning; a real camera's 50 to 100 ms of latency has no setting; and the robot's clock is
  taken for this machine's.
- **The fixed frame is the robot's base**, so a base that moves makes everyone move, and the
  camera's travel is left out of the turn.
- **Monocular depth is trusted for where to look**, then held back by two constants: a hand no
  more than 0.6 m in front of its owner's face, in depth only and on the near side only, and a
  point no nearer than 0.5 m. Aim by bearing, with depth for the parallax alone.
- **A pinhole camera with guessed intrinsics**: no distortion, and the C920's and D455's are
  from their datasheets, which every distance scales with. Thor has no depth at all, so nothing
  is placed in 3D and the attention finds nobody.
- **Two people of the same name are merged** by the face database, and a face is embedded
  inside the lock the event loop takes to read the faces.

### Attention and the head

- **People are tracks in the image, forgotten after 2 s out of it.** With a narrower camera, or
  people spread wide, looking at one makes another leave: disengaged, greeted again, and new to
  the attention. Keep people in the base frame, and tell out of view from gone. It is the cause
  of "someone unknown who leaves the view and comes back", above.
- **The attention's `moving` adds up the path of raw 3D points**, so depth noise reads as speed
  (1.7 cm of jitter a frame is 0.5 m/s) and half the frame rate halves it. The path is meant: a
  hand waving in place goes nowhere. To tune on recordings, on bearings rather than depth.
- **A raised hand is a pose, not an event**: a phone at the ear or a chin on a hand scores like
  a wave, and takes the head back each time its habituation recovers.
- **A pose the head cannot reach holds every axis**: someone close to a tabletop robot
  saturates the pitch, and the yaw stops following too.
- **The head controller's waking, recentring and holding are in `rva-app`, untested**, and it
  wakes on the pose it sent, not the one measured.
- **Three angles for eye contact** (12 and 18 degrees, 15, 20) in the vision, the interaction and
  the attention, and a second, older attention in `rva-vision-faces` that picks whose hands are
  fitted, apart from the one the head follows.
- **A step of the head covers 0.1 s at most**, a constant: a robot that reports slower than 10
  times a second would move slower than its limits.
- **Whoever speaks draws the head three times as much in the Reachy profile only.** The default
  profile's weight of 1 "often left the head on a listener", and was not changed.

### Starting it all

- **A robot in the profile means a simulation to `lune`**: it refuses to start if the robot's
  topics exist, starts `lune-sim` and its scenario. The real Reachy Mini's bridge would be
  refused. `rva-app` and the head controller also import `rva-robot-sim`, Genesis and all, for
  the robot's description. Say in the profile whether the robot is simulated or there already,
  and move the description out of the simulation's package.
- **What `lune` starts is a list in the code**, the mock among it: probed at the body's address
  and started at the VLA's, so a body slow to answer gets a mock on a real VLA's address. `lune`
  does not handle SIGTERM either, and under systemd would leave the LLM server and the
  simulation running.
- **`cli.py` builds the perception four times**, a little differently each, and the voice and
  text agents repeat each other.
