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
- **The head's own motion looks like the people's.** The attention measures `moving` and
  `centrality` in the camera's frame. Centering a face is meant to favour it until it habituates,
  but a head turn also makes every face in view seem to move, by up to 0.2 of score for a turn of
  40°. With two still people in the simulation, the face just switched to jumps from about 0.26
  to 0.42 and decays over two or three seconds. The minimum dwell keeps this from flapping, but
  measuring motion in the base frame, from the head's pose at the frame's time as the controller
  does, would leave only the people's own.
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
    `person_pose` mailbox of the simulation's `sim` node, by name, and its `npc` CLI. Nothing yet
    tells them, while the sim runs, to play a clip, to talk or to look somewhere.
  - *Executing a scenario*: a script per person of what they do and say, over time, to test the
    interactive loops (attention, visual servoing, engagement, the dialogue) by hand and in
    integration tests. Built for fixed times (`rva_robot_sim.scenarios`, played by
    `rva_robot_sim.actor`), picked by name by `sim.yaml`'s `scenario`, `lune --scenario` or
    `lune-sim --scenario`; a person's script is scripted or live, someone at the microphone
    speaking for them. The people idle until `lune` has everything up, the agent last, and
    starts the scenario through the sim's `scenario_start` mailbox (or `npc start`). Not yet: steps that wait on the robot
    (its answer, its gaze), walking and looking somewhere, and checks of what the robot did.
- **The simulated people move and speak, but do not walk.** In `greeting` they idle, talk
  with gestures and listen, blinking, skinned in numpy and shown by Nyx through a patch
  ([AVATAR_ANIMATION.md](AVATAR_ANIMATION.md)). Ana says her lines at their times, gesturing,
  synthesized once by Kokoro and played into the `rva_sim_room` PipeWire sink the agent listens
  to, her mouth opening as loud as the line is: one viseme, `AA_VI_10_aa`, not the shapes of the
  sounds yet. In `greeting_as_ana` or `greeting_as_ben` the real microphone speaks for one of
  them instead, the mouth following its level, with no gesture. `npc walk` still slides one
  sideways without walk clips, and they do not turn to the robot or look at it. Clips start and
  end without blending, so the pose jumps when a gesture cuts in or ends.
- **Onboarding in the simulation.** The Reachy profile has a face database and an admins file of
  its own, so the avatar is a stranger who meets Reachy as a new user would, and the first person
  named becomes an operator. The profile sets `active_speaker: null`, which hands each
  utterance to the face in front. In `greeting` Ana's mouth now follows her lines, so
  Light-ASD could see her speak: **try putting it back in `config/reachy_mini_sim/runtime.yaml`**.
  Someone talking through the real microphone does so as a live person in the scene, whose mouth
  moves with their voice, so Light-ASD sees them speak too; how well it does on a mouth that only
  opens and shuts is untested.
- **Reachy's gestures and the agent's gaze go to the mock's body API**, not to the simulated
  head, which follows the attention alone: a body client over kickmsg is to replace it, and to
  decide with the attention who has the head.
- **The head controller does not notice a robot that stopped** until its commands pile up.
- **The servos' gearboxes are guessed.** Their inertia and friction are Pollen's estimate for the
  XC330 ("would need to re-identify"), which leaves each simulated servo a dead band of about
  3 degrees and the head a few degrees off its goal; their speed and torque are ROBOTIS'. A step
  response measured on a Reachy Mini would settle them.
- **The simulated direction of arrival is exact.** The simulated microphones hear the loudest
  person speaking from their head, with no error, delay or echoes, where the reSpeaker is off
  by several degrees and 60 ms late. Light-ASD is back on in the Reachy profile with it; how
  well the two pick the face speaking among the simulated people is not measured yet.
- **The attention swings between two people every few seconds.** In `greeting` it goes from Ana
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
