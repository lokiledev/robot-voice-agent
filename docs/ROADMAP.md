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
- **Voices out of view are ignored.** A microphone array that tells where a sound comes from
  (ReSpeaker) is planned, to place speakers the camera does not see.

## Robot body

- **Visual servoing.** A controller that keeps a point in space, the attention's target, in the
  middle of the head camera, from its position and velocity, knowing nothing of the attention.
  It needs Reachy Mini's head kinematics: a head pose to the six Stewart motors' angles
  (Pollen's `reachy-mini-rust-kinematics` does it analytically, without their SDK).
- **The camera's pose and intrinsics** are not published: the controller and the perception will
  need where the camera is in the robot's base frame and its focal length. The simulation knows
  both (the MJCF's mount and field of view); the head pose could be published as ground truth too.
- **The simulated camera into the perception stack**, in place of the webcam, so that `lune see`
  and the agent run on the simulation.
- **A restarted simulation** replaces its topics: clients of the old one stop hearing from it and
  raise once silent too long, rather than reopening the new ones. Restart them with it.
- **The real Reachy Mini** needs a bridge from Pollen's daemon to the same topics, so that the
  controller and the clients do not change.

## Sound

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
