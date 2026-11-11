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
  `rva-robot-kinematics` gives the motors' angles for a head pose, and the head's pose from the
  joints' (forward kinematics, for the camera's pose). In the simulation, the head's yaw strays by
  a few degrees when the platform turns it near Pollen's 65° from the body: the motors' small
  errors weigh more there.
- **The camera's pose is not published**: the controller will need where the camera is in the
  robot's base frame. The simulation knows it (the MJCF's mount), and could publish the head pose
  as ground truth too. Its intrinsics are in the profile, computed from the MJCF's field of view.
- **The simulated people stand still, in an A-pose.** Idle motion, walking in and out of view,
  turning to the robot and a talking mouth would exercise tracking, engagement and the active
  speaker: [AVATAR_ANIMATION.md](AVATAR_ANIMATION.md) proposes how, with Rocketbox's own clips
  skinned in numpy and a patch to Nyx.
- **Onboarding in the simulation.** The Reachy profile has a face database and an admins file of
  its own, so the avatar is a stranger who meets Reachy as a new user would, and the first person
  named becomes an operator. But what is said to Reachy is dropped as said by nobody in view: the
  avatar's lips do not move, so the active speaker never sees it speak. Until they do (see
  [AVATAR_ANIMATION.md](AVATAR_ANIMATION.md)), `active_speaker: null` in the profile hands each
  utterance to the face in front.
- **Reachy's gestures and gaze go to the mock's body API**, not to the simulated head: the head
  controller and a body client over kickmsg are to replace it.
- **The head controller goes straight to each pose** it is sent, with no trajectory or speed
  limit, and does not notice a robot that stopped until its commands pile up.
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
