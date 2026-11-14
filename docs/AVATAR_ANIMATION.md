# Proposal: people that move in the simulation

The people around the simulated Reachy Mini stand still in an A-pose. The perception has more to
work with in people who move: a sway while they stand, walking in and out of view, turning to the
robot or away, gestures, and a mouth that moves while they talk. This proposes how to animate
them, from what was tried. The first part is built, as an experiment apart from the simulation:
`uv run animate` plays clips on one avatar, skinned in numpy (`rva_robot_sim.skinning`), shown by
Genesis' rasterizer or by Nyx through a patch of its plugin (`rva_robot_sim.nyx_skinned`), and
films it. In the simulation, the Reachy profile's scenario, `greeting`, has two people idle,
gesture and talk, as `rva_robot_sim.scenarios` scripts them; `still` keeps them still.

## What was tried

A prototype, outside the repository, animated a Rocketbox avatar in Genesis and rendered it with
Nyx: Female_Adult_01 playing a talking gesture of Rocketbox's own, her mouth opened by one of its
face shapes, at the mesh's full quality, without seams. Three ways were compared.

| Approach | Nyx | Looks | Cost per person and frame |
|---|---|---|---|
| Skinning in numpy, the vertices written into Genesis | With a patch to the plugin | The full mesh, no seams | ~0.5 ms |
| A rigid body per bone, posed each step | Out of the box | Cracks at the waist, jaw and knees | ~0.4 ms |
| A mesh per frame, baked and swapped | Loads every frame's textures | Full mesh, no head turns or face | Memory: ~30 MB per 15 s clip |

**Skinning in numpy wins.** Genesis lets the vertices of a visual-only mesh be set at each step
(`gs.morphs.Mesh(enable_custom_vverts=True)`, `RigidEntity.set_vverts`), and its rasterizer shows
them. Nyx 0.1.10 does not: it streams only the poses of rigid bodies, and the vertices of soft
bodies. It has what is needed, though, a `DynamicMesh` instance and a vertex buffer it shares with
the GPU: a patch of about 150 lines exports a mesh with custom vertices as a `DynamicMesh`, with
its texture, and copies its vertices into that buffer at each update. It costs 0.15 ms more a
frame. The patch replaces private methods of the plugin, so it pins its version until the change
is made upstream (Apache-2.0, github.com/Genesis-Embodied-AI/genesis-nyx).

The rigid parts need no patch but crack where the skin bends, on the face and the mouth among
other places, which the perception looks at most. The baked frames cost memory for nothing and
cannot turn a head towards the robot.

## Where the animations come from

Rocketbox's own, from the same repository at the same pinned commit, MIT-licensed, downloaded
without an account: 417 FBX clips at 30 frames a second, on the avatars' skeleton, bone for bone.

- 326 in place: idling, breathing, looking around, talking and listening with gestures, waving.
- 71 with root motion on the floor: walks and runs.
- 74 with root motion in space: turns from 60 to 180 degrees, starting to walk, sitting down.
- For each avatar a face FBX with the same mesh and 175 face shapes: 15 visemes, the 52 of ARKit
  (`JawOpen`, the blinks...) and FACS units. `AA_VI_10_aa` opens the mouth by up to 22 mm.

One catch: the clips' rest pose has the arms down, the avatars' an A-pose, so giving a clip to an
avatar as it is leaves it in its A-pose. Blender retargets them with a constraint per bone copying
its rotation in world space, and the pelvis's position; this worked for Female_Adult_01, and is
not checked on the male avatars or across them.

Other banks need a retargeting of their skeleton too, and their licences rule most out: CMU's
motion capture (no restriction) and 100STYLE (CC BY 4.0) could do; Bandai Namco's and LaFAN1 are
non-commercial, AMASS needs a registration, Mixamo an Adobe account.

## The proposal

**Baked once, at first use**, by Blender like the avatars, into the same cache:

- per avatar, its GLB from the face FBX, and next to it its skin: the four heaviest bones of each
  vertex and their weights, the bones' rest poses, and the offsets of a few face shapes
  (`AA_VI_10_aa`, `JawOpen`, the blinks);
- per avatar and clip, the bones' rotations relative to their parents and the pelvis's position,
  frame by frame, retargeted. Relative rotations rather than poses in the world, so that the sim
  can turn a head towards the robot, put a walk on a path of its own, or face a person anywhere.

A first set: idling, looking around, talking with gestures, walking, and the turns; 15 to 25 MB of
FBX. Nothing of it is committed.

**Played in the simulation's loop.** Each person is a mesh with custom vertices, still at the
origin: the sim places it by moving its vertices. Before each frame, at 30 Hz rather than at each
physics step, the sim samples each person's clip, turns their head as asked, computes their bones
and their skin, adds the mouth's shape, and writes the vertices. About 0.5 ms a person, in a frame
of 33 ms. The patch for Nyx loads with the scene; the rasterizer needs none.

**A talking mouth.** Built: a scene's people are scripted or live. A scripted person has a
Kokoro voice and lines, each said at a time of the simulation. `lune-sim` synthesizes them before
it starts, kept under `speech.directory` by voice and text (`rva_robot_sim.speech`), and plays
them as they come due (`rva_robot_sim.room`) into `speech.room`, the `rva_sim_room` PipeWire sink
of `config/pipewire/61-rva-sim-room.conf`, and out of `speech.loudspeaker` for whoever watches.
A live person, one at most, speaks with the real microphone's voice, which the room relays into
the sink in the same audio callback that reads it. The agent listens to the sink's monitor as
its microphone, which is recorded as any other. The weight of `AA_VI_10_aa` follows the line's
loudness, in steps of 10 ms, or the microphone's level between `speech.live_mouth`'s bounds, in
blocks of 10 ms; the visemes of the sounds are next: kokoro-onnx's `create_timed` may give their
timings, else a forced aligner or an audio-to-viseme model, which a live voice would need.

**Told what to do by a scenario first.** Built: a scene (`rva_robot_sim.scenes`) places the
table and the people, each with a name and a voice; a scenario (`rva_robot_sim.scenarios`),
picked by name, gives each of them a script. A script loops idle clips and, at fixed times,
plays a clip or says a line with a gesture; the gesture lasts as long as the line, repeated or
cut, since Rocketbox's talking clips are takes of half a minute. An action cuts the idle clip
playing, and idling goes on after it: the pose jumps at both ends, with no blending yet. Not
yet: a path to walk with the walk clip, facing the robot or a direction. It is deterministic,
which tests want. Later, a `person_command` mailbox of the sim, like `joint_command`, lets a test
or an operator drive them: play a clip, walk to a point, face the robot, talk with a given
loudness. The sim would then publish where each person is, their head and whether they talk, as
ground truth to score the perception against.

## What the experiment found

- The clips need no Blender constraints to retarget. The clips' skeleton and the avatars' are
  both 3ds Max Bipeds with the same bone axes, so each bone's rotation in its parent, taken from
  the clip, poses the avatar as the clip's actor with the avatar's own bone lengths: the arms come
  down from the A-pose. The pelvis's path is scaled by the ratio of their pelvis heights (0.92 m
  to 0.919 for Female_Adult_01). The face's bones (eyes, jaw) keep the avatar's rest pose; the
  face shapes move them.
- The face FBX has a mesh of 4,734 vertices, 80 bones and 175 shapes; Genesis loads it as 5,563,
  split at the texture's seams, each at a skin vertex's exact rest position.
- Posing and skinning take about 0.55 ms a person in numpy, and 1.3 ms a frame with Genesis'
  `set_vverts`. Clips chain through their root motion: a 90-degree turn leaves the next clip
  facing the new way.
- Genesis' rasterizer moves the vertices but keeps the rest pose's normals, which the shading of
  bent limbs barely shows.

## What Nyx needed

- The patch builds on the plugin's buffers shared with torch (0.1.10): the skinned parts are a
  section of the soft bodies' vertex buffer, and Genesis' custom vertices are copied into it
  before each render, GPU to GPU. It wraps five private methods and replaces one, and pins the
  plugin's version.
- Nyx draws an instance it is told is disabled: the GLB's own instance, which the vgeoms' rigid
  references must point at, becomes an empty box instead.
- Genesis keeps a GLB texture's alpha apart, as the surface's opacity texture; it is put back into
  the PNG the patch writes for Nyx, whose materials cut out by the colour texture's alpha. The
  eyelashes' cards are soft-edged: cut at 0.5 their edge darkens the eyes, at 0.75 a little still
  does. The plugin's own path for a GLB draws them solid, as black rims.
- The FBX's materials import fully metallic, which a path tracer shows as skin and cloth mirroring
  the room: the skin's bake sets them non-metallic. The still people's bake (`people.py`) does not
  yet.
- Path traced, the person and its shadows come out right; the mouth's inside shows mid-grey.

## ozz-animation, parked

[ozz-animation](https://github.com/guillaumeblanc/ozz-animation) (MIT, C++) samples, blends
(partial, additive), extracts and blends root motion, solves two-bone and aim IK and skins with
SIMD; its glTF importer needs no Autodesk SDK. It has no Python bindings and no face shapes. It
is parked until numpy is too slow: a binding of a few calls (nanobind) would replace sampling,
posing and skinning, the face shapes staying in numpy; the data is already laid out as ozz's,
rotations in parents and parents before children. A separate process streaming poses over kickmsg
in lockstep with the sim's steps is not worth its protocol: what crosses kickmsg should be
commands to the sim (play a clip, walk somewhere, look at the robot, talk), not frames. Python's
own libraries (pymotion, fairmotion, PyMO) read BVH and do forward kinematics, no more than
`skinning.py`.

## Not known yet

- How the people's animation weighs on the Reachy simulation, which steps its physics at 100 Hz
  and renders at 30: whether ozz becomes worth it.
- The patch over long runs, with several people, and Nyx's cost of rebuilding a deforming mesh's
  acceleration structure at 1280×720 with 16 samples a pixel.
- Whether the hair's and eyelashes' cut-out material renders right everywhere.
- The male avatars and clips, giving a clip to the other sex, and how much feet slide once a
  walk's root motion follows a path instead.
