# Proposal: people that move in the simulation

The people around the simulated Reachy Mini stand still in an A-pose. The perception has more to
work with in people who move: a sway while they stand, walking in and out of view, turning to the
robot or away, gestures, and a mouth that moves while they talk. This proposes how to animate
them, from what was tried; nothing of it is built yet.

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

**A talking mouth.** The weight of `AA_VI_10_aa` follows the loudness of the speech: of a clip the
test plays into the agent's microphone, on the same clock, or a syllable-like 4 to 5 Hz
modulation. The sim plays no sound yet, which a test of the active speaker needs.

**Told what to do by `sim.yaml` first.** Each placed person gets a behaviour: a clip looped at a
speed, a path to walk with the walk clip, facing the robot or a direction. It is deterministic,
which tests want. Later, a `person_command` mailbox of the sim, like `joint_command`, lets a test
or an operator drive them: play a clip, walk to a point, face the robot, talk with a given
loudness. The sim would then publish where each person is, their head and whether they talk, as
ground truth to score the perception against.

## Not known yet

- The patch over long runs, with several people, and Nyx's cost of rebuilding a deforming mesh's
  acceleration structure at 1280×720 with 16 samples a pixel.
- Whether the hair's and eyelashes' cut-out material renders right everywhere.
- The male avatars and clips, giving a clip to the other sex, and how much feet slide once a
  walk's root motion follows a path instead.
