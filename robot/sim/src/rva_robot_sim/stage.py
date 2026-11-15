"""The simulation as the stage of a scenario: its people saying lines and playing clips."""

from collections.abc import Mapping
from typing import Final

from loguru import logger

from rva_robot_sim.actor import Actor
from rva_robot_sim.animation_assets import ClipName
from rva_robot_sim.room import Room
from rva_robot_sim.speech import ScriptedMouth, SpokenLine
from rva_schemas.audio import AudioClip


class SimStage:
    """Implements `steps.Stage` with the simulation's actors, their mouths and the room's sound.

    Its clock is set by the simulation's loop before each tick of the scenario.
    """

    def __init__(
        self,
        actors: Mapping[str, Actor],
        mouths: Mapping[str, ScriptedMouth],
        lines: Mapping[tuple[str, str], AudioClip],
        room: Room,
    ) -> None:
        """Nothing starts yet.

        Args:
            actors: Everyone in the scene, by name.
            mouths: Of those who say the script's lines, by name: everyone but a live person.
            lines: Every line of the script as it sounds, by who says it and its text.
            room: Where the lines are heard.
        """
        self._actors: Final = actors
        self._mouths: Final = mouths
        self._lines: Final = lines
        self._room: Final = room
        # The line each person is saying, or said last.
        self._saying: dict[str, AudioClip] = {}
        self.time = 0.0
        """Seconds of simulated time."""

    def now(self) -> float:
        return self.time

    def say(self, who: str, text: str, gesture: ClipName) -> float:
        logger.info("{:.1f} s: {} says {!r}", self.time, who, text)
        line = self._lines[who, text]
        self._saying[who] = line
        self._mouths[who].say(SpokenLine(at_s=self.time, clip=line))
        self._actors[who].gesture(gesture, line.duration, self.time)
        self._room.play(line)
        return line.duration

    def play(self, who: str, clips: tuple[ClipName, ...]) -> float:
        logger.info("{:.1f} s: {} plays {}", self.time, who, ", ".join(clip.name for clip in clips))
        return self._actors[who].play(clips, self.time)

    def cut(self, who: str) -> None:
        logger.info("{:.1f} s: {} is cut", self.time, who)
        self._actors[who].cut(self.time)
        if who in self._saying:
            self._mouths[who].shut()
            self._room.cut(self._saying.pop(who))
