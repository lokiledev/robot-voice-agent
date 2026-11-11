from typing import Literal

from pydantic import Field, model_validator

from rva_pydantic import RVABaseModel


class EngagementConfig(RVABaseModel):
    """Who in view pays attention to the robot.

    A person's score moves towards `closeness * staying` while they look at the robot, and
    towards 0 while they do not. They are engaged once it reaches `enter`, until it falls to
    `leave`.
    """

    near_m: float = Field(gt=0)
    """Distance up to which closeness is 1: within reach of a conversation."""
    far_m: float = Field(gt=0)
    """Distance from which closeness is 0."""
    moving_window_s: float = Field(gt=0)
    """How far back a person's speed is measured, from where they were to where they are."""
    walking_m_s: float = Field(gt=0)
    """Speed from which staying is 0: someone walking past."""
    rise_s: float = Field(gt=0)
    """Time looking at the robot for the score to cover two thirds of the way up."""
    fall_s: float = Field(gt=0)
    """Time not looking for it to fall by two thirds. Longer than `rise_s`: a listener looks away
    now and then."""
    enter: float = Field(gt=0, le=1)
    leave: float = Field(ge=0, lt=1)
    max_engaged: int = Field(gt=0)
    """How many people are engaged at once, at most; the others wait for one to disengage."""

    @model_validator(mode="after")
    def _ordered(self) -> "EngagementConfig":
        if self.near_m >= self.far_m:
            raise ValueError(f"near_m {self.near_m} is not under far_m {self.far_m}")
        if self.leave >= self.enter:
            raise ValueError(f"leave {self.leave} is not under enter {self.enter}")
        return self


class InteractionConfig(RVABaseModel):
    """How what the camera and the microphone perceive shapes the conversation."""

    eye_contact_deg: float = Field(gt=0)
    """Angle of the gaze off the camera up to which a person looks at the robot."""
    history_s: float = Field(gt=0)
    """How far back who was speaking and looking is kept: longer than the longest utterance."""
    min_speaking_share: float = Field(gt=0, le=1)
    """Share of an utterance's frames a face must be seen speaking in to have said it. Pauses
    between words count against it, so it sits well below 1."""
    min_segment_s: float = Field(gt=0)
    """The shortest piece a turn is cut into where the face speaking changes; a shorter stretch
    of someone else speaking, a word thrown in, stays in the piece around it."""
    engaged_min_gaze: float = Field(ge=0, le=1)
    """Share of an utterance an engaged speaker must look at the robot for it to be said to the
    robot. Low: they are trusted unless they face someone else throughout."""
    bystander_min_gaze: float = Field(ge=0, le=1)
    """The same for a speaker who is not engaged, who must look at the robot for most of it."""
    barge_in: Literal["anyone", "engaged"]
    """Whose speech stops the robot's: anyone's, or only an engaged person's seen speaking; anyone's
    still while nobody is engaged."""
    engagement: EngagementConfig
