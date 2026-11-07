from pydantic import Field, model_validator

from rva_pydantic import RVABaseModel


class FeatureWeights(RVABaseModel):
    """How much each feature, from 0 to 1, counts in an entity's score."""

    closeness: float = Field(ge=0)
    centrality: float = Field(ge=0)
    speaking: float = Field(ge=0)
    eye_contact: float = Field(ge=0)
    moving: float = Field(ge=0)
    known: float = Field(ge=0)


class NoveltyConfig(RVABaseModel):
    """The bonus an entity gets for having just appeared."""

    amplitude: float = Field(ge=0)
    """The bonus at the first frame, in the score's units."""
    decay_s: float = Field(gt=0)
    """Time for the bonus to fall to a third."""


class HabituationConfig(RVABaseModel):
    """How an entity's score fades while it is attended, and recovers while it is not."""

    floor: float = Field(ge=0, le=1)
    """What is left of the weighted features once fully habituated."""
    decay_s: float = Field(gt=0)
    """Time attended for the habituation to cover two thirds of the way to the floor."""
    recovery_s: float = Field(gt=0)
    """Time not attended for it to recover two thirds of the way back to 1."""


class VisualAttentionConfig(RVABaseModel):
    """Scores every face and hand in view and in 3D, and picks the one to attend to.

    A candidate's score is its weighted features, times its habituation, plus its novelty. The
    best takes over from the current target only after `min_dwell_s`, when it is better by
    `margin`.
    """

    weights: FeatureWeights
    far_m: float = Field(gt=0)
    """Distance from which closeness is 0."""
    edge_deg: float = Field(gt=0)
    """Angle off the camera's axis from which centrality is 0."""
    eye_contact_deg: float = Field(gt=0)
    """Angle of the gaze off the camera from which eye contact is 0."""
    moving_window_s: float = Field(gt=0)
    """How far back the speed is measured."""
    speed_m_s: float = Field(gt=0)
    """Speed from which moving is 1."""
    novelty: NoveltyConfig
    face: HabituationConfig
    hand: HabituationConfig
    """Faster than a face's: a waving hand draws the eye, then yields to its owner's face."""
    margin: float = Field(ge=0)
    """How much better than the current target a candidate must score to take over."""
    min_dwell_s: float = Field(ge=0)
    """How long a target is kept, at least, before another can take over."""
    hold_s: float = Field(ge=0)
    """How long the target is kept, at its last pose, while it is not seen: a face the mesh or
    the depth missed for a frame has not left. Longer than that, it is dropped at once."""
    min_score: float = Field(ge=0)
    """Below this, nothing is attended: a still person far at the edge is as good as nobody."""
    lifetime_s: float = Field(gt=0)
    """How long an entity is remembered after it was last seen, with its history, novelty and
    habituation: a face lost for a moment comes back as the same entity, without the novelty
    bonus again. Only entities seen in the frame are attended."""

    @model_validator(mode="after")
    def _within_lifetime(self) -> "VisualAttentionConfig":
        for name in ("moving_window_s", "hold_s"):
            if getattr(self, name) > self.lifetime_s:
                raise ValueError(
                    f"{name} {getattr(self, name)} is beyond lifetime_s {self.lifetime_s}"
                )
        return self
