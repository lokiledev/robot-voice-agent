from pydantic import Field

from rva_pydantic import RVABaseModel


class AgentConfig(RVABaseModel):
    max_history_messages: int = Field(gt=0)
    """Messages kept besides the system prompt; older ones are dropped."""
    max_tool_rounds: int = Field(gt=0)
    """LLM calls allowed per turn when it keeps calling tools."""
    stop_words: frozenset[str]
    """Whole utterances that cancel the running task without asking the LLM, e.g. "stop"."""
    mute_phrases: frozenset[str]
    """Phrases that silence the robot wherever they appear in an utterance, e.g. "stop talking"."""
    unmute_phrases: frozenset[str]
    """Phrases that let a muted robot talk again, e.g. "you can talk again"; operators only."""
    regreet_after_s: float = Field(ge=0)
    """A known person coming back into view sooner than this after leaving is not greeted again."""
