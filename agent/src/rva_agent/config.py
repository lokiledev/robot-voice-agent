from pydantic import Field

from rva_pydantic import RVABaseModel


class AgentConfig(RVABaseModel):
    max_history_messages: int = Field(gt=0)
    """Messages kept besides the system prompt; older ones are dropped."""
    max_tool_rounds: int = Field(gt=0)
    """LLM calls allowed per turn when it keeps calling tools."""
    stop_words: frozenset[str]
    """Whole utterances that cancel the running task without asking the LLM, e.g. "stop"."""
