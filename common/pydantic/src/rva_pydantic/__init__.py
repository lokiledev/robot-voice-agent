from pathlib import Path

import yaml
from pydantic import BaseModel


class RVABaseModel(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """An immutable pydantic model that rejects unknown fields. Base every schema and config on it.

    Attribute docstrings become field descriptions, which the LLM reads in tool schemas.
    """


def load_yaml[TModel: BaseModel](model: type[TModel], path: Path) -> TModel:
    """Validate a YAML file against `model`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        pydantic.ValidationError: If the content does not match `model`.
    """
    return model.model_validate(yaml.safe_load(path.read_text()))
