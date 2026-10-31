from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class RVABaseModel(
    BaseModel,
    extra="forbid",
    frozen=True,
    ser_json_inf_nan="strings",
    use_attribute_docstrings=True,
    validate_default=True,
):
    """A frozen Pydantic model with the project defaults. Base every schema and config on it."""

    model_config = ConfigDict(serialize_by_alias=True)


def load_yaml[TModel: BaseModel](model: type[TModel], path: Path) -> TModel:
    """Validate a YAML file against `model`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        pydantic.ValidationError: If the content does not match `model`.
    """
    return model.model_validate(yaml.safe_load(path.read_text()))
