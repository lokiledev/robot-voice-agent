from typing import Final

import jinja2

from rva_schemas.profile import Profile

_ENVIRONMENT: Final = jinja2.Environment(
    loader=jinja2.PackageLoader("rva_agent", "."),
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=True,
)


def render_system_prompt(profile: Profile) -> str:
    return _ENVIRONMENT.get_template("system_prompt.md.j2").render(
        character=profile.character,
        environment=profile.environment,
        capabilities=profile.capabilities,
    )
