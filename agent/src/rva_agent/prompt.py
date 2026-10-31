from typing import Final

import jinja2

from rva_schemas.profile import Profile

_ENVIRONMENT: Final = jinja2.Environment(
    loader=jinja2.PackageLoader("rva_agent", "."),
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=True,
)


def render_system_prompt(profile: Profile, *, vision: bool) -> str:
    """Render the system prompt.

    Args:
        profile: Who the robot is and where.
        vision: Whether the robot has a camera and the tools to remember people.
    """
    return _ENVIRONMENT.get_template("system_prompt.md.j2").render(
        vision=vision,
        character=profile.character,
        environment=profile.environment,
        capabilities=profile.capabilities,
    )
