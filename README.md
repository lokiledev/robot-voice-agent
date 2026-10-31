# robot-voice-agent

Local voice agent for a humanoid robot: listens to the user, reasons with a local LLM, talks back, and sends short-horizon natural-language tasks to the robot's VLA policy.

Everything runs on this machine (RTX 4090, 64 GB RAM). Nothing calls a cloud API.

- Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Conventions and package layout: [AGENTS.md](AGENTS.md)
- Robot profiles: [config/](config/)

```bash
uv sync
uv run pytest
```
