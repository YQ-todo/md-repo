"""Agent registrations: all agents declared here."""
from __future__ import annotations

from .agent import AGENTS, Agent, register
from .tools import TOOLS


def _register_all() -> None:
    """Register all agents. Called once at import time."""
    if "vault-assistant" not in AGENTS:
        from .prompts import SYSTEM_PROMPT
        register(Agent(
            name="vault-assistant",
            system_prompt=SYSTEM_PROMPT,
            tools=list(TOOLS.keys()),
        ))


_register_all()
