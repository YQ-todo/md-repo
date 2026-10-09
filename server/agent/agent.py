"""Agent declaration: dataclass + global registry."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Agent:
    """Declarative agent configuration.

    A new agent = one Agent instance + register() call.
    """
    name: str
    system_prompt: str
    tools: list[str]
    workspace: str | None = None
    auto_tools: set[str] = field(default_factory=set)
    max_rounds: int = 15
    context_window: int = 16384
    read_doc_max_chars: int = 4000


AGENTS: dict[str, Agent] = {}


def register(agent: Agent) -> None:
    """Register an agent into the global table."""
    AGENTS[agent.name] = agent


def get_agent(name: str) -> Agent | None:
    """Look up a registered agent by name."""
    return AGENTS.get(name)
