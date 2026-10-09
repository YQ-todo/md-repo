"""AgentContext: runtime parameter bundle for agent_loop."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .agent import Agent


@dataclass
class AgentContext:
    """Bundles all runtime parameters for agent_loop into one object."""
    agent: Agent
    messages: list[dict]
    pending_confirms: dict
    vault_dir: Path
    db_conn: sqlite3.Connection | None
    last_read_ts: dict = field(default_factory=dict)
    ai_config: object = None  # AIConfig (avoid import cycle)
