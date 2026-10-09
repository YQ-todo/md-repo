"""Pre-execution validation of tool calls (pure functions, no side effects)."""
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from .tools import TOOL_ARGS


def validate_tool_call(
    name: str, args: dict, vault_dir: Path, workspace: str | None = None
) -> str | None:
    """Validate args against the tool's Pydantic model + vault boundary check."""
    model = TOOL_ARGS.get(name)
    if model is None:
        return f"Error: unknown tool '{name}'"
    try:
        model(**args)
    except ValidationError as e:
        parts = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) if err["loc"] else ""
            parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
        return f"Error: {'; '.join(parts)}"
    # vault_dir boundary check (needs filesystem context)
    for key in ("path", "from_path", "to_path"):
        p = args.get(key)
        if p is None:
            continue
        full = (vault_dir / p).resolve()
        if not str(full).startswith(str(vault_dir.resolve())):
            return f"Error: path '{p}' resolves outside vault"
    # workspace prefix check (top-level path args)
    if workspace:
        for key in ("path", "from_path", "to_path"):
            p = args.get(key)
            if p is None:
                continue
            if not p.startswith(workspace):
                return f"Error: path '{p}' outside agent workspace '{workspace}'"
    # workspace prefix check (move_docs: each item in moves list)
    if workspace and "moves" in args:
        for m in args["moves"]:
            for key in ("from_path", "to_path"):
                p = m.get(key)
                if p is not None and not p.startswith(workspace):
                    return f"Error: path '{p}' outside agent workspace '{workspace}'"
    return None


def parse_since(s: str) -> float:
    """Parse a human-friendly time string to a Unix timestamp.

    Supports: "5m", "1h", "2d", ISO 8601 ("2026-10-02T14:00:00"), or epoch seconds.
    Returns time.time() - delta for relative strings.
    """
    s = s.strip().lower()
    m = re.match(r"^(\d+)\s*([mhd])$", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = n * {"m": 60, "h": 3600, "d": 86400}[unit]
        return time.time() - delta
    try:
        return float(s)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except ValueError:
        return 0.0
