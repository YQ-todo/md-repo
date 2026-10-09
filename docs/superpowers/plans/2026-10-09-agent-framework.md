# Agent 框架抽取 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将单 agent 硬编码实现重构为声明式多 agent 框架，使添加新 agent 变成配置级操作。

**Architecture:** 引入 `Agent` dataclass（声明式配置）+ `ToolDef`（工具自带 confirm 策略）+ `AgentContext`（参数对象），工具执行从 if/elif → 注册表分发。现有 vault-assistant 行为不变，新 agent 只需声明 prompt + 工具子集 + workspace + auto_tools。

**Tech Stack:** Python 3.12, Pydantic v2, OpenAI SDK (AsyncOpenAI), pytest

**Design Spec:** `docs/superpowers/specs/2026-10-09-agent-framework-design.md`

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `server/agent/agent.py` | Create | `Agent` dataclass + `AGENTS` 注册表 + `register()`/`get_agent()` |
| `server/agent/context.py` | Create | `AgentContext` dataclass（运行时参数打包） |
| `server/agent/tools.py` | Modify | 增加 `ToolDef` dataclass，`TOOLS` 从 list → dict[str, ToolDef]，增加 `needs_confirm()` |
| `server/agent/execute.py` | Modify | if/elif → 注册表分发，handler 签名改为 `(args, ctx)` |
| `server/agent/validate.py` | Modify | `validate_tool_call` 增加 `workspace` 参数 |
| `server/agent/loop.py` | Modify | `agent_loop` 参数 → `ctx: AgentContext`，confirm 判断用 `needs_confirm()` |
| `server/agent/__init__.py` | Modify | 适配新导出，保留兼容 re-export |
| `server/api.py` | Modify | agent_chat 路由增加 `agent` 字段，构造 AgentContext |
| `tests/test_agent.py` | Modify | 适配新签名 + 新增 workspace/auto_tools/agent 注册测试 |
| `tests/test_agent_framework.py` | Create | Agent 注册、needs_confirm、workspace 校验的专项测试 |

---

### Task 1: Agent dataclass + 注册表

**Files:**
- Create: `server/agent/agent.py`
- Test: `tests/test_agent_framework.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_framework.py
"""Tests for Agent dataclass, registry, and confirm strategy."""
from __future__ import annotations

import pytest


def test_agent_dataclass_defaults():
    from server.agent.agent import Agent

    a = Agent(name="test", system_prompt="You are test.", tools=["read_doc"])
    assert a.name == "test"
    assert a.system_prompt == "You are test."
    assert a.tools == ["read_doc"]
    assert a.workspace is None
    assert a.auto_tools == set()
    assert a.max_rounds == 15
    assert a.context_window == 16384
    assert a.read_doc_max_chars == 4000


def test_agent_registry_register_and_get():
    from server.agent.agent import Agent, register, get_agent

    a = Agent(name="test-agent-1", system_prompt="p", tools=["read_doc"])
    register(a)
    assert get_agent("test-agent-1") is a
    assert get_agent("nonexistent") is None


def test_agent_registry_overwrite():
    from server.agent.agent import Agent, register, get_agent

    a1 = Agent(name="dup", system_prompt="first", tools=[])
    a2 = Agent(name="dup", system_prompt="second", tools=["read_doc"])
    register(a1)
    register(a2)
    assert get_agent("dup").system_prompt == "second"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_framework.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'server.agent.agent'"

- [ ] **Step 3: Write minimal implementation**

```python
# server/agent/agent.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_framework.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add server/agent/agent.py tests/test_agent_framework.py
git commit -m "feat(agent): Agent dataclass + global registry"
```

---

### Task 2: ToolDef + TOOLS dict 改造

**Files:**
- Modify: `server/agent/tools.py`
- Test: `tests/test_agent_framework.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_agent_framework.py

def test_tooldef_requires_confirm():
    from server.agent.tools import TOOLS

    assert TOOLS["read_doc"].requires_confirm is False
    assert TOOLS["write_doc"].requires_confirm is True
    assert TOOLS["edit_doc"].requires_confirm is True
    assert TOOLS["delete_doc"].requires_confirm is True
    assert TOOLS["move_docs"].requires_confirm is True
    assert TOOLS["search_docs"].requires_confirm is False
    assert TOOLS["list_files"].requires_confirm is False


def test_tools_dict_has_nine_entries():
    from server.agent.tools import TOOLS

    assert len(TOOLS) == 9
    assert set(TOOLS.keys()) == {
        "read_doc", "search_docs", "list_files",
        "write_doc", "edit_doc", "delete_doc", "move_docs",
        "get_related_docs", "recent_changes",
    }


def test_needs_confirm_with_agent():
    from server.agent.tools import needs_confirm
    from server.agent.agent import Agent

    agent_default = Agent(name="a", system_prompt="p", tools=["write_doc"])
    assert needs_confirm("write_doc", agent_default) is True

    agent_auto = Agent(name="b", system_prompt="p", tools=["write_doc"], auto_tools={"write_doc"})
    assert needs_confirm("write_doc", agent_auto) is False

    # Read tools never need confirm regardless of auto_tools
    assert needs_confirm("read_doc", agent_default) is False


def test_tools_schema_generation():
    """TOOLS dict entries can generate OpenAI function schema."""
    from server.agent.tools import TOOLS, tool_schema

    schema = tool_schema("read_doc")
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "read_doc"
    assert "path" in schema["function"]["parameters"]["properties"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_framework.py::test_tooldef_requires_confirm -v`
Expected: FAIL with "AttributeError" or "ImportError"

- [ ] **Step 3: Modify tools.py**

在 `server/agent/tools.py` 中：

1. 顶部增加 import：
```python
from dataclasses import dataclass, field
from typing import Callable
```

2. 在 `# ── Tool registry ──` 注释前增加 ToolDef：
```python
# ── Tool definition ─────────────────────────────────────────────────────

@dataclass
class ToolDef:
    """A tool's declaration: schema, confirm policy, and handler."""
    name: str
    args_model: type[BaseModel]
    requires_confirm: bool = False
    handler: Callable | None = None


def tool_schema(name: str) -> dict:
    """Generate the OpenAI function-calling schema for a tool."""
    t = openai.pydantic_function_tool(TOOLS[name].args_model, name=name)
    t["function"].pop("strict", None)
    return t


def needs_confirm(tool_name: str, agent: "Agent") -> bool:
    """Determine if a tool call requires user confirmation under this agent."""
    from .agent import Agent  # avoid circular at module level
    tool = TOOLS.get(tool_name)
    if tool is None:
        return False
    return tool.requires_confirm and tool_name not in agent.auto_tools
```

3. 替换 `TOOL_ARGS` dict + `_tool()` + `TOOLS` list 为新的 `TOOLS` dict：
```python
# ── Tool registry ───────────────────────────────────────────────────────

TOOLS: dict[str, ToolDef] = {
    "read_doc": ToolDef("read_doc", ReadDocArgs, requires_confirm=False),
    "search_docs": ToolDef("search_docs", SearchDocsArgs, requires_confirm=False),
    "list_files": ToolDef("list_files", ListFilesArgs, requires_confirm=False),
    "get_related_docs": ToolDef("get_related_docs", GetRelatedDocsArgs, requires_confirm=False),
    "recent_changes": ToolDef("recent_changes", RecentChangesArgs, requires_confirm=False),
    "write_doc": ToolDef("write_doc", WriteDocArgs, requires_confirm=True),
    "edit_doc": ToolDef("edit_doc", EditDocArgs, requires_confirm=True),
    "delete_doc": ToolDef("delete_doc", DeleteDocArgs, requires_confirm=True),
    "move_docs": ToolDef("move_docs", MoveDocsArgs, requires_confirm=True),
}

# Backward-compat: TOOL_ARGS for validate.py
TOOL_ARGS: dict[str, type[BaseModel]] = {name: t.args_model for name, t in TOOLS.items()}

# Backward-compat: TOOLS_LIST for LLM tools param
def get_tools_list(names: list[str] | None = None) -> list[dict]:
    """Generate the OpenAI tools param (list of function schemas)."""
    if names is None:
        return [tool_schema(n) for n in TOOLS]
    return [tool_schema(n) for n in names if n in TOOLS]

# Backward-compat: READ_TOOLS / WRITE_TOOLS (derived)
READ_TOOLS = {n for n, t in TOOLS.items() if not t.requires_confirm}
WRITE_TOOLS = {n for n, t in TOOLS.items() if t.requires_confirm}
```

4. 删除旧的 `_tool()` 函数和旧的 `TOOLS = [...]` list。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_framework.py -v`
Expected: All PASS (including Task 1 tests)

- [ ] **Step 5: Run existing agent tests to check compat**

Run: `pytest tests/test_agent.py -v`
Expected: Some may fail due to `TOOLS` being dict now (test_tools_schema_has_nine_tools uses list comprehension). Fix those in Task 6.

- [ ] **Step 6: Commit**

```bash
git add server/agent/tools.py tests/test_agent_framework.py
git commit -m "feat(agent): ToolDef dataclass + TOOLS dict registry + needs_confirm"
```

---

### Task 3: AgentContext dataclass

**Files:**
- Create: `server/agent/context.py`
- Test: `tests/test_agent_framework.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_agent_framework.py

def test_agent_context_fields():
    from server.agent.context import AgentContext
    from server.agent.agent import Agent
    from pathlib import Path

    agent = Agent(name="test", system_prompt="p", tools=["read_doc"])
    ctx = AgentContext(
        agent=agent,
        messages=[{"role": "user", "content": "hi"}],
        pending_confirms={},
        vault_dir=Path("/tmp/vault"),
        db_conn=None,
        last_read_ts={},
        ai_config=MagicMock(base_url="http://x", api_key="k", model="m"),
    )
    assert ctx.agent.name == "test"
    assert ctx.vault_dir == Path("/tmp/vault")
    assert ctx.db_conn is None
    assert ctx.last_read_ts == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_framework.py::test_agent_context_fields -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write implementation**

```python
# server/agent/context.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_framework.py::test_agent_context_fields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/agent/context.py tests/test_agent_framework.py
git commit -m "feat(agent): AgentContext parameter object"
```

---

### Task 4: validate.py workspace 支持

**Files:**
- Modify: `server/agent/validate.py:14-35`
- Test: `tests/test_agent_framework.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_agent_framework.py

def test_validate_workspace_allows_prefix(tmp_path):
    from server.agent.validate import validate_tool_call

    vault = tmp_path / "vault"
    vault.mkdir()
    result = validate_tool_call("read_doc", {"path": "essays/note.md"}, vault, workspace="essays/")
    assert result is None


def test_validate_workspace_rejects_outside(tmp_path):
    from server.agent.validate import validate_tool_call

    vault = tmp_path / "vault"
    vault.mkdir()
    result = validate_tool_call("read_doc", {"path": "notes/other.md"}, vault, workspace="essays/")
    assert result is not None
    assert "workspace" in result


def test_validate_workspace_none_means_no_limit(tmp_path):
    from server.agent.validate import validate_tool_call

    vault = tmp_path / "vault"
    vault.mkdir()
    result = validate_tool_call("read_doc", {"path": "any/path.md"}, vault, workspace=None)
    assert result is None


def test_validate_workspace_move_docs(tmp_path):
    """move_docs from_path and to_path both checked against workspace."""
    from server.agent.validate import validate_tool_call

    vault = tmp_path / "vault"
    vault.mkdir()
    # from_path outside workspace
    result = validate_tool_call(
        "move_docs",
        {"moves": [{"from_path": "notes/a.md", "to_path": "essays/a.md"}]},
        vault, workspace="essays/",
    )
    assert result is not None
    assert "workspace" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_framework.py::test_validate_workspace_allows_prefix -v`
Expected: FAIL with "TypeError: validate_tool_call() got an unexpected keyword argument 'workspace'"

- [ ] **Step 3: Modify validate.py**

```python
# server/agent/validate.py — modify validate_tool_call signature and add workspace check

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
    # workspace prefix check
    if workspace:
        for key in ("path", "from_path", "to_path"):
            p = args.get(key)
            if p is None:
                continue
            if not p.startswith(workspace):
                return f"Error: path '{p}' outside agent workspace '{workspace}'"
    # move_docs: check each item in moves list
    if workspace and "moves" in args:
        for m in args["moves"]:
            for key in ("from_path", "to_path"):
                p = m.get(key)
                if p is not None and not p.startswith(workspace):
                    return f"Error: path '{p}' outside agent workspace '{workspace}'"
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_framework.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing validate tests**

Run: `pytest tests/test_agent.py -k "validate" -v`
Expected: All PASS (workspace defaults to None, existing behavior unchanged)

- [ ] **Step 6: Commit**

```bash
git add server/agent/validate.py tests/test_agent_framework.py
git commit -m "feat(agent): workspace prefix validation in validate_tool_call"
```

---

### Task 5: execute.py 注册表分发改造

**Files:**
- Modify: `server/agent/execute.py`
- Test: `tests/test_agent_framework.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_agent_framework.py

def test_execute_tool_via_registry(tmp_path):
    """execute_tool dispatches through TOOLS registry handler."""
    from server.agent.execute import execute_tool
    from server.agent.context import AgentContext
    from server.agent.agent import Agent

    vault = tmp_path / "vault"
    notes = vault / "notes"
    notes.mkdir(parents=True)
    (notes / "test.md").write_bytes(b"# Hello")

    agent = Agent(name="t", system_prompt="p", tools=["read_doc"])
    ctx = AgentContext(
        agent=agent,
        messages=[],
        pending_confirms={},
        vault_dir=vault,
        db_conn=None,
        last_read_ts={},
        ai_config=MagicMock(),
    )

    result = execute_tool("read_doc", {"path": "notes/test.md"}, ctx)
    assert "# Hello" in result


def test_execute_tool_unknown(tmp_path):
    from server.agent.execute import execute_tool
    from server.agent.context import AgentContext
    from server.agent.agent import Agent

    vault = tmp_path / "vault"
    vault.mkdir()
    agent = Agent(name="t", system_prompt="p", tools=[])
    ctx = AgentContext(
        agent=agent, messages=[], pending_confirms={},
        vault_dir=vault, db_conn=None, last_read_ts={}, ai_config=MagicMock(),
    )
    result = execute_tool("nonexistent", {}, ctx)
    assert "unknown tool" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_framework.py::test_execute_tool_via_registry -v`
Expected: FAIL (execute_tool still has old signature)

- [ ] **Step 3: Rewrite execute.py**

将 `server/agent/execute.py` 改造为：

```python
"""Tool execution: registry dispatch to handler functions."""
from __future__ import annotations

import time
from pathlib import Path

from .. import responses
from .context import AgentContext
from .tools import TOOLS
from .validate import parse_since


def execute_tool(name: str, args: dict, ctx: AgentContext) -> str:
    """Execute a tool call via registry dispatch. Returns result string for LLM."""
    tool = TOOLS.get(name)
    if tool is None or tool.handler is None:
        return f"Error: unknown tool '{name}'"
    return tool.handler(args, ctx)


# ── Handler functions ───────────────────────────────────────────────────

def _read_doc(args: dict, ctx: AgentContext) -> str:
    vault_dir = ctx.vault_dir
    fp = vault_dir / args["path"]
    if not fp.is_file():
        return responses._file_not_found_hint(args["path"], ctx.db_conn, tool="read_doc")
    content = fp.read_bytes().decode("utf-8", errors="replace")
    lines = content.splitlines(keepends=True)
    total = len(lines)
    offset = max(int(args.get("offset", 1)), 1)
    limit = args.get("limit")
    if limit is not None:
        limit = int(limit)
    if offset > total:
        return responses.read_offset_exceeds(offset, total)
    if limit is not None:
        selected = lines[offset - 1: offset - 1 + limit]
    else:
        selected = lines[offset - 1:]

    MAX_CHARS = ctx.agent.read_doc_max_chars
    char_count = 0
    cut = len(selected)
    truncated = False
    for i, line in enumerate(selected):
        char_count += len(line)
        if char_count > MAX_CHARS:
            cut = i
            truncated = True
            break

    shown_lines = selected[:cut]
    text = "".join(shown_lines)
    end_line = offset + len(shown_lines) - 1

    if not truncated and offset == 1 and len(selected) == total:
        return f"File: {args['path']} ({total} lines, {len(content)} chars)\n<content>\n{text}</content>"

    header = f"File: {args['path']} (lines {offset}-{end_line} of {total})"
    if truncated:
        remaining = len(selected) - cut
        next_offset = offset + cut
        header += f"\n[truncated — {remaining} more lines, use read_doc with offset={next_offset} to continue]"
    return f"{header}\n<content>\n{text}</content>"


def _search_docs(args: dict, ctx: AgentContext) -> str:
    import html as _html
    from ..ai import rag_search

    expand = args.get("expand", True)
    try:
        results = rag_search(str(ctx.vault_dir), args["query"], top_k=args.get("top_k", 5), expand=expand)
    except Exception as e:
        if "syntax error" in str(e).lower() or "fts5" in str(e).lower():
            return responses.fts_syntax_error(args["query"])
        raise
    if not results:
        return responses.search_no_match(args["query"])
    lines = [f'<results count="{len(results)}" query="{_html.escape(args["query"], quote=False)}">']
    for r in results:
        lines.append(f'  <result path="{_html.escape(r["path"], quote=False)}" score="{r["score"]:.2f}">')
        for seg in r["segments"]:
            heading = _html.escape(seg["heading_path"], quote=False) if seg["heading_path"] else ""
            kind = seg["kind"]
            if expand and seg["content"]:
                content = _html.escape(seg["content"].replace("\n", " "), quote=False)
                lines.append(f'    <segment heading="{heading}" kind="{kind}">{content}</segment>')
            else:
                lines.append(f'    <segment heading="{heading}" kind="{kind}"/>')
        lines.append('  </result>')
    lines.append('</results>')
    return "\n".join(lines)


def _list_files(args: dict, ctx: AgentContext) -> str:
    vault_dir = ctx.vault_dir
    base = vault_dir / args.get("dir", "")
    files = [
        f.relative_to(vault_dir).as_posix()
        for f in base.rglob("*.md") if ".sync" not in f.parts
    ]
    return "\n".join(sorted(files)) if files else responses.list_empty(args.get("dir", ""))


def _write_doc(args: dict, ctx: AgentContext) -> str:
    vault_dir = ctx.vault_dir
    fp = vault_dir / args["path"]
    existed = fp.is_file()
    fp.parent.mkdir(parents=True, exist_ok=True)
    from ..write_registry import register_write

    register_write(args["path"], "agent")
    fp.write_bytes(args["content"].encode("utf-8"))
    if existed:
        return f"Saved: {args['path']} ({len(args['content'])} chars, overwritten)"
    return f"Saved: {args['path']} ({len(args['content'])} chars)"


def _delete_doc(args: dict, ctx: AgentContext) -> str:
    vault_dir = ctx.vault_dir
    fp = vault_dir / args["path"]
    if not fp.is_file():
        return responses._file_not_found_hint(args["path"], ctx.db_conn, tool="delete_doc")
    from ..write_registry import register_write
    from ..db.file_ops import apply_change

    register_write(args["path"], "agent")
    fp.unlink()
    if ctx.db_conn:
        apply_change(ctx.db_conn, vault_dir, args["path"], "deleted", source="agent")
    return f"Deleted: {args['path']}"


def _move_docs(args: dict, ctx: AgentContext) -> str:
    vault_dir = ctx.vault_dir
    db_conn = ctx.db_conn
    from ..db.file_ops import _update_references_batch
    from ..write_registry import register_write

    moves = [(m["from_path"], m["to_path"]) for m in args["moves"]]

    for from_path, to_path in moves:
        src = vault_dir / from_path
        if not src.is_file():
            return responses._file_not_found_hint(from_path, db_conn, tool="move_docs")
        dst = vault_dir / to_path
        if dst.is_file():
            return responses.move_dest_exists(to_path)

    ref_count = _update_references_batch(db_conn, vault_dir, moves, pre_rename=True)

    for from_path, to_path in moves:
        src = vault_dir / from_path
        dst = vault_dir / to_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        register_write(to_path, "agent")
        src.rename(dst)

    lines = [f"Moved {len(moves)} file(s):"]
    for from_path, to_path in moves:
        lines.append(f"  {from_path} → {to_path}")
    if ref_count:
        lines.append(f"Updated references in {ref_count} file(s)")
    return "\n".join(lines)


def _edit_doc(args: dict, ctx: AgentContext) -> str:
    vault_dir = ctx.vault_dir
    from ..db.file_ops import edit_file
    from ..write_registry import register_write

    fp = vault_dir / args["path"]
    if not fp.is_file():
        return responses._file_not_found_hint(args["path"], ctx.db_conn, tool="edit_doc")

    register_write(args["path"], "agent")
    result = edit_file(vault_dir, args["path"], args["old_string"], args["new_string"])
    return result


def _get_related_docs(args: dict, ctx: AgentContext) -> str:
    vault_dir = ctx.vault_dir
    db_conn = ctx.db_conn
    from ..db.graph import graph_get_related

    fp = vault_dir / args["path"]
    if not fp.is_file():
        return responses.related_no_file(args["path"])

    results = graph_get_related(db_conn, args["path"], hops=1)
    if not results:
        return responses.related_no_links(args["path"])
    lines = [f"{r['path']} (via {r['type']})" for r in results]
    return f"Related documents for {args['path']}:\n" + "\n".join(lines)


def _recent_changes(args: dict, ctx: AgentContext) -> str:
    db_conn = ctx.db_conn
    from ..db.changes import get_changes

    if not db_conn:
        return "Error: database not available"
    since_raw = args.get("since")
    if isinstance(since_raw, str) and since_raw.strip().lower() == "none":
        since_raw = None
    since_ts = parse_since(since_raw) if since_raw else 0.0
    path_raw = args.get("path")
    if isinstance(path_raw, str) and path_raw.strip().lower() == "none":
        path_raw = None
    source_raw = args.get("source")
    if isinstance(source_raw, str) and source_raw.strip().lower() == "none":
        source_raw = None
    changes = get_changes(
        db_conn,
        since=since_ts,
        path=path_raw,
        source=source_raw,
        limit=args.get("limit", 20),
    )
    if not changes:
        return responses.changes_empty()
    lines = []
    for c in changes:
        ts_str = time.strftime("%H:%M", time.localtime(c["ts"]))
        path_str = c["path"]
        if c.get("from_path"):
            path_str += f" (was: {c['from_path']})"
        lines.append(f"{ts_str} {c['source']:<10} {c['action']:<8} {path_str}")
    return "\n".join(lines)


# ── Register handlers ───────────────────────────────────────────────────

TOOLS["read_doc"].handler = _read_doc
TOOLS["search_docs"].handler = _search_docs
TOOLS["list_files"].handler = _list_files
TOOLS["write_doc"].handler = _write_doc
TOOLS["delete_doc"].handler = _delete_doc
TOOLS["move_docs"].handler = _move_docs
TOOLS["edit_doc"].handler = _edit_doc
TOOLS["get_related_docs"].handler = _get_related_docs
TOOLS["recent_changes"].handler = _recent_changes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_framework.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add server/agent/execute.py tests/test_agent_framework.py
git commit -m "refactor(agent): execute_tool registry dispatch + handler functions"
```

---

### Task 6: loop.py AgentContext 改造

**Files:**
- Modify: `server/agent/loop.py`
- Test: `tests/test_agent.py`（适配新签名）

- [ ] **Step 1: Modify loop.py**

将 `agent_loop` 签名从散装参数改为 `ctx: AgentContext`：

```python
"""Agent loop: LLM streaming + tool-call orchestration. Yields SSE events."""
from __future__ import annotations

import asyncio
import json
import logging
import time

from openai import AsyncOpenAI

from .context import AgentContext
from .execute import execute_tool
from .prompts import sse
from .tools import TOOLS, needs_confirm, get_tools_list
from .validate import validate_tool_call

logger = logging.getLogger("md-vault.agent")


async def agent_loop(ctx: AgentContext):
    """Core agent loop. Yields SSE events."""
    agent = ctx.agent
    client = AsyncOpenAI(base_url=ctx.ai_config.base_url, api_key=ctx.ai_config.api_key)
    tools = get_tools_list(agent.tools)

    for _round in range(agent.max_rounds):
        # In the last 3 rounds, inject a temporary reminder (not persisted to messages)
        if _round >= agent.max_rounds - 3:
            call_messages = ctx.messages + [{"role": "system", "content": "You are approaching the round limit. Summarize your progress and provide a final answer."}]
        else:
            call_messages = ctx.messages

        stream = await client.chat.completions.create(
            model=ctx.ai_config.model,
            messages=call_messages,
            tools=tools if tools else None,
            stream=True,
            stream_options={"include_usage": True},
        )

        content = ""
        tool_calls = {}
        usage = None

        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                content += delta.content
                yield sse("token", {"text": delta.content})

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    acc = tool_calls.setdefault(
                        tc.index, {"id": None, "name": None, "arguments": ""}
                    )
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function and tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        acc["arguments"] += tc.function.arguments

        # Emit token usage from the API response
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                current_tokens = prompt_tokens + completion_tokens
                yield sse("token_usage", {
                    "used": current_tokens,
                    "total": agent.context_window,
                    "percent": round(current_tokens / agent.context_window * 100, 1) if agent.context_window else 0,
                })

        if not tool_calls:
            ctx.messages.append({"role": "assistant", "content": content})
            yield sse("done", {})
            return

        ctx.messages.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls.values()
            ],
        })

        for tc in tool_calls.values():
            try:
                args = json.loads(tc["arguments"])
            except (json.JSONDecodeError, TypeError):
                err_msg = "Error: malformed tool arguments"
                ctx.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": err_msg})
                yield sse("tool_result", {"id": tc["id"], "result": err_msg, "error": True})
                continue

            yield sse("tool_call", {"id": tc["id"], "name": tc["name"], "args": args})
            logger.info("tool_call %s args=%s", tc["name"], json.dumps(args, ensure_ascii=False))

            error = validate_tool_call(tc["name"], args, ctx.vault_dir, workspace=agent.workspace)
            if error:
                logger.warning("tool_call %s rejected: %s", tc["name"], error)
                ctx.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": error})
                yield sse("tool_result", {"id": tc["id"], "result": error, "error": True})
                continue

            # Pre-check existence for delete/move before asking user to confirm
            if tc["name"] == "delete_doc":
                if not (ctx.vault_dir / args["path"]).is_file():
                    err = f"Error: file not found: {args['path']}"
                    ctx.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": err})
                    yield sse("tool_result", {"id": tc["id"], "result": err, "error": True})
                    continue
            elif tc["name"] == "move_docs":
                missing = [m["from_path"] for m in args.get("moves", []) if not (ctx.vault_dir / m["from_path"]).is_file()]
                if missing:
                    err = f"Error: file(s) not found: {', '.join(missing)}"
                    ctx.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": err})
                    yield sse("tool_result", {"id": tc["id"], "result": err, "error": True})
                    continue

            logger.info("tool_call %s: needs_confirm=%s", tc["name"], needs_confirm(tc["name"], agent))
            if needs_confirm(tc["name"], agent):
                entry = {"event": asyncio.Event(), "approved": None}
                ctx.pending_confirms[tc["id"]] = entry
                confirm_sse = sse("confirm_request", {
                    "id": tc["id"], "name": tc["name"], "args": args,
                })
                logger.info("yielding confirm_request for %s", tc["id"])
                yield confirm_sse
                logger.info("waiting for user confirmation on %s", tc["id"])
                await entry["event"].wait()
                logger.info("user confirmed/rejected %s, approved=%s", tc["id"], entry["approved"])
                del ctx.pending_confirms[tc["id"]]
                approved = entry["approved"]
                if not approved:
                    result = "User rejected this operation."
                    logger.info("tool_call %s rejected by user", tc["name"])
                    ctx.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    yield sse("tool_result", {"id": tc["id"], "result": result, "rejected": True})
                    continue

            result = execute_tool(tc["name"], args, ctx)
            logger.info("tool_result %s => %s", tc["name"], result[:500])
            yield sse("tool_result", {"id": tc["id"], "result": result})
            ctx.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

            # Track last read timestamp for change detection
            if tc["name"] == "read_doc" and not result.startswith("Error"):
                path = args.get("path", "")
                if path:
                    ctx.last_read_ts[path] = time.time()
```

- [ ] **Step 2: Update existing tests in test_agent.py**

现有测试调用 `agent_loop(messages=..., tools=..., ai_config=..., pending_confirms=..., vault_dir=..., db_conn=...)` 需要改为构造 `AgentContext`。

在 `tests/test_agent.py` 顶部增加 helper：

```python
def _make_ctx(vault_dir, db_conn=None, messages=None, pending_confirms=None, max_rounds=15):
    """Build an AgentContext for testing."""
    from server.agent.context import AgentContext
    from server.agent.agent import Agent

    agent = Agent(
        name="test-agent",
        system_prompt="You are a test agent.",
        tools=list(TOOLS.keys()),
        max_rounds=max_rounds,
    )
    return AgentContext(
        agent=agent,
        messages=messages or [{"role": "user", "content": "hi"}],
        pending_confirms=pending_confirms or {},
        vault_dir=vault_dir,
        db_conn=db_conn,
        last_read_ts={},
        ai_config=MagicMock(base_url="http://x", api_key="k", model="m"),
    )
```

然后将所有 `agent_loop(messages=..., tools=..., ai_config=..., pending_confirms=..., vault_dir=..., db_conn=...)` 调用替换为 `agent_loop(_make_ctx(...))`。

具体替换模式（每个测试）：
```python
# Before:
agent_loop(
    messages=[{"role": "user", "content": "hi"}],
    tools=[],
    ai_config=MagicMock(base_url="http://x", api_key="k", model="m"),
    pending_confirms={},
    vault_dir=vault,
    db_conn=None,
)

# After:
agent_loop(_make_ctx(vault, db_conn=None, messages=[{"role": "user", "content": "hi"}]))
```

对于 `test_agent_loop_max_rounds` 和 `test_agent_loop_round_limit_reminder`，需要传 `max_rounds` 参数：
```python
agent_loop(_make_ctx(MagicMock(), max_rounds=3))
```

对于 `test_agent_loop_write_requires_confirm` 等需要 `pending_confirms` 引用的测试：
```python
pending = {}
ctx = _make_ctx(vault, pending_confirms=pending)
# ... agent_loop(ctx) ...
```

同时修复 `test_tools_schema_has_nine_tools`：
```python
def test_tools_schema_has_nine_tools():
    from server.agent.tools import TOOLS

    assert len(TOOLS) == 9
    assert set(TOOLS.keys()) == {
        "read_doc", "search_docs", "list_files",
        "write_doc", "edit_doc", "delete_doc", "move_docs",
        "get_related_docs", "recent_changes",
    }
```

- [ ] **Step 3: Run all agent tests**

Run: `pytest tests/test_agent.py tests/test_agent_framework.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add server/agent/loop.py tests/test_agent.py
git commit -m "refactor(agent): agent_loop accepts AgentContext, uses needs_confirm + workspace"
```

---

### Task 7: __init__.py 兼容层适配

**Files:**
- Modify: `server/agent/__init__.py`

- [ ] **Step 1: Update __init__.py**

```python
"""Agent package: prompts, tool definitions, validation, execution, loop.

Compatibility layer: re-exports every symbol of the former monolithic
``server/agent.py`` so existing imports (``from server.agent import X``)
keep working. New code should import from the submodules directly, e.g.
``from server.agent.loop import agent_loop``.
"""
# Re-exported so ``patch("server.agent.AsyncOpenAI")`` in tests keeps working.
from openai import AsyncOpenAI

from .agent import AGENTS, Agent, get_agent, register
from .context import AgentContext
from .execute import execute_tool
from .loop import agent_loop
from .prompts import SYSTEM_PROMPT, _build_time_block, sse
from .session import SessionManager
from .tools import (
    DeleteDocArgs,
    EditDocArgs,
    GetRelatedDocsArgs,
    ListFilesArgs,
    MoveDocItem,
    MoveDocsArgs,
    READ_TOOLS,
    RecentChangesArgs,
    ReadDocArgs,
    SearchDocsArgs,
    TOOLS,
    TOOL_ARGS,
    ToolDef,
    WRITE_TOOLS,
    WriteDocArgs,
    _check_md_path,
    _check_optional_path,
    get_tools_list,
    needs_confirm,
    tool_schema,
)
from .validate import parse_since, validate_tool_call

__all__ = [
    # agent
    "Agent", "AGENTS", "register", "get_agent",
    # context
    "AgentContext",
    # prompts
    "SYSTEM_PROMPT",
    "_build_time_block",
    "sse",
    # tools (definitions)
    "ReadDocArgs",
    "SearchDocsArgs",
    "ListFilesArgs",
    "WriteDocArgs",
    "DeleteDocArgs",
    "MoveDocItem",
    "MoveDocsArgs",
    "EditDocArgs",
    "GetRelatedDocsArgs",
    "RecentChangesArgs",
    "TOOL_ARGS",
    "ToolDef",
    "TOOLS",
    "READ_TOOLS",
    "WRITE_TOOLS",
    "_check_md_path",
    "_check_optional_path",
    "get_tools_list",
    "needs_confirm",
    "tool_schema",
    # validate
    "validate_tool_call",
    "parse_since",
    # execute
    "execute_tool",
    # loop
    "agent_loop",
    # session
    "SessionManager",
]
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS（mcp_server.py 使用 `from server.agent import validate_tool_call` 仍工作）

- [ ] **Step 3: Commit**

```bash
git add server/agent/__init__.py
git commit -m "refactor(agent): __init__ compat layer for new module structure"
```

---

### Task 8: api.py 路由适配

**Files:**
- Modify: `server/api.py:253-333`

- [ ] **Step 1: Modify agent_chat route**

在 `server/api.py` 的 `agent_chat` 函数中：

1. 增加 `agent` 字段解析：
```python
@app.post("/api/agent/chat")
async def agent_chat(request: Request):
    if ai_config is None:
        raise HTTPException(status_code=503, detail="AI not configured")
    body = await request.json()
    message = body.get("message", "")
    session_id = body.get("session_id", "default")
    doc_path = body.get("doc_path", "")
    agent_name = body.get("agent", "vault-assistant")

    from .agent import get_agent
    agent = get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"unknown agent '{agent_name}'")

    session_mgr.cleanup()
    session = session_mgr.get_or_create(session_id)
    session["last_active"] = time.time()

    # ... doc_context 逻辑不变 ...

    lock = session_mgr.get_lock(session_id)
    await lock.acquire()

    from .agent import AgentContext

    async def event_generator():
        try:
            # ... doc_context 注入（不变）...

            session["messages"].append({"role": "user", "content": message})

            ctx = AgentContext(
                agent=agent,
                messages=session["messages"],
                pending_confirms=session["pending_confirms"],
                vault_dir=vault_dir,
                db_conn=db_conn,
                last_read_ts=session.setdefault("last_read_ts", {}),
                ai_config=ai_config,
            )

            async for event in agent_loop(ctx):
                yield event
            session_mgr.save(session_id)
        except (GeneratorExit, asyncio.CancelledError):
            logger.error(
                "SSE stream disconnected (session=%s, pending_confirms=%d)",
                session_id, len(session["pending_confirms"]),
                exc_info=True,
            )
            raise
        finally:
            lock.release()

    from fastapi.responses import StreamingResponse

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

2. 删除旧的 `from .agent import TOOLS, agent_loop`（TOOLS 不再直接传入）。

- [ ] **Step 2: Register vault-assistant agent**

在 `server/api.py` 的 `create_app` 函数开头（或模块级）注册默认 agent：

```python
from .agent import Agent, register

# Register the default agent (idempotent)
if "vault-assistant" not in AGENTS:
    from .agent.prompts import SYSTEM_PROMPT as _SP
    register(Agent(
        name="vault-assistant",
        system_prompt=_SP,
        tools=list(TOOLS.keys()),
    ))
```

或者更干净地放在 `server/agent/agents.py`（新文件）：

```python
# server/agent/agents.py
"""Agent registrations: all agents declared here."""
from __future__ import annotations

from pathlib import Path

from .agent import Agent, register
from .tools import TOOLS

_PROMPTS = Path(__file__).parent / "prompts"


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
```

在 `server/agent/__init__.py` 末尾增加：
```python
from . import agents  # noqa: F401 — triggers registration
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add server/api.py server/agent/agents.py server/agent/__init__.py
git commit -m "feat(agent): api route supports agent field + vault-assistant registration"
```

---

### Task 9: 最终验证 + 设计文档提交

**Files:**
- Modify: `docs/superpowers/specs/2026-10-09-agent-framework-design.md`（已在分支上）

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS, no regressions

- [ ] **Step 2: Verify mcp_server still works**

Run: `python -c "from server.agent import validate_tool_call, execute_tool; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit design doc + plan**

```bash
git add docs/superpowers/specs/2026-10-09-agent-framework-design.md docs/superpowers/plans/2026-10-09-agent-framework.md
git commit -m "docs: agent framework design spec + implementation plan"
```

- [ ] **Step 4: Final commit (squash or merge)**

确认所有测试通过后，分支可合并到 main。
