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


def test_get_tools_list():
    from server.agent.tools import get_tools_list

    all_schemas = get_tools_list()
    assert len(all_schemas) == 9

    subset = get_tools_list(["read_doc", "write_doc"])
    assert len(subset) == 2
    names = {s["function"]["name"] for s in subset}
    assert names == {"read_doc", "write_doc"}


def test_read_write_tools_derived():
    from server.agent.tools import READ_TOOLS, WRITE_TOOLS

    assert READ_TOOLS == {"read_doc", "search_docs", "list_files", "get_related_docs", "recent_changes"}
    assert WRITE_TOOLS == {"write_doc", "edit_doc", "delete_doc", "move_docs"}
    assert READ_TOOLS.isdisjoint(WRITE_TOOLS)


def test_execute_tool_via_registry(tmp_path):
    """execute_tool dispatches through TOOLS registry handler."""
    from server.agent.execute import execute_tool
    from server.agent.context import AgentContext
    from server.agent.agent import Agent
    from unittest.mock import MagicMock

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
    from unittest.mock import MagicMock

    vault = tmp_path / "vault"
    vault.mkdir()
    agent = Agent(name="t", system_prompt="p", tools=[])
    ctx = AgentContext(
        agent=agent, messages=[], pending_confirms={},
        vault_dir=vault, db_conn=None, last_read_ts={}, ai_config=MagicMock(),
    )
    result = execute_tool("nonexistent", {}, ctx)
    assert "unknown tool" in result
