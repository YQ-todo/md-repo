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
