# Agent 框架抽取设计

## 目标

将当前硬编码的单 agent 实现重构为声明式多 agent 框架，使"添加新 agent"变成配置级操作（声明 prompt + 工具子集 + workspace + 放行列表），而非修改 loop/execute/api 代码。

首个新 agent 用例：定时触发的随笔 vault 碎片整理 agent（workspace 限定、后台执行 + 通知）。

## 设计决策

| 维度 | 决策 | 理由 |
|------|------|------|
| Agent 定义 | `Agent` dataclass + 全局注册表 | 声明式，新 agent = 10 行配置 |
| Tool 定义 | `ToolDef` dataclass + 全局注册表，自带 `requires_confirm` | confirm 策略跟随 tool 本身，不单独声明 |
| Confirm 覆盖 | Agent 的 `auto_tools` 放行列表 | 白名单语义：列出该 agent 下自动执行的工具 |
| 数据范围 | `workspace: str \| None`，路径前缀校验 | 简单直接，不需要虚拟映射 |
| 执行分发 | 注册表 + handler 函数（替代 if/elif） | 新工具 = 注册一个 handler，不改分发逻辑 |
| 参数传递 | `AgentContext` 参数对象（替代 9 个散装参数） | agent_loop 签名简化，扩展不破坏调用方 |
| 分层 | 不做 vault/ + services/ 大分层 | 本次只做框架抽取，分层作为后续独立项目 |

## 核心数据结构

### Agent（`server/agent/agent.py`，新文件）

```python
@dataclass
class Agent:
    name: str                          # 唯一标识，如 "vault-assistant"
    system_prompt: str                 # 系统提示词（从文件加载或内联）
    tools: list[str]                   # 工具名子集（从 TOOLS 注册表选）
    workspace: str | None = None       # 路径前缀限制（None = 全 vault）
    auto_tools: set[str] = field(default_factory=set)  # 放行列表（覆盖 requires_confirm）
    max_rounds: int = 15
    context_window: int = 16384
    read_doc_max_chars: int = 4000

AGENTS: dict[str, Agent] = {}

def register(agent: Agent) -> None:
    """注册一个 agent 到全局表。"""
    AGENTS[agent.name] = agent

def get_agent(name: str) -> Agent | None:
    return AGENTS.get(name)
```

### ToolDef（`server/agent/tools.py`，改造）

```python
@dataclass
class ToolDef:
    name: str
    args_model: type[BaseModel]        # Pydantic 参数模型（生成 LLM schema）
    requires_confirm: bool = False     # 默认是否需要用户确认
    handler: Callable | None = None    # 执行函数 (args, ctx) -> str

TOOLS: dict[str, ToolDef] = {
    "read_doc":       ToolDef("read_doc", ReadDocArgs, requires_confirm=False),
    "search_docs":    ToolDef("search_docs", SearchDocsArgs, requires_confirm=False),
    "list_files":     ToolDef("list_files", ListFilesArgs, requires_confirm=False),
    "get_related_docs": ToolDef("get_related_docs", GetRelatedDocsArgs, requires_confirm=False),
    "recent_changes": ToolDef("recent_changes", RecentChangesArgs, requires_confirm=False),
    "write_doc":      ToolDef("write_doc", WriteDocArgs, requires_confirm=True),
    "edit_doc":       ToolDef("edit_doc", EditDocArgs, requires_confirm=True),
    "delete_doc":     ToolDef("delete_doc", DeleteDocArgs, requires_confirm=True),
    "move_docs":      ToolDef("move_docs", MoveDocsArgs, requires_confirm=True),
}

def needs_confirm(tool_name: str, agent: Agent) -> bool:
    """判断某工具在某 agent 下是否需要用户确认。"""
    tool = TOOLS.get(tool_name)
    if tool is None:
        return False
    return tool.requires_confirm and tool_name not in agent.auto_tools
```

`READ_TOOLS` / `WRITE_TOOLS` 集合废弃，改为从 `requires_confirm` 派生（需要时）。

### AgentContext（`server/agent/context.py`，新文件）

```python
@dataclass
class AgentContext:
    agent: Agent                        # 声明式配置
    messages: list[dict]                # 会话消息（含 system prompt）
    pending_confirms: dict              # 待确认的 tool calls {id: {event, approved}}
    vault_dir: Path                     # vault 根目录
    db_conn: sqlite3.Connection | None  # SQLite 连接
    last_read_ts: dict                  # 变更检测 {path: timestamp}
    ai_config: AIConfig                 # LLM 连接配置（base_url, api_key, model...）
```

## 模块改造

### loop.py

```python
async def agent_loop(ctx: AgentContext) -> AsyncIterator[str]:
    """核心 agent loop。Yields SSE events。"""
    client = AsyncOpenAI(base_url=ctx.ai_config.base_url, api_key=ctx.ai_config.api_key)
    agent = ctx.agent

    for _round in range(agent.max_rounds):
        # ... LLM streaming（不变）...

        for tc in tool_calls.values():
            # 校验（传入 workspace）
            error = validate_tool_call(tc["name"], args, ctx.vault_dir, agent.workspace)
            ...

            # 确认判断（跟随 tool + agent 放行列表）
            if needs_confirm(tc["name"], agent):
                # ... confirm_request SSE + await ...

            # 执行（注册表分发）
            result = execute_tool(tc["name"], args, ctx)
            ...
```

关键变化：
- 参数从 9 个 → `ctx: AgentContext`
- `validate_tool_call` 增加 `workspace` 参数
- confirm 判断从 `tc["name"] in WRITE_TOOLS` → `needs_confirm(tc["name"], agent)`
- `execute_tool` 签名改为 `(name, args, ctx)`

### execute.py

```python
def execute_tool(name: str, args: dict, ctx: AgentContext) -> str:
    """注册表分发。"""
    tool = TOOLS.get(name)
    if tool is None or tool.handler is None:
        return f"Error: unknown tool '{name}'"
    return tool.handler(args, ctx)

# 每个 handler 是独立函数，签名统一：(args: dict, ctx: AgentContext) -> str
def _read_doc(args: dict, ctx: AgentContext) -> str: ...
def _search_docs(args: dict, ctx: AgentContext) -> str: ...
def _write_doc(args: dict, ctx: AgentContext) -> str: ...
# ...

# 注册（模块加载时）
TOOLS["read_doc"].handler = _read_doc
TOOLS["write_doc"].handler = _write_doc
# ...
```

关键变化：
- if/elif 链 → 注册表分发
- handler 函数接收 `ctx`（从 ctx 取 vault_dir, db_conn, read_doc_max_chars）
- 新工具 = 写一个 handler + 在 TOOLS 注册

### validate.py

```python
def validate_tool_call(
    name: str, args: dict, vault_dir: Path, workspace: str | None = None
) -> str | None:
    """Pydantic 验证 + vault 边界检查 + workspace 前缀检查。"""
    # ... 现有逻辑不变 ...

    # 新增：workspace 前缀检查
    if workspace:
        for key in ("path", "from_path", "to_path"):
            p = args.get(key)
            if p is None:
                continue
            if not p.startswith(workspace):
                return f"Error: path '{p}' outside agent workspace '{workspace}'"
    return None
```

### api.py

```python
@app.post("/api/agent/chat")
async def agent_chat(request: Request):
    body = await request.json()
    agent_name = body.get("agent", "vault-assistant")
    agent = get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"unknown agent '{agent_name}'")

    # ... session 管理（不变）...

    ctx = AgentContext(
        agent=agent,
        messages=session["messages"],
        pending_confirms=session["pending_confirms"],
        vault_dir=vault_dir,
        db_conn=db_conn,
        last_read_ts=session.setdefault("last_read_ts", {}),
        ai_config=ai_config,
    )

    async def event_generator():
        # ... doc_context 注入（不变）...
        async for event in agent_loop(ctx):
            yield event
        session_mgr.save(session_id)
```

关键变化：
- 请求体增加 `agent` 字段（默认 "vault-assistant"）
- 构造 `AgentContext` 替代散装参数传递

## 文件结构（改造后）

```
server/agent/
├── __init__.py       # re-export（兼容层保留）
├── agent.py          # Agent dataclass + AGENTS 注册表 + register/get_agent
├── context.py        # AgentContext dataclass
├── loop.py           # agent_loop(ctx) — 核心循环
├── tools.py          # ToolDef + TOOLS 注册表（含 requires_confirm, handler）
├── execute.py        # handler 函数 + execute_tool 注册表分发
├── validate.py       # validate_tool_call（+ workspace 检查）
├── session.py        # SessionManager（不变）
└── prompts.py        # sse() + time block（不变）
```

## 新 Agent 添加示例

```python
# server/agent/agents.py（新文件，集中注册所有 agent）

from pathlib import Path
from .agent import Agent, register

_PROMPTS = Path(__file__).parent / "prompts"

# 当前 agent（从 api.py 硬编码迁入）
register(Agent(
    name="vault-assistant",
    system_prompt=(_PROMPTS / "system.md").read_text(encoding="utf-8"),
    tools=list(TOOLS.keys()),  # 全部工具
))

# 碎片整理 agent
register(Agent(
    name="essay-organizer",
    system_prompt=(_PROMPTS / "essay_organizer.md").read_text(encoding="utf-8"),
    tools=["read_doc", "search_docs", "list_files", "write_doc", "edit_doc", "move_docs"],
    workspace="essays/",
    auto_tools={"move_docs"},  # 碎片整理中 move 是安全的，放行
))
```

## 确认策略

判断逻辑：`tool.requires_confirm and tool.name not in agent.auto_tools`

| 场景 | requires_confirm | auto_tools | 结果 |
|------|-----------------|------------|------|
| vault-assistant 的 write_doc | True | ∅ | 需要确认 |
| essay-organizer 的 move_docs | True | {move_docs} | 自动执行 |
| essay-organizer 的 write_doc | True | {move_docs} | 需要确认 |
| vault-assistant 的 read_doc | False | ∅ | 自动执行 |

## 兼容性

- `server/agent/__init__.py` 保留 re-export，现有 `from server.agent import X` 不破坏
- `TOOLS` 从 list → dict，`__init__.py` 中 re-export 的 `TOOLS` 需适配（或保留一个 `TOOLS_LIST` 兼容属性）
- `READ_TOOLS` / `WRITE_TOOLS` 废弃，但 `__init__.py` 中保留为派生集合（过渡期）
- API 请求体新增 `agent` 字段，不传时默认 "vault-assistant"，前端无需改动

## 测试策略

- `agent_loop` 现有行为不变（vault-assistant 走相同路径）
- 新增测试：workspace 校验（路径越界拒绝）、auto_tools 放行（confirm 跳过）、未知 agent 404
- handler 函数可独立测试（传入 mock ctx）

## 不在本次范围

- vault/ + services/ 大分层（decoupling-roadmap）
- 定时触发机制（scheduler/cron）— 碎片整理 agent 的触发是独立问题
- 多 agent 协作（agent 间调用）
- MCP server 的工具重复实现清理
