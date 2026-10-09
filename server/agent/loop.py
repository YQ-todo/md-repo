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
