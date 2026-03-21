"""ReAct-style agent: Anthropic Claude + retrieval tools to answer over transcripts."""

from __future__ import annotations

from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.config import ANTHROPIC_API_KEY, LLM_MODEL
from src.retrieval.tools import AGENT_TOOLS

SYSTEM_PROMPT = """You are a podcast knowledge assistant. You have access to transcripts from
multiple podcasts and YouTube videos. When answering:
- ALWAYS cite the source: [Episode Title | MM:SS | link]
- If comparing across episodes, organize by speaker/episode
- If asked for notes or summary, structure with headers and bullet points
- If you don't find relevant content, say so honestly
"""


def _run_tool(name: str, args: dict) -> str:
    tool_by_name = {t.name: t for t in AGENT_TOOLS}
    if name not in tool_by_name:
        return f"Unknown tool: {name}"
    return str(tool_by_name[name].invoke(args))


def _tool_call_id(tc: object) -> str:
    return getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else "")


def _tool_call_name(tc: object) -> str:
    return getattr(tc, "name", None) or (tc.get("name", "") if isinstance(tc, dict) else "")


def _tool_call_args(tc: object) -> dict:
    return getattr(tc, "args", None) or (tc.get("args", {}) if isinstance(tc, dict) else {})


def ask(question: str, max_tool_rounds: int = 15) -> str:
    """Run the agent on one question; returns final answer text."""
    if not ANTHROPIC_API_KEY:
        return "Error: ANTHROPIC_API_KEY not set in .env"

    llm = ChatAnthropic(
        model=LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
    ).bind_tools(AGENT_TOOLS)

    messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)
    rounds = 0

    while getattr(response, "tool_calls", None) and rounds < max_tool_rounds:
        rounds += 1
        tool_messages: List[ToolMessage] = []
        for tc in response.tool_calls:
            name = _tool_call_name(tc)
            args = _tool_call_args(tc) or {}
            result = _run_tool(name, args)
            tool_messages.append(
                ToolMessage(content=result, tool_call_id=_tool_call_id(tc))
            )
        messages = [*messages, response, *tool_messages]
        response = llm.invoke(messages)

    if hasattr(response, "content") and response.content:
        # response.content can be a string OR a list of blocks
        if isinstance(response.content, str):
            return response.content
        # blocks are dicts like {"type": "text", "text": "..."} 
        # or {"type": "tool_use", ...}
        parts = []
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts) if parts else "No text in response."

    return "No response generated."