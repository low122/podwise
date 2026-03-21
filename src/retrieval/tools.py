"""High-level retrieval helpers over SupabaseStore.

Plain functions for CLI; LangChain tools below for the agent.
"""

from __future__ import annotations

from typing import List

from langchain_core.tools import tool

from src.storage.supabase_store import SupabaseStore


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def search_transcripts(query: str, top_k: int = 5) -> str:
    """Vector search across all ingested content.

    Returns a human-readable string with:
    [Title | HH:MM:SS | link] then the chunk text.
    """
    store = SupabaseStore()
    results = store.search(query, top_k=top_k)
    if not results:
        return "No results found in the current index."

    lines: List[str] = [f"Top {len(results)} results for: {query!r}", ""]
    for i, r in enumerate(results, start=1):
        doc = r.document
        meta = doc.metadata
        ts = _format_timestamp(float(meta.get("start_time", 0.0)))
        link = meta.get("url_with_timestamp", "")
        title = meta.get("title", "")
        lines.append(f"{i}. [{title} | {ts} | {link}]  (score={r.score:.3f})")
        lines.append(f"   {doc.page_content.strip()[:400]}")  # trim long chunks
        lines.append("")
    return "\n".join(lines)


def get_episode_list() -> str:
    """List all ingested episodes with basic metadata."""
    store = SupabaseStore()
    episodes = store.list_episodes()
    if not episodes:
        return "No episodes indexed yet. Try ingesting a YouTube URL first."

    lines: List[str] = ["Episodes in index:", ""]
    for e in episodes:
        lines.append(
            f"- {e['video_id']} | {e['title']} | {e['channel']} | {e.get('language', '')}"
        )
    return "\n".join(lines)


def get_episode_context(video_id: str, timestamp: float, window: int = 2) -> str:
    """Get a window of chunks around a timestamp for one episode.

    window=2 -> 2 before + the main chunk + 2 after = up to 5 chunks.
    """
    store = SupabaseStore()
    chunks = store.get_episode_chunks(video_id)
    if not chunks:
        return f"No chunks found for video_id={video_id!r}."

    # find the chunk whose [start_time, end_time] contains timestamp
    idx = 0
    for i, doc in enumerate(chunks):
        meta = doc.metadata
        start = float(meta.get("start_time", 0.0))
        end = float(meta.get("end_time", start))
        if start <= timestamp <= end:
            idx = i
            break

    start_i = max(0, idx - window)
    end_i = min(len(chunks), idx + window + 1)
    selected = chunks[start_i:end_i]

    lines: List[str] = [
        f"Context for video_id={video_id}, timestamp={timestamp:.1f}s "
        f"(chunks {start_i}..{end_i-1} of {len(chunks)}):",
        "",
    ]

    for i, doc in enumerate(selected, start=start_i):
        meta = doc.metadata
        ts = _format_timestamp(float(meta.get("start_time", 0.0)))
        link = meta.get("url_with_timestamp", "")
        title = meta.get("title", "")
        lines.append(f"[{i}] [{title} | {ts} | {link}]")
        lines.append(doc.page_content.strip())
        lines.append("")

    return "\n".join(lines)


# ----- LangChain tools for the Anthropic agent -----


@tool
def search_transcripts_tool(query: str, top_k: int = 5) -> str:
    """Search across all ingested podcast/YouTube transcripts by meaning.
    Use this to find relevant quotes and segments when answering the user's question.
    Returns chunks with [Episode Title | Timestamp | YouTube link] and snippet."""
    return search_transcripts(query=query, top_k=top_k)


@tool
def get_episode_list_tool() -> str:
    """List all ingested episodes (video_id, title, channel, language).
    Call this when the user asks what's in the index or which podcasts are available."""
    return get_episode_list()


@tool
def get_episode_context_tool(video_id: str, timestamp: float, window: int = 2) -> str:
    """Get more context around a specific timestamp in one episode.
    video_id: YouTube video ID (e.g. rIwgZWzUKm8). timestamp: time in seconds.
    Use after search when you need surrounding sentences at that moment."""
    return get_episode_context(video_id=video_id, timestamp=timestamp, window=window)


AGENT_TOOLS = [search_transcripts_tool, get_episode_list_tool, get_episode_context_tool]


if __name__ == "__main__":
    # Tiny manual demo:
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "search":
        print(search_transcripts(" ".join(sys.argv[2:])))
    elif len(sys.argv) == 2 and sys.argv[1] == "list":
        print(get_episode_list())
    else:
        print(
            "Usage:\n"
            "  python -m src.retrieval.tools list\n"
            '  python -m src.retrieval.tools search "your question here"\n'
        )

