"""Build LangChain Documents from cleaned transcript blocks with metadata.

This version adds a semantic chunking step on top of time-based blocks.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from src.embedding.embedder import get_embedding_model
from src.ingestion.cleaner import CleanBlock


def _timestamp_url(video_id: str, start_sec: float) -> str:
    """YouTube link that jumps to start_sec (integer seconds)."""
    t = int(start_sec)
    return f"https://www.youtube.com/watch?v={video_id}&t={t}"

def semantic_chunk_documents(
    blocks: List[CleanBlock],
    video_id: str,
    title: str,
    channel: str,
    language: Optional[str] = None,
) -> List[Document]:
    """
    Semantic chunking on top of time-based blocks.
    For now we:
    - run SemanticChunker on EACH CleanBlock's text (topic-ish sub-chunks)
    - keep the parent's time window for all sub-chunks
    """
    # Reuse the same embedding model we use for storage
    embedding_model = get_embedding_model()
    chunker = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
    )

    docs: List[Document] = []
    chunk_idx = 0

    for block in blocks:
        pieces = chunker.split_text(block.text)
        for piece in pieces:
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "video_id": video_id,
                        "title": title,
                        "channel": channel,
                        "language": language or "",
                        "chunk_index": chunk_idx,
                        "start_time": block.start,
                        "end_time": block.end,
                        "url_with_timestamp": _timestamp_url(video_id, block.start),
                    },
                )
            )
            chunk_idx += 1

    return docs

