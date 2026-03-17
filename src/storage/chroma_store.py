"""ChromaDB wrapper for storing and querying podcast chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import CHROMA_PATH, COLLECTION_NAME
from src.embedding.embedder import get_embedding_model


@dataclass
class SearchResult:
    document: Document
    score: float


class ChromaStore:
    """Small helper around Chroma for this project."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_path: str | None = None,
    ) -> None:
        embedding = get_embedding_model()
        self._chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=str(persist_path or CHROMA_PATH),
        )

    # --------- write ----------
    def upsert_documents(self, documents: Sequence[Document]) -> None:
        """Add or update documents in the collection."""
        if not documents:
            return
        self._chroma.add_documents(list(documents))

    # --------- read / query ----------
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_by_episode: Optional[str] = None,
    ) -> List[SearchResult]:
        """Semantic search over all content, optionally filtered by video_id."""
        search_filter: Optional[Dict[str, str]] = None
        if filter_by_episode:
            search_filter = {"video_id": filter_by_episode}

        docs_and_scores: List[Tuple[Document, float]] = (
            self._chroma.similarity_search_with_score(
                query, k=top_k, filter=search_filter
            )
        )
        return [
            SearchResult(document=doc, score=score) for doc, score in docs_and_scores
        ]

    def list_episodes(self) -> List[Dict[str, str]]:
        """Return distinct episodes with basic metadata."""
        raw = self._chroma.get(include=["metadatas"])
        seen: Dict[str, Dict[str, str]] = {}
        for meta in raw.get("metadatas", []) or []:
            video_id = meta.get("video_id")
            if not video_id or video_id in seen:
                continue
            seen[video_id] = {
                "video_id": video_id,
                "title": meta.get("title", ""),
                "channel": meta.get("channel", ""),
                "language": meta.get("language", ""),
            }
        return list(seen.values())

    def delete_episode(self, video_id: str) -> None:
        """Delete all chunks for a given video_id."""
        self._chroma.delete(filter={"video_id": video_id})

    def get_episode_chunks(self, video_id: str) -> List[Document]:
        """Return all documents for a video_id, ordered by chunk_index."""
        res = self._chroma.get(
            where={"video_id": video_id},
            include=["metadatas", "documents"],
        )
        docs: List[Document] = []
        for text, meta in zip(res.get("documents", []) or [], res.get("metadatas", []) or []):
            docs.append(Document(page_content=text, metadata=meta))
        docs.sort(key=lambda d: int(d.metadata.get("chunk_index", 0)))
        return docs

