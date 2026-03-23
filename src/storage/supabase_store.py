"""Supabase (pgvector) wrapper for storing and querying podcast chunks.

Uses SupabaseVectorStore for writes (add_documents handles embedding +
insert) and direct RPC calls for reads (avoids langchain-community /
postgrest version incompatibilities).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

from src.config import (
    SUPABASE_QUERY_NAME,
    SUPABASE_SERVICE_KEY,
    SUPABASE_TABLE_NAME,
    SUPABASE_URL,
)
from src.embedding.embedder import get_embedding_model


@dataclass
class SearchResult:
    document: Document
    score: float


class SupabaseStore:
    """Vector store backed by Supabase pgvector."""

    def __init__(
        self,
        table_name: str = SUPABASE_TABLE_NAME,
        query_name: str = SUPABASE_QUERY_NAME,
    ) -> None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set. "
                "Create a project at supabase.com and add keys to .env"
            )
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        self._embedding = get_embedding_model()
        self._store = SupabaseVectorStore(
            client=client,
            embedding=self._embedding,
            table_name=table_name,
            query_name=query_name,
        )
        self._client = client
        self._table_name = table_name
        self._query_name = query_name

    # --------- write ----------
    def upsert_documents(self, documents: Sequence[Document]) -> None:
        """Add documents to the collection."""
        if not documents:
            return
        self._store.add_documents(list(documents))

    def upsert_user(self, email: str, provider: str, provider_id: str) -> str:
        """Insert user or return existing. Returns the user's UUID."""
        result = (
            self._client.table("users")
            .upsert(
                {"email": email, "provider": provider, "provider_id": provider_id},
                on_conflict="email",
            )
            .execute()
        )
        return result.data[0]["id"]

    # --------- read / query ----------
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_by_episode: Optional[str] = None,
    ) -> List[SearchResult]:
        """Semantic search via match_documents RPC."""
        query_vector = self._embedding.embed_query(query)
        params: Dict = {
            "query_embedding": query_vector,
            "match_count": top_k,
        }
        if filter_by_episode:
            params["filter"] = {"video_id": filter_by_episode}
        result = self._client.rpc(self._query_name, params).execute()
        return [
            SearchResult(
                document=Document(
                    page_content=r.get("content", ""),
                    metadata=r.get("metadata") or {},
                ),
                score=r.get("similarity", 0.0),
            )
            for r in (result.data or [])
        ]

    def list_episodes(self) -> List[Dict[str, str]]:
        """Return distinct episodes with basic metadata. Uses list_episodes RPC."""
        result = self._client.rpc("list_episodes").execute()
        rows = result.data or []
        return [
            {
                "video_id": r["video_id"],
                "title": r.get("title", ""),
                "channel": r.get("channel", ""),
                "language": r.get("language", ""),
            }
            for r in rows
        ]

    def delete_episode(self, video_id: str) -> None:
        """Delete all chunks for a given video_id. Uses delete_by_video_id RPC."""
        self._client.rpc("delete_by_video_id", {"p_video_id": video_id}).execute()

    def get_episode_chunks(self, video_id: str) -> List[Document]:
        """Return all documents for a video_id, ordered by chunk_index."""
        result = self._client.rpc(
            "get_chunks_by_video_id", {"p_video_id": video_id}
        ).execute()
        rows = result.data or []
        return [
            Document(page_content=r.get("content", ""), metadata=r.get("metadata") or {})
            for r in rows
        ]
