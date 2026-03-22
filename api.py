"""FastAPI app for Podwise: ingest videos, ask over transcripts."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when running as uvicorn api:app
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.agent.agent import ask
from src.index import ingest_youtube
from src.storage.supabase_store import SupabaseStore

app = FastAPI(title="Podwise", description="Agentic RAG over podcast transcripts")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    question: str
    max_tool_rounds: int = 15


class UploadLinkRequest(BaseModel):
    url: str


@app.get("/")
def root():
    """Serve the frontend."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/episodes")
def list_episodes() -> dict:
    """Return all indexed episodes for the frontend."""
    try:
        store = SupabaseStore()
        episodes = store.list_episodes()
        return {"episodes": episodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_endpoint(request: AskRequest) -> dict:
    answer = ask(question=request.question, max_tool_rounds=request.max_tool_rounds)
    return {"answer": answer}


@app.post("/upload_link")
def upload_link_endpoint(request: UploadLinkRequest) -> dict:
    """Ingest a YouTube video: fetch transcript, chunk, embed, store in index."""
    import traceback
    try:
        result = ingest_youtube(request.url)
        return {"status": "ok", **result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
