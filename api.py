"""FastAPI app for Podwise: ingest videos, ask over transcripts."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when running as uvicorn api:app
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from src.agent.agent import ask
from src.index import ingest_youtube
from src.storage.supabase_store import SupabaseStore


def _get_real_client_ip(request: Request) -> str:
    """Extract real client IP behind Render's reverse proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_get_real_client_ip)
app = FastAPI(title="Podwise", description="Agentic RAG over podcast transcripts")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


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
@limiter.limit("3/30minutes")
def ask_endpoint(request: Request, body: AskRequest) -> dict:
    answer = ask(question=body.question, max_tool_rounds=body.max_tool_rounds)
    return {"answer": answer}


@app.post("/upload_link")
@limiter.limit("2/hour")
def upload_link_endpoint(request: Request, body: UploadLinkRequest) -> dict:
    """Ingest a YouTube video: fetch transcript, chunk, embed, store in index."""
    import traceback
    try:
        result = ingest_youtube(body.url)
        return {"status": "ok", **result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
