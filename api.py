"""FastAPI app for Podwise: ingest videos, ask over transcripts."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when running as uvicorn api:app
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import traceback

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from src.agent.agent import ask
from src.auth.router import router as auth_router, get_current_user
from src.index import ingest_youtube_stream
from src.storage.supabase_store import SupabaseStore


def _get_user_id_from_jwt(request: Request) -> str:
    """Extract user_id from JWT for per-user rate limiting."""
    from jose import jwt as jose_jwt, JWTError
    from src.config import JWT_SECRET
    # Check Authorization header first (POST endpoints)
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        # Fall back to query param (SSE endpoint)
        token = request.query_params.get("token", "")
    try:
        payload = jose_jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["sub"]
    except (JWTError, KeyError):
        return "anonymous"


limiter = Limiter(key_func=_get_user_id_from_jwt)
app = FastAPI(title="Podwise", description="Agentic RAG over podcast transcripts")
app.state.limiter = limiter
app.include_router(auth_router)


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
@limiter.limit("3/30minutes", key_func=_get_user_id_from_jwt)
def ask_endpoint(request: Request, body: AskRequest, user=Depends(get_current_user)) -> dict:
    answer = ask(question=body.question, max_tool_rounds=body.max_tool_rounds)
    return {"answer": answer}



def _get_user_from_token_query(request: Request, token: str = Query(...)) -> dict:
    """Verify JWT from query param. EventSource can't send headers."""
    from jose import jwt as jose_jwt, JWTError
    from src.config import JWT_SECRET
    try:
        payload = jose_jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return {"user_id": payload["sub"], "email": payload["email"]}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@app.get("/upload_link/stream")
@limiter.limit("2/hour", key_func=_get_user_id_from_jwt)
def upload_link_stream(request: Request, url: str = Query(...), user=Depends(_get_user_from_token_query)):
    """SSE endpoint that streams ingestion progress events."""
    def event_generator():
        try:
            for event in ingest_youtube_stream(url):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
