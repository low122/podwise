from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from src.config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT, JWT_SECRET
from src.storage.supabase_store import SupabaseStore

router = APIRouter(prefix="/auth")
bearer = HTTPBearer()

# GOOGLE_REDIRECT is the base URL (e.g. https://podwise-jcgn.onrender.com)
# The full callback URL that Google redirects to after login:
CALLBACK_URL = f"{GOOGLE_REDIRECT}/auth/google/callback"

JWT_EXPIRY_DAYS = 7


def _make_jwt(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRY_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
) -> Dict:
    """Decode JWT from Authorization header. Use as a FastAPI dependency."""
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=["HS256"])
        return {"user_id": payload["sub"], "email": payload["email"]}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.get("/google")
def google_login():
    """Redirect user to Google's OAuth consent screen."""
    url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={CALLBACK_URL}"
        "&response_type=code"
        "&scope=openid email"
        "&access_type=offline"
    )
    return RedirectResponse(url)


@router.get("/google/callback")
async def google_callback(code: str):
    """Google redirects here with an auth code. We exchange it for user info."""
    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": CALLBACK_URL,
                "grant_type": "authorization_code",
            },
        )
        if token_res.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange code with Google")
        access_token = token_res.json()["access_token"]

        user_res = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if user_res.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch user info from Google")
        info = user_res.json()

    store = SupabaseStore()
    user_id = store.upsert_user(
        email=info["email"],
        provider="google",
        provider_id=info["sub"],
    )

    token = _make_jwt(user_id, info["email"])
    # Redirect back to the frontend with the JWT as a query param
    return RedirectResponse(f"{GOOGLE_REDIRECT}#token={token}")
