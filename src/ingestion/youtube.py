"""YouTube transcript ingestion utilities."""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)


@dataclass
class TranscriptSegment:
    text: str
    start: float
    duration: float


@dataclass
class VideoMetadata:
    video_id: str
    title: str
    channel: str
    language_code: Optional[str]


def _extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a URL.

    Handles:
    - normal watch URLs with extra params: https://www.youtube.com/watch?v=ID&t=7608s
    - copied shell-escaped URLs like 'watch\\?v\\=ID\\&t\\=7608s'
    - youtu.be short links and /shorts/ URLs
    """
    # Fix common shell-escaped copies where ? and & are prefixed with backslashes
    cleaned = url.replace(r"\?", "?").replace(r"\&", "&").replace(r"\=", "=")

    parsed = urllib.parse.urlparse(cleaned)
    if parsed.hostname in {"youtu.be"}:
        return parsed.path.lstrip("/")
    if parsed.hostname and "youtube.com" in parsed.hostname:
        query = urllib.parse.parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]
        # Shorts or embed URLs
        parts = parsed.path.split("/")
        for part in parts:
            if len(part) == 11:
                return part
    raise ValueError(f"Could not extract video id from URL: {url}")


def _fetch_page_html(video_id: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (podwise)"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _parse_title_and_channel(page_html: str) -> Dict[str, str]:
    """Parse video title and channel name from YouTube HTML."""
    title = ""
    channel = ""

    m_title = re.search(
        r'<meta property="og:title" content="([^"]+)"', page_html
    )
    if not m_title:
        m_title = re.search(r'<meta name="title" content="([^"]+)"', page_html)
    if m_title:
        title = html.unescape(m_title.group(1))

    m_channel = re.search(
        r'<meta itemprop="author" content="([^"]+)"', page_html
    )
    if m_channel:
        channel = html.unescape(m_channel.group(1))

    return {"title": title or "Unknown title", "channel": channel or "Unknown channel"}


def fetch_transcript(url: str, languages: Optional[List[str]] = None) -> Dict[str, object]:
    """Fetch transcript segments and metadata for a YouTube URL.

    Returns dict with:
      - metadata: VideoMetadata
      - segments: list[TranscriptSegment]
    """
    video_id = _extract_video_id(url)

    api = YouTubeTranscriptApi()
    try:
        transcript = api.fetch(
            video_id,
            languages=languages or ["en", "zh-Hans", "zh-Hant"],
        )
    except (NoTranscriptFound, TranscriptsDisabled) as exc:
        raise RuntimeError(f"No transcript available for video {video_id}") from exc

    page_html = _fetch_page_html(video_id)
    meta = _parse_title_and_channel(page_html)

    metadata = VideoMetadata(
        video_id=video_id,
        title=meta["title"],
        channel=meta["channel"],
        language_code=transcript.language_code or None,
    )

    segments = [
        TranscriptSegment(text=s.text.strip(), start=s.start, duration=s.duration)
        for s in transcript
        if s.text
    ]

    return {"metadata": metadata, "segments": segments}

