"""Merge short transcript segments into coherent paragraphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

MAX_WORDS_PER_BLOCK = 200
PAUSE_THRESHOLD_SEC = 2.0


class SegmentLike(Protocol):
    text: str
    start: float
    duration: float


@dataclass
class CleanBlock:
    """One merged paragraph with start/end timestamps."""
    text: str
    start: float
    end: float


def _word_count(s: str) -> int:
    return len(s.split())


def merge_segments(segments: List[SegmentLike]) -> List[CleanBlock]:
    """Merge tiny segments into blocks of ~MAX_WORDS_PER_BLOCK or until a >PAUSE_THRESHOLD_SEC gap."""
    if not segments:
        return []

    blocks: List[CleanBlock] = []
    current_text: List[str] = []
    current_start = segments[0].start
    prev_end = segments[0].start

    for seg in segments:
        gap = seg.start - prev_end
        words_so_far = _word_count(" ".join(current_text))
        # Start new block on big gap or when we'd exceed ~200 words
        if current_text and (gap >= PAUSE_THRESHOLD_SEC or words_so_far + _word_count(seg.text) > MAX_WORDS_PER_BLOCK):
            blocks.append(CleanBlock(
                text=" ".join(current_text).strip(),
                start=current_start,
                end=prev_end,
            ))
            current_text = []
            current_start = seg.start

        current_text.append(seg.text.strip())
        prev_end = seg.start + seg.duration

    if current_text:
        blocks.append(CleanBlock(
            text=" ".join(current_text).strip(),
            start=current_start,
            end=prev_end,
        ))

    return blocks
