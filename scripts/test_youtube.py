"""Quick test: fetch transcript for a YouTube URL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.youtube import fetch_transcript

URL = "https://www.youtube.com/watch?v=rIwgZWzUKm8"
data = fetch_transcript(URL)
print("ID:", data["metadata"].video_id)
print("Title:", data["metadata"].title)
print("Channel:", data["metadata"].channel)
print("Language:", data["metadata"].language_code)
print("Segments:", len(data["segments"]))
print("First segment:", data["segments"][0])
