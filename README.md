# podwise

Agentic RAG pipeline for YouTube and podcast transcripts: ingest videos, then ask questions and get answers with clickable timestamp links. Built with LangChain, ChromaDB, Ollama (embeddings), and Claude (Anthropic).

## What it does

1. **Ingest** — Give a YouTube URL; the pipeline downloads the transcript, merges short segments into paragraphs, runs semantic chunking, embeds with Ollama (`nomic-embed-text`), and stores in ChromaDB.
2. **Ask** — Ask any question; a Claude agent uses retrieval tools (search transcripts, list episodes, get context) and answers with citations in the form `[Episode Title | MM:SS | link]`.
3. Supports **English and Mandarin** transcripts (YouTube provides them; no extra config).

<img width="855" height="900" alt="Example1" src="https://github.com/user-attachments/assets/87839cc0-ac2b-4385-9fe9-652d8c46121f" />


## Prerequisites

- **Python 3.11+**
- **Ollama** (local, for embeddings): [ollama.ai](https://ollama.ai). After install, run:
  ```bash
  ollama pull nomic-embed-text
  ```
- **Anthropic API key** (for the Q&A agent): set `ANTHROPIC_API_KEY` in `.env`.

## Setup

```bash
git clone <repo>
cd podwise
uv sync   # or: pip install -e ".[dev]"

cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (required for ask). Others have defaults.
```

## Usage

**Ingest a video (transcript → chunks → ChromaDB):**

```bash
uv run python -m src.index "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Ask a question over all ingested content:**

```bash
uv run python main.py ask "What does Saining Xie say about world models?"
uv run python main.py ask "Compare what was said about sleep across episodes"
```

Answers are printed in the terminal with Markdown formatting and citation links.

**List indexed episodes (optional):**

```bash
uv run python -m src.retrieval.tools list
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PATH` | `./data/chroma` | ChromaDB persistence directory |
| `COLLECTION_NAME` | `podwise` | Chroma collection name |
| `TRANSCRIPTS_PATH` | `./data/transcripts` | Raw transcript cache (optional) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server (embeddings) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `ANTHROPIC_API_KEY` | — | **Required** for `ask` |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model for the agent |

## Project layout

```
podwise/
├── main.py                 # CLI: ask
├── src/
│   ├── config.py           # Env settings
│   ├── index.py            # Ingest: YouTube → clean → chunk → embed → Chroma
│   ├── ingestion/          # youtube, cleaner, chunker (semantic)
│   ├── embedding/         # Ollama embedder
│   ├── storage/            # ChromaDB wrapper
│   ├── retrieval/         # search_transcripts, get_episode_list, get_episode_context (+ LangChain tools)
│   └── agent/              # Claude + tools (ReAct-style)
├── data/chroma/            # Vector store (gitignored)
└── .env.example
```

## Tech stack

| Layer        | Choice |
|-------------|--------|
| Transcripts | `youtube-transcript-api` |
| Chunking    | Time-based merge + LangChain `SemanticChunker` (Ollama) |
| Embeddings  | Ollama `nomic-embed-text` |
| Vector DB   | ChromaDB (`langchain-chroma`) |
| LLM / Agent | Anthropic Claude via `langchain-anthropic`; tools for search/list/context |
| CLI         | Typer + Rich |
