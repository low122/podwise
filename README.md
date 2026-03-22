# podwise

Agentic RAG pipeline for YouTube and podcast transcripts: ingest videos, then ask questions and get answers with clickable timestamp links. Built with LangChain, Supabase (pgvector), Voyage (embeddings), and Claude (Anthropic).

<img width="1239" height="856" alt="Screenshot 2026-03-22 at 02 54 50" src="https://github.com/user-attachments/assets/b77521d1-0a08-4786-bbb4-b74593c83a7b" />

## What it does

1. **Ingest** — Give a YouTube URL; the pipeline downloads the transcript, merges short segments into paragraphs, runs semantic chunking, embeds with Voyage, and stores in Supabase.
2. **Ask** — Ask any question; a Claude agent uses retrieval tools (search transcripts, list episodes, get context) and answers with citations in the form `[Episode Title | MM:SS | link]`.
3. Supports **English and Mandarin** transcripts (YouTube provides them; no extra config).

## Setup

```bash
git clone <repo>
cd podwise
uv sync   # or: pip install -e ".[dev]"

cp .env.example .env
# Edit .env: set SUPABASE_URL, SUPABASE_SERVICE_KEY, VOYAGE_API_KEY, ANTHROPIC_API_KEY.
```

## Usage

**Ingest a video (transcript → chunks → Supabase):**

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
| `SUPABASE_URL` | — | **Required** Supabase project URL |
| `SUPABASE_SERVICE_KEY` | — | **Required** Supabase service role key |
| `SUPABASE_TABLE_NAME` | `documents` | Table for vector chunks |
| `SUPABASE_QUERY_NAME` | `match_documents` | RPC for similarity search |
| `TRANSCRIPTS_PATH` | `./data/transcripts` | Raw transcript cache (optional) |
| `VOYAGE_API_KEY` | — | **Required** Voyage API key |
| `VOYAGE_MODEL` | `voyage-3` | Voyage model (1024 dims) |
| `ANTHROPIC_API_KEY` | — | **Required** for `ask` |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model for the agent |

## Project layout

```
podwise/
├── main.py                 # CLI: ask
├── src/
│   ├── config.py           # Env settings
│   ├── index.py            # Ingest: YouTube → clean → chunk → embed → Supabase
│   ├── ingestion/          # youtube, cleaner, chunker (semantic)
│   ├── embedding/         # Voyage embedder
│   ├── storage/            # Supabase (pgvector) wrapper
│   ├── retrieval/         # search_transcripts, get_episode_list, get_episode_context (+ LangChain tools)
│   └── agent/              # Claude + tools (ReAct-style)
└── .env.example
```

## Tech stack

| Layer        | Choice |
|-------------|--------|
| Transcripts | `youtube-transcript-api` |
| Chunking    | Time-based merge + LangChain `SemanticChunker` (Voyage) |
| Embeddings  | Voyage `voyage-3` |
| Vector DB   | Supabase (pgvector, `langchain-community`) |
| LLM / Agent | Anthropic Claude via `langchain-anthropic`; tools for search/list/context |
| CLI         | Typer + Rich |
