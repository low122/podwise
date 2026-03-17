"""CLI for podwise: ingest, ask, list, etc."""

import sys
from pathlib import Path

# Ensure project root is on path when running as python main.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

app = typer.Typer(rich_markup_mode="rich")
console = Console()


@app.command()
def ask(
    question: list[str] = typer.Argument(
        ...,
        help="Question to ask (pass words after 'ask'; use quotes for multi-word).",
    ),
) -> None:
    """Ask a question over all ingested transcripts. Uses Claude + retrieval tools."""
    from src.agent.agent import ask as agent_ask

    q = " ".join(question).strip()
    # In some shells/invocation the word "ask" can end up as first token; drop it
    if q.lower().startswith("ask "):
        q = q[4:].strip()
    if not q:
        raise SystemExit("Please provide a question (e.g. main.py ask \"What is RAG?\")")

    console.print()
    console.print("[bold]Question:[/bold]", q)
    console.print(Rule(style="dim"))

    answer = agent_ask(q)

    console.print(Panel(Markdown(answer), title="Answer", border_style="green", padding=(1, 2)))
    console.print()


if __name__ == "__main__":
    app()
