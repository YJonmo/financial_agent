"""
Main typer app for ConvFinQA
"""

import asyncio
import typer
from rich import print as rich_print

from src.agent import AgentType, build_agent
from src.data import get_record
from src.evaluate import run_evaluation
from src.logger import get_logger
from src.retrieval import table_to_markdown

app = typer.Typer(
    name="main",
    help="Agentic RAG CLI for ConvFinQA",
    add_completion=True,
    no_args_is_help=True,
)

logger = get_logger(__name__)


def run_async(coro):
    """Run an async coroutine safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(coro)
    else:
        # We're in an async context, use run_until_complete on a new loop
        # This is hacky but works
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

@app.command()
def chat(  # Keep it synchronous
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
    provider: str = typer.Option(
        "openai", help="LLM backend to use: 'openai' (paid) or 'groq' (free)"
    ),
    model: str = typer.Option(
        None,
        help="Model name for the chosen provider (defaults to gpt-4o-mini for "
        "openai, openai/gpt-oss-120b for groq)",
    ),
    agent: AgentType = typer.Option(
        AgentType.direct,
        "--agent",
        help="Agentic approach: 'direct', 'react', 'cot', or 'planner'",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show the agent's tool calls as it reasons"
    ),
) -> None:
    """Ask conversational questions about a specific ConvFinQA record."""
    try:
        record = get_record(record_id)
    except KeyError as exc:
        rich_print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    rich_print(f"[bold]Document {record.id}[/bold]")
    rich_print(f"[dim]{record.doc.pre_text[:400]}...[/dim]")
    rich_print(table_to_markdown(record.doc.table))
    rich_print(f"[dim]{record.doc.post_text[:400]}...[/dim]")
    rich_print(
        "[dim]Ask questions about this document below. Type 'exit' or 'quit' to stop.[/dim]\n"
    )

    try:
        conv_agent = build_agent(
            record, provider=provider, model=model, agent_type=agent, verbose=verbose
        )
    except RuntimeError as exc:
        rich_print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        try:
            response, final_value, rationale = run_async(conv_agent.ask(message))
        except Exception as exc:
            logger.exception("Agent failed to answer question")
            rich_print(f"[red]error: {exc}[/red]")
            continue

        rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")
        if final_value is not None:
            rich_print(f"[green]  final answer: {final_value}  ({rationale})[/green]")

@app.command()
def evaluate(
    split: str = typer.Option("dev", help="Dataset split to sample conversations from"),
    n: int = typer.Option(25, help="Number of conversations to sample"),
    seed: int = typer.Option(42, help="Random seed for the sample"),
    provider: str = typer.Option(
        "openai", help="LLM backend to use: 'openai' (paid) or 'groq' (free)"
    ),
    model: str = typer.Option(
        None,
        help="Model name for the chosen provider (defaults to gpt-4o-mini for "
        "openai, openai/gpt-oss-120b for groq)",
    ),
    agent: AgentType = typer.Option(
        AgentType.direct,
        "--agent",
        help="Agentic approach: 'direct', 'react', 'cot', or 'planner'",
    ),
) -> None:
    """Run the agent over a sample of conversations and report execution accuracy."""
    run_evaluation(split=split, n=n, seed=seed, provider=provider, model=model, agent_type=agent)


if __name__ == "__main__":
    import asyncio
    asyncio.run(app())  # Run the async app