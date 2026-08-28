"""Evaluation harness: execution accuracy of the agent over ConvFinQA conversations.

Metric choice ("execution accuracy", Exe Acc):
We compare the agent's final numeric answer for each turn against the
dataset's `executed_answers` (the golden DSL program's execution result),
rather than the raw `conv_answers` strings. This mirrors the metric used in
the original ConvFinQA/FinQA papers, and sidesteps formatting differences
(e.g. "14.1%" vs 0.14136) that a pure string match would incorrectly
penalise. A prediction counts as correct if it is within 1% relative (or
0.001 absolute, whichever is larger) of the gold value, to allow for
reasonable rounding while still catching real reasoning errors. Yes/no
answers are compared as case-insensitive exact string matches.

Conversation history is NOT seeded with gold answers between turns: each
conversation is replayed turn-by-turn against the SAME agent instance, so
later turns depend on the agent's own (possibly wrong) earlier answers. This
mirrors how the system would actually be used, and lets us reproduce the
paper's finding that errors compound across turns (see REPORT.md).
"""

import asyncio
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from src.agent import AgentType, build_agent
from src.data import iter_records
from src.llm_providers import resolve_model_name
from src.logger import get_logger
from src.models import ConvFinQARecord

logger = get_logger(__name__)
console = Console()

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"


def parse_numeric(value: str) -> float | None:
    """Parse a model-produced answer string into a float, or None if not numeric."""
    s = value.strip().lower().replace("$", "").replace(",", "")
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]
    if s.endswith("%"):
        s = s[:-1]
        scale = 0.01
    else:
        scale = 1.0
    try:
        parsed = float(s) * scale
    except ValueError:
        return None
    return -parsed if negative else parsed


def is_correct(predicted: str | None, gold: float | str, rel_tol: float = 0.01) -> bool:
    if predicted is None:
        return False
    if isinstance(gold, str):
        return predicted.strip().lower() == gold.strip().lower()
    predicted_value = parse_numeric(predicted)
    if predicted_value is None:
        return False
    tolerance = max(abs(gold) * rel_tol, 1e-3)
    return abs(predicted_value - gold) <= tolerance


@dataclass
class TurnResult:
    conversation_id: str
    turn_index: int
    question: str
    gold: float | str
    predicted: str | None
    correct: bool
    is_type2_turn: bool
    turn_program: str


@dataclass
class ConversationResult:
    conversation_id: str
    has_type2_question: bool
    num_turns: int
    turns: list[TurnResult]

    @property
    def all_correct(self) -> bool:
        return all(t.correct for t in self.turns)


async def run_conversation(  # Made async
    record: ConvFinQARecord,
    provider: str,
    model: str | None,
    agent_type: AgentType = AgentType.direct,
) -> ConversationResult:
    conv_agent = build_agent(
        record, provider=provider, model=model, agent_type=agent_type, verbose=False
    )

    turns: list[TurnResult] = []
    for i, question in enumerate(record.dialogue.conv_questions):
        gold = record.dialogue.executed_answers[i]
        try:
            _, value, _ = await conv_agent.ask(question)  # Added await
        except Exception:  # noqa: BLE001
            logger.exception("Turn failed for %s turn %d", record.id, i)
            value = None
        turns.append(
            TurnResult(
                conversation_id=record.id,
                turn_index=i,
                question=question,
                gold=gold,
                predicted=value,
                correct=is_correct(value, gold),
                is_type2_turn=record.dialogue.qa_split[i],
                turn_program=record.dialogue.turn_program[i],
            )
        )
    return ConversationResult(
        conversation_id=record.id,
        has_type2_question=record.features.has_type2_question,
        num_turns=record.features.num_dialogue_turns,
        turns=turns,
    )


def _turn_bucket(turn_index: int) -> str:
    position = turn_index + 1
    return f"turn {position}" if position < 5 else "turn 5+"


def summarize(results: list[ConversationResult]) -> dict[str, Any]:
    all_turns = [t for r in results for t in r.turns]

    def acc(turns: list[TurnResult]) -> float | None:
        return sum(t.correct for t in turns) / len(turns) if turns else None

    turns_by_position: dict[str, list[TurnResult]] = defaultdict(list)
    for t in all_turns:
        turns_by_position[_turn_bucket(t.turn_index)].append(t)
    by_turn_position = {k: acc(v) for k, v in turns_by_position.items()}

    type1_turns = [t for r in results if not r.has_type2_question for t in r.turns]
    type2_turns = [t for r in results if r.has_type2_question for t in r.turns]

    return {
        "num_conversations": len(results),
        "num_turns": len(all_turns),
        "turn_execution_accuracy": acc(all_turns),
        "conversation_accuracy": (
            sum(r.all_correct for r in results) / len(results) if results else None
        ),
        "accuracy_by_turn_position": by_turn_position,
        "accuracy_type1_conversations": acc(type1_turns),
        "accuracy_type2_conversations": acc(type2_turns),
    }


async def run_evaluation_async(  # New async function
    split: str = "dev",
    n: int = 25,
    seed: int = 42,
    provider: str = "openai",
    model: str | None = None,
    agent_type: AgentType = AgentType.direct,
    out_path: Path | None = None,
) -> dict[str, Any]:
    records = list(iter_records(split))
    sample = random.Random(seed).sample(records, min(n, len(records)))
    resolved_model = resolve_model_name(provider, model)

    results: list[ConversationResult] = []
    start = time.time()
    for i, record in enumerate(sample, start=1):
        console.print(f"[dim]({i}/{len(sample)}) evaluating {record.id}...[/dim]")
        result = await run_conversation(record, provider=provider, model=model, agent_type=agent_type)
        results.append(result)
    elapsed = time.time() - start

    summary = summarize(results)
    summary["split"] = split
    summary["sample_size"] = len(sample)
    summary["seed"] = seed
    summary["provider"] = provider
    summary["model"] = resolved_model
    summary["agent_type"] = agent_type.value
    summary["elapsed_seconds"] = round(elapsed, 1)

    _print_summary(summary)

    out_path = out_path or (
        RESULTS_DIR / f"eval_{split}_{provider}_{resolved_model}_{agent_type.value}_{len(sample)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "conversations": [
            {**asdict(r), "turns": [asdict(t) for t in r.turns]} for r in results
        ],
        "generated_at": datetime.now().isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    console.print(f"\n[bold]Results written to {out_path}[/bold]")
    return summary


def run_evaluation(  # Keep as sync wrapper for backward compatibility
    split: str = "dev",
    n: int = 25,
    seed: int = 42,
    provider: str = "openai",
    model: str | None = None,
    agent_type: AgentType = AgentType.direct,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Run the agent over a sample of conversations and report execution accuracy."""
    return asyncio.run(
        run_evaluation_async(
            split=split,
            n=n,
            seed=seed,
            provider=provider,
            model=model,
            agent_type=agent_type,
            out_path=out_path,
        )
    )


def _print_summary(summary: dict[str, Any]) -> None:
    table = Table(title="ConvFinQA Agentic RAG - Execution Accuracy")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row(
        "Split / provider / model / agent",
        f"{summary['split']} / {summary['provider']} / {summary['model']} / {summary['agent_type']}",
    )
    table.add_row("Conversations / turns", f"{summary['num_conversations']} / {summary['num_turns']}")
    table.add_row("Turn execution accuracy", f"{summary['turn_execution_accuracy']:.1%}")
    table.add_row("Conversation accuracy (all turns correct)", f"{summary['conversation_accuracy']:.1%}")
    for bucket, value in sorted(summary["accuracy_by_turn_position"].items()):
        table.add_row(f"  accuracy @ {bucket}", f"{value:.1%}" if value is not None else "n/a")
    if summary["accuracy_type1_conversations"] is not None:
        table.add_row("Type I (simple) accuracy", f"{summary['accuracy_type1_conversations']:.1%}")
    if summary["accuracy_type2_conversations"] is not None:
        table.add_row("Type II (hybrid) accuracy", f"{summary['accuracy_type2_conversations']:.1%}")
    table.add_row("Elapsed", f"{summary['elapsed_seconds']}s")
    console.print(table)


if __name__ == "__main__":
    run_evaluation()