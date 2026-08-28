"""Tools exposed to the ConvFinQA agent."""

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass, field

from llama_index.core.tools import FunctionTool

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return float(_BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_safe_eval(node.operand)))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def safe_calculate(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression (+ - * / ** and parentheses)."""
    cleaned = expression.replace("$", "").replace(",", "").replace("%", "")
    tree = ast.parse(cleaned, mode="eval")
    return _safe_eval(tree.body)


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(206588 - 181001) / 181001'.

    Use this for every subtraction, division, percentage-change or growth-rate
    calculation instead of doing arithmetic in your head, to avoid mistakes.
    """
    try:
        result = safe_calculate(expression)
    except Exception as exc:  # noqa: BLE001
        return f"error: could not evaluate {expression!r}: {exc}"
    return str(result)


def make_calculator_tool() -> FunctionTool:
    return FunctionTool.from_defaults(fn=calculator, name="calculator")


def make_document_tool(document: str) -> FunctionTool:
    def read_document() -> str:
        """Read the complete financial document.

        Use this whenever you need narrative or table context.
        """
        return document

    return FunctionTool.from_defaults(fn=read_document, name="read_document")


@dataclass
class FinalAnswerCapture:
    """Captures the structured final answer the agent submits for a turn."""

    value: str | None = field(default=None)
    rationale: str | None = field(default=None)

    def reset(self) -> None:
        self.value = None
        self.rationale = None

    def make_tool(self) -> FunctionTool:
        def record_final_answer(value: str, rationale: str) -> str:
            """Submit the final answer for the current question.

            Args:
                value: The final numeric answer as a plain number (no $ signs,
                    no % signs, no commas - e.g. '25587' or '0.1414' for a 14.14%
                    change). For yes/no questions, use exactly 'yes' or 'no'.
                rationale: A one-sentence explanation of how the answer was derived.

            You MUST call this exactly once, as your last action, for every question.
            """
            self.value = value
            self.rationale = rationale
            return "Final answer recorded."

        return FunctionTool.from_defaults(fn=record_final_answer, name="record_final_answer")
