"""Agent architectures for answering ConvFinQA conversational questions.

Several agentic approaches are compared on the same tool set (read_document,
calculator, record_final_answer):

- `direct`: answer directly, using tools as needed. Uses the LLM's native
  function-calling API (`FunctionAgent`, llama-index's workflow-based
  function-calling agent - the successor to the now-deprecated
  `FunctionCallingAgent`).
- `react`: same tools, but uses llama-index's workflow-based `ReActAgent`
  instead - the classic Thought/Action/Observation text-based agent loop,
  which parses the model's own text rather than relying on native
  tool-calling.
- `cot`: prompted to explicitly reason step by step before using tools.
- `planner`: a separate planner LLM call first produces an ordered list of
  steps, then an executor agent carries out that plan using tools.

All agent types here are `llama_index.core.agent.workflow` agents (the
successor to the deprecated `AgentRunner`-based `FunctionCallingAgent` /
`ReActAgent`), which run async and use an explicit `Context` per conversation
to persist chat memory across turns - see `ConvFinQAAgent.ask` below for how
that's bridged into this module's synchronous call interface.
"""

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum

from llama_index.core.agent.workflow import BaseWorkflowAgent, FunctionAgent, ReActAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Context
from pydantic import BaseModel, Field

from src.llm_providers import build_llm
from src.models import ConvFinQARecord
from src.retrieval import build_document_text
from src.tools import FinalAnswerCapture, make_calculator_tool, make_document_tool


class AgentType(str, Enum):
    """Which agentic approach to use."""

    direct = "direct"
    react = "react"
    cot = "cot"
    planner = "planner"


DIRECT_SYSTEM_PROMPT = """\
You are a financial analyst assistant. You answer a conversational sequence \
of questions about ONE financial document (report excerpt + data table). \
Later questions often depend on earlier turns in the conversation (e.g. \
"what about in 2008?", "what is the percentage change?") - use the \
conversation history to resolve what they refer to.

Rules:
- Always ground numbers in the `read_document` tool output. Do not guess \
figures from memory.
- Always use the `calculator` tool for arithmetic (subtraction, division, \
percentage change, growth rates). Do not compute by hand.
- Express ratios/percentage changes as decimals (e.g. 0.1414), not as \
percentage strings (not "14.14%"), unless the question is yes/no.
- For yes/no questions, answer exactly "yes" or "no".
- You MUST call `record_final_answer` exactly once, as your last step, for \
every question, with the plain final value and a one-sentence rationale.

Examples:

Example 1:
Question: What was the revenue in 2010?
Read the document, find the 2010 revenue value, then call \
record_final_answer with that value.

Example 2:
Question: What was the percentage change in revenue from 2008 to 2010?
Read the document, find revenue for both 2008 and 2010, use the \
calculator with (2010 revenue - 2008 revenue) / 2008 revenue, and call \
record_final_answer with the decimal result (not multiplied by 100). \
Double-check the sign of the result matches whether 2010 is higher or \
lower than 2008 before finalizing - a common mistake is flipping the sign \
or the order of subtraction.
"""

COT_SYSTEM_PROMPT = """\
You are a financial reasoning assistant. You answer a conversational \
sequence of questions about ONE financial document (report excerpt + data \
table). Later questions often depend on earlier turns in the conversation \
(e.g. "what about in 2008?", "what is the percentage change?") - use the \
conversation history to resolve what they refer to.

Before using tools:
1. Identify the entities involved (which years, line items, or metrics the
   question refers to).
2. Determine the required calculations.
3. Decide which information is needed from the document.
4. Verify the result.

Do not answer until this reasoning is complete. Use tools when required.

Rules:
- Always ground numbers in the `read_document` tool output. Do not guess \
figures from memory.
- Always use the `calculator` tool for arithmetic (subtraction, division, \
percentage change, growth rates). Do not compute by hand.
- Express ratios/percentage changes as decimals (e.g. 0.1414), not as \
percentage strings (not "14.14%"), unless the question is yes/no.
- For yes/no questions, answer exactly "yes" or "no".
- Finally, you MUST call `record_final_answer` exactly once, as your last \
step, with the plain final value and a one-sentence rationale.

Examples:

Example 1:
Question: What was the revenue in 2010?
Reasoning: (1) Entities: revenue, year 2010. (2) Calculation needed: none, \
this is a direct lookup. (3) Information needed: the revenue figure for \
2010 from the document. (4) Verify: after reading the document, confirm \
the value read is specifically the 2010 revenue figure, not a nearby year \
or a different line item.
Then: read the document, find the 2010 revenue value, call \
record_final_answer with that value.

Example 2:
Question: What was the percentage change in revenue from 2008 to 2010?
Reasoning: (1) Entities: revenue, years 2008 and 2010. (2) Calculation: \
percentage change = (2010 revenue - 2008 revenue) / 2008 revenue. (3) \
Information needed: revenue for both 2008 and 2010. (4) Verify: check that \
the sign of the result matches whether 2010 is higher or lower than 2008 \
before finalizing - a common mistake is flipping the sign or the order of \
subtraction.
Then: read the document, find both values, use the calculator with \
(2010 revenue - 2008 revenue) / 2008 revenue, and call record_final_answer \
with the decimal result (not multiplied by 100).
"""

PLAN_PROMPT = PromptTemplate(
    "Create a plan to answer a question about a financial document. Return "
    "an ordered list of concrete steps needed to answer it - steps only, do "
    "not answer the question itself.\n\n"
    "Important: you may reference which year/period/line item is involved "
    "(e.g. '2010 revenue'), but never write down the actual numeric VALUE "
    "for any of them, even if it appears earlier in the conversation - the "
    "executor must always look every value up fresh from the document. You "
    "may describe a calculation symbolically (e.g. '(2010 revenue - 2008 "
    "revenue) / 2008 revenue'), but never include a scaling factor like "
    "'* 100' - the executor decides output formatting, not the plan.\n\n"
    "Examples:\n\n"
    "Example 1:\n"
    "Conversation: None\n"
    "Question: What was the revenue in 2010?\n"
    "Plan:\n"
    '["Find the revenue value for 2010 in the financial document"]\n\n'
    "Example 2:\n"
    "Conversation:\n"
    "Q: What was operating income in 2008?\n"
    "A: Operating income in 2008 was 100.\n"
    "Question: What about 2010?\n"
    "Plan:\n"
    '["Resolve that the question refers to operating income", '
    '"Find the operating income value for 2010 in the financial document"]\n\n'
    "Example 3:\n"
    "Conversation: None\n"
    "Question: What was the percentage increase in revenue from 2008 to 2010?\n"
    "Plan:\n"
    '["Find the revenue value for 2008 in the financial document", '
    '"Find the revenue value for 2010 in the financial document", '
    '"Calculate the percentage change using (2010 revenue - 2008 revenue) / 2008 revenue"]\n\n'
    "Example 4:\n"
    "Conversation: None\n"
    "Question: Was the company's net income higher in 2011 than in 2010?\n"
    "Plan:\n"
    '["Find the net income value for 2010 in the financial document", '
    '"Find the net income value for 2011 in the financial document", '
    '"Compare the two values and determine whether 2011 is higher than 2010"]\n\n'
    "Example 5 (the conversation switches to an unrelated topic):\n"
    "Conversation:\n"
    "Q: What was the revenue in 2008?\n"
    "A: Revenue in 2008 was 500.\n"
    "Q: What was the revenue in 2010?\n"
    "A: Revenue in 2010 was 600.\n"
    "Q: What was the percentage increase in revenue from 2008 to 2010?\n"
    "A: The percentage increase was 0.2.\n"
    "Question: What was the operating margin in 2010?\n"
    "Plan:\n"
    '["Recognize this question shifts to operating margin, a new topic '
    'unrelated to the prior revenue discussion", '
    '"Find the operating margin value for 2010 in the financial document"]\n\n'
    "Example 6 (a pronoun must resolve to the MOST RECENT topic, not an "
    "earlier one from the same conversation):\n"
    "Conversation:\n"
    "Q: What was the revenue in 2008?\n"
    "A: Revenue in 2008 was 500.\n"
    "Q: What was the revenue in 2010?\n"
    "A: Revenue in 2010 was 600.\n"
    "Q: What was the operating margin in 2010?\n"
    "A: Operating margin in 2010 was 0.15.\n"
    "Question: And what was it in 2008?\n"
    "Plan:\n"
    '["Resolve that \'it\' refers to operating margin, the most recently '
    'discussed topic, not revenue", '
    '"Find the operating margin value for 2008 in the financial document"]\n\n'
    "Now create a plan for this question.\n\n"
    "Conversation so far:\n{history}\n\n"
    "Question: {question}\n\n"
    "Return steps only."
)


class Plan(BaseModel):
    """An ordered plan of steps to answer a financial question."""

    steps: list[str] = Field(
        description="Ordered list of concrete steps needed to answer the question."
    )


PLANNER_EXECUTOR_SYSTEM_PROMPT = """\
You are a financial analyst assistant executing a pre-made plan to answer \
a question about ONE financial document (report excerpt + data table). \
Follow the plan's steps in order, using tools as needed for each step. \
Later questions often depend on earlier turns in the conversation (e.g. \
"what about in 2008?", "what is the percentage change?") - use the \
conversation history to resolve what they refer to.

Rules:
- Always ground numbers in the `read_document` tool output. Do not guess \
figures from memory.
- Always use the `calculator` tool for arithmetic (subtraction, division, \
percentage change, growth rates). Do not compute by hand.
- Express ratios/percentage changes as decimals (e.g. 0.1414), not as \
percentage strings (not "14.14%"), unless the question is yes/no.
- For yes/no questions, answer exactly "yes" or "no".
- You MUST call `record_final_answer` exactly once, as your last step, for \
every question, with the plain final value and a one-sentence rationale.
"""

# ReActAgent builds its own Thought/Action/Observation prompt internally;
# task-specific instructions are passed via `system_prompt` like any other
# workflow agent, and folded into the ReAct formatter's context automatically.
REACT_CONTEXT = """\
You answer a conversational sequence of questions about ONE financial \
document (report excerpt + data table). Later questions often depend on \
earlier turns in the conversation (e.g. "what about in 2008?", "what is \
the percentage change?") - use the conversation history to resolve what \
they refer to.

Rules:
- Always ground numbers in the `read_document` tool output. Do not guess \
figures from memory.
- Always use the `calculator` tool for arithmetic (subtraction, division, \
percentage change, growth rates). Do not compute by hand.
- Express ratios/percentage changes as decimals (e.g. 0.1414), not as \
percentage strings (not "14.14%"), unless the question is yes/no.
- For yes/no questions, answer exactly "yes" or "no".
- You MUST call `record_final_answer` exactly once, as your last step, for \
every question, with the plain final value and a one-sentence rationale.

Examples:

Example 1:
Question: What was the revenue in 2010?
This is a direct lookup - no calculation needed. Read the document, find \
the 2010 revenue value, and call record_final_answer with that value.

Example 2:
Question: What was the percentage change in revenue from 2008 to 2010?
Read the document to find both the 2008 and 2010 revenue values, then use \
the calculator with (2010 revenue - 2008 revenue) / 2008 revenue. Before \
calling record_final_answer, double check the sign of the result matches \
whether 2010 is higher or lower than 2008 - a common mistake is flipping \
the sign or the subtraction order. Report the decimal result (not \
multiplied by 100).
"""

_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _extract_fallback_answer(text: str) -> str:
    """Best-effort extraction of a final value from free text.

    Used when the model skips `record_final_answer` and just answers in a
    full sentence (e.g. "The percentage change ... is 0.1414."). Takes the
    LAST number or yes/no mentioned, since the actual answer is typically
    stated at the end of the sentence - not perfectly reliable (a trailing
    year mention would win over an earlier value), but far better than
    treating the whole sentence as the value.
    """
    numbers: list[str] = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1]
    yes_no: list[str] = _YES_NO_RE.findall(text)
    if yes_no:
        return yes_no[-1].lower()
    return text.strip()


@dataclass
class ConvFinQAAgent:
    agent: BaseWorkflowAgent
    capture: FinalAnswerCapture
    # Workflow agents have no synchronous `.chat()` - a `Context` must be
    # created once per conversation (`Context(agent)`) and passed to every
    # `.run()` call so chat memory persists across turns.
    ctx: Context
    # Forwarded to `.run()` on every call; only takes effect on the first
    # call for a given `ctx` (see `BaseWorkflowAgent._init_context`). `None`
    # falls back to the class default (20).
    max_iterations: int | None = None

    async def ask(self, question: str) -> tuple[str, str | None, str | None]:
        """Ask a question in the ongoing conversation.

        Returns (natural_language_response, final_value, rationale).
        """
        self.capture.reset()
        response = await self.agent.run(question, ctx=self.ctx, max_iterations=self.max_iterations)
        text = str(response)

        value, rationale = self.capture.value, self.capture.rationale
        if value is None and text.strip():
            # Some free/local models occasionally skip the required
            # `record_final_answer` call and just answer in plain text -
            # fall back to that text rather than losing an otherwise-correct
            # answer to a missed tool call.
            value = _extract_fallback_answer(text)
            rationale = "(model answered directly without calling record_final_answer)"

        return text, value, rationale


@dataclass
class ConvFinQAAgentPlanner:
    """Plan-then-execute agent.

    A separate planner LLM call first produces an ordered list of steps for
    the question, then a tool-calling executor agent (same tools as the
    direct agent) carries out that plan to produce the final answer.
    """

    llm: FunctionCallingLLM
    executor: FunctionAgent
    capture: FinalAnswerCapture
    ctx: Context
    history: list[tuple[str, str | None]] = field(default_factory=list)

    def _make_plan(self, question: str) -> Plan:
        history_block = "\n".join(f"Q: {q}\nA: {a}" for q, a in self.history) or "(none yet)"
        plan = self.llm.structured_predict(Plan, PLAN_PROMPT, history=history_block, question=question)
        assert isinstance(plan, Plan)
        return plan

    async def ask(self, question: str) -> tuple[str, str | None, str | None]:  # Added async
        """Ask a question in the ongoing conversation.

        Returns (natural_language_response, final_value, rationale).
        """
        plan = self._make_plan(question)
        plan_block = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan.steps))

        self.capture.reset()
        response = await self.executor.run(f"Plan:\n{plan_block}\n\nQuestion: {question}", ctx=self.ctx)  # Added await
        text = str(response)

        value, rationale = self.capture.value, self.capture.rationale
        if value is None and text.strip():
            value = _extract_fallback_answer(text)
            rationale = "(model answered directly without calling record_final_answer)"

        self.history.append((question, value))
        return text, value, rationale


def build_agent(
    record: ConvFinQARecord,
    provider: str = "openai",
    model: str | None = None,
    llm: FunctionCallingLLM | None = None,
    agent_type: AgentType = AgentType.direct,
    verbose: bool = False,
) -> ConvFinQAAgent | ConvFinQAAgentPlanner:
    """Build a fresh agent + tool set scoped to a single ConvFinQA record.

    `provider`/`model` pick the backend (e.g. provider="groq" for a free,
    hosted alternative to OpenAI); pass `llm` directly instead to bypass the
    provider factory entirely. `agent_type` picks the agentic approach - see
    `AgentType`.
    """
    llm = llm or build_llm(provider, model)

    document_text = build_document_text(record.doc)
    capture = FinalAnswerCapture()

    tools: list[BaseTool] = [
        make_document_tool(document_text),
        make_calculator_tool(),
        capture.make_tool(),
    ]

    if agent_type == AgentType.planner:
        executor = FunctionAgent(
            tools=tools,
            llm=llm,
            system_prompt=PLANNER_EXECUTOR_SYSTEM_PROMPT,
            verbose=verbose,
            allow_parallel_tool_calls=False,
        )
        return ConvFinQAAgentPlanner(llm=llm, executor=executor, capture=capture, ctx=Context(executor))

    if agent_type == AgentType.react:
        react_agent = ReActAgent(
            tools=tools,
            llm=llm,
            system_prompt=REACT_CONTEXT,
            verbose=verbose,
        )
        return ConvFinQAAgent(
            agent=react_agent,
            capture=capture,
            ctx=Context(react_agent),
            # Class default is 20; empirically, capping at 15 does better on
            # this dataset (see REPORT.md's ReAct comparison) - a looser cap
            # gave harder (hybrid) conversations more room to lose track of
            # cross-turn coreference before ever producing a final answer.
            max_iterations=15,
        )

    system_prompt = COT_SYSTEM_PROMPT if agent_type == AgentType.cot else DIRECT_SYSTEM_PROMPT

    agent = FunctionAgent(
        tools=tools,
        llm=llm,
        system_prompt=system_prompt,
        verbose=verbose,
        # OpenAI parallel tool calls raced with the old FunctionCallingAgent's
        # early-exit once `record_final_answer` was called, leaving a
        # dangling tool_call in memory that broke the *next* turn's request.
        # FunctionAgent's step-based tool handling looks more careful about
        # this, but sequential tool calls are kept as the safe default.
        allow_parallel_tool_calls=False,
    )
    return ConvFinQAAgent(agent=agent, capture=capture, ctx=Context(agent))
