"""The finance assistant: an LLM (temperature 0) that answers only from tool results.

How answers stay grounded in the database:
  * every figure is computed by a tool (agent_tools.py); the prompt forbids the model from doing math or recalling numbers;
  * changes are only ever *staged* by tools and applied after the user approves them;
  * before an answer is shown, every number and date in it is checked against the tool results of this
    conversation (check_grounding). Unverifiable figures send the model back to fetch real data once or twice;
    if it still can't back them up, the answer is shown with a visible warning.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

import agent_tools as at
import llm
from config import AGENT_MAX_STEPS, SHARED_SPLIT

MAX_GROUNDING_RETRIES = 2
MAX_HISTORY_MESSAGES = 60


def system_prompt() -> str:
    today = date.today()
    return f"""You are the assistant inside a personal finance app. Today is {today.isoformat()} (current month {today:%Y-%m}).
You work for one person, using only the app's database through your tools.

GROUNDING RULES (most important):
1. Every number, date, id, category, merchant or bank you state must come from a tool result in this conversation. \
Never estimate, guess, round differently, or remember figures. If you need a number, call a tool.
2. Never do arithmetic yourself. Totals, averages, highs, lows, counts, differences and trends come from the aggregate, \
trend, budget_vs_actual and other tools. If no tool gives the figure, say you can't compute it.
3. If a tool returns nothing, an error, or only part of what was asked, say so plainly. Do not fill gaps. \
If a result says truncated or shows fewer rows than matched, say that.
4. If the question is ambiguous (which month? which category?), either ask, or state the assumption you used \
(for example "this month = {today:%Y-%m}") and use it in the tool call.
5. Look up ids with query_transactions or list_review_queue; never invent them. Use exact category and bank names \
from get_overview.
6. Changing data: write tools only PROPOSE an action. Nothing changes until the user clicks Approve. Say "I proposed ..." \
and never "I changed ..." until a message tells you the action was approved and applied.
7. Say which figure you used when it matters: "my share" (shared rows count {SHARED_SPLIT:.0%}) vs full amount. \
Budget difference = Budget - Actual, positive = under budget.
8. You cannot see why an earlier reply failed. If asked about your own errors, rely only on "[app notice]" messages \
and otherwise say you don't know; never guess at a cause.
9. Be concise. Quote figures exactly as the tools return them, as dollars with 2 decimals. Use short tables or bullets \
for lists. Give no financial advice beyond what the data shows."""


# ---------------------------------------------------------------- grounding check
NUMBER = re.compile(r"(?<![\w.])[-−]?\$?[-−]?\d[\d,]*(?:\.\d+)?%?")
DATE_TOKEN = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
LIST_MARKER = re.compile(r"(?m)^\s*\d{1,2}[.)]\s")
SMALL_INT = 20   # bare integers up to this are treated as counts/ordinals, not figures


def _evidence_numbers(texts: list[str]) -> list[float]:
    nums = []
    for t in texts:
        nums += [float(m) for m in re.findall(r"\d+(?:\.\d+)?", t.replace(",", ""))]
    return nums


def check_grounding(answer: str, evidence: list[str]) -> list[str]:
    """Numbers and dates in `answer` that appear nowhere in `evidence` (tool results, the user's messages and
    approval notes). A figure matches if it equals an evidence number at the precision the answer shows
    ($1,234.6 matches 1234.56; 12% matches 0.12 or 12). Bare integers up to 20 are exempt (counts, ordinals)."""
    blob = " ".join(evidence)
    missing = []
    for d in DATE_TOKEN.findall(answer):
        if d not in blob and d not in missing:
            missing.append(d)
    text = LIST_MARKER.sub(" ", DATE_TOKEN.sub(" ", answer))
    known = _evidence_numbers(evidence)
    for tok in NUMBER.findall(text):
        raw = tok.replace("−", "-").replace("$", "").replace(",", "").rstrip("%")
        try:
            x = abs(float(raw))
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if decimals == 0 and "$" not in tok and not tok.endswith("%") and x <= SMALL_INT:
            continue
        tol = 0.5 * 10 ** -decimals + 1e-9
        pct = tok.endswith("%")
        ok = any(abs(e - x) <= tol or (pct and abs(e * 100 - x) <= tol) for e in known)
        if not ok and tok not in missing:
            missing.append(tok)
    return missing


# ---------------------------------------------------------------- text-form tool calls and reasoning blocks
THINK = re.compile(r"<think>.*?</think>", re.S)
TEXT_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _strip_think(text: str) -> str:
    return THINK.sub("", text or "").strip()


def _text_tool_calls(content: str) -> list[dict]:
    """Some local models print tool calls as <tool_call>{...}</tool_call> text instead of using the tools field."""
    calls = []
    for i, m in enumerate(TEXT_CALL.findall(content or "")):
        try:
            d = json.loads(m)
            args = d.get("arguments", d.get("parameters", {}))
            calls.append({"id": f"text-{i}", "type": "function",
                          "function": {"name": d["name"], "arguments": json.dumps(args) if isinstance(args, dict) else args}})
        except (json.JSONDecodeError, KeyError):
            continue
    return calls


# ---------------------------------------------------------------- the agent
@dataclass
class ToolCall:
    name: str
    args: dict
    result: dict


@dataclass
class Turn:
    """What one question produced: the answer, the tool calls behind it, and any grounding warning."""
    answer: str
    calls: list[ToolCall] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)   # what went wrong along the way (empty replies, rejected figures)
    staged: list[int] = field(default_factory=list)   # ids of actions proposed this turn (not yet applied)
    error: str | None = None


class Agent:
    def __init__(self, model: str, chat=llm.chat):
        self.model = model
        self.chat = chat
        self.ctx = at.Context()
        self.messages: list[dict] = []
        self.evidence: list[str] = [date.today().isoformat(), date.today().strftime("%Y-%m")]

    # -- conversation state
    def note(self, text: str) -> None:
        """Tell the model something that happened outside the chat (e.g. the user approved an action)."""
        self.messages.append({"role": "user", "content": f"[app notice] {text}"})
        self.evidence.append(text)

    def _compact_history(self) -> None:
        """Earlier turns keep their conclusions but drop bulky tool output; the model re-fetches if it needs it."""
        for m in self.messages:
            if m["role"] == "tool" and not m["content"].startswith("[earlier"):
                m["content"] = "[earlier tool output omitted; call the tool again if you need the data]"
        if len(self.messages) > MAX_HISTORY_MESSAGES:
            del self.messages[:len(self.messages) - MAX_HISTORY_MESSAGES]
            while self.messages and self.messages[0]["role"] != "user":   # never start on an orphaned tool reply
                self.messages.pop(0)

    # -- one question
    def ask(self, question: str) -> Turn:
        self._compact_history()
        self.messages.append({"role": "user", "content": question})
        self.evidence.append(question)
        turn = Turn(answer="")
        candidate: tuple[str, list[str]] | None = None   # newest non-empty answer that failed the figures check
        retries = 0
        for _ in range(AGENT_MAX_STEPS):
            try:
                msg = self.chat([{"role": "system", "content": system_prompt()}] + self.messages,
                                tools=at.schemas(), model=self.model)
            except llm.LLMError as e:
                turn.error = str(e)
                turn.answer = "I couldn't reach the language model, so I can't answer right now."
                return turn
            content = _strip_think(msg.get("content"))
            calls = msg.get("tool_calls") or _text_tool_calls(msg.get("content") or "")

            if calls:
                self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for c in calls:
                    self._run_call(c, turn)
                continue

            if content:
                self.messages.append({"role": "assistant", "content": content})
                bad = check_grounding(content, self.evidence)
                if not bad:
                    return self._finish(turn, content, [])
                candidate = (content, bad)
                turn.events.append(f"Figures not found in the data were rejected: {', '.join(bad)}.")
                nudge = (f"These figures in your answer do not appear in any tool result: {', '.join(bad)}. "
                         "Do not state numbers you did not get from a tool (and do not add up figures yourself). "
                         "Fetch the real values with the right tool, or remove the claim and say the data isn't "
                         "available. Then answer again, briefly.")
            else:   # reasoning models can spend the whole token budget thinking and return no answer text
                turn.events.append("The model returned an empty reply.")
                nudge = "Your last reply was empty. Give the final answer now, briefly, using only the tool results above."
            if retries >= MAX_GROUNDING_RETRIES:
                break
            retries += 1
            self.messages.append({"role": "user", "content": f"[app notice] {nudge}"})

        if candidate:   # better to show the newest answer with a visible warning than nothing
            return self._finish(turn, *candidate)
        turn.answer = ("The model didn't produce a readable answer"
                       + (", though it did fetch data (see Data used)." if turn.calls else ".") + " Please try again.")
        return self._finish(turn, turn.answer, [])

    def _finish(self, turn: Turn, answer: str, ungrounded: list[str]) -> Turn:
        turn.answer, turn.ungrounded = answer, ungrounded
        if turn.events:   # so the model can answer "what went wrong?" from facts, not guesses
            self.note("About your previous answer: " + " ".join(turn.events)
                      + (" The figures shown to the user carry a warning." if ungrounded else ""))
        return turn

    def _run_call(self, call: dict, turn: Turn) -> None:
        fn = call["function"]
        try:
            args = json.loads(fn.get("arguments") or "{}") if isinstance(fn.get("arguments"), str) else (fn.get("arguments") or {})
        except json.JSONDecodeError:
            args, result = {}, {"error": "Tool arguments were not valid JSON."}
        else:
            result = at.run_tool(self.ctx, fn["name"], args)
        text = at.evidence_text(result)
        self.evidence.append(text)
        turn.calls.append(ToolCall(fn["name"], args, result))
        if result.get("status") == "pending_user_approval":
            turn.staged.append(result["action_id"])
        self.messages.append({"role": "tool", "tool_call_id": call.get("id", fn["name"]), "content": text})
