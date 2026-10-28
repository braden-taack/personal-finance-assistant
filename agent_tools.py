"""Tools the assistant agent can call. No Streamlit and no LLM in here, so everything is testable.

Read tools compute every figure with pandas from the database (the model never does arithmetic).
Write tools never touch the database: they validate the request against real data, then *stage* an action.
Nothing changes until the user approves the staged action (apply_action), so a wrong guess by the model
can't alter the books.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

import classifier
import db
import reports
from config import AGENT_MAX_ROWS, SHARED_SPLIT

MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
MAX_BULK_ROWS = 500
UNCATEGORIZED = "Uncategorized"


class ToolError(Exception):
    """A request the tool can't honour; the message goes back to the model so it can correct itself."""


# ---------------------------------------------------------------- registry
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable
    write: bool


TOOLS: dict[str, Tool] = {}


def tool(description: str, props: dict | None = None, required: tuple = (), write: bool = False):
    def deco(fn):
        TOOLS[fn.__name__] = Tool(fn.__name__, description,
                                  {"type": "object", "properties": props or {}, "required": list(required)}, fn, write)
        return fn
    return deco


def schemas() -> list[dict]:
    """The tool list in OpenAI function-calling format."""
    return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in TOOLS.values()]


class Context:
    """Per-conversation state shared by tools: the staged (pending) actions."""

    def __init__(self):
        self.pending: list[dict] = []
        self._next = 1

    def stage(self, tool_name: str, args: dict, summary: str, preview: list[dict] | None = None,
              rows_affected: int | None = None) -> dict:
        for a in self.pending:   # asking twice for the same thing must not queue it twice
            if a["status"] == "pending" and a["tool"] == tool_name and a["args"] == args:
                return self._ack(a)
        action = {"id": self._next, "tool": tool_name, "args": args, "summary": summary, "preview": preview or [],
                  "rows_affected": rows_affected, "status": "pending", "result": None}
        self._next += 1
        self.pending.append(action)
        return self._ack(action)

    @staticmethod
    def _ack(a: dict) -> dict:
        return {"status": "pending_user_approval", "action_id": a["id"], "summary": a["summary"],
                "rows_affected": a["rows_affected"], "preview": a["preview"][:5],
                "note": "NOTHING HAS CHANGED YET. The user must approve this action in the Pending actions panel."}

    def open_actions(self) -> list[dict]:
        return [a for a in self.pending if a["status"] == "pending"]


def run_tool(ctx: Context, name: str, args: dict) -> dict:
    """Execute a tool call; every failure becomes {"error": ...} for the model to read."""
    t = TOOLS.get(name)
    if t is None:
        return {"error": f"Unknown tool '{name}'. Available tools: {sorted(TOOLS)}"}
    if not isinstance(args, dict):
        return {"error": "Tool arguments must be a JSON object."}
    try:
        return _py(t.fn(ctx, **args))
    except ToolError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}. Expected parameters: {sorted(t.parameters['properties'])}"}
    except Exception as e:   # a bug in a tool must not crash the conversation
        return {"error": f"{name} failed: {type(e).__name__}: {e}"}


def _py(x):
    """Make a value JSON-safe: numpy -> python, NaN/NaT -> None, Timestamp -> date string."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, pd.Timestamp):
        return None if pd.isna(x) else x.strftime("%Y-%m-%d")
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) or math.isinf(x) else round(float(x), 4)
    if x is pd.NaT or x is None:
        return None
    return x


# ---------------------------------------------------------------- argument helpers
def _bool(v, name: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    raise ToolError(f"'{name}' must be true or false.")


def _num(v, name: str) -> float:
    try:
        x = float(str(v).replace("$", "").replace(",", "")) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        raise ToolError(f"'{name}' must be a number.") from None
    if math.isnan(x) or math.isinf(x):
        raise ToolError(f"'{name}' must be a finite number.")
    return x


def _month(v, name: str = "month") -> str:
    if not isinstance(v, str) or not MONTH_RE.match(v.strip()):
        raise ToolError(f"'{name}' must look like 2026-03 (yyyy-mm).")
    return v.strip()


def _iso(v, name: str) -> str:
    try:
        return date.fromisoformat(str(v).strip()).isoformat()
    except ValueError:
        raise ToolError(f"'{name}' must be a real date like 2026-03-15 (yyyy-mm-dd).") from None


def _all_categories() -> list[str]:
    return db.get_categories(kind=None, active_only=False)


def _pick(value: str, options: list[str], what: str) -> str:
    """Case-insensitive match of `value` against `options`, or a ToolError that lists the real options."""
    lookup = {o.lower(): o for o in options}
    hit = lookup.get(str(value).strip().lower())
    if hit is None:
        raise ToolError(f"Unknown {what} '{value}'. Valid {what}s: {options}")
    return hit


# ---------------------------------------------------------------- filtering
FILTERS = {
    "month": {"type": "string", "description": "Only this month, yyyy-mm."},
    "start_date": {"type": "string", "description": "From this date (inclusive), yyyy-mm-dd."},
    "end_date": {"type": "string", "description": "Up to this date (inclusive), yyyy-mm-dd."},
    "category": {"type": "string", "description": "A category name, or 'Uncategorized'."},
    "bank": {"type": "string", "description": "A bank name (e.g. AMEX), or '(none)'."},
    "search": {"type": "string", "description": "Text contained in the description (case-insensitive)."},
    "shared": {"type": "boolean", "description": "true = only shared-with-roommate rows, false = only unshared."},
    "settled": {"type": "boolean", "description": "Settlement status of shared rows."},
    "min_amount": {"type": "number", "description": "Full transaction amount is at least this."},
    "max_amount": {"type": "number", "description": "Full transaction amount is at most this."},
    "kind": {"type": "string", "enum": ["expense", "income", "all"],
             "description": "Default is expense (what budgets count). Income rows are legacy imports."},
    "source": {"type": "string", "enum": ["ml", "llm", "manual", "import", "rule"],
               "description": "How the category was assigned."},
    "needs_review": {"type": "boolean", "description": "Rows waiting in the review queue."},
    "ids": {"type": "array", "items": {"type": "integer"}, "description": "Specific transaction ids."},
}


def _frame(f: dict, default_kind: str | None = "expense") -> pd.DataFrame:
    """All transactions matching the filters, with `my_share` (amount after the roommate split) and `month`."""
    unknown = set(f) - set(FILTERS)
    if unknown:
        raise ToolError(f"Unknown filter(s) {sorted(unknown)}. Valid filters: {sorted(FILTERS)}")
    df = db.fetch_transactions()
    df["date"] = pd.to_datetime(df["date"])
    df["my_share"] = db.effective_amount(df)
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["category_label"] = df["category"].where(df["category"].notna() & (df["category"] != ""), UNCATEGORIZED)

    kind = f.get("kind", default_kind)
    if kind not in (None, "all"):
        df = df[df["kind"] == _pick(kind, ["expense", "income"], "kind")]
    if "ids" in f:
        df = df[df["id"].isin([int(i) for i in f["ids"]])]
    if "month" in f:
        df = df[df["month"] == _month(f["month"])]
    if "start_date" in f:
        df = df[df["date"] >= pd.Timestamp(_iso(f["start_date"], "start_date"))]
    if "end_date" in f:
        df = df[df["date"] <= pd.Timestamp(_iso(f["end_date"], "end_date"))]
    if "category" in f:
        if str(f["category"]).strip().lower() == UNCATEGORIZED.lower():
            df = df[df["category_label"] == UNCATEGORIZED]
        else:
            df = df[df["category"] == _pick(f["category"], _all_categories(), "category")]
    if "bank" in f:
        if str(f["bank"]).strip().lower() == "(none)":
            df = df[df["bank"].isna() | (df["bank"] == "")]
        else:
            df = df[df["bank"] == _pick(f["bank"], db.get_banks(), "bank")]
    if "search" in f:
        df = df[df["description"].str.contains(str(f["search"]), case=False, regex=False, na=False)]
    if "shared" in f:
        df = df[(df["shared"] == 1) == _bool(f["shared"], "shared")]
    if "settled" in f:
        df = df[(df["shared_settled"] == 1) == _bool(f["settled"], "settled")]
    if "min_amount" in f:
        df = df[df["amount"] >= _num(f["min_amount"], "min_amount")]
    if "max_amount" in f:
        df = df[df["amount"] <= _num(f["max_amount"], "max_amount")]
    if "source" in f:
        df = df[df["cat_source"] == _pick(f["source"], ["ml", "llm", "manual", "import", "rule"], "source")]
    if "needs_review" in f:
        df = df[(df["needs_review"] == 1) == _bool(f["needs_review"], "needs_review")]
    return df


def _row(r) -> dict:
    return {"id": int(r.id), "date": r.date.strftime("%Y-%m-%d"), "bank": r.bank or None, "description": r.description,
            "category": r.category_label, "amount": round(float(r.amount), 2), "my_share": round(float(r.my_share), 2),
            "kind": r.kind, "shared": bool(r.shared), "settled": bool(r.shared_settled),
            "source": r.cat_source, "confidence": None if pd.isna(r.cat_confidence) else round(float(r.cat_confidence), 2),
            "needs_review": bool(r.needs_review)}


def _rows(df: pd.DataFrame, limit: int) -> list[dict]:
    return [_row(r) for r in df.head(limit).itertuples()]


def _field(field: str) -> str:
    if field not in ("my_share", "amount"):
        raise ToolError("'field' must be 'my_share' (what the dashboard counts) or 'amount' (full amount).")
    return field


FIELD_PROP = {"type": "string", "enum": ["my_share", "amount"],
              "description": "my_share (default) = what counts toward my budget (shared rows count "
                             f"{SHARED_SPLIT:.0%}); amount = full transaction amount."}


# ---------------------------------------------------------------- read tools
@tool("Orientation: today's date, how much data exists and its date range, the category and bank names, review-queue "
      "size and how amounts are defined. Call this first when unsure what data or names exist.")
def get_overview(ctx: Context) -> dict:
    with db.connect() as c:
        n, first, last = c.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM transactions").fetchone()
        uncategorized = c.execute("SELECT COUNT(*) FROM transactions WHERE category IS NULL OR category=''").fetchone()[0]
    cats = db.category_table()
    today = date.today()
    return {
        "today": today.isoformat(), "current_month": today.strftime("%Y-%m"),
        "transactions": {"count": n, "first_date": first, "last_date": last, "uncategorized": uncategorized},
        "months_with_expenses": sorted(db.transaction_months()),
        "categories": [{"name": r["name"], "is_savings": bool(r["is_savings"])} for _, r in cats.iterrows()
                       if r["active"] and r["kind"] == "expense"],
        "retired_categories": [r["name"] for _, r in cats.iterrows() if not r["active"]],
        "banks": db.get_banks(),
        "review_queue_count": len(db.fetch_review_queue()),
        "definitions": {
            "amount": "Full transaction amount; positive = money spent, negative = refund.",
            "my_share": f"amount after the roommate split: shared rows count {SHARED_SPLIT:.0%}, others 100%. "
                        "The dashboard's Actual figures use my_share.",
            "budget_difference": "Budget minus Actual; positive = under budget, negative = over budget.",
        },
    }


@tool("List individual transactions matching filters, sorted. Returns the matching row count and totals for ALL "
      f"matches plus up to {AGENT_MAX_ROWS} rows. Use it to find ids and see details.",
      {**FILTERS,
       "sort_by": {"type": "string", "enum": ["date", "amount", "my_share", "description", "category"]},
       "descending": {"type": "boolean"},
       "limit": {"type": "integer", "description": f"Rows to return (default 25, max {AGENT_MAX_ROWS})."}})
def query_transactions(ctx: Context, sort_by: str = "date", descending=True, limit=25, **filters) -> dict:
    if sort_by not in ("date", "amount", "my_share", "description", "category"):
        raise ToolError("sort_by must be one of date, amount, my_share, description, category.")
    df = _frame(filters)
    limit = max(1, min(int(limit), AGENT_MAX_ROWS))
    df = df.sort_values("category_label" if sort_by == "category" else sort_by,
                        ascending=not _bool(descending, "descending"), kind="stable")
    return {"filters": filters, "matching_rows": len(df), "returned_rows": min(len(df), limit),
            "total_amount": round(float(df["amount"].sum()), 2), "total_my_share": round(float(df["my_share"].sum()), 2),
            "rows": _rows(df, limit)}


GROUPS = {
    "none": None,
    "category": lambda d: d["category_label"],
    "bank": lambda d: d["bank"].fillna("(none)").replace("", "(none)"),
    "month": lambda d: d["month"],
    "year": lambda d: d["date"].dt.strftime("%Y"),
    "merchant": lambda d: d["description"].map(classifier.clean).replace("", "(blank)"),
    "shared": lambda d: d["shared"].map({1: "shared", 0: "not shared"}),
    "source": lambda d: d["cat_source"].fillna("(none)"),
}
METRICS = {"sum": "sum", "avg": "mean", "min": "min", "max": "max", "count": "count", "median": "median"}


@tool("Do math over transactions: sum, avg, min, max, median or count, optionally grouped (by category, bank, month, "
      "year, merchant, shared, source), with the same filters as query_transactions. ALWAYS use this for totals, "
      "averages, highs and lows; never calculate yourself. avg is per transaction. For average per MONTH use the "
      "trend tool.",
      {**FILTERS,
       "metric": {"type": "string", "enum": list(METRICS)},
       "field": FIELD_PROP,
       "group_by": {"type": "string", "enum": list(GROUPS), "description": "Default none = one overall figure."},
       "top_n": {"type": "integer", "description": "Max groups to return (default 25)."}},
      required=("metric",))
def aggregate(ctx: Context, metric: str, field: str = "my_share", group_by: str = "none", top_n=25, **filters) -> dict:
    if metric not in METRICS:
        raise ToolError(f"metric must be one of {list(METRICS)}.")
    if group_by not in GROUPS:
        raise ToolError(f"group_by must be one of {list(GROUPS)}.")
    field = _field(field)
    df = _frame(filters)
    out = {"metric": metric, "field": field, "filters": filters, "matching_rows": len(df)}
    if df.empty:
        return {**out, "overall": None, "groups": [], "note": "No transactions match these filters."}
    out["overall"] = {"value": round(float(df[field].agg(METRICS[metric])), 2), "count": len(df)}
    if metric in ("min", "max"):
        idx = df[field].idxmin() if metric == "min" else df[field].idxmax()
        out["overall"]["transaction"] = _row(next(df.loc[[idx]].itertuples()))
    if GROUPS[group_by]:
        g = df.groupby(GROUPS[group_by](df))[field]
        table = pd.DataFrame({"value": g.agg(METRICS[metric]).round(2), "count": g.size()})
        ordered = (table.sort_index() if group_by in ("month", "year")
                   else table.sort_values("value", ascending=(metric == "min")))
        top_n = max(1, int(top_n))
        out["groups_total"] = len(ordered)
        out["groups"] = [{"group": k, "value": r["value"], "count": int(r["count"])} for k, r in ordered.head(top_n).iterrows()]
    return out


def _series_stats(s: pd.Series, partial_last: bool) -> dict:
    """Stats for a month-indexed series (zeros already filled in). Direction ignores a partial last month."""
    complete = s.iloc[:-1] if partial_last and len(s) > 1 else s
    out = {"total": round(float(s.sum()), 2), "months_counted": len(s), "mean_per_month": round(float(s.mean()), 2),
           "highest_month": {"month": s.idxmax(), "value": round(float(s.max()), 2)},
           "lowest_month": {"month": s.idxmin(), "value": round(float(s.min()), 2)},
           "first_month_value": round(float(complete.iloc[0]), 2), "last_month_value": round(float(complete.iloc[-1]), 2)}
    out["change_abs"] = round(out["last_month_value"] - out["first_month_value"], 2)
    out["change_pct"] = round(out["change_abs"] / out["first_month_value"] * 100, 1) if out["first_month_value"] else None
    if len(complete) >= 3:
        slope = float(np.polyfit(np.arange(len(complete)), complete.values, 1)[0])
        mean = float(complete.mean())
        out["slope_per_month"] = round(slope, 2)
        out["direction"] = "flat" if abs(slope) <= 0.02 * abs(mean) else ("up" if slope > 0 else "down")
    else:
        out["slope_per_month"], out["direction"] = None, "not enough months (need 3+)"
    out["direction_months_used"] = len(complete)
    return out


@tool("Month-by-month totals over time with stats: mean per month (months with no spending count as 0), highest and "
      "lowest month, first-to-last change, trend slope and direction (up/down/flat). Same filters as "
      "query_transactions. Use group_by=category for a per-category trend summary.",
      {**FILTERS, "field": FIELD_PROP,
       "group_by": {"type": "string", "enum": ["none", "category"]}})
def trend(ctx: Context, field: str = "my_share", group_by: str = "none", **filters) -> dict:
    if group_by not in ("none", "category"):
        raise ToolError("group_by must be 'none' or 'category'.")
    field = _field(field)
    df = _frame(filters)
    if df.empty:
        return {"filters": filters, "months": [], "note": "No transactions match these filters."}
    lo = _month(filters["start_date"][:7], "start_date") if "start_date" in filters else df["month"].min()
    hi = _month(filters["end_date"][:7], "end_date") if "end_date" in filters else df["month"].max()
    index = [str(p) for p in pd.period_range(lo, hi, freq="M")]
    partial = hi == date.today().strftime("%Y-%m")

    def series(d):
        return d.groupby("month")[field].sum().reindex(index, fill_value=0.0)

    total = series(df)
    out = {"field": field, "filters": filters, "first_month": index[0], "last_month": index[-1],
           "last_month_partial": partial,
           "months": [{"month": m, "value": round(float(v), 2), "count": int((df["month"] == m).sum())}
                      for m, v in total.items()],
           "stats": _series_stats(total, partial)}
    if group_by == "category":
        per = {c: _series_stats(series(d), partial) for c, d in df.groupby("category_label")}
        ranked = sorted(per.items(), key=lambda kv: -abs(kv[1]["total"]))
        out["categories"] = [{"category": c, **s} for c, s in ranked[:15]]
        out["categories_total"] = len(per)
    if partial:
        out["note"] = "The last month is still in progress, so direction and change exclude it."
    return out


def _months_arg(month, year) -> list[str]:
    if month and year:
        raise ToolError("Give either month or year, not both.")
    if year:
        return reports.months_of_year(int(year))
    return [_month(month or date.today().strftime("%Y-%m"))]


@tool("The dashboard's Summary by Category for a month or a whole year: Budget, Actual, Difference per category, the "
      "total, and Table 1 (401k Saved, Liquid Saved, Net Leftover). Difference = Budget - Actual (positive = under "
      "budget). Defaults to the current month.",
      {"month": {"type": "string", "description": "yyyy-mm"}, "year": {"type": "integer"}})
def budget_vs_actual(ctx: Context, month: str | None = None, year: int | None = None) -> dict:
    months = _months_arg(month, year)
    summary = reports.summary_by_category(months)
    body = summary.drop(index=reports.TOTAL)
    body = body[(body["Budget"] != 0) | (body["Actual"] != 0)]
    t1 = reports.table1(months, summary)
    return {"months": months,
            "categories": [{"category": c, "budget": r["Budget"], "actual": r["Actual"], "difference": r["Difference"],
                            "over_budget": bool(r["Difference"] < 0)} for c, r in body.iterrows()],
            "total": {"budget": summary.loc[reports.TOTAL, "Budget"], "actual": summary.loc[reports.TOTAL, "Actual"],
                      "difference": summary.loc[reports.TOTAL, "Difference"]},
            "table1": {row: t1.loc[row].to_dict() for row in t1.index}}


@tool("The budget in force for a month (a month with nothing saved uses the most recent earlier month).",
      {"month": {"type": "string", "description": "yyyy-mm (default current month)"}})
def get_budget(ctx: Context, month: str | None = None) -> dict:
    month = _month(month or date.today().strftime("%Y-%m"))
    saved = db.fetch_budgets(month)
    eff = saved if not saved.empty else db.effective_budget(month)
    if eff.empty:
        return {"month": month, "categories": [], "note": "No budget saved for this or any earlier month."}
    return {"month": month, "saved_for_this_month": not saved.empty, "inherited_from_month": eff["month"].iloc[0],
            "categories": [{"category": r.category, "amount": r.amount} for r in eff.itertuples()],
            "total": round(float(eff["amount"].sum()), 2)}


@tool("The two paychecks for a month (carried forward from the latest earlier month if none saved) with gross, "
      "taxes, deductions, 401k, employer match and take-home per paycheck and for the month.",
      {"month": {"type": "string", "description": "yyyy-mm (default current month)"}})
def get_paychecks(ctx: Context, month: str | None = None) -> dict:
    month = _month(month or date.today().strftime("%Y-%m"))
    saved = db.fetch_paychecks(month)
    lines = saved if not saved.empty else db.effective_paychecks(month)
    if lines.empty:
        return {"month": month, "paychecks": [], "note": "No paychecks saved for this or any earlier month."}
    periods = {}
    for period, p in lines.groupby("period", sort=False):
        periods[period] = {"lines": [{"category": r.category, "subcategory": r.subcategory, "amount": r.amount}
                                     for r in p.itertuples()], "totals": db.paycheck_totals(p)}
    return {"month": month, "saved_for_this_month": not saved.empty, "inherited_from_month": lines["month"].iloc[0],
            "paychecks": periods, "month_take_home": round(sum(p["totals"]["take_home"] for p in periods.values()), 2)}


@tool("Shared expenses with the roommate for a month: shared total, what the roommate owes, settled and still owed "
      "per category, plus still owed from earlier months and across all months.",
      {"month": {"type": "string", "description": "yyyy-mm (default current month)"}})
def shared_summary(ctx: Context, month: str | None = None) -> dict:
    month = _month(month or date.today().strftime("%Y-%m"))
    t = reports.roommate_by_category(month)
    body = t.drop(index=reports.TOTAL)
    return {"month": month, "roommate_share": round(1 - SHARED_SPLIT, 2),
            "categories": [{"category": c, "shared_total": r["Shared total"], "roommate_owes": r["Roommate owes"],
                            "settled": r["Settled"], "still_owed": r["Still owed"]} for c, r in body.iterrows()],
            "month_total": {"shared_total": t.loc[reports.TOTAL, "Shared total"],
                            "roommate_owes": t.loc[reports.TOTAL, "Roommate owes"],
                            "settled": t.loc[reports.TOTAL, "Settled"], "still_owed": t.loc[reports.TOTAL, "Still owed"]},
            "still_owed_from_earlier_months": reports.still_owed(before=month),
            "still_owed_all_months": reports.still_owed()}


@tool("Transactions waiting in the review queue, each with the best guess already filled in (category, where it came "
      "from and its confidence). Newest first.",
      {"limit": {"type": "integer", "description": f"Rows to return (default 25, max {AGENT_MAX_ROWS})."}})
def list_review_queue(ctx: Context, limit=25) -> dict:
    qd = db.fetch_review_queue()
    limit = max(1, min(int(limit), AGENT_MAX_ROWS))
    return {"total_in_queue": len(qd), "returned_rows": min(len(qd), limit),
            "rows": [{"id": int(r.id), "date": r.date, "bank": r.bank or None, "description": r.description,
                      "amount": round(float(r.amount), 2),
                      "suggested_category": r.category or None, "suggestion_source": r.cat_source,
                      "suggestion_confidence": None if pd.isna(r.cat_confidence) else round(float(r.cat_confidence), 2)}
                     for r in qd.head(limit).itertuples()]}


# ---------------------------------------------------------------- write tools (staged, never applied here)
CHANGE_PROPS = {
    "date": {"type": "string", "description": "yyyy-mm-dd"},
    "bank": {"type": ["string", "null"]},
    "description": {"type": "string"},
    "amount": {"type": "number"},
    "kind": {"type": "string", "enum": ["expense", "income"]},
    "category": {"type": ["string", "null"], "description": "An active category name; null clears it."},
    "shared": {"type": "boolean"},
    "shared_settled": {"type": "boolean"},
}
WHERE_PROP = {"type": "object", "properties": {k: v for k, v in FILTERS.items() if k != "ids"},
              "description": "Select rows by filters instead of ids (at least one filter required)."}


def _clean_changes(changes: dict) -> dict:
    if not isinstance(changes, dict) or not changes:
        raise ToolError("'changes' must be an object with at least one field to change.")
    bad = set(changes) - set(db.EDITABLE_FIELDS)
    if bad:
        raise ToolError(f"Can't change {sorted(bad)}. Editable fields: {list(db.EDITABLE_FIELDS)}")
    out = {}
    for k, v in changes.items():
        if k == "date":
            out[k] = _iso(v, "date")
        elif k == "amount":
            out[k] = round(_num(v, "amount"), 2)
        elif k == "description":
            if not str(v or "").strip():
                raise ToolError("description can't be empty.")
            out[k] = str(v).strip()
        elif k == "kind":
            out[k] = _pick(v, ["expense", "income"], "kind")
        elif k == "category":
            out[k] = None if v in (None, "") or str(v).lower() == UNCATEGORIZED.lower() \
                else _pick(v, db.get_categories(kind=None), "category")
        elif k == "bank":
            out[k] = None if v in (None, "") else _pick(v, db.get_banks(), "bank")
        else:
            out[k] = _bool(v, k)
    return out


def _targets(ids, where) -> pd.DataFrame:
    if bool(ids) == bool(where):
        raise ToolError("Give exactly one of 'ids' or 'where'.")
    if where and not (set(where) - {"kind"}):
        raise ToolError("'where' needs at least one real filter (not just kind), so it can't match every row.")
    df = _frame({"ids": ids} if ids else where, default_kind=None)
    if ids:
        missing = sorted(set(int(i) for i in ids) - set(df["id"]))
        if missing:
            raise ToolError(f"No transaction with id {missing[:10]}. Look the ids up with query_transactions.")
    if df.empty:
        raise ToolError("No transactions match, so there is nothing to change.")
    if len(df) > MAX_BULK_ROWS:
        raise ToolError(f"That would touch {len(df)} rows; the limit is {MAX_BULK_ROWS} per action. Narrow the filters.")
    return df


def _label(v) -> str:
    return "none" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


@tool("Propose changing fields on transactions (pick rows by ids OR by where-filters). Covers re-categorizing, "
      "fixing dates/amounts/descriptions/banks, marking shared or settled. The change is only PROPOSED; the user "
      "approves it in the UI.",
      {"changes": {"type": "object", "properties": CHANGE_PROPS}, "ids": FILTERS["ids"], "where": WHERE_PROP},
      required=("changes",), write=True)
def update_transactions(ctx: Context, changes: dict, ids: list | None = None, where: dict | None = None) -> dict:
    changes = _clean_changes(changes)
    df = _targets(ids, where)
    col = {"shared_settled": "shared_settled"}
    preview = []
    for r in df.head(8).itertuples():
        row = {"id": int(r.id), "description": r.description[:40]}
        for k, v in changes.items():
            old = getattr(r, col.get(k, k))
            old = bool(old) if k in ("shared", "shared_settled") else old
            row[k] = f"{_label(old)} -> {_label(v)}"
        preview.append(row)
    desc = ", ".join(f"{k} = {_label(v)}" for k, v in changes.items())
    return ctx.stage("update_transactions", {"ids": [int(i) for i in df["id"]], "changes": changes},
                     f"Set {desc} on {len(df)} transaction(s)", preview, len(df))


@tool("Propose permanently deleting transactions (by ids OR where-filters; filters must not be empty). Only PROPOSED; "
      "the user approves it in the UI.",
      {"ids": FILTERS["ids"], "where": WHERE_PROP}, write=True)
def delete_transactions(ctx: Context, ids: list | None = None, where: dict | None = None) -> dict:
    df = _targets(ids, where)
    preview = [{"id": r["id"], "date": r["date"], "description": r["description"], "amount": r["amount"]}
               for r in _rows(df, 8)]
    return ctx.stage("delete_transactions", {"ids": [int(i) for i in df["id"]]},
                     f"DELETE {len(df)} transaction(s)", preview, len(df))


@tool("Propose confirming review-queue items. Each item is {id, category}; omit category to accept the pre-filled "
      "guess. Confirmed labels become training data. Only PROPOSED; the user approves it in the UI.",
      {"items": {"type": "array", "items": {"type": "object", "properties": {
          "id": {"type": "integer"}, "category": {"type": "string"}}, "required": ["id"]}}},
      required=("items",), write=True)
def confirm_review(ctx: Context, items: list) -> dict:
    if not items or not isinstance(items, list):
        raise ToolError("'items' must be a non-empty list of {id, category}.")
    if len(items) > MAX_BULK_ROWS:
        raise ToolError(f"At most {MAX_BULK_ROWS} items per action.")
    queue = db.fetch_review_queue().set_index("id")
    cats = db.get_categories()
    resolved, preview = [], []
    for it in items:
        tid = int(it["id"])
        if tid not in queue.index:
            raise ToolError(f"Transaction {tid} is not in the review queue. See list_review_queue.")
        guess = queue.at[tid, "category"]
        cat = it.get("category") or (guess if guess in cats else None)
        if not cat:
            raise ToolError(f"Transaction {tid} has no usable suggestion; give it a category.")
        cat = _pick(cat, cats, "category")
        resolved.append({"id": tid, "category": cat})
        preview.append({"id": tid, "description": str(queue.at[tid, "description"])[:40], "category": cat})
    return ctx.stage("confirm_review", {"items": resolved}, f"Confirm {len(resolved)} review-queue item(s)",
                     preview[:8], len(resolved))


@tool("Propose setting budget amounts for a month: {category: amount}. Other categories keep their current amounts. "
      "Only PROPOSED; the user approves it in the UI.",
      {"month": {"type": "string", "description": "yyyy-mm"},
       "amounts": {"type": "object", "additionalProperties": {"type": "number"}}},
      required=("month", "amounts"), write=True)
def set_budget(ctx: Context, month: str, amounts: dict) -> dict:
    month = _month(month)
    if not isinstance(amounts, dict) or not amounts:
        raise ToolError("'amounts' must map category names to amounts.")
    cats = db.get_categories()
    clean = {}
    for c, a in amounts.items():
        a = round(_num(a, f"amount for {c}"), 2)
        if a < 0:
            raise ToolError(f"Budget for {c} can't be negative.")
        clean[_pick(c, cats, "category")] = a
    current = db.fetch_budgets(month)
    base = current if not current.empty else db.effective_budget(month)
    old = dict(zip(base["category"], base["amount"])) if not base.empty else {}
    preview = [{"category": c, "budget": f"{old.get(c, 0.0):.2f} -> {a:.2f}"} for c, a in clean.items()]
    return ctx.stage("set_budget", {"month": month, "amounts": clean},
                     f"Set {len(clean)} budget amount(s) for {month}", preview, len(clean))


@tool("Propose copying one month's budget to another (existing rows in the target are left alone). Only PROPOSED.",
      {"from_month": {"type": "string"}, "to_month": {"type": "string"}}, required=("from_month", "to_month"), write=True)
def copy_budget_forward(ctx: Context, from_month: str, to_month: str) -> dict:
    a, b = _month(from_month, "from_month"), _month(to_month, "to_month")
    if db.fetch_budgets(a).empty:
        raise ToolError(f"No budget is saved for {a}.")
    return ctx.stage("copy_budget_forward", {"from_month": a, "to_month": b}, f"Copy the budget from {a} to {b}")


@tool("Propose changing one paycheck line amount (the line must already exist, see get_paychecks). Only PROPOSED.",
      {"month": {"type": "string"}, "period": {"type": "string", "enum": ["mid", "end"]},
       "category": {"type": "string"}, "subcategory": {"type": "string"}, "amount": {"type": "number"}},
      required=("month", "period", "category", "subcategory", "amount"), write=True)
def set_paycheck_line(ctx: Context, month: str, period: str, category: str, subcategory: str, amount) -> dict:
    month = _month(month)
    period = _pick(period, ["mid", "end"], "period")
    amount = round(_num(amount, "amount"), 2)
    if amount < 0:
        raise ToolError("amount can't be negative.")
    lines = db.effective_paychecks(month)
    p = lines[lines["period"] == period]
    hit = p[(p["category"].str.lower() == str(category).lower()) & (p["subcategory"].str.lower() == str(subcategory).lower())]
    if hit.empty:
        raise ToolError(f"No {period} paycheck line {category}/{subcategory} for {month}. Existing lines: "
                        f"{[f'{r.category}/{r.subcategory}' for r in p.itertuples()]}")
    r = hit.iloc[0]
    new = p.assign(amount=p["amount"].where(~((p["category"] == r["category"]) & (p["subcategory"] == r["subcategory"])), amount))
    before, after = db.paycheck_totals(p)["take_home"], db.paycheck_totals(new)["take_home"]
    return ctx.stage("set_paycheck_line", {"month": month, "period": period, "category": r["category"],
                                           "subcategory": r["subcategory"], "amount": amount},
                     f"Set {period} paycheck {r['category']}/{r['subcategory']} for {month} to {amount:.2f}",
                     [{"line": f"{r['category']}/{r['subcategory']}", "amount": f"{r['amount']:.2f} -> {amount:.2f}",
                       "take_home_of_this_paycheck": f"{before:.2f} -> {after:.2f}"}])


@tool("Propose copying one month's paychecks to another (existing lines in the target are left alone). Only PROPOSED.",
      {"from_month": {"type": "string"}, "to_month": {"type": "string"}}, required=("from_month", "to_month"), write=True)
def copy_paychecks_forward(ctx: Context, from_month: str, to_month: str) -> dict:
    a, b = _month(from_month, "from_month"), _month(to_month, "to_month")
    if db.fetch_paychecks(a).empty:
        raise ToolError(f"No paychecks are saved for {a}.")
    return ctx.stage("copy_paychecks_forward", {"from_month": a, "to_month": b}, f"Copy paychecks from {a} to {b}")


@tool("Propose a category change: add a new one, rename, retire (kept on old transactions) or flag as a savings "
      "category. Only PROPOSED.",
      {"action": {"type": "string", "enum": ["add", "rename", "retire", "set_savings"]},
       "name": {"type": "string"}, "new_name": {"type": "string", "description": "For rename."},
       "is_savings": {"type": "boolean", "description": "For add and set_savings."}},
      required=("action", "name"), write=True)
def manage_category(ctx: Context, action: str, name: str, new_name: str | None = None, is_savings=False) -> dict:
    allc, active = _all_categories(), db.get_categories(kind=None)
    name = str(name).strip()
    if action == "add":
        if name.lower() in (c.lower() for c in active):
            raise ToolError(f"Category '{name}' already exists.")
        args, summary = {"action": action, "name": name, "is_savings": _bool(is_savings, "is_savings")}, f"Add category '{name}'"
    elif action == "rename":
        old = _pick(name, allc, "category")
        if not new_name or not str(new_name).strip():
            raise ToolError("rename needs 'new_name'.")
        if str(new_name).strip().lower() in (c.lower() for c in allc):
            raise ToolError(f"A category named '{new_name}' already exists.")
        args, summary = {"action": action, "name": old, "new_name": str(new_name).strip()}, f"Rename category '{old}' to '{new_name}'"
    elif action == "retire":
        old = _pick(name, active, "category")
        args, summary = {"action": action, "name": old}, f"Retire category '{old}'"
    elif action == "set_savings":
        old = _pick(name, active, "category")
        args = {"action": action, "name": old, "is_savings": _bool(is_savings, "is_savings")}
        summary = f"Mark '{old}' as {'a savings' if args['is_savings'] else 'not a savings'} category"
    else:
        raise ToolError("action must be add, rename, retire or set_savings.")
    return ctx.stage("manage_category", args, summary)


@tool("Propose retraining the ML categorizer on the manual and imported labels. Only PROPOSED.", write=True)
def retrain_classifier(ctx: Context) -> dict:
    return ctx.stage("retrain_classifier", {}, "Retrain the ML categorizer")


# ---------------------------------------------------------------- applying approved actions
def _apply_set_budget(a: dict) -> str:
    month = a["month"]
    if db.fetch_budgets(month).empty:   # keep the other categories' inherited amounts instead of dropping them
        eff = db.effective_budget(month)
        if not eff.empty:
            db.copy_budget_forward(eff["month"].iloc[0], month)
    for c, amt in a["amounts"].items():
        db.set_budget(month, c, amt)
    return f"Saved {len(a['amounts'])} budget amount(s) for {month}."


def _apply_paycheck_line(a: dict) -> str:
    month = a["month"]
    lines = db.fetch_paychecks(month)
    if lines.empty:
        lines = db.effective_paychecks(month)
    hit = ((lines["period"] == a["period"]) & (lines["category"] == a["category"])
           & (lines["subcategory"] == a["subcategory"]))
    if not hit.any():
        raise ValueError("That paycheck line no longer exists.")
    lines = lines.assign(amount=lines["amount"].where(~hit, a["amount"]))
    db.save_paychecks(month, lines[["period", "category", "subcategory", "amount"]])
    return f"Saved paycheck line {a['category']}/{a['subcategory']} for {month}."


def _apply_category(a: dict) -> str:
    act, name = a["action"], a["name"]
    if act == "add":
        db.add_category(name, "expense", a["is_savings"])
        return f"Added category '{name}'."
    if act == "rename":
        db.rename_category(name, a["new_name"])
        return f"Renamed '{name}' to '{a['new_name']}' (transactions and budgets updated)."
    if act == "retire":
        db.deactivate_category(name)
        return f"Retired '{name}'."
    db.set_category_savings(name, a["is_savings"])
    return f"Updated the savings flag on '{name}'."


def _apply_retrain(a: dict) -> str:
    m = classifier.train()
    acc = f"{m['cv_accuracy']:.1%}" if m["cv_accuracy"] is not None else "n/a"
    return f"Trained on {m['rows']} rows, {m['categories']} categories. CV accuracy: {acc}."


def _apply_confirm(a: dict) -> str:
    for it in a["items"]:
        db.set_category(it["id"], it["category"])
    return f"Confirmed {len(a['items'])} review-queue item(s)."


EXECUTORS: dict[str, Callable[[dict], str]] = {
    "update_transactions": lambda a: f"Updated {db.update_transactions(a['ids'], a['changes'])} transaction(s).",
    "delete_transactions": lambda a: f"Deleted {db.delete_transactions(a['ids'])} transaction(s).",
    "confirm_review": _apply_confirm,
    "set_budget": _apply_set_budget,
    "copy_budget_forward": lambda a: f"Copied {db.copy_budget_forward(a['from_month'], a['to_month'])} budget row(s).",
    "set_paycheck_line": _apply_paycheck_line,
    "copy_paychecks_forward": lambda a: f"Copied {db.copy_paychecks_forward(a['from_month'], a['to_month'])} paycheck line(s).",
    "manage_category": _apply_category,
    "retrain_classifier": _apply_retrain,
}


def apply_action(action: dict) -> tuple[bool, str]:
    """Run an approved action against the database. Returns (ok, message) and records both on the action."""
    if action["status"] != "pending":
        return False, f"Action {action['id']} is already {action['status']}."
    try:
        message = EXECUTORS[action["tool"]](action["args"])
        action.update(status="applied", result=message)
        return True, message
    except Exception as e:
        action.update(status="failed", result=f"{type(e).__name__}: {e}")
        return False, action["result"]


def reject_action(action: dict) -> None:
    if action["status"] == "pending":
        action.update(status="rejected", result="Rejected by the user.")


def evidence_text(result: dict) -> str:
    """Tool output as the text the grounding check searches for numbers in."""
    return json.dumps(result, default=str)
