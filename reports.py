"""Calculations behind the dashboard, paychecks and shared-expense pages (no Streamlit here, so it is testable).

All "actual" figures use db.effective_amount() via the `effective` column of db.fetch_expenses().
"""
from __future__ import annotations

from datetime import date

import pandas as pd

import classifier
import db
from config import PAYCHECK_PERIODS, SHARED_SPLIT

TOTAL = "Total"


def months_of_year(year: int) -> list[str]:
    """Months a year view aggregates: all twelve, or up to this month for the current year."""
    today = date.today()
    last = today.month if year == today.year else 12
    return [f"{year}-{m:02d}" for m in range(1, last + 1)]


def expenses_for(months: list[str]) -> pd.DataFrame:
    tx = db.fetch_expenses(db.month_bounds(min(months))[0], db.month_bounds(max(months))[1])
    if tx.empty:
        return tx
    tx = tx[tx["month"].isin(months)].copy()
    tx["category"] = tx["category"].fillna("Uncategorized")
    return tx


def budget_by_category(months: list[str]) -> pd.Series:
    parts = [db.effective_budget(m).set_index("category")["amount"] for m in months]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts).groupby(level=0).sum() if parts else pd.Series(dtype=float)


def paycheck_summary(months: list[str]) -> dict:
    """Sum db.paycheck_totals over both paychecks of every month in `months` (with carry-forward)."""
    keys = ["gross", "taxes", "other", "retirement", "employer_match", "deductions", "take_home"]
    out = {k: 0.0 for k in keys}
    for m in months:
        lines = db.effective_paychecks(m)
        for period in PAYCHECK_PERIODS:
            p = lines[lines["period"] == period]
            if p.empty:
                continue
            t = db.paycheck_totals(p)
            for k in keys:
                out[k] += t[k]
    return {k: round(v, 2) for k, v in out.items()}


def summary_by_category(months: list[str], tx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Category | Budget | Actual | Difference (= Budget - Actual), in sheet order, plus a Total row.

    Categories with spending but no budget line (legacy names, Uncategorized) are appended so the
    total covers ALL spending.
    """
    tx = expenses_for(months) if tx is None else tx
    budget = budget_by_category(months)
    actual = tx.groupby("category")["effective"].sum() if not tx.empty else pd.Series(dtype=float)
    order = db.get_categories()
    order += [c for c in budget.index if c not in order]
    order += [c for c in actual.index if c not in order]
    df = pd.DataFrame(index=pd.Index(order, name="Category"))
    df["Budget"] = budget.reindex(df.index).fillna(0.0)
    df["Actual"] = actual.reindex(df.index).fillna(0.0)
    df["Difference"] = df["Budget"] - df["Actual"]
    df.loc[TOTAL] = df.sum()
    return df.round(2)


def table1(months: list[str], summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows Budgeted / Actual; columns 401k Saved, Liquid Saved, Net Leftover.

    Net Leftover = take-home of both paychecks - budget/actual over ALL categories (incl. savings).
    Liquid Saved = the is_savings categories (Investing + Savings).
    401k Saved   = employee 401K + employer match, both paychecks (same figure in both rows).
    """
    summary = summary_by_category(months) if summary is None else summary
    pay = paycheck_summary(months)
    body = summary.drop(index=TOTAL)
    savings = [c for c in db.savings_categories() if c in body.index]
    k401 = pay["retirement"] + pay["employer_match"]
    df = pd.DataFrame(
        {
            "401k Saved": [k401, k401],
            "Liquid Saved": [body.loc[savings, "Budget"].sum(), body.loc[savings, "Actual"].sum()],
            "Net Leftover": [pay["take_home"] - body["Budget"].sum(), pay["take_home"] - body["Actual"].sum()],
        },
        index=["Budgeted", "Actual"],
    )
    return df.round(2)


# ---------- shared expenses ----------
def shared_rows(months: list[str] | None = None) -> pd.DataFrame:
    """Shared transactions (all months when `months` is None) with a `roommate` column."""
    tx = db.fetch_expenses() if months is None else expenses_for(months)
    if tx.empty:
        tx["roommate"] = pd.Series(dtype=float)
        return tx
    tx = tx[tx["shared"] == 1].copy()
    tx["category"] = tx["category"].fillna("Uncategorized")
    tx["roommate"] = db.roommate_amount(tx)
    return tx


def roommate_by_category(month: str) -> pd.DataFrame:
    """Category | Shared total | Roommate owes | Settled | Still owed, plus a Total row."""
    rows = shared_rows([month])
    cols = ["Shared total", "Roommate owes", "Settled", "Still owed"]
    if rows.empty:
        df = pd.DataFrame(columns=cols, index=pd.Index([], name="Category"), dtype=float)
    else:
        rows["settled_amt"] = rows["roommate"].where(rows["shared_settled"] == 1, 0.0)
        g = rows.groupby("category")
        df = pd.DataFrame({
            "Shared total": g["amount"].sum(),
            "Roommate owes": g["roommate"].sum(),
            "Settled": g["settled_amt"].sum(),
        })
        df["Still owed"] = df["Roommate owes"] - df["Settled"]
        order = [c for c in db.get_categories() if c in df.index] + [c for c in df.index if c not in db.get_categories()]
        df = df.loc[order]
        df.index.name = "Category"
    df.loc[TOTAL] = df.sum() if not df.empty else 0.0
    return df.astype(float).round(2)


def still_owed(before: str | None = None) -> float:
    """Unsettled roommate balance across all months (or only months before `before`, yyyy-mm)."""
    rows = shared_rows()
    if rows.empty:
        return 0.0
    rows = rows[rows["shared_settled"] == 0]
    if before:
        rows = rows[rows["month"] < before]
    return round(float(rows["roommate"].sum()), 2)


def roommate_by_month(year: int) -> pd.DataFrame:
    """Per-month totals for `year` plus a running balance that carries unsettled amounts forward."""
    months = months_of_year(year)
    carried = still_owed(before=months[0])
    out = []
    for m in months:
        t = roommate_by_category(m).loc[TOTAL]
        carried += t["Still owed"]
        out.append({"Month": m, **t.to_dict(), "Running still owed": round(carried, 2)})
    return pd.DataFrame(out).set_index("Month")


def roommate_year_breakdown(year: int) -> pd.DataFrame:
    """Long table: Month, Category and the four amounts, for every month of `year` with shared rows."""
    parts = []
    for m in months_of_year(year):
        t = roommate_by_category(m).drop(index=TOTAL)
        if not t.empty:
            parts.append(t.reset_index().assign(Month=m))
    if not parts:
        return pd.DataFrame(columns=["Month", "Category", "Shared total", "Roommate owes", "Settled", "Still owed"])
    return pd.concat(parts)[["Month", "Category", "Shared total", "Roommate owes", "Settled", "Still owed"]]


def itemized(month: str) -> pd.DataFrame:
    """The rows behind the roommate table: date, bank, description, category, full amount, roommate's half."""
    rows = shared_rows([month])
    if rows.empty:
        return pd.DataFrame(columns=["id", "date", "bank", "description", "category", "amount", "roommate",
                                     "settled", "settled_on"])
    out = rows.rename(columns={"shared_settled": "settled", "shared_settled_date": "settled_on"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["settled"] = out["settled"].astype(bool)
    return out[["id", "date", "bank", "description", "category", "amount", "roommate", "settled", "settled_on"]] \
        .sort_values(["category", "date"])


def request_message(month: str) -> str:
    """Plain-text Venmo/text request for what's still owed in `month`."""
    t = roommate_by_category(month)
    body = t.drop(index=TOTAL)
    body = body[body["Still owed"].abs() >= 0.005]
    label = pd.Timestamp(f"{month}-01").strftime("%B %Y")
    if body.empty:
        return f"Shared expenses for {label}: nothing outstanding."
    pct = round((1 - SHARED_SPLIT) * 100)
    lines = [f"Shared expenses for {label} (your {pct}% share):"]
    lines += [f"- {cat}: ${r['Still owed']:,.2f}" for cat, r in body.iterrows()]
    lines.append(f"Total: ${body['Still owed'].sum():,.2f}")
    earlier = still_owed(before=month)
    if earlier >= 0.005:
        lines.append(f"(Plus ${earlier:,.2f} still open from earlier months, ${earlier + body['Still owed'].sum():,.2f} in all.)")
    return "\n".join(lines)


def shared_merchant_hints() -> set[str]:
    """Cleaned merchant names previously marked shared (for a non-binding hint only)."""
    rows = shared_rows()
    return set() if rows.empty else {classifier.clean(d) for d in rows["description"]}


# ---------- Sankey ----------
def sankey(year: int) -> dict | None:
    """Gross -> taxes / other deductions / 401k / take-home; take-home -> categories (+ leftover).

    Returns {labels, source, target, value} or None when there is no paycheck or spending data.
    """
    months = months_of_year(year)
    pay = paycheck_summary(months)
    tx = expenses_for(months)
    spend = tx.groupby("category")["effective"].sum() if not tx.empty else pd.Series(dtype=float)
    spend = spend[spend > 0].sort_values(ascending=False)
    if pay["gross"] <= 0 and spend.empty:
        return None

    labels: list[str] = []
    src: list[int] = []
    tgt: list[int] = []
    val: list[float] = []

    def node(name: str) -> int:
        if name not in labels:
            labels.append(name)
        return labels.index(name)

    def link(a: str, b: str, v: float) -> None:
        if v > 0.005:
            src.append(node(a))
            tgt.append(node(b))
            val.append(round(v, 2))

    gross, take = "Gross income", "Take-home"
    link(gross, "Taxes", pay["taxes"])
    link(gross, "Other deductions", pay["other"])
    link(gross, "401k", pay["retirement"])
    link("Employer 401k match", "401k", pay["employer_match"])
    link(gross, take, pay["take_home"])
    shortfall = spend.sum() - pay["take_home"]
    if shortfall > 0.005:
        link("Shortfall (spent more than take-home)", take, shortfall)
    for cat, v in spend.items():
        link(take, cat, float(v))
    if shortfall < -0.005:
        link(take, "Leftover", -shortfall)
    return {"labels": labels, "source": src, "target": tgt, "value": val}
