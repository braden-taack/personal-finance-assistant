"""SQLite data layer."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterable

import pandas as pd

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE,
    kind      TEXT NOT NULL DEFAULT 'expense' CHECK (kind IN ('expense','income')),
    active    INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS transactions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    date           TEXT NOT NULL,              -- ISO yyyy-mm-dd
    description    TEXT NOT NULL,
    amount         REAL NOT NULL,              -- positive = money spent / negative = refund; income uses kind
    kind           TEXT NOT NULL DEFAULT 'expense' CHECK (kind IN ('expense','income')),
    account        TEXT DEFAULT '',
    category       TEXT,                       -- NULL until categorized
    cat_source     TEXT,                       -- 'ml' | 'llm' | 'manual' | 'import'
    cat_confidence REAL,
    needs_review   INTEGER NOT NULL DEFAULT 0,
    import_batch   TEXT,
    created_at     TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_tx_cat ON transactions(category);

-- Budgets can change month to month: one row per (month, category).
CREATE TABLE IF NOT EXISTS budgets (
    month     TEXT NOT NULL,                   -- yyyy-mm
    category  TEXT NOT NULL,
    amount    REAL NOT NULL,
    PRIMARY KEY (month, category)
);

CREATE TABLE IF NOT EXISTS dedupe_decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    new_desc     TEXT, new_date TEXT, new_amount REAL,
    existing_id  INTEGER,
    decision     TEXT,                         -- 'skip' | 'import'
    decided_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


# ---------- categories ----------
def get_categories(kind: str | None = None, active_only: bool = True) -> list[str]:
    q = "SELECT name FROM categories WHERE 1=1"
    params: list = []
    if active_only:
        q += " AND active=1"
    if kind:
        q += " AND kind=?"
        params.append(kind)
    q += " ORDER BY name"
    with connect() as c:
        return [r["name"] for r in c.execute(q, params)]


def add_category(name: str, kind: str = "expense") -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO categories(name, kind, active) VALUES(?,?,1) "
            "ON CONFLICT(name) DO UPDATE SET active=1, kind=excluded.kind",
            (name.strip(), kind),
        )


def deactivate_category(name: str) -> None:
    with connect() as c:
        c.execute("UPDATE categories SET active=0 WHERE name=?", (name,))


def rename_category(old: str, new: str) -> None:
    with connect() as c:
        c.execute("UPDATE categories SET name=? WHERE name=?", (new, old))
        c.execute("UPDATE transactions SET category=? WHERE category=?", (new, old))
        c.execute("UPDATE budgets SET category=? WHERE category=?", (new, old))


# ---------- transactions ----------
def insert_transactions(rows: Iterable[dict], batch: str) -> int:
    n = 0
    with connect() as c:
        for r in rows:
            c.execute(
                """INSERT INTO transactions
                   (date, description, amount, kind, account, category, cat_source,
                    cat_confidence, needs_review, import_batch)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    r["date"], r["description"], float(r["amount"]), r.get("kind", "expense"),
                    r.get("account", ""), r.get("category"), r.get("cat_source"),
                    r.get("cat_confidence"), int(r.get("needs_review", 0)), batch,
                ),
            )
            n += 1
    return n


def fetch_transactions(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    q = "SELECT * FROM transactions WHERE 1=1"
    params: list = []
    if start:
        q += " AND date>=?"
        params.append(start)
    if end:
        q += " AND date<=?"
        params.append(end)
    q += " ORDER BY date DESC, id DESC"
    with connect() as c:
        df = pd.read_sql_query(q, c, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_labeled() -> pd.DataFrame:
    """Rows usable as training data: manual labels and imported labels, plus confirmed ML/LLM rows."""
    with connect() as c:
        return pd.read_sql_query(
            """SELECT description, category FROM transactions
               WHERE category IS NOT NULL AND needs_review=0
                 AND cat_source IN ('manual','import')""",
            c,
        )


def fetch_review_queue() -> pd.DataFrame:
    with connect() as c:
        return pd.read_sql_query(
            "SELECT * FROM transactions WHERE needs_review=1 ORDER BY date DESC", c
        )


def set_category(tx_id: int, category: str, source: str = "manual", confidence: float | None = 1.0) -> None:
    with connect() as c:
        c.execute(
            "UPDATE transactions SET category=?, cat_source=?, cat_confidence=?, needs_review=0 WHERE id=?",
            (category, source, confidence, tx_id),
        )


def delete_transaction(tx_id: int) -> None:
    with connect() as c:
        c.execute("DELETE FROM transactions WHERE id=?", (tx_id,))


# ---------- budgets ----------
def set_budget(month: str, category: str, amount: float) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO budgets(month, category, amount) VALUES(?,?,?) "
            "ON CONFLICT(month, category) DO UPDATE SET amount=excluded.amount",
            (month, category, amount),
        )


def copy_budget_forward(from_month: str, to_month: str) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT OR IGNORE INTO budgets(month, category, amount) "
            "SELECT ?, category, amount FROM budgets WHERE month=?",
            (to_month, from_month),
        )
        return cur.rowcount


def fetch_budgets(month: str | None = None) -> pd.DataFrame:
    q, params = "SELECT * FROM budgets", []
    if month:
        q += " WHERE month=?"
        params.append(month)
    q += " ORDER BY month DESC, category"
    with connect() as c:
        return pd.read_sql_query(q, c, params=params)


def effective_budget(month: str) -> pd.DataFrame:
    """Budget in force for `month`: uses that month's rows, else the most recent earlier month."""
    with connect() as c:
        row = c.execute(
            "SELECT MAX(month) AS m FROM budgets WHERE month<=?", (month,)
        ).fetchone()
    if not row or not row["m"]:
        return pd.DataFrame(columns=["month", "category", "amount"])
    return fetch_budgets(row["m"])
