"""SQLite data layer.

Schema changes go through MIGRATIONS, tracked with PRAGMA user_version, so an existing
finance.db upgrades in place. Before upgrading a database that already holds tables, a
backup copy is written next to it.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

import config
from config import DB_PATH, SHARED_SPLIT

# Version 1: the original schema (databases created before migrations existed report user_version 0).
SCHEMA_V1 = """
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
    amount         REAL NOT NULL,              -- positive = money spent / negative = refund
    kind           TEXT NOT NULL DEFAULT 'expense' CHECK (kind IN ('expense','income')),
    account        TEXT DEFAULT '',            -- legacy, superseded by bank
    category       TEXT,                       -- NULL until categorized
    cat_source     TEXT,                       -- 'ml' | 'llm' | 'manual' | 'import' | 'rule'
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


def _columns(c: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in c.execute(f"PRAGMA table_info({table})")}


def _add_column(c: sqlite3.Connection, table: str, col: str, ddl: str) -> None:
    if col not in _columns(c, table):
        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


def _migrate_v1(c: sqlite3.Connection) -> None:
    for stmt in SCHEMA_V1.split(";"):
        if stmt.strip():
            c.execute(stmt)


def _migrate_v2(c: sqlite3.Connection) -> None:
    """Banks, budget categories with is_savings, shared expenses, paychecks."""
    _add_column(c, "categories", "is_savings", "INTEGER NOT NULL DEFAULT 0")
    _add_column(c, "categories", "sort_order", "INTEGER NOT NULL DEFAULT 1000")
    _add_column(c, "transactions", "bank", "TEXT")
    _add_column(c, "transactions", "shared", "INTEGER NOT NULL DEFAULT 0")
    _add_column(c, "transactions", "shared_settled", "INTEGER NOT NULL DEFAULT 0")
    _add_column(c, "transactions", "shared_settled_date", "TEXT")
    _add_column(c, "dedupe_decisions", "new_bank", "TEXT")
    c.execute("CREATE INDEX IF NOT EXISTS idx_tx_bank ON transactions(bank)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_tx_shared ON transactions(shared)")
    c.execute("""CREATE TABLE IF NOT EXISTS banks (
        name             TEXT PRIMARY KEY,
        sort_order       INTEGER NOT NULL DEFAULT 1000,
        mapping          TEXT NOT NULL DEFAULT '{}',   -- JSON: field -> CSV column
        charges_negative INTEGER NOT NULL DEFAULT 0
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS paychecks (
        month       TEXT NOT NULL,                     -- yyyy-mm
        period      TEXT NOT NULL CHECK (period IN ('mid','end')),
        category    TEXT NOT NULL,
        subcategory TEXT NOT NULL,
        amount      REAL NOT NULL,
        sort_order  INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (month, period, category, subcategory)
    )""")

    for i, b in enumerate(config.DEFAULT_BANKS):
        c.execute("INSERT OR IGNORE INTO banks(name, sort_order, mapping, charges_negative) VALUES(?,?,?,?)",
                  (b["name"], i, json.dumps(b["mapping"]), b["charges_negative"]))
    # Old free-text account values that name one of the banks become the bank.
    names = [b["name"] for b in config.DEFAULT_BANKS]
    c.execute(f"UPDATE transactions SET bank=UPPER(TRIM(account)) WHERE bank IS NULL "
              f"AND UPPER(TRIM(account)) IN ({','.join('?' * len(names))})", names)

    for i, name in enumerate(config.DEFAULT_CATEGORIES):
        c.execute(
            "INSERT INTO categories(name, kind, active, is_savings, sort_order) VALUES(?, 'expense', 1, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET active=1, kind='expense', is_savings=excluded.is_savings, "
            "sort_order=excluded.sort_order",
            (name, int(name in config.SAVINGS_CATEGORIES), i),
        )

    month = date.today().strftime("%Y-%m")
    if not c.execute("SELECT 1 FROM budgets WHERE month=?", (month,)).fetchone():
        for name in config.DEFAULT_CATEGORIES:
            c.execute("INSERT INTO budgets(month, category, amount) VALUES(?,?,?)",
                      (month, name, float(config.DEFAULT_BUDGET.get(name, 0.0))))
    if not c.execute("SELECT 1 FROM paychecks").fetchone():
        for period in ('mid', 'end'):   # the commission paycheck is added by _migrate_v6
            for i, (cat, sub, amt) in enumerate(config.DEFAULT_PAYCHECK):
                c.execute("INSERT INTO paychecks VALUES(?,?,?,?,?,?)", (month, period, cat, sub, amt, i))


def _migrate_v3(c: sqlite3.Connection) -> None:
    """App settings chosen in the UI (e.g. the LLM model)."""
    c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")


def _migrate_v4(c: sqlite3.Connection) -> None:
    """Partly shared transactions: shared_amount is the portion split with the roommate (NULL = the whole amount)."""
    _add_column(c, "transactions", "shared_amount", "REAL")


def _migrate_v5(c: sqlite3.Connection) -> None:
    """Commission: an Income/Commission line (0) right after Salary on every saved paycheck."""
    sub = config.COMMISSION_SUBCATEGORY
    for month, period, order in c.execute(
            "SELECT month, period, sort_order FROM paychecks WHERE category=? AND subcategory='Salary' "
            "AND NOT EXISTS (SELECT 1 FROM paychecks p2 WHERE p2.month=paychecks.month AND p2.period=paychecks.period "
            "AND p2.category=? AND p2.subcategory=?)", (config.INCOME_CATEGORY,) * 2 + (sub,)).fetchall():
        c.execute("UPDATE paychecks SET sort_order=sort_order+1 WHERE month=? AND period=? AND sort_order>?",
                  (month, period, order))
        c.execute("INSERT INTO paychecks VALUES(?,?,?,?,0,?)", (month, period, config.INCOME_CATEGORY, sub, order + 1))


def _migrate_v6(c: sqlite3.Connection) -> None:
    """Third paycheck period 'commission' (widen the CHECK) with every line at 0 for each saved month."""
    c.execute("ALTER TABLE paychecks RENAME TO paychecks_old")
    c.execute("""CREATE TABLE paychecks (
        month       TEXT NOT NULL,                     -- yyyy-mm
        period      TEXT NOT NULL CHECK (period IN ('mid','end','commission')),
        category    TEXT NOT NULL,
        subcategory TEXT NOT NULL,
        amount      REAL NOT NULL,
        sort_order  INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (month, period, category, subcategory)
    )""")
    c.execute("INSERT INTO paychecks SELECT * FROM paychecks_old")
    c.execute("INSERT INTO paychecks SELECT month, 'commission', category, subcategory, 0, sort_order "
              "FROM paychecks_old WHERE period='mid'")
    c.execute("DROP TABLE paychecks_old")


def _migrate_v7(c: sqlite3.Connection) -> None:
    """Split State Taxes into SITW (the old Withholding amount), NYPFL and NY Disability (0 for saved paychecks)."""
    for month, period, order in c.execute(
            "SELECT month, period, sort_order FROM paychecks WHERE category='State Taxes' "
            "AND subcategory='Withholding'").fetchall():
        c.execute("UPDATE paychecks SET sort_order=sort_order+2 WHERE month=? AND period=? AND sort_order>?",
                  (month, period, order))
        c.execute("UPDATE paychecks SET subcategory='SITW' WHERE month=? AND period=? AND category='State Taxes' "
                  "AND subcategory='Withholding'", (month, period))
        for i, sub in enumerate(("NYPFL", "NY Disability"), 1):
            c.execute("INSERT OR IGNORE INTO paychecks VALUES(?,?,?,?,0,?)", (month, period, "State Taxes", sub, order + i))


MIGRATIONS = [_migrate_v1, _migrate_v2, _migrate_v3, _migrate_v4, _migrate_v5, _migrate_v6, _migrate_v7]   # index + 1 == version reached after running it
SCHEMA_VERSION = len(MIGRATIONS)


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


def schema_version() -> int:
    with connect() as c:
        return c.execute("PRAGMA user_version").fetchone()[0]


def init_db() -> None:
    """Create or upgrade the database. Each migration runs in its own transaction."""
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        has_tables = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone()
        if version == 0 and has_tables:
            version = 1   # pre-migration database: already on the original schema
        if version < SCHEMA_VERSION and has_tables:
            backup_path = DB_PATH.with_name(f"{DB_PATH.stem}.pre-v{SCHEMA_VERSION}-backup.db")
            if not backup_path.exists():
                with sqlite3.connect(backup_path) as dst:
                    conn.backup(dst)
        for v in range(version, SCHEMA_VERSION):
            conn.execute("BEGIN")
            try:
                MIGRATIONS[v](conn)
                conn.execute(f"PRAGMA user_version = {v + 1}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
    finally:
        conn.close()


# ---------- categories ----------
def get_categories(kind: str | None = "expense", active_only: bool = True) -> list[str]:
    q = "SELECT name FROM categories WHERE 1=1"
    params: list = []
    if active_only:
        q += " AND active=1"
    if kind:
        q += " AND kind=?"
        params.append(kind)
    q += " ORDER BY sort_order, name"
    with connect() as c:
        return [r["name"] for r in c.execute(q, params)]


def category_table() -> pd.DataFrame:
    with connect() as c:
        return pd.read_sql_query(
            "SELECT name, kind, active, is_savings FROM categories ORDER BY sort_order, name", c)


def savings_categories() -> list[str]:
    with connect() as c:
        return [r["name"] for r in c.execute("SELECT name FROM categories WHERE is_savings=1 ORDER BY sort_order")]


def add_category(name: str, kind: str = "expense", is_savings: bool = False) -> None:
    with connect() as c:
        nxt = c.execute("SELECT COALESCE(MAX(sort_order), 0) + 1 FROM categories WHERE sort_order < 1000").fetchone()[0]
        c.execute(
            "INSERT INTO categories(name, kind, active, is_savings, sort_order) VALUES(?,?,1,?,?) "
            "ON CONFLICT(name) DO UPDATE SET active=1, kind=excluded.kind",
            (name.strip(), kind, int(is_savings), nxt),
        )


def set_category_savings(name: str, is_savings: bool) -> None:
    with connect() as c:
        c.execute("UPDATE categories SET is_savings=? WHERE name=?", (int(is_savings), name))


def deactivate_category(name: str) -> None:
    with connect() as c:
        c.execute("UPDATE categories SET active=0 WHERE name=?", (name,))


def rename_category(old: str, new: str) -> None:
    with connect() as c:
        c.execute("UPDATE categories SET name=? WHERE name=?", (new, old))
        c.execute("UPDATE transactions SET category=? WHERE category=?", (new, old))
        c.execute("UPDATE budgets SET category=? WHERE category=?", (new, old))


# ---------- settings ----------
def get_setting(key: str, default: str | None = None) -> str | None:
    with connect() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def set_setting(key: str, value: str) -> None:
    with connect() as c:
        c.execute("INSERT INTO settings(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, str(value)))


# ---------- banks ----------
def get_banks() -> list[str]:
    with connect() as c:
        return [r["name"] for r in c.execute("SELECT name FROM banks ORDER BY sort_order, name")]


def get_bank(name: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM banks WHERE name=?", (name,)).fetchone()
    if not r:
        return None
    return {"name": r["name"], "mapping": json.loads(r["mapping"] or "{}"),
            "charges_negative": bool(r["charges_negative"])}


def save_bank(name: str, mapping: dict, charges_negative: bool) -> None:
    clean = {k: v for k, v in mapping.items() if v}
    with connect() as c:
        c.execute(
            "INSERT INTO banks(name, mapping, charges_negative) VALUES(?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET mapping=excluded.mapping, charges_negative=excluded.charges_negative",
            (name.strip().upper(), json.dumps(clean), int(charges_negative)),
        )


# ---------- transactions ----------
def insert_transactions(rows: Iterable[dict], batch: str) -> int:
    n = 0
    with connect() as c:
        for r in rows:
            c.execute(
                """INSERT INTO transactions
                   (date, bank, description, amount, kind, account, category, cat_source,
                    cat_confidence, needs_review, import_batch, shared, shared_settled, shared_settled_date,
                    shared_amount)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r["date"], r.get("bank"), r["description"], float(r["amount"]), r.get("kind", "expense"),
                    r.get("account", ""), r.get("category"), r.get("cat_source"),
                    r.get("cat_confidence"), int(r.get("needs_review", 0)), batch,
                    int(r.get("shared", 0)), int(r.get("shared_settled", 0)), r.get("shared_settled_date"),
                    r.get("shared_amount"),
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


def effective_amount(df: pd.DataFrame) -> pd.Series:
    """The part of each transaction that is mine: SHARED_SPLIT of shared rows, all of the rest.

    Every budget actual, leftover, savings figure and chart goes through this function.
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    amount = df["amount"].astype(float)
    return amount - shared_portion(df) * (1 - SHARED_SPLIT)


def shared_portion(df: pd.DataFrame) -> pd.Series:
    """The part of each row that is split with the roommate: 0 if not shared, else shared_amount (default: all)."""
    amount = df["amount"].astype(float)
    if "shared" not in df:
        return amount * 0.0
    shared = df["shared"].fillna(0).astype(int) == 1
    part = df["shared_amount"].astype(float) if "shared_amount" in df else pd.Series(np.nan, index=df.index)
    return pd.Series(np.where(shared, part.fillna(amount), 0.0), index=df.index)


def roommate_amount(df: pd.DataFrame) -> pd.Series:
    """What the roommate owes per row: the non-mine part of shared rows, 0 otherwise."""
    return shared_portion(df) * (1 - SHARED_SPLIT)


def month_bounds(month: str) -> tuple[str, str]:
    return f"{month}-01", f"{month}-31"


def fetch_expenses(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Expense transactions with an `effective` column (share-adjusted amount).

    Legacy rows imported as kind='income' before paychecks existed are excluded.
    """
    tx = fetch_transactions(start, end)
    if tx.empty:
        tx["effective"] = pd.Series(dtype=float)
        return tx
    tx = tx[tx["kind"] == "expense"].copy()
    tx["effective"] = effective_amount(tx)
    tx["month"] = tx["date"].dt.strftime("%Y-%m")
    return tx


def transaction_months() -> list[str]:
    with connect() as c:
        return [r[0] for r in c.execute(
            "SELECT DISTINCT substr(date, 1, 7) FROM transactions WHERE kind='expense' ORDER BY 1 DESC")]


def fetch_labeled() -> pd.DataFrame:
    """Rows usable as training data: manual and imported labels only (never the model's own guesses)."""
    with connect() as c:
        return pd.read_sql_query(
            """SELECT description, category FROM transactions
               WHERE category IS NOT NULL AND needs_review=0 AND kind='expense'
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


def set_date(tx_id: int, iso_date: str) -> None:
    with connect() as c:
        c.execute("UPDATE transactions SET date=? WHERE id=?", (iso_date, tx_id))


def delete_transaction(tx_id: int) -> None:
    with connect() as c:
        c.execute("DELETE FROM transactions WHERE id=?", (tx_id,))


EDITABLE_FIELDS = ("date", "bank", "description", "amount", "kind", "category", "shared", "shared_amount",
                   "shared_settled")


def update_transactions(ids: Iterable[int], changes: dict) -> int:
    """Apply `changes` (any of EDITABLE_FIELDS) to every row in `ids`; returns rows touched.

    A category change is a manual label (so it trains the model and leaves the review queue).
    Un-sharing clears the settlement, and only shared rows can be settled.
    """
    unknown = set(changes) - set(EDITABLE_FIELDS)
    if unknown:
        raise ValueError(f"Not editable: {sorted(unknown)}")
    ids = [int(i) for i in ids]
    plain = {k: v for k, v in changes.items() if k not in ("category", "shared", "shared_settled")}
    if "amount" in plain:
        plain["amount"] = float(plain["amount"])
    if plain.get("shared_amount") is not None:
        plain["shared_amount"] = float(plain["shared_amount"])
    sets, vals = [f"{k}=?" for k in plain], list(plain.values())
    if "category" in changes:
        cat = changes["category"] or None
        sets += ["category=?", "cat_source=?", "cat_confidence=?", "needs_review=0"]
        vals += [cat, "manual" if cat else None, 1.0 if cat else None]
    with connect() as c:
        for i in ids:
            if sets:
                c.execute(f"UPDATE transactions SET {', '.join(sets)} WHERE id=?", [*vals, i])
            if "shared" in changes:
                if changes["shared"]:
                    c.execute("UPDATE transactions SET shared=1 WHERE id=?", (i,))
                else:
                    c.execute("UPDATE transactions SET shared=0, shared_amount=NULL, shared_settled=0, shared_settled_date=NULL WHERE id=?", (i,))
            if "shared_settled" in changes:
                if changes["shared_settled"]:
                    c.execute("UPDATE transactions SET shared_settled=1, shared_settled_date=COALESCE(shared_settled_date, ?) "
                              "WHERE id=? AND shared=1", (date.today().isoformat(), i))
                else:
                    c.execute("UPDATE transactions SET shared_settled=0, shared_settled_date=NULL WHERE id=?", (i,))
    return len(ids)


def delete_transactions(ids: Iterable[int]) -> int:
    ids = [(int(i),) for i in ids]
    with connect() as c:
        c.executemany("DELETE FROM transactions WHERE id=?", ids)
    return len(ids)


# ---------- shared expenses ----------
def set_shared(ids: Iterable[int], shared: bool) -> None:
    """Mark rows shared / not shared. Un-sharing a row also clears its settlement."""
    ids = [int(i) for i in ids]
    with connect() as c:
        if shared:
            c.executemany("UPDATE transactions SET shared=1 WHERE id=?", [(i,) for i in ids])
        else:
            c.executemany("UPDATE transactions SET shared=0, shared_amount=NULL, shared_settled=0, shared_settled_date=NULL WHERE id=?",
                          [(i,) for i in ids])


def set_settled(ids: Iterable[int], settled: bool = True, on: str | None = None) -> int:
    """Mark shared rows settled (stamped with `on`, default today) or unsettled. Non-shared rows are ignored."""
    on = on or date.today().isoformat()
    n = 0
    with connect() as c:
        for i in ids:
            cur = c.execute(
                "UPDATE transactions SET shared_settled=?, shared_settled_date=? WHERE id=? AND shared=1",
                (int(settled), on if settled else None, int(i)),
            )
            n += cur.rowcount
    return n


# ---------- budgets ----------
def set_budget(month: str, category: str, amount: float) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO budgets(month, category, amount) VALUES(?,?,?) "
            "ON CONFLICT(month, category) DO UPDATE SET amount=excluded.amount",
            (month, category, amount),
        )


def replace_budget(month: str, amounts: dict[str, float]) -> None:
    """Make `amounts` the whole budget for `month` (categories not listed are removed)."""
    with connect() as c:
        c.execute("DELETE FROM budgets WHERE month=?", (month,))
        c.executemany("INSERT INTO budgets(month, category, amount) VALUES(?,?,?)",
                      [(month, cat, float(amt)) for cat, amt in amounts.items()])


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


# ---------- paychecks ----------
def fetch_paychecks(month: str) -> pd.DataFrame:
    with connect() as c:
        return pd.read_sql_query(
            "SELECT * FROM paychecks WHERE month=? ORDER BY period, sort_order", c, params=[month])


def effective_paychecks(month: str) -> pd.DataFrame:
    """Paychecks in force for `month`: that month's rows, else carried forward from the latest earlier month."""
    with connect() as c:
        row = c.execute("SELECT MAX(month) AS m FROM paychecks WHERE month<=?", (month,)).fetchone()
    if not row or not row["m"]:
        return pd.DataFrame(columns=["month", "period", "category", "subcategory", "amount", "sort_order"])
    lines = fetch_paychecks(row["m"])
    if row["m"] != month:   # the occasional commission check is not repeated month after month
        lines.loc[lines["period"] == "commission", "amount"] = 0.0
    return lines


def save_paychecks(month: str, lines: pd.DataFrame) -> None:
    """Replace `month`'s paycheck lines with `lines` (columns: period, category, subcategory, amount)."""
    with connect() as c:
        c.execute("DELETE FROM paychecks WHERE month=?", (month,))
        for period, grp in lines.groupby("period", sort=False):
            for i, r in enumerate(grp.itertuples(index=False)):
                c.execute("INSERT INTO paychecks VALUES(?,?,?,?,?,?)",
                          (month, period, r.category, r.subcategory, float(r.amount), i))


def copy_paychecks_forward(from_month: str, to_month: str) -> int:
    """Copy the regular (mid/end) paychecks; the commission paycheck is never copied, it starts at 0."""
    with connect() as c:
        n = c.execute(
            "INSERT OR IGNORE INTO paychecks SELECT ?, period, category, subcategory, amount, sort_order "
            "FROM paychecks WHERE month=? AND period<>'commission'", (to_month, from_month)).rowcount
        c.execute(
            "INSERT OR IGNORE INTO paychecks SELECT ?, 'commission', category, subcategory, 0, sort_order "
            "FROM paychecks WHERE month=? AND period='mid'", (to_month, from_month))
        return n


def paycheck_totals(lines: pd.DataFrame) -> dict:
    """Break one paycheck's lines into gross / taxes / other deductions / 401k / employer match / take-home."""
    cat, sub, amt = lines["category"], lines["subcategory"], lines["amount"].astype(float)
    employer = cat == config.EMPLOYER_CATEGORY
    income = cat == config.INCOME_CATEGORY
    deductions = ~employer & ~income
    retirement = deductions & (sub == config.RETIREMENT_SUBCATEGORY)
    taxes = deductions & cat.isin(config.TAX_CATEGORIES)
    gross = float(amt[income].sum())
    commission = float(amt[income & (sub == config.COMMISSION_SUBCATEGORY)].sum())
    total_ded = float(amt[deductions].sum())
    return {
        "gross": gross,
        "commission": commission,
        "taxes": float(amt[taxes].sum()),
        "other": float(amt[deductions & ~taxes & ~retirement].sum()),
        "retirement": float(amt[retirement].sum()),
        "employer_match": float(amt[employer & (sub == config.EMPLOYER_MATCH_SUBCATEGORY)].sum()),
        "deductions": total_ded,
        "take_home": round(gross - total_ded, 2),
    }
