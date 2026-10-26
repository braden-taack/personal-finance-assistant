"""CSV import (with flexible column mapping) and duplicate detection."""
from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

import db
from config import DUP_DATE_WINDOW_DAYS, DUP_DESC_SIMILARITY


def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file, dtype=str, keep_default_na=False)


def guess_mapping(columns: list[str]) -> dict:
    """Best-guess mapping from CSV headers to our fields."""
    lower = {c.lower().strip(): c for c in columns}

    def pick(*names):
        for n in names:
            for k, orig in lower.items():
                if n in k:
                    return orig
        return None

    return {
        "date": pick("date", "posted", "transaction date"),
        "description": pick("description", "merchant", "payee", "memo", "name", "details"),
        "amount": pick("amount", "debit", "value"),
        "category": pick("category"),
        "account": pick("account", "card"),
    }


def normalize(df: pd.DataFrame, mapping: dict, invert_sign: bool = False) -> pd.DataFrame:
    """Return a clean frame: date (ISO), description, amount, kind, account, category(optional).

    Convention: expenses are positive amounts. Pass invert_sign=True when the bank exports
    spending as negative numbers (common for credit cards/checking).
    """
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[mapping["date"]], errors="coerce").dt.strftime("%Y-%m-%d")
    out["description"] = df[mapping["description"]].astype(str).str.strip()
    amt = (
        df[mapping["amount"]]
        .astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    out["amount"] = pd.to_numeric(amt, errors="coerce")
    if invert_sign:
        out["amount"] = -out["amount"]
    out["kind"] = "expense"
    out.loc[out["amount"] < 0, "kind"] = "income"   # after normalization negative = money in
    out["amount"] = out["amount"].abs()
    out["account"] = df[mapping["account"]].astype(str) if mapping.get("account") else ""
    if mapping.get("category"):
        out["category"] = df[mapping["category"]].replace("", None)
    else:
        out["category"] = None
    out = out.dropna(subset=["date", "amount"])
    out = out[out["description"] != ""]
    return out.reset_index(drop=True)


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_duplicates(new: pd.DataFrame) -> list[dict]:
    """Compare each new row to existing DB rows and to earlier rows in the same file.

    A pair is a *probable* duplicate when amounts match exactly, dates are within
    DUP_DATE_WINDOW_DAYS, and descriptions are similar (>= DUP_DESC_SIMILARITY).
    Exact same date + amount + description is flagged as 'exact'.
    Returns a list of {new_idx, match_source ('db'|'file'), match (dict), score, exact}.
    """
    existing = db.fetch_transactions()
    hits: list[dict] = []
    new = new.copy()
    new["_dt"] = pd.to_datetime(new["date"])

    for i, row in new.iterrows():
        candidates = []
        if not existing.empty:
            window = existing[
                (existing["amount"].sub(row["amount"]).abs() < 0.005)
                & ((existing["date"] - row["_dt"]).abs().dt.days <= DUP_DATE_WINDOW_DAYS)
            ]
            for _, e in window.iterrows():
                candidates.append(("db", e["id"], e["date"].strftime("%Y-%m-%d"), e["description"], e["amount"]))
        earlier = new.loc[: i - 1] if i > 0 else new.iloc[0:0]
        if not earlier.empty:
            w2 = earlier[
                (earlier["amount"].sub(row["amount"]).abs() < 0.005)
                & ((earlier["_dt"] - row["_dt"]).abs().dt.days <= DUP_DATE_WINDOW_DAYS)
            ]
            for j, e in w2.iterrows():
                candidates.append(("file", j, e["date"], e["description"], e["amount"]))

        best = None
        for src, ref, d, desc, amt in candidates:
            s = _sim(row["description"], desc)
            if s >= DUP_DESC_SIMILARITY and (best is None or s > best["score"]):
                best = {
                    "new_idx": i,
                    "match_source": src,
                    "match": {"ref": ref, "date": d, "description": desc, "amount": amt},
                    "score": round(s, 3),
                    "exact": s == 1.0 and d == row["date"],
                }
        if best:
            hits.append(best)
    return hits
