"""CSV import (flexible column mapping, per-bank settings), transfer flagging and duplicate detection."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

import db
from config import (BANK_FILENAME_HINTS, DUP_DATE_WINDOW_DAYS, DUP_DESC_SIMILARITY, SAVINGS_TRANSFER_RULES,
                    TRANSFER_PATTERNS)

FIELDS = ["date", "description", "amount", "debit", "credit", "category", "bank", "shared"]


def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file, dtype=str, keep_default_na=False)


def guess_mapping(columns: list[str]) -> dict:
    """Best-guess mapping from CSV headers to our fields.

    `amount` is a single signed column; `debit` + `credit` is the two-column layout (e.g. Capital One).
    """
    lower = {c.lower().strip(): c for c in columns}

    def pick(*names):
        for n in names:
            for k, orig in lower.items():
                if n in k:
                    return orig
        return None

    m = {
        "date": pick("transaction date", "trans. date", "date", "posted"),
        "description": pick("description", "merchant", "payee", "memo", "name", "details"),
        "amount": pick("amount", "value"),
        "debit": pick("debit"),
        "credit": pick("credit"),
        "category": pick("category"),
        "bank": pick("bank"),
        "shared": pick("shared"),
    }
    if m["amount"] and m["debit"] and m["credit"]:
        m["debit"] = m["credit"] = None   # prefer the single signed column when both exist
    return m


def saved_mapping_fits(mapping: dict, columns: list[str]) -> bool:
    """True when every column a bank's saved mapping names exists in this file."""
    used = [v for v in mapping.values() if v]
    return bool(used) and all(v in columns for v in used)


def guess_bank(filename: str, raw: pd.DataFrame, mapping: dict) -> tuple[str | None, str]:
    """Pre-select a bank: Bank column -> filename -> the one bank whose saved mapping fits. Returns (bank, why)."""
    banks = db.get_banks()
    col = mapping.get("bank")
    if col and col in raw:
        vals = raw[col].astype(str).str.strip().str.upper()
        vals = vals[vals.isin(banks)]
        if not vals.empty:
            return vals.mode().iloc[0], f"most common value in the '{col}' column"
    name = (filename or "").lower()
    for bank in banks:
        for hint in BANK_FILENAME_HINTS.get(bank, [bank.lower()]):
            if re.search(rf"(^|[^a-z0-9]){re.escape(hint)}([^a-z0-9]|$)", name):
                return bank, "filename"
    fits = [b for b in banks if saved_mapping_fits((db.get_bank(b) or {}).get("mapping", {}), list(raw.columns))]
    if len(fits) == 1:
        return fits[0], "column headers match its saved mapping"
    return None, "could not tell — please choose"


def _parse_money(s: pd.Series) -> pd.Series:
    s = (
        s.astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    return pd.to_numeric(s, errors="coerce")


def normalize(df: pd.DataFrame, mapping: dict, bank: str, charges_negative: bool = False) -> pd.DataFrame:
    """Return a clean frame: date (ISO or None), bank, description, amount, category + flag columns.

    Convention: expenses are positive, refunds negative. With a single amount column, pass
    charges_negative=True when the bank exports spending as negative numbers. With debit/credit
    columns, amount = debit - credit.

    `bank` applies to every row unless the file has a mapped Bank column; per-row values are
    validated against the banks table and invalid/blank ones fall back to `bank` (bank_invalid=1).
    Rows with a blank or unreadable date are kept with date=None and date_missing=1.
    An optional Shared column (1/yes/true/x) marks rows split 50/50 with the roommate.
    Category values are matched case-insensitively to known categories; unknown ones become None
    (original text kept in category_raw) so the cascade categorizes them.
    """
    out = pd.DataFrame(index=df.index)
    raw_dates = df[mapping["date"]].astype(str).str.strip()
    parsed = pd.to_datetime(raw_dates.where(raw_dates != ""), errors="coerce", format="mixed")
    out["date"] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)
    out["date_missing"] = parsed.isna().astype(int)
    out["date_raw"] = raw_dates

    banks = db.get_banks()
    out["bank"] = bank
    out["bank_invalid"] = 0
    if mapping.get("bank") and mapping["bank"] in df:
        vals = df[mapping["bank"]].astype(str).str.strip().str.upper()
        ok = vals.isin(banks)
        out.loc[ok, "bank"] = vals[ok]
        out.loc[~ok, "bank_invalid"] = 1

    out["description"] = df[mapping["description"]].astype(str).str.strip()
    if mapping.get("amount"):
        amt = _parse_money(df[mapping["amount"]])
        if charges_negative:
            amt = -amt
    else:
        debit = _parse_money(df[mapping["debit"]])
        credit = _parse_money(df[mapping["credit"]])
        amt = debit.fillna(0) - credit.fillna(0)
        amt[debit.isna() & credit.isna()] = float("nan")
    out["amount"] = amt
    out["kind"] = "expense"
    out["shared"] = 0
    if mapping.get("shared") and mapping["shared"] in df:   # optional: rows split 50/50 with the roommate
        flag = df[mapping["shared"]].astype(str).str.strip().str.lower()
        out["shared"] = flag.isin(["1", "1.0", "true", "yes", "y", "x"]).astype(int)

    out["category"] = None
    out["category_raw"] = ""
    if mapping.get("category") and mapping["category"] in df:
        lookup = {c.lower(): c for c in db.get_categories()}
        raw_cat = df[mapping["category"]].astype(str).str.strip()
        out["category_raw"] = raw_cat
        out["category"] = raw_cat.str.lower().map(lookup).astype(object).where(lambda s: s.notna(), None)

    out = out[out["amount"].notna() & (out["description"] != "")]
    return out.reset_index(drop=True)


def savings_rule(description: str) -> str | None:
    """Category for legitimate savings moves (e.g. Fidelity 'Investing Transfer'), else None."""
    for pat, cat in SAVINGS_TRANSFER_RULES.items():
        if re.search(pat, description, flags=re.I):
            return cat
    return None


def flag_transfers(df: pd.DataFrame) -> list[int]:
    """Indices of rows that look like card payments / transfers between my own accounts."""
    hits = []
    for i, desc in df["description"].items():
        if savings_rule(desc):
            continue
        if any(re.search(p, desc, flags=re.I) for p in TRANSFER_PATTERNS):
            hits.append(i)
    return hits


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _same_bank(a, b) -> bool:
    """Banks must match. A blank bank (legacy rows from before banks existed) matches any bank."""
    if not a or not b or pd.isna(a) or pd.isna(b):
        return True
    return a == b


def find_duplicates(new: pd.DataFrame) -> list[dict]:
    """Compare each new row to existing DB rows and to earlier rows in the same file.

    A pair is a *probable* duplicate when the bank is the same, amounts match exactly, dates
    are within DUP_DATE_WINDOW_DAYS, and descriptions are similar (>= DUP_DESC_SIMILARITY).
    Exact same date + amount + description is flagged as 'exact'. Rows without a date are skipped.
    Returns a list of {new_idx, match_source ('db'|'file'), match (dict), score, exact}.
    """
    existing = db.fetch_transactions()
    hits: list[dict] = []
    new = new.copy()
    new["_dt"] = pd.to_datetime(new["date"])
    if "bank" not in new:
        new["bank"] = None

    for i, row in new.iterrows():
        if pd.isna(row["_dt"]):
            continue
        candidates = []
        if not existing.empty:
            window = existing[
                (existing["amount"].sub(row["amount"]).abs() < 0.005)
                & ((existing["date"] - row["_dt"]).abs().dt.days <= DUP_DATE_WINDOW_DAYS)
            ]
            for _, e in window.iterrows():
                if _same_bank(row["bank"], e.get("bank")):
                    candidates.append(("db", e["id"], e["date"].strftime("%Y-%m-%d"), e["description"],
                                       e["amount"], e.get("bank")))
        earlier = new.loc[: i - 1] if i > 0 else new.iloc[0:0]
        if not earlier.empty:
            w2 = earlier[
                (earlier["amount"].sub(row["amount"]).abs() < 0.005)
                & ((earlier["_dt"] - row["_dt"]).abs().dt.days <= DUP_DATE_WINDOW_DAYS)
            ]
            for j, e in w2.iterrows():
                if _same_bank(row["bank"], e["bank"]):
                    candidates.append(("file", j, e["date"], e["description"], e["amount"], e["bank"]))

        best = None
        for src, ref, d, desc, amt, bank in candidates:
            s = _sim(row["description"], desc)
            if s >= DUP_DESC_SIMILARITY and (best is None or s > best["score"]):
                best = {
                    "new_idx": i,
                    "match_source": src,
                    "match": {"ref": ref, "date": d, "description": desc, "amount": amt, "bank": bank},
                    "score": round(s, 3),
                    "exact": s == 1.0 and d == row["date"],
                }
        if best:
            hits.append(best)
    return hits
