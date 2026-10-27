"""Extract transactions, monthly budgets and paychecks from yearly Apple Numbers budget files.

Each file is one year ("2024 Personal Budget.numbers") with one sheet per month. Per month sheet it reads:
  Transactions           -> transactions.csv  (Date, Bank, Description, Category, Amount + provenance columns)
  Summary by Category    -> budgets.csv       (Month, Category, Budget)
  *Paycheck tables       -> paychecks.csv     (Month, Period, Category, Subcategory, Amount)

The month comes from the sheet name and the year from the file name; they win over a transaction's own
date. A row dated 2019-08-01 on the "March" sheet of the 2024 file becomes 2024-03-01 (the day is kept and
clamped to the month's length), and the original value is kept in "Original Date". Rows without a usable
date keep a blank Date so the Import page asks for one.

    python tools/numbers_import.py numbers_files/                 # write CSVs to data/numbers_export/
    python tools/numbers_import.py numbers_files/ --load-db       # also load budgets + paychecks into the app DB
    python tools/numbers_import.py numbers_files/ --undated-day 1 # put undated rows on the 1st of their month

Transactions are not written to the database: import transactions.csv on the Import CSV page so they get
the bank confirmation, duplicate and transfer checks and the ML -> LLM -> review cascade.
"""
from __future__ import annotations

import argparse
import calendar
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from numbers_parser import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTH_ABBR = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
SUMMARY_ROW = re.compile(r"^(total|sum)\b", re.I)          # "Total", "TOTAL", "Total Shared Amt", "sum"
PAYCHECK_PERIODS = {"mid month paycheck": "mid", "end of month paycheck": "end"}
# Unambiguous spellings / bank-export names -> budget category. Anything else unknown is left as-is,
# and the Import page sends it through the ML -> LLM -> review cascade.
CATEGORY_ALIASES = {
    "bar": "Bars",
    "dining": "Restaurants",
    "other travel": "Travel",
    "grocerires": "Groceries",
    "merchandise & supplies-groceries": "Groceries",
}
# Venmo and BILT activity is tracked under the Fidelity account.
BANK_ALIASES = {"venmo": "FIDELITY", "michelle venmo": "FIDELITY", "bilt": "FIDELITY"}


# ---------------------------------------------------------------- parsing helpers
def sheet_month(file_year: int, sheet_name: str) -> tuple[int, int] | None:
    """(year, month) for a month sheet: month from the sheet name, year from the file. None for other sheets."""
    first = sheet_name.strip().split()[0].lower() if sheet_name.strip() else ""
    return (file_year, MONTHS[first]) if first in MONTHS else None


def original_date(value) -> tuple[int | None, date | None]:
    """(day, full date if one was written) from a Date cell, or (None, None) when there's nothing usable."""
    if isinstance(value, datetime):
        return value.day, value.date()
    if not isinstance(value, str):
        return None, None
    s = value.strip()
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s):
        y, mo, d = map(int, m.groups())
    elif m := re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", s):
        mo, d, y = map(int, m.groups())
        y += 2000 if y < 100 else 0
    elif m := re.fullmatch(r"([A-Za-z]{3,9})\.? (\d{1,2})", s):              # "Apr 24": no year
        d = int(m.group(2))
        return (d, None) if m.group(1).lower()[:3] in MONTH_ABBR and 1 <= d <= 31 else (None, None)
    else:
        return None, None
    try:
        return d, date(y, mo, d)
    except ValueError:
        return (d, None) if 1 <= d <= 31 else (None, None)


def on_sheet_month(day: int, year: int, month: int) -> date:
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def table_dicts(table) -> list[dict]:
    """Non-empty body rows as {header: value}, header names stripped."""
    names = [str(table.cell(0, c).value or "").strip() for c in range(table.num_cols)]
    out = []
    for r in range(max(1, table.num_header_rows), table.num_rows):
        vals = [table.cell(r, c).value for c in range(table.num_cols)]
        if all(v is None or (isinstance(v, str) and not v.strip()) for v in vals):
            continue
        out.append(dict(zip(names, vals)))
    return out


def text(v) -> str:
    return "" if v is None else str(v).strip()


def number(v) -> float | None:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------- extraction
def extract(paths: list[Path], include_future: bool, undated_day: int | None = None):
    tx, budgets, pays = [], [], []
    stats: dict[str, Counter] = defaultdict(Counter)
    unknown_cats, odd_banks, budget_cats = Counter(), Counter(), []
    this_month = date.today().replace(day=1)

    for path in paths:
        m = re.search(r"(19|20)\d{2}", path.stem)
        if not m:
            print(f"skipping {path.name}: no year in the file name")
            continue
        year, st = int(m.group(0)), stats[path.name]
        doc = Document(str(path))
        for sheet in doc.sheets:
            ym = sheet_month(year, sheet.name)
            if not ym:
                st["sheets skipped (not a month)"] += 1
                continue
            month_key = f"{ym[0]}-{ym[1]:02d}"
            tables = {t.name.strip(): t for t in sheet.tables}
            st["month sheets"] += 1

            # ---- budget
            sheet_cats = {}
            if "Summary by Category" in tables:
                for row in table_dicts(tables["Summary by Category"]):
                    cat = text(row.get("Category"))
                    if not cat or SUMMARY_ROW.match(cat):
                        continue
                    sheet_cats[cat.lower()] = cat
                    budgets.append({"Month": month_key, "Category": cat, "Budget": number(row.get("Budget")) or 0.0})
                    if cat not in budget_cats:
                        budget_cats.append(cat)

            # ---- paychecks
            for name, table in tables.items():
                if "paycheck" not in name.lower():
                    continue
                period = PAYCHECK_PERIODS.get(name.lower(), re.sub(r"\W+", "_", name.lower().replace(" paycheck", "")))
                group = ""
                for i, row in enumerate(table_dicts(table)):
                    cat = text(row.get("Category")) or group
                    sub_key = next((k for k in row if k.replace(" ", "").lower() == "sub-category"), None)
                    sub, amt = text(row.get(sub_key)), number(row.get("Amount"))
                    group = cat
                    if SUMMARY_ROW.match(cat) or not sub or amt is None:
                        continue
                    pays.append({"Month": month_key, "Period": period, "Category": cat, "Subcategory": sub,
                                 "Amount": amt, "Order": i})

            # ---- employer 401K match: whatever the sheet's own "401k Saved" (Table 1, Budgeted row) counts
            # beyond my contributions. The sheets counted a 1:1 match through 2023 and none from 2024 on.
            sheet_401k = None
            for row in table_dicts(tables["Table 1"]) if "Table 1" in tables else []:
                if text(row.get("")).lower().startswith("budget"):
                    sheet_401k = number(row.get("401k Saved"))
            mine = {p["Period"]: 0.0 for p in pays if p["Month"] == month_key and p["Period"] in ("mid", "end")}
            for p in pays:
                if p["Month"] == month_key and p["Period"] in mine and p["Subcategory"].upper() == "401K":
                    mine[p["Period"]] += p["Amount"]
            total_mine = sum(mine.values())
            if sheet_401k is None:
                match_total = total_mine
                st["months without a 401k Saved figure (match assumed = contribution)"] += 1
            else:
                match_total = max(0.0, round(sheet_401k - total_mine, 2))
            for period, amount in mine.items():
                share = amount / total_mine if total_mine else 1 / len(mine)
                pays.append({"Month": month_key, "Period": period, "Category": "Employer", "Subcategory": "401K Match",
                             "Amount": round(match_total * share, 2), "Order": 999})

            # ---- transactions
            if "Transactions" not in tables:
                continue
            future = date(*ym, 1) > this_month
            for row in table_dicts(tables["Transactions"]):
                desc, cat_raw = text(row.get("Description")), text(row.get("Category"))
                amt = number(row.get("Amount"))
                # Summary rows are marked in Category (the sheet totals with SUMIF on Category); a description
                # like "Total Wine" is a real purchase, so only a bare "Total"/"Sum" description counts.
                if SUMMARY_ROW.match(cat_raw) or re.fullmatch(r"(total|sum):?", desc, flags=re.I):
                    st["total/sum rows skipped"] += 1
                    continue
                if not desc:
                    st["rows without a description skipped"] += 1
                    continue
                if amt is None:
                    st["rows without an amount skipped"] += 1
                    continue
                if future and not include_future:
                    st["future-month rows left out"] += 1
                    continue
                day, orig = original_date(row.get("Date"))
                if day is None and undated_day:
                    final = on_sheet_month(undated_day, *ym).isoformat()
                    st[f"rows without a usable date put on day {undated_day}"] += 1
                elif day is None:
                    final = ""
                    st["rows without a usable date (fix on import)"] += 1
                else:
                    final = on_sheet_month(day, *ym).isoformat()
                    if orig is None or (orig.year, orig.month) != ym:
                        st["dates moved to the sheet's month"] += 1
                cat = sheet_cats.get(cat_raw.lower()) or CATEGORY_ALIASES.get(cat_raw.lower()) or cat_raw
                if cat and cat.lower() not in sheet_cats and cat not in CATEGORY_ALIASES.values():
                    unknown_cats[cat] += 1
                bank_raw = text(row.get("Bank"))
                bank = BANK_ALIASES.get(bank_raw.lower(), bank_raw.upper()) if bank_raw.lower() != "none" else ""
                if bank and bank not in {"FIDELITY", "C1", "AMEX", "DISCOVER"}:
                    odd_banks[bank] += 1
                raw_date = row.get("Date")
                tx.append({
                    "Date": final, "Bank": bank, "Description": desc, "Category": cat, "Amount": amt,
                    "Original Date": raw_date.date().isoformat() if isinstance(raw_date, datetime) else text(raw_date),
                    "Source File": path.name, "Sheet": sheet.name.strip(), "Sheet Month": month_key,
                })
                st["transactions"] += 1
    return tx, budgets, pays, stats, unknown_cats, odd_banks, budget_cats


# ---------------------------------------------------------------- loading budgets + paychecks
def load_db(budgets: pd.DataFrame, pays: pd.DataFrame, budget_cats: list[str]) -> None:
    import config
    import db
    db.init_db()
    existing = set(db.get_categories(kind=None, active_only=False))
    added = [c for c in budget_cats if c not in existing]
    for c in added:
        db.add_category(c)
    for c in budget_cats:
        if c in config.SAVINGS_CATEGORIES:
            db.set_category_savings(c, True)
    for month, grp in budgets.groupby("Month"):
        db.replace_budget(month, dict(zip(grp["Category"], grp["Budget"])))
    loaded, skipped = 0, sorted(set(pays.loc[~pays["Period"].isin(config.PAYCHECK_PERIODS), "Period"]))
    for month, grp in pays[pays["Period"].isin(config.PAYCHECK_PERIODS)].groupby("Month"):
        lines = grp.sort_values(["Period", "Order"], key=lambda s: s.map({"mid": 0, "end": 1}) if s.name == "Period" else s)
        db.save_paychecks(month, lines.rename(columns={"Period": "period", "Category": "category",
                                                       "Subcategory": "subcategory", "Amount": "amount"}))
        loaded += 1
    print(f"\nLoaded into {db.DB_PATH}: budgets for {budgets['Month'].nunique()} months, paychecks for {loaded} months."
          + (f" Added categories: {', '.join(added)}." if added else "")
          + (f" Not loaded (the app has only mid/end paychecks): {', '.join(skipped)}." if skipped else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help=".numbers files or folders containing them")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "data" / "numbers_export")
    ap.add_argument("--include-future", action="store_true", help="also export transactions on months after this one")
    ap.add_argument("--undated-day", type=int, metavar="N",
                    help="put rows with no usable date on day N of their sheet's month (default: leave blank to fix on import)")
    ap.add_argument("--load-db", action="store_true", help="load budgets + paychecks into the app database")
    args = ap.parse_args()

    paths = sorted({p for i in args.inputs for p in ([i] if i.is_file() else i.rglob("*.numbers"))
                    if not p.name.startswith("._")})
    if not paths:
        sys.exit("No .numbers files found.")
    tx, budgets, pays, stats, unknown_cats, odd_banks, budget_cats = extract(paths, args.include_future, args.undated_day)

    args.out.mkdir(parents=True, exist_ok=True)
    tx_df = pd.DataFrame(tx).sort_values(["Sheet Month", "Date"], kind="stable")
    bud_df, pay_df = pd.DataFrame(budgets), pd.DataFrame(pays)
    tx_df.to_csv(args.out / "transactions.csv", index=False)
    bud_df.to_csv(args.out / "budgets.csv", index=False)
    pay_df.drop(columns="Order").to_csv(args.out / "paychecks.csv", index=False)

    for name, st in stats.items():
        print(f"\n{name}")
        for k, v in st.items():
            print(f"  {k}: {v}")
    print(f"\nTotals: {len(tx_df)} transactions ({tx_df['Sheet Month'].min()} to {tx_df['Sheet Month'].max()}), "
          f"budgets for {bud_df['Month'].nunique()} months, paychecks for {pay_df['Month'].nunique()} months "
          f"(months per period: { {k: int(v) for k, v in pay_df.groupby('Period')['Month'].nunique().items()} }).")
    if unknown_cats:
        print("Transaction categories that aren't budget categories (sent through the cascade on import):",
              dict(unknown_cats.most_common()))
    if odd_banks:
        print("Bank values other than FIDELITY/C1/AMEX/DISCOVER (add them as banks in the app, or they fall back "
              "to the bank you confirm on import):", dict(odd_banks.most_common()))
    print(f"Wrote {args.out}/transactions.csv, budgets.csv, paychecks.csv")
    if args.load_db:
        load_db(bud_df, pay_df, budget_cats)


if __name__ == "__main__":
    main()
