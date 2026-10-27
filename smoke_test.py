"""End-to-end check on throwaway DBs: migration -> seed -> import -> dedupe -> cascade -> shared -> paychecks -> review."""
import os
import sqlite3
import tempfile
from datetime import date
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
os.environ["FT_DB_PATH"] = str(tmp / "t.db")
os.environ["FT_MODEL_PATH"] = str(tmp / "m.joblib")
os.environ["FT_LLM_URL"] = "http://localhost:9/v1"  # unreachable on purpose

import classifier, config, db, importer, pipeline, reports  # noqa: E402
import pandas as pd  # noqa: E402

THIS_MONTH = date.today().strftime("%Y-%m")


def close(a, b):
    return abs(float(a) - float(b)) < 0.005


# ---------------------------------------------------------------- 1. migration from the old schema
old = tmp / "old.db"
with sqlite3.connect(old) as c:
    c.executescript(db.SCHEMA_V1)   # exactly what pre-migration versions created; user_version stays 0
    c.execute("INSERT INTO categories(name, kind) VALUES('Groceries','expense'), ('Dining','expense'), ('Pay','income')")
    c.executemany("INSERT INTO transactions(date, description, amount, kind, account, category, cat_source) "
                  "VALUES(?,?,?,?,?,?,?)",
                  [("2026-01-05", "KROGER 0932", 40.0, "expense", "amex", "Groceries", "import"),
                   ("2026-01-06", "CHIPOTLE 2231", 12.0, "expense", "Visa", "Dining", "manual"),
                   ("2026-01-15", "PAYROLL", 2000.0, "income", "", "Pay", "import")])
    c.execute("INSERT INTO budgets VALUES('2026-01','Groceries',300)")
assert sqlite3.connect(old).execute("PRAGMA user_version").fetchone()[0] == 0
db.DB_PATH = old
db.init_db()
db.init_db()  # second run must be a no-op
assert db.schema_version() == db.SCHEMA_VERSION == 3
tx_old = db.fetch_transactions().set_index("description")
assert len(tx_old) == 3, "migration lost rows"
assert tx_old.loc["KROGER 0932", "bank"] == "AMEX" and pd.isna(tx_old.loc["CHIPOTLE 2231", "bank"])
assert (tx_old["shared"] == 0).all() and (tx_old["shared_settled"] == 0).all()
assert tx_old.loc["CHIPOTLE 2231", "category"] == "Dining"  # legacy category kept
assert db.fetch_budgets("2026-01")["amount"].tolist() == [300.0]
assert set(config.DEFAULT_CATEGORIES) <= set(db.get_categories())
assert "PAYROLL" not in db.fetch_expenses()["description"].tolist()  # legacy income rows stay out of actuals
assert (tmp / f"old.pre-v{db.SCHEMA_VERSION}-backup.db").exists()
print("migration: OK")

# ---------------------------------------------------------------- 2. fresh database + seeds
db.DB_PATH = tmp / "t.db"
db.init_db()
assert db.get_categories() == config.DEFAULT_CATEGORIES
assert db.savings_categories() == ["Investing", "Savings"]
assert db.get_banks() == ["FIDELITY", "C1", "AMEX", "DISCOVER"]
assert close(db.effective_budget(THIS_MONTH)["amount"].sum(), 5299.13)

pay = db.effective_paychecks(THIS_MONTH)
for period in ("mid", "end"):
    assert close(db.paycheck_totals(pay[pay["period"] == period])["take_home"], 2335.21)
t1 = reports.table1([THIS_MONTH])
print(t1)
assert close(t1.loc["Budgeted", "Net Leftover"], -628.71)        # 4670.42 - 5299.13
assert close(t1.loc["Budgeted", "Liquid Saved"], 1000)
assert close(t1.loc["Budgeted", "401k Saved"], 3875.92)          # 4 x 968.98 (employer match assumption)
print("seeds + paychecks + Net Leftover (budgeted): OK")

# Carry-forward: a later month with nothing saved uses this month's budget and paychecks.
assert close(reports.table1(["2031-06"]).loc["Budgeted", "Net Leftover"], -628.71)

# ---------------------------------------------------------------- 3. training data
data = {
    "Groceries": ["WHOLE FOODS #123", "TRADER JOES 552", "KROGER 0932", "SAFEWAY STORE 12", "ALDI 889"],
    "Restaurants": ["CHIPOTLE 2231", "PANERA BREAD 44", "OLIVE GARDEN 18", "CHEESECAKE FACTORY", "NOODLES & CO"],
    "Gas": ["SHELL OIL 5521", "EXXON 9981", "BP GAS 1123", "SPEEDWAY 0042", "MARATHON PETRO"],
    "Phone": ["VERIZON WIRELESS", "TMOBILE AUTOPAY BILL", "ATT MOBILITY", "MINT MOBILE", "VISIBLE WIRELESS"],
}
rows = []
for i, (cat, names) in enumerate(data.items()):
    for j, n in enumerate(names * 2):
        rows.append(dict(date=f"2025-0{1 + (j % 9)}-1{i + 1}", bank="AMEX", description=n, amount=10 + j + i,
                         category=cat, cat_source="import", cat_confidence=1.0))
db.insert_transactions(rows, "seed")
print("trained:", classifier.train())

# ---------------------------------------------------------------- 4. bank-confirmed import (Capital One debit/credit)
c1_csv = pd.DataFrame({
    "Transaction Date": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"],
    "Posted Date": ["2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"],
    "Card No.": ["1234"] * 4,
    "Description": ["WHOLE FOODS #999", "CAPITAL ONE AUTOPAY PYMT", "ZZZ UNKNOWN VENDOR", "WHOLE FOODS REFUND"],
    "Category": ["Merchandise", "Payment/Credit", "Other", "Merchandise"],  # C1's own categories: not ours
    "Debit": ["54.10", "", "33.00", ""],
    "Credit": ["", "500.00", "", "10.00"],
}).astype(str)
guess = importer.guess_mapping(list(c1_csv.columns))
assert guess["debit"] == "Debit" and guess["credit"] == "Credit" and guess["amount"] is None
bank, why = importer.guess_bank("capital_one_2026-10.csv", c1_csv, guess)
assert bank == "C1", (bank, why)
saved = db.get_bank("C1")
assert importer.saved_mapping_fits(saved["mapping"], list(c1_csv.columns))
clean = importer.normalize(c1_csv, saved["mapping"], bank="C1", charges_negative=saved["charges_negative"])
assert (clean["bank"] == "C1").all()
assert clean["amount"].tolist() == [54.10, -500.0, 33.0, -10.0]   # debit - credit; refund negative
assert importer.flag_transfers(clean) == [1]                      # autopay flagged
assert clean["category"].isna().all()                             # C1 categories aren't budget categories

# Bank fallback order: filename beats nothing; unknown filename + no Bank column -> saved mapping headers.
assert importer.guess_bank("export.csv", c1_csv, guess)[0] == "C1"

# Fidelity: negative = spending; savings transfers are legitimate and categorized by rule.
fid = pd.DataFrame({"Date": ["10/05/2026", "10/06/2026", "10/07/2026"],
                    "Name": ["Investing Transfer", "Savings Transfer", "TRANSFER FROM BROKERAGE"],
                    "Amount": ["-1000.00", "-500.00", "250.00"]})
f_clean = importer.normalize(fid, db.get_bank("FIDELITY")["mapping"], bank="FIDELITY",
                             charges_negative=db.get_bank("FIDELITY")["charges_negative"])
assert f_clean["amount"].tolist() == [1000.0, 500.0, -250.0]
assert importer.flag_transfers(f_clean) == [2]
f_out = pipeline.categorize_frame(f_clean.drop(index=[2]), use_llm=False)
assert f_out["category"].tolist() == ["Investing", "Savings"] and (f_out["cat_source"] == "rule").all()
print("bank-confirmed import + transfer flagging: OK")

# ---------------------------------------------------------------- 5. same-bank duplicate detection
db.insert_transactions([dict(date="2026-10-01", bank="AMEX", description="WHOLE FOODS #999", amount=54.10,
                             category="Groceries", cat_source="manual")], "dupseed")
assert importer.find_duplicates(clean.iloc[[0]]) == []            # same amount/date but C1 vs AMEX
amex_row = clean.iloc[[0]].assign(bank="AMEX")
d = importer.find_duplicates(amex_row)
assert len(d) == 1 and d[0]["exact"] and d[0]["match_source"] == "db"
in_file = pd.concat([clean.iloc[[0]], clean.iloc[[0]], clean.iloc[[0]].assign(bank="DISCOVER")], ignore_index=True)
d = importer.find_duplicates(in_file)
assert [(x["new_idx"], x["match_source"]) for x in d] == [(1, "file")]  # DISCOVER copy is not a duplicate
print("same-bank duplicates: OK")

out = pipeline.categorize_frame(clean.drop(index=[1]), use_llm=True)
print(out[["description", "amount", "category", "cat_source", "cat_confidence", "needs_review"]])
assert out["needs_review"].sum() >= 1
db.insert_transactions(out.to_dict("records"), "b1")

# ---------------------------------------------------------------- 6. Numbers export (Date, Bank, Description, Category, Amount)
numbers = pd.DataFrame({
    "Date": ["Oct 8, 2026", "", "10/09/2026", "2026-10-10"],
    "Bank": ["AMEX", "DISCOVER", "CHASE", ""],
    "Description": ["TRADER JOES 552", "SHELL OIL 5521", "NETFLIX.COM", "GOLF GALAXY"],
    "Category": ["groceries", "Gas", "Coffee", "Golf"],
    "Amount": ["$62.40", "$41.00", "$15.49", "$1,200.00"],
    "Shared": ["1", "", "0", "yes"],
    "Settled": ["1", "1", "", ""],
})
nm = importer.guess_mapping(list(numbers.columns))
assert nm["bank"] == "Bank" and nm["amount"] == "Amount"
n_clean = importer.normalize(numbers, nm, bank="AMEX")
assert len(n_clean) == 4                                           # blank date kept, not dropped
assert n_clean["date_missing"].tolist() == [0, 1, 0, 0]
assert n_clean["bank"].tolist() == ["AMEX", "DISCOVER", "AMEX", "AMEX"]
assert n_clean["bank_invalid"].tolist() == [0, 0, 1, 1]           # CHASE / blank fall back to the confirmed bank
assert n_clean["category"].tolist()[:2] == ["Groceries", "Gas"] and n_clean["category"][2] is None
assert n_clean["amount"].tolist() == [62.40, 41.00, 15.49, 1200.00]
assert nm["shared"] == "Shared" and n_clean["shared"].tolist() == [1, 0, 0, 1]   # optional Shared column
assert n_clean["shared_settled"].tolist() == [1, 0, 0, 0]                      # settled only counts on shared rows
assert n_clean["shared_settled_date"][0] == date.today().isoformat() and n_clean["shared_settled_date"][1] is None
assert importer.normalize(numbers.drop(columns="Shared"), nm | {"shared": None}, bank="AMEX")["shared"].sum() == 0
print("Numbers export: OK")

# ---------------------------------------------------------------- 7. Net Leftover (actual) mirrors the sheet
M = "2030-01"
db.copy_budget_forward(THIS_MONTH, M)
db.copy_paychecks_forward(THIS_MONTH, M)
sheet_actuals = {"Rent": 1675, "Investing": 1000, "Savings": 500}  # 3175.00 in all
db.insert_transactions([dict(date=f"{M}-05", bank="FIDELITY", description=f"{c} payment", amount=a,
                             category=c, cat_source="manual") for c, a in sheet_actuals.items()], "sheet")
t1 = reports.table1([M])
assert close(t1.loc["Actual", "Net Leftover"], 1495.42)            # 4670.42 - 3175.00
assert close(t1.loc["Actual", "Liquid Saved"], 1500)
assert close(t1.loc["Budgeted", "Net Leftover"], -628.71)
s = reports.summary_by_category([M])
assert close(s.loc["Total", "Budget"], 5299.13) and close(s.loc["Total", "Actual"], 3175)
assert close(s.loc["Rent", "Difference"], 0) and close(s.loc["Investing", "Difference"], 0)
print("Net Leftover (actual): OK")

# ---------------------------------------------------------------- 8. shared expenses
M2 = "2030-02"
db.insert_transactions([dict(date=f"{M2}-03", bank="AMEX", description="COSTCO WHOLESALE", amount=100.0,
                             category="Groceries", cat_source="manual"),
                        dict(date=f"{M2}-04", bank="C1", description="KROGER 0932", amount=30.0,
                             category="Groceries", cat_source="manual")], "shared")
ids = db.fetch_transactions(f"{M2}-01", f"{M2}-31").set_index("description")["id"]
costco = int(ids["COSTCO WHOLESALE"])
assert close(reports.summary_by_category([M2]).loc["Groceries", "Actual"], 130)  # nothing auto-shared
db.set_shared([costco], True)
assert close(reports.summary_by_category([M2]).loc["Groceries", "Actual"], 80)   # $100 shared counts $50
rt = reports.roommate_by_category(M2)
print(rt)
assert close(rt.loc["Groceries", "Shared total"], 100) and close(rt.loc["Groceries", "Roommate owes"], 50)
assert close(rt.loc["Groceries", "Still owed"], 50) and close(rt.loc["Total", "Still owed"], 50)
assert close(reports.still_owed(), 50)
assert "Groceries: $50.00" in reports.request_message(M2)
assert db.set_settled([int(ids["KROGER 0932"])]) == 0               # non-shared rows can't be settled
assert db.set_settled([costco], on="2030-02-20") == 1
rt = reports.roommate_by_category(M2)
assert close(rt.loc["Groceries", "Still owed"], 0) and close(rt.loc["Groceries", "Settled"], 50)
assert close(rt.loc["Groceries", "Roommate owes"], 50)              # still in the monthly history
items = reports.itemized(M2)
assert len(items) == 1 and items.iloc[0]["settled"] and items.iloc[0]["settled_on"] == "2030-02-20"
assert close(reports.still_owed(), 0)
assert close(reports.summary_by_category([M2]).loc["Groceries", "Actual"], 80)   # settling doesn't change my share
yr = reports.roommate_by_month(2030)
assert close(yr.loc[M2, "Settled"], 50) and close(yr.loc[M2, "Running still owed"], 0)
print("shared expenses: OK")

sk = reports.sankey(2030)
assert sk and "Gross income" in sk["labels"] and "Take-home" in sk["labels"]

# ---------------------------------------------------------------- 8b. LLM model choice + parallel requests
import threading  # noqa: E402
import time  # noqa: E402

import llm  # noqa: E402

offered = [{"id": "qwen/qwen3.8-27b", "loaded": True}, {"id": "nvidia/nemotron-3-nano-4b", "loaded": False}]
assert config.LLM_MODEL == "nvidia/nemotron-3-nano-4b" and config.LLM_PARALLEL == 4
assert llm.resolve_model(None, offered) == "nvidia/nemotron-3-nano-4b"           # default when offered
assert llm.resolve_model(None, offered[:1]) == "qwen/qwen3.8-27b"               # else whatever is loaded
assert llm.resolve_model("gone/model", [{"id": "a", "loaded": False}, {"id": "b", "loaded": True}]) == "b"
assert llm.resolve_model(None, []) is None                                     # LM Studio offline
assert llm.resolve_model() is None                                             # (FT_LLM_URL is unreachable here)

active, peak, lock = 0, 0, threading.Lock()


def fake_categorize(desc, amount, cats, hint=None, model=None, examples=None, glossary=None):
    global active, peak
    with lock:
        active += 1
        peak = max(peak, active)
    time.sleep(0.15)
    with lock:
        active -= 1
    return ("Golf", 0.9) if "GOLF" in desc else ("", 0.0)


real_categorize, real_resolve = llm.categorize, llm.resolve_model
llm.categorize, llm.resolve_model = fake_categorize, lambda preferred=None, models=None: "fake/model"
try:
    items = [(f"ROW {i}{' GOLF' if i % 3 == 0 else ''}", 1.0, None) for i in range(10)]
    t = time.time()
    res = llm.categorize_many(items, ["Golf"], workers=4)
    assert peak == 4 and time.time() - t < 0.15 * 10 / 2                        # really concurrent, capped at 4
    assert [r[0] for r in res] == ["Golf" if i % 3 == 0 else "" for i in range(10)]  # input order kept

    db.set_setting("llm_parallel", 2)
    peak = 0
    frame = pd.DataFrame({"description": [f"QQQ MYSTERY GOLF {i}" for i in range(6)], "amount": 20.0,
                          "category": None, "date": "2026-10-20", "bank": "AMEX"})
    out = pipeline.categorize_frame(frame, use_llm=True)
    assert peak == 2 and out.attrs["llm_model"] == "fake/model"                 # setting overrides the default 4
    assert (out["cat_source"] == "llm").all() and (out["category"] == "Golf").all()
    db.set_setting("llm_model", "picked/in-ui")
    assert pipeline.llm_model() == "picked/in-ui"
finally:
    llm.categorize, llm.resolve_model = real_categorize, real_resolve
print("LLM model choice + parallel requests: OK")

# ---------------------------------------------------------------- 8c. LLM context from my own history
import retrieval  # noqa: E402

index = retrieval.ExampleIndex(db.fetch_labeled())
near = index.similar("WHOLE FOODS #555")
assert near and near[0][1] == "Groceries" and near[0][0].startswith("WHOLE FOODS")   # store numbers ignored
assert index.similar("") == [] and "Groceries" in index.glossary() and len(index.glossary()["Gas"]) <= 4
system, user = llm.build_prompt("WHOLE FOODS #555", 12.0, db.get_categories(), hint="Groceries",
                                examples=near, glossary=index.glossary())
assert "follow how they used them" in system and '-> Groceries' in user and "- Groceries: " in user
assert "Most similar" not in llm.build_prompt("X", 1.0, ["Gas"])[1]                  # no context -> plain prompt
seen = {}
llm.categorize, llm.resolve_model = (lambda d, a, c, hint=None, model=None, examples=None, glossary=None:
                                     seen.setdefault(d, (examples, glossary)) and ("Groceries", 0.9)), \
                                    (lambda preferred=None, models=None: "fake/model")
real_predict, classifier.predict = classifier.predict, lambda descs, model=None: [("", 0.0)] * len(descs)  # ML unsure
try:
    pipeline.categorize_frame(pd.DataFrame({"description": ["QQQ WHOLE FOODS 777"], "amount": [5.0], "category": [None],
                                            "date": ["2026-10-20"], "bank": ["AMEX"]}))
    ex, gl = seen["QQQ WHOLE FOODS 777"]
    assert ex and ex[0][1] == "Groceries" and "Groceries" in gl                     # pipeline passes the context
finally:
    llm.categorize, llm.resolve_model, classifier.predict = real_categorize, real_resolve, real_predict
print("LLM history context: OK")

# ---------------------------------------------------------------- 9. manual label -> retrain picks it up
rid = int(db.fetch_review_queue().iloc[0]["id"])
db.set_category(rid, "Superlatives")
print("review left:", len(db.fetch_review_queue()))
print("retrained:", classifier.train())
print("OK")
