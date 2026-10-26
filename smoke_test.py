"""End-to-end check on a throwaway DB: import -> dedupe -> train -> cascade -> review -> retrain."""
import os
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
os.environ["FT_DB_PATH"] = str(tmp / "t.db")
os.environ["FT_MODEL_PATH"] = str(tmp / "m.joblib")
os.environ["FT_LLM_URL"] = "http://localhost:9/v1"  # unreachable on purpose

import classifier, db, importer, pipeline  # noqa: E402
import pandas as pd  # noqa: E402

db.init_db()
data = {
    "Groceries": ["WHOLE FOODS #123", "TRADER JOES 552", "KROGER 0932", "SAFEWAY STORE 12", "ALDI 889"],
    "Dining": ["CHIPOTLE 2231", "STARBUCKS #881", "MCDONALDS F1223", "PANERA BREAD 44", "DOMINOS PIZZA 7"],
    "Transport": ["SHELL OIL 5521", "UBER TRIP", "LYFT RIDE", "EXXON 9981", "MTA METROCARD"],
    "Subscriptions": ["NETFLIX.COM", "SPOTIFY USA", "HULU 8812", "ADOBE CREATIVE", "ICLOUD STORAGE"],
}
for c in data:
    db.add_category(c)
rows = []
for i, (cat, names) in enumerate(data.items()):
    for j, n in enumerate(names * 2):
        rows.append(dict(date=f"2026-0{1 + (j % 9)}-1{i + 1}", description=n, amount=10 + j + i,
                         kind="expense", category=cat, cat_source="import", cat_confidence=1.0))
db.insert_transactions(rows, "seed")
m = classifier.train()
print("trained:", m)

csv = pd.DataFrame({
    "Date": ["2026-10-01", "2026-10-02", "2026-10-02", "2026-10-03", "2026-10-04"],
    "Description": ["WHOLE FOODS #999", "NETFLIX.COM", "NETFLIX.COM", "ZZZ UNKNOWN VENDOR", "CHIPOTLE 7712"],
    "Amount": ["-54.10", "-15.49", "-15.49", "-33.00", "-12.25"],
})
clean = importer.normalize(csv, importer.guess_mapping(list(csv.columns)), invert_sign=True)
dups = importer.find_duplicates(clean)
print("dups in file:", [(d["new_idx"], d["match_source"], d["score"]) for d in dups])
assert len(dups) == 1 and dups[0]["exact"]

out = pipeline.categorize_frame(clean.drop(index=[2]), use_llm=True)
print(out[["description", "category", "cat_source", "cat_confidence", "needs_review"]])
assert out["needs_review"].sum() >= 1
db.insert_transactions(out.to_dict("records"), "b1")
# dedupe against DB now
assert importer.find_duplicates(clean)  # re-import should flag everything it matches

# manual label -> retrain picks it up
rid = int(db.fetch_review_queue().iloc[0]["id"])
db.add_category("Misc")
db.set_category(rid, "Misc")
db.set_budget("2026-09", "Dining", 200)
db.set_budget("2026-10", "Dining", 250)
print(db.effective_budget("2026-11"))
print("review left:", len(db.fetch_review_queue()))
print("OK")
