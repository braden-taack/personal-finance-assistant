# Finance Tracker

Local personal-finance app: SQLite + scikit-learn + LM Studio + Streamlit.
Everything runs on your machine; the only network call is to your local LM Studio server.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Start LM Studio's local server (Developer tab -> Start Server, default `http://localhost:1234/v1`) and load a model.
Optional env vars: `FT_LLM_URL`, `FT_LLM_MODEL`, `FT_ML_THRESHOLD` (0.70), `FT_LLM_THRESHOLD` (0.60),
`FT_DUP_WINDOW` (3 days), `FT_DUP_SIM` (0.85), `FT_DB_PATH`.

## Migrating from Apple Numbers
In Numbers: File -> Export To -> CSV. Import it on the **Import CSV** page. If the export has a category column,
those rows are stored as trusted labels and become the model's first training data.
Set up categories first (Categories & Model page), then import, then **Retrain model** (needs 20+ labeled rows).

## How categorization works
1. **ML model** (TF-IDF word + char n-grams -> logistic regression). Confidence >= 70% -> accepted.
2. **LM Studio LLM** is asked to choose from your category list and return JSON with a confidence. >= 60% -> accepted.
3. **Manual review queue**: anything left is flagged with the best guess pre-selected. Your confirmations are saved
   as `manual` labels and feed the next retrain. The model only trains on `manual`/`import` labels, never its own guesses.

Adding/renaming/retiring categories is on the Categories & Model page; relabel, then click **Retrain model**.

## Duplicate check
On import, each row is compared to the database and to earlier rows in the same file: same amount, date within 3 days,
description similarity >= 85%. You choose skip/import per pair (or a default for all). Decisions are logged in `dedupe_decisions`.
Note: two legitimate identical purchases (e.g. two coffees) will be flagged; choose "Import anyway".

## Budgets
One row per (month, category). A month without saved budgets uses the most recent earlier month. "Copy forward" clones a month.

## Sign convention
Expenses are positive; negative amounts become income/refunds-in. Tick "my bank exports spending as negative" when needed.

## Access from elsewhere
Keep it local for privacy. For phone/remote access, run on a home machine and use Tailscale (or similar private VPN)
rather than exposing the port publicly; add `--server.address 0.0.0.0` only on a trusted network.

## Test
`python smoke_test.py` runs the pipeline on a temp database.
