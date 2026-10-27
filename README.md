# Finance Tracker

Local personal-finance app: SQLite + scikit-learn + LM Studio + Streamlit, laid out to mirror my Apple Numbers
budget sheet. Everything runs on your machine; the only network call is to your local LM Studio server.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Start LM Studio's local server (Developer tab -> Start Server, default `http://localhost:1234/v1`) and load a model.
Optional env vars: `FT_LLM_URL`, `FT_LLM_MODEL`, `FT_ML_THRESHOLD` (0.70), `FT_LLM_THRESHOLD` (0.60),
`FT_DUP_WINDOW` (3 days), `FT_DUP_SIM` (0.85), `FT_DB_PATH`, `FT_LLM_MAX_TOKENS` (1024: reasoning models such as
Qwen3 think before answering, and a low cap returns no answer, sending everything to manual review).

Data lives in `data/` (git-ignored): `finance.db` and `classifier.joblib`.

## Schema and upgrades
The schema version is stored in `PRAGMA user_version` and upgraded by `db.MIGRATIONS` on every start, so an
existing `finance.db` upgrades in place without losing rows. Before upgrading a database that already has tables,
a copy is saved as `data/finance.pre-v<N>-backup.db`.

- **v1**: the original schema.
- **v2**: `banks` table, `categories.is_savings`, `transactions.bank / shared / shared_settled / shared_settled_date`,
  `paychecks` table. It seeds the 22 budget categories, the 4 banks, and the current month's budget and paychecks.
  Old `account` values that name a bank (e.g. `amex`) become that bank. Old rows stored as `kind='income'` are kept
  but left out of budgets, since income now lives in `paychecks`.
- **v3**: `settings` table (LLM model choice, parallel request count).

## Pages
- **Dashboard**: Month or Year view. Shows Summary by Category (Budget / Actual / Difference, green under budget,
  red over), Table 1 (401k Saved / Liquid Saved / Net Leftover), both paychecks, the transactions, a pie chart,
  budget-vs-actual bars, trend lines and a year Sankey (gross pay -> taxes / deductions / 401k / take-home ->
  categories + leftover).
- **Budgets**: one row per (month, category). A month with nothing saved uses the most recent earlier month.
- **Paychecks**: Mid Month and End of Month paychecks side by side, with editable amounts and a computed take-home
  total. These carry forward the same way budgets do.
- **Shared expenses**: mark transactions shared with my roommate, track what they owe by category, settle up,
  copy a request message and download an itemized CSV.
- **Import CSV**, **Review queue**, **Transactions**, **Categories & Model** (categories, `is_savings`, bank settings,
  retraining).

## Banks and CSV import
Banks are FIDELITY, C1, AMEX and DISCOVER, stored in `banks` with a saved column mapping and sign convention each.
Only date, description and amount are required. Amount can be one signed column, or separate **Debit** and
**Credit** columns (Capital One), in which case amount = debit - credit.

1. Upload a file. The bank is pre-selected from a Bank column in the file, then the filename, then whichever bank's
   saved mapping matches the headers. Confirm or change it; the bank applies to every row. If the file has a Bank
   column, each row uses its own value, and blank or unknown values fall back to the confirmed bank.
2. The column mapping and the "charges are negative" setting are pre-filled from that bank's saved settings and can
   be saved back.
3. **Analyze file** then flags:
   - **Missing dates.** Rows are kept, not dropped, and you fill in the dates before importing (or choose to leave
     those rows out).
   - **Card payments and transfers** ("payment thank you", "autopay", "transfer from", ...), skipped by default
     because they double-count card purchases. Fidelity "Investing Transfer" / "Savings Transfer" are not flagged;
     they map straight to Investing / Savings.
   - **Duplicates**, against the database and within the file. A duplicate needs the same bank, the same amount,
     dates within 3 days and description similarity >= 85%. Legacy rows with no bank match any bank. You choose
     skip or import per pair.

**Numbers export**: in Numbers, select the Transactions table, then File -> Export To -> CSV
(Date, Bank, Description, Category, Amount). Categories that match a budget category (case-insensitive) are
stored as trusted `import` labels; anything else goes through the cascade.

## LLM model
The default is `nvidia/nemotron-3-nano-4b` (`FT_LLM_MODEL`). If LM Studio doesn't offer it, the app uses whichever
model is loaded there. Pick another model with **LLM model** in the sidebar (● = loaded); the choice is saved
in the `settings` table (schema v3). The import message names the model that was used.

## How categorization works
1. A CSV category that matches a budget category is kept as an `import` label.
2. **Savings rules**: Investing / Savings transfers are categorized by rule (`rule`).
3. **ML model** (TF-IDF word + char n-grams -> logistic regression). Confidence >= 70% is accepted.
4. **LM Studio LLM** picks from the category list and returns JSON with a confidence. >= 60% is accepted.
   All rows the ML model was unsure about are sent together, up to 4 requests at once (Categories & Model page,
   or `FT_LLM_PARALLEL`). LM Studio only runs them concurrently when its **Max Concurrent Predictions** setting
   (model load settings) is at least that high; otherwise they queue.
5. **Manual review queue**: anything left is flagged with the best guess pre-selected. Confirmed labels are saved
   as `manual` and feed the next retrain. The model trains only on `manual` / `import` labels, never on its own guesses.

## Shared expenses (roommate)
The split is always 50/50 (`SHARED_SPLIT = 0.5` in `config.py`, my share). There's no per-row split and no UI
setting. Marking is manual; a "shared before" hint appears for merchants marked shared in another month, but
nothing is marked automatically.

- Every budget figure (Actual, Net Leftover, Liquid Saved, charts) uses `db.effective_amount()`: a shared $100
  charge counts $50 toward its category.
- The roommate table (Category | Shared total | Roommate owes | Settled | Still owed + Total) has month and year
  views. "Total still owed by roommate across all months" carries unsettled balances forward.
- Settling (selected rows, a category or the whole month) stamps `shared_settled_date`. Settled rows stay in the
  history and only leave "Still owed".

## Paychecks and Table 1
Each paycheck has Income / Salary, then FED, State, Local and Other deductions. Take-home = gross - deductions
= 2335.21 per paycheck by default. There is also an `Employer / 401K Match` line, which is **not** part of
take-home.

- **Net Leftover** = both take-home paychecks - budget (or actual) over all categories, including Investing and
  Savings.
- **Liquid Saved** = the `is_savings` categories (Investing + Savings).
- **401k Saved** = employee 401K + employer match for both paychecks. The match defaults to the employee amount
  (968.98), an assumption that reproduces the sheet's 3875.92 = 4 x 968.98. Edit it on the Paychecks page if your
  match differs.

## Access from elsewhere
`.streamlit/config.toml` binds the app to `localhost`. Streamlit otherwise listens on every network interface.
For phone/remote access, run on a home machine and use Tailscale (or a similar private VPN) rather than exposing
the port publicly; pass `--server.address 0.0.0.0` only on a trusted network.

## Tests
`python smoke_test.py` runs everything on temp databases: migration from the old schema, seeds, bank-confirmed
import, same-bank duplicates, transfer flagging, Numbers export, paycheck take-home, Net Leftover, shared expenses,
settling, LLM model choice and parallel requests.

`python tests/e2e_playwright.py` drives the real app in a headless browser (Playwright) on a throwaway database.
It covers paychecks, both import flows (bank pre-selection, debit/credit, transfers, duplicates, fixing a blank
date), sharing and settling, dashboard totals, and the LLM selector. It generates its own sample CSVs dated in the
current month and starts and stops the app itself. Run `python -m playwright install chromium` once first.
LM Studio is optional: when it's unreachable, the LLM checks are skipped. Options: `--llm-url`, `--headed`
(watch it run), `--shots DIR` (save screenshots).
