# Finance Tracker

Local personal-finance app: SQLite + scikit-learn + LM Studio + Streamlit, laid out to mirror my Apple Numbers
budget sheet. Everything runs on your machine; the only network call is to your local LM Studio server.

## Setup
```bash
./run.sh                    # first run creates .venv and installs requirements, then opens the app
./run.sh install-launcher   # optional: adds "Finance Tracker" (with a Stop action) to the application menu
```
`./run.sh` starts the app in the background at http://localhost:8501 and opens your browser; if it's already running
it just opens the browser. It also starts LM Studio's server through `lms` if that isn't up. Other commands:
`./run.sh stop`, `restart`, `status`, `logs` (logs are in `data/streamlit.log`). `PORT=8600 ./run.sh` uses another port.

Manual setup instead:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Start LM Studio's local server (Developer tab -> Start Server, default `http://localhost:1234/v1`) and load a model.
Optional env vars: `FT_LLM_URL`, `FT_LLM_MODEL`, `FT_ML_THRESHOLD` (0.90), `FT_LLM_THRESHOLD` (1.01 = the LLM never auto-accepts),
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

## Assistant
The **Assistant** is a chat bubble fixed to the bottom-right of every page (click to expand, **—** to shrink it back). It is an agent that uses the same LM Studio model as the rest of the app, at temperature 0
(`agent.py`, `agent_tools.py`, `llm.chat`). It has no data of its own; it works through tools:

- **Read tools** compute every figure with pandas from the database: overview, transaction search, sum / avg / min /
  max / median / count (optionally grouped by category, bank, month, year, merchant, ...), month-by-month trends
  (mean per month, high/low, change, slope and direction), budget vs actual and Table 1, budgets, paychecks, shared
  expenses and the review queue.
- **Write tools** cover what the other pages do: edit or delete transactions (by id or filter), confirm review-queue
  items, set or copy budgets, edit paycheck lines, add / rename / retire categories, retrain the classifier. They only
  **propose** an action. It is validated against real data (unknown categories, bad ids, empty filters are refused),
  shown under *Pending actions* with a before/after preview, and applied only when you click **Approve**.
- **Grounding**: the prompt forbids estimating or doing math; before an answer is shown, every number and date in it is
  checked against the tool results from the conversation. Unverifiable figures send the model back for real data, and
  if it still can't back them up the answer is shown with a warning. Each answer has a *Data used* expander listing
  the exact tool calls and results. The check is a heuristic: it catches invented figures, not a wrong reading of a
  real one, and bare integers up to 20 are treated as counts.

Not covered: importing CSV files (use the Import page). Tool calling needs a model LM Studio can call tools with.
Settings: `FT_AGENT_MAX_TOKENS` (4096), `FT_AGENT_MAX_STEPS` (10), `FT_AGENT_TIMEOUT` (180).
Tests: `python tests/agent_test.py` (scripted fake LLM, throwaway database, no LM Studio needed).

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

**Numbers files (one file per year, one sheet per month)**: `tools/numbers_import.py` reads `.numbers` files directly,
with no Numbers app needed:
```bash
python tools/numbers_import.py numbers_files/                  # writes data/numbers_export/*.csv
python tools/numbers_import.py numbers_files/ --undated-day 1  # put rows with no usable date on the 1st
python tools/numbers_import.py numbers_files/ --load-db        # also load budgets + paychecks into the app
```
It writes `transactions.csv` (from each Transactions table), `budgets.csv` (Summary by Category, Budget column) and
`paychecks.csv` (the Mid/End of Month Paycheck tables). The month comes from the sheet name and the year from the file
name, and they override a transaction's own date; the original is kept in `Original Date`. Template and yearly-total
sheets, total/sum rows, and rows without a description or amount are skipped. Transactions on months after the
current one are left out unless you pass `--include-future`. Venmo and BILT rows are filed under FIDELITY. The
employer 401K match for each month is whatever the sheet's own 401k Saved (Table 1) counts beyond your
contribution (the sheets counted a 1:1 match through 2023 and none from 2024 on). `--load-db` replaces each
month's budget, adds any budget categories the app doesn't have yet (Roth IRA is flagged as savings), and saves
the mid and end paychecks; other paycheck tables (e.g. a one-off promotion paycheck) are not loaded.
Shared costs: the sheets record the roommate's half as a negative row whose formula halves specific rows (e.g.
`SUM(E31,E29)÷-2`). The script turns each of those into a `Shared` = 1 flag on the rows it references and drops the
negative row, so the app's 50% split gives the same totals. Formulas that aren't a plain half (the roommate paying
all of it, a quarter, fixed amounts mixed in) stay as ordinary rows. `--settled-through YYYY-MM` adds a `Settled`
column marking shared rows on that month and earlier as already paid back. Rent was paid in full but recorded at
half, with the roommate's half added through the month's total row (`SUM(...)-1675` or `-E4`); rent rows matched
that way are doubled and marked `Shared`, so your share is unchanged and the roommate's half is tracked. Months
whose total row doesn't subtract rent keep the rent row as recorded. The Import page reads both columns
(settled rows get the import date as their settled date). Import `transactions.csv` on the **Import CSV** page. `numbers_files/` and `*.numbers` are
git-ignored.

To export a single sheet by hand instead: in Numbers, select the Transactions table, then File -> Export To -> CSV
(Date, Bank, Description, Category, Amount). Categories that match a budget category (case-insensitive) are
stored as trusted `import` labels; anything else goes through the cascade.

## LLM model
The default is `nvidia/nemotron-3-nano-4b` (`FT_LLM_MODEL`). If LM Studio doesn't offer it, the app uses whichever
model is loaded there. Pick another model with **LLM model** in the sidebar (● = loaded); the choice is saved
in the `settings` table (schema v3). The import message names the model that was used.

## How categorization works
1. A CSV category that matches a budget category is kept as an `import` label.
2. **Savings rules**: Investing / Savings transfers are categorized by rule (`rule`).
3. **ML model** (TF-IDF word + char n-grams -> logistic regression). Confidence >= 90% is accepted (about 95% of those were right in calibration).
4. **LM Studio LLM** picks from the category list and returns JSON with a confidence. By default it only
   *suggests*: its answer pre-fills the review queue (threshold 1.01), because a calibration run on my data
   showed its stated confidence doesn't predict correctness (about 55% right when it claimed >= 80%).
   The prompt includes your own history (`retrieval.py`): the most similar past transactions and how you
   categorized them, plus typical merchants for each category, so it follows your conventions rather than
   guessing what a category like Superlatives means.
   All rows the ML model was unsure about are sent together, up to 4 requests at once (Categories & Model page,
   or `FT_LLM_PARALLEL`). LM Studio only runs them concurrently when its **Max Concurrent Predictions** setting
   (model load settings) is at least that high; otherwise they queue.
5. **Manual review queue**: anything left is flagged with the best guess (LLM's, else ML's) pre-selected. Confirmed labels are saved
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
