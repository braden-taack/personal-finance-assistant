"""Browser end-to-end test: drives the real Streamlit app with Playwright on a throwaway database.

    python -m playwright install chromium     # once
    python tests/e2e_playwright.py            # LM Studio optional; LLM checks are skipped if it's offline

Options: --llm-url (default http://127.0.0.1:1234/v1), --headed, --shots DIR (save screenshots).
Sample CSVs are generated at run time (dated in the current month), so no data files live in the repo.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--llm-url", default="http://127.0.0.1:1234/v1")
parser.add_argument("--headed", action="store_true")
parser.add_argument("--shots", type=Path, help="directory to save screenshots in")
args = parser.parse_args()

TMP = Path(tempfile.mkdtemp(prefix="ft-e2e-"))
os.environ.update(FT_DB_PATH=str(TMP / "e2e.db"), FT_MODEL_PATH=str(TMP / "m.joblib"), FT_LLM_URL=args.llm_url)
sys.path.insert(0, str(ROOT))

import classifier  # noqa: E402
import db  # noqa: E402
import llm  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

TODAY = date.today()
M = TODAY.strftime("%Y-%m")


def day(d: int, fmt: str = "%Y-%m-%d") -> str:
    return TODAY.replace(day=d).strftime(fmt)


# ---------------------------------------------------------------- fixtures
def seed() -> None:
    """Training labels (dated last year) so the ML step has a model."""
    db.init_db()
    data = {
        "Groceries": ["WHOLE FOODS #123", "TRADER JOES 552", "KROGER 0932", "SAFEWAY STORE 12", "ALDI 889"],
        "Restaurants": ["CHIPOTLE 2231", "PANERA BREAD 44", "OLIVE GARDEN 18", "CHEESECAKE FACTORY", "NOODLES & CO"],
        "Gas": ["SHELL OIL 5521", "EXXON 9981", "BP GAS 1123", "SPEEDWAY 0042", "MARATHON PETRO"],
        "Rent": ["AVALON APTS RENT", "RENT PAYMENT PORTAL", "APARTMENT RENT", "BILT RENT", "ZILLOW RENT"],
    }
    y = TODAY.year - 1
    db.insert_transactions([dict(date=f"{y}-0{1 + j % 9}-1{i}", bank="AMEX", description=n, amount=10 + j,
                                 category=c, cat_source="import")
                            for i, (c, ns) in enumerate(data.items()) for j, n in enumerate(ns * 2)], "seed")
    classifier.train()


def write_csvs() -> tuple[Path, Path]:
    c1 = TMP / f"capital_one_{M}.csv"
    c1.write_text(
        "Transaction Date,Posted Date,Card No.,Description,Category,Debit,Credit\n"
        f"{day(2)},{day(3)},1234,WHOLE FOODS #999,Merchandise,54.10,\n"
        f"{day(3)},{day(4)},1234,CAPITAL ONE AUTOPAY PYMT,Payment/Credit,,500.00\n"
        f"{day(4)},{day(5)},1234,PLANET FITNESS,Health,24.99,\n"
        f"{day(5)},{day(6)},1234,COSTCO WHOLESALE,Merchandise,100.00,\n"
        f"{day(6)},{day(7)},1234,TRADER JOES 552,Merchandise,30.00,\n"
        f"{day(6)},{day(7)},1234,TRADER JOES 552,Merchandise,30.00,\n")
    numbers = TMP / "numbers_transactions.csv"
    numbers.write_text(
        "Date,Bank,Description,Category,Amount\n"
        f'"{day(8, "%b %-d, %Y")}",AMEX,AVALON APTS RENT,Rent,"$1,675.00"\n'
        ",DISCOVER,SHELL OIL 5521,gas,$41.00\n"
        f'{day(10, "%m/%d/%Y")},FIDELITY,Investing Transfer,Investing,"$1,000.00"\n'
        f'{day(11, "%m/%d/%Y")},CHASE,TOPGOLF CHICAGO,Golf,$85.00\n')
    return c1, numbers


# My share of everything imported: 54.10 + 24.99 + 100/2 (Costco, shared) + 30 + 1675 + 41 + 1000 + 85
EXPECTED_SPENT = 2960.09
EXPECTED_LEFTOVER = round(4670.42 - EXPECTED_SPENT, 2)


# ---------------------------------------------------------------- app server
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_app(port: int) -> subprocess.Popen:
    log = open(TMP / "streamlit.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port),
                             "--server.address", "127.0.0.1", "--server.headless", "true"],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    import requests
    for _ in range(120):
        try:
            if requests.get(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).ok:
                return proc
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    proc.kill()
    sys.exit(f"Streamlit didn't start; see {TMP / 'streamlit.log'}")


# ---------------------------------------------------------------- helpers
results: list[tuple[str, bool, str]] = []
console_errors: list[str] = []

BUSY = """() => {
    const w = document.querySelector('[data-testid="stStatusWidget"]');
    return !!((w && /Running|Stop/.test(w.innerText)) || document.querySelector('[data-testid="stSpinner"]'));
}"""


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def skip(name: str, why: str) -> None:
    print(f"SKIP {name} — {why}", flush=True)


def idle(page, timeout=300_000) -> None:
    """Wait for the Streamlit rerun to start (~1s after an interaction), then to finish."""
    try:
        page.wait_for_function(BUSY, timeout=2500)
    except Exception:
        pass
    page.wait_for_function(f"() => !({BUSY})()", timeout=timeout)


def goto(page, name: str) -> None:
    page.locator('[data-testid="stSidebar"]').get_by_text(name, exact=True).click()
    idle(page)


def exceptions(page) -> list[str]:
    return [e.inner_text()[:300] for e in page.locator('[data-testid="stException"]').all()]


def metric(page, label: str) -> str:
    m = page.locator('[data-testid="stMetric"]').filter(has_text=label).first
    return m.locator('[data-testid="stMetricValue"]').inner_text().strip()


def main_text(page) -> str:
    return page.locator('[data-testid="stMain"]').inner_text()


def shot(page, name: str) -> None:
    if args.shots:
        args.shots.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(args.shots / f"{name}.png"), full_page=True)


def code_text(page) -> str:
    code = page.locator('[data-testid="stCode"]')
    return code.inner_text() if code.count() else ""


# ---------------------------------------------------------------- the run
def run(page, c1_csv: Path, numbers_csv: Path, llm_model: str | None) -> None:
    page.goto(URL)
    page.wait_for_selector('[data-testid="stSidebar"]', timeout=60_000)
    idle(page)

    # Dashboard before imports: seeded budget + paychecks for this month
    shot(page, "01-dashboard-empty")
    check("dashboard renders", not exceptions(page), "; ".join(exceptions(page)))
    check("take-home metric = $4,670.42", metric(page, "Take-home pay") == "$4,670.42", metric(page, "Take-home pay"))

    # Paychecks
    goto(page, "Paychecks")
    shot(page, "02-paychecks")
    vals = [v.inner_text() for v in page.locator('[data-testid="stMetric"]').filter(has_text="Total (take-home)")
            .locator('[data-testid="stMetricValue"]').all()]
    check("both paychecks take-home $2,335.21", vals == ["$2,335.21", "$2,335.21"], str(vals))
    check("month take-home $4,670.42", metric(page, "Take-home for the month") == "$4,670.42")
    check("side-by-side headings", "Mid Month Paycheck" in main_text(page) and "End of Month Paycheck" in main_text(page))

    # Import: Capital One debit/credit file
    goto(page, "Import CSV")
    page.locator('input[type="file"]').set_input_files(str(c1_csv))
    idle(page)
    txt = main_text(page)
    bank_val = page.locator('[data-testid="stSelectbox"]').filter(has_text="Bank — applies to every row") \
        .locator("input").input_value()
    check("C1 pre-selected", bank_val == "C1", bank_val)
    check("pre-selected from filename", "Pre-selected from: filename" in txt)
    check("saved C1 mapping used", "Using the saved column mapping for C1" in txt)
    check("debit/credit layout selected", page.get_by_role("radio", name="Separate Debit and Credit columns").is_checked())
    page.get_by_role("button", name="Analyze file").click()
    idle(page)
    shot(page, "03-import-c1-analyzed")
    txt = main_text(page)
    counts = re.search(r"\d+ rows ·[^\n]*", txt)
    check("C1 analysis counts", "6 rows" in txt and "1 possible duplicate(s)" in txt and "1 payment/transfer(s)" in txt,
          counts.group(0) if counts else txt[:200])
    check("autopay listed as transfer", "CAPITAL ONE AUTOPAY PYMT" in txt)
    page.get_by_role("button", name="Categorize & import").click()
    idle(page)
    m = re.search(r"Imported \d+ rows[^\n]*", main_text(page))
    check("C1 import: 4 rows, autopay + duplicate skipped", bool(m) and m.group(0).startswith("Imported 4 rows (2 skipped)"),
          m.group(0) if m else "; ".join(exceptions(page)))
    if llm_model:
        check("LLM used, with the expected model", bool(m) and "'llm'" in m.group(0) and f"LLM: {llm_model}." in m.group(0),
              m.group(0) if m else "")
    else:
        skip("LLM used", "LM Studio unreachable")
    shot(page, "04-import-c1-done")

    # Import: Numbers export (Bank column, blank date, unknown bank)
    page.locator('input[type="file"]').set_input_files(str(numbers_csv))
    idle(page)
    check("Numbers file: Bank column notice", "The file has a Bank column" in main_text(page))
    page.get_by_role("button", name="Analyze file").click()
    idle(page)
    txt = main_text(page)
    check("Numbers analysis: 1 missing date", "1 missing date(s)" in txt)
    check("Numbers analysis: invalid bank warned", "blank or unknown Bank value" in txt)
    import_btn = page.get_by_role("button", name="Categorize & import")
    check("import blocked until date fixed", import_btn.is_disabled())
    grid = page.locator('[data-testid="stDataFrame"]').last          # the date-fix editor (canvas grid)
    grid.scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    box = grid.bounding_box()
    page.mouse.dblclick(box["x"] + 80, box["y"] + 35 + 17)            # first data row, date column
    page.wait_for_timeout(800)
    date_input = page.locator('input[type="date"]')
    if date_input.count():
        date_input.first.fill(day(9))
        date_input.first.press("Enter")
        idle(page)
    check("date filled in via editor", import_btn.is_enabled())
    shot(page, "05-import-numbers-datefix")
    import_btn.click()
    idle(page)
    m = re.search(r"Imported \d+ rows[^\n]*", main_text(page))
    check("Numbers import: 4 rows, all trusted labels",
          bool(m) and m.group(0).startswith("Imported 4 rows (0 skipped). Sources: {'import': 4}"),
          m.group(0) if m else "; ".join(exceptions(page)))

    goto(page, "Transactions")
    check("transactions page renders", not exceptions(page), "; ".join(exceptions(page)))

    # Shared expenses
    goto(page, "Shared expenses")
    search = page.get_by_label("Search description")
    search.fill("COSTCO")
    search.press("Enter")
    idle(page)
    page.get_by_text(re.compile(r"Select all 1 shown")).click()
    idle(page)
    page.get_by_role("button", name="Mark 1 selected shared").click()
    idle(page)
    shot(page, "06-shared-marked")
    check("roommate owes $50.00 after sharing $100", metric(page, "Still owed for") == "$50.00", metric(page, "Still owed for"))
    check("total across months $50.00", metric(page, "Total still owed by roommate") == "$50.00")
    check("Groceries expander itemized",
          page.get_by_text(re.compile(r"Groceries — 1 item\(s\), roommate owes \$50\.00")).count() == 1)
    page.get_by_role("button", name="Copy request message").click()
    idle(page)
    code = code_text(page)
    check("request message", "Groceries: $50.00" in code and "Total: $50.00" in code, code.replace("\n", " / "))
    with page.expect_download() as d:
        page.get_by_role("button", name="Download itemized CSV").click()
    csv_text = Path(d.value.path()).read_text()
    check("itemized CSV download", "COSTCO WHOLESALE" in csv_text, csv_text.splitlines()[0])
    page.get_by_role("button", name="Mark whole month settled").click()
    idle(page)
    shot(page, "07-shared-settled")
    check("still owed $0.00 after settling", metric(page, "Still owed for") == "$0.00", metric(page, "Still owed for"))
    check("settled item stays in history", page.get_by_text(re.compile(r"Groceries — 1 item\(s\)")).count() == 1)
    check("request message updates after settling", "nothing outstanding" in code_text(page),
          code_text(page).replace("\n", " / "))
    page.locator("label").filter(has_text=re.compile(r"^Year$")).last.click()
    idle(page)
    check("shared year view renders", not exceptions(page), "; ".join(exceptions(page)))

    # Dashboard after imports
    goto(page, "Dashboard")
    shot(page, "08-dashboard-month")
    check("dashboard month renders", not exceptions(page), "; ".join(exceptions(page)))
    check(f"spent (my share) = ${EXPECTED_SPENT:,.2f}", metric(page, "Spent (my share)") == f"${EXPECTED_SPENT:,.2f}",
          metric(page, "Spent (my share)"))
    check(f"Net Leftover = ${EXPECTED_LEFTOVER:,.2f}", metric(page, "Net Leftover") == f"${EXPECTED_LEFTOVER:,.2f}",
          metric(page, "Net Leftover"))
    page.locator("label").filter(has_text=re.compile(r"^Year$")).first.click()
    idle(page)
    shot(page, "09-dashboard-year")
    check("dashboard year renders (incl. Sankey)", not exceptions(page) and "Money flow" in main_text(page),
          "; ".join(exceptions(page)))

    for name in ("Review queue", "Categories & Model", "Budgets"):
        goto(page, name)
        check(f"{name} renders", not exceptions(page), "; ".join(exceptions(page)))

    # LLM selector: default, change, persists across a reload
    side = page.locator('[data-testid="stSidebar"]')
    if not llm_model:
        skip("LLM selector", "LM Studio unreachable")
        return
    box = side.locator('[data-testid="stSelectbox"]').filter(has_text="LLM model")
    current = box.locator("input").input_value()
    check("LLM selector shows the resolved default", current.startswith(llm_model), current)
    check("sidebar shows 4 parallel requests", "4 parallel requests" in side.inner_text())
    box.click()
    others = page.get_by_role("option").filter(has_not_text=llm_model)
    if not others.count():
        page.keyboard.press("Escape")
        skip("LLM choice persists", "LM Studio offers only one model")
    else:
        target = others.first.inner_text().removesuffix(" ●").strip()
        others.first.click()
        idle(page)
        page.reload()
        page.wait_for_selector('[data-testid="stSidebar"]', timeout=60_000)
        idle(page)
        after = side.locator('[data-testid="stSelectbox"]').filter(has_text="LLM model").locator("input").input_value()
        check("LLM choice persists after reload", after.startswith(target), after)
    goto(page, "Categories & Model")
    check("parallel setting shown", page.locator('input[aria-label="Parallel LLM requests"]').input_value() == "4")


if __name__ == "__main__":
    seed()
    c1_csv, numbers_csv = write_csvs()
    llm_model = llm.resolve_model()
    print(f"temp dir: {TMP}\nLLM: {llm_model or 'offline — LLM checks skipped'}", flush=True)
    port = free_port()
    URL = f"http://127.0.0.1:{port}"
    server = start_app(port)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not args.headed)
            page = browser.new_page(viewport={"width": 1500, "height": 1000})
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
            try:
                run(page, c1_csv, numbers_csv, llm_model)
            finally:
                browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    check("no browser console errors", not console_errors, "; ".join(console_errors[:5]))
    fails = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(fails)}/{len(results)} passed")
    if fails:
        print(f"Kept {TMP} (database + streamlit.log) for debugging.")
    else:
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if fails else 0)
