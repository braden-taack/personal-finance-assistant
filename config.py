"""Central configuration. Override via environment variables or a .env-style edit here."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = Path(os.environ.get("FT_DB_PATH", DATA_DIR / "finance.db"))
MODEL_PATH = Path(os.environ.get("FT_MODEL_PATH", DATA_DIR / "classifier.joblib"))

# LM Studio exposes an OpenAI-compatible server (default port 1234).
LLM_BASE_URL = os.environ.get("FT_LLM_URL", "http://localhost:1234/v1")
# Preferred model; if LM Studio doesn't offer it, whatever model is loaded there is used instead.
# The sidebar's model selector overrides this (stored in the settings table).
LLM_MODEL = os.environ.get("FT_LLM_MODEL", "nvidia/nemotron-3-nano-4b")
# Concurrent LLM requests during categorization. LM Studio only runs them in parallel when its
# "Max Concurrent Predictions" setting is above 1; otherwise they queue (still correct, just not faster).
LLM_PARALLEL = int(os.environ.get("FT_LLM_PARALLEL", "4"))
LLM_TIMEOUT = float(os.environ.get("FT_LLM_TIMEOUT", "60"))
# Reasoning models (e.g. Qwen3) spend tokens thinking before they answer; too low a cap returns no answer.
LLM_MAX_TOKENS = int(os.environ.get("FT_LLM_MAX_TOKENS", "1024"))

# Confidence cascade
ML_THRESHOLD = float(os.environ.get("FT_ML_THRESHOLD", "0.80"))    # below -> ask LLM
LLM_THRESHOLD = float(os.environ.get("FT_LLM_THRESHOLD", "0.80"))  # below -> manual review

# Duplicate detection
DUP_DATE_WINDOW_DAYS = int(os.environ.get("FT_DUP_WINDOW", "3"))
DUP_DESC_SIMILARITY = float(os.environ.get("FT_DUP_SIM", "0.85"))

# Minimum labeled rows needed before the ML model is trained
MIN_TRAIN_ROWS = 20

# Shared expenses with the roommate: always 50/50. SHARED_SPLIT is *my* share of a shared
# transaction; the roommate owes the rest. Deliberately not configurable from the UI.
SHARED_SPLIT = 0.5

# Banks (seeded into the `banks` table; settings are editable on the Categories & Model page).
# mapping keys: date, description, amount | debit+credit, category, bank.
# charges_negative: 1 when the bank exports spending as negative numbers (single amount column only).
DEFAULT_BANKS = [
    {"name": "FIDELITY", "charges_negative": 1, "mapping": {"date": "Date", "description": "Name", "amount": "Amount"}},
    {"name": "C1", "charges_negative": 0,
     "mapping": {"date": "Transaction Date", "description": "Description", "debit": "Debit", "credit": "Credit",
                 "category": None}},
    {"name": "AMEX", "charges_negative": 0, "mapping": {"date": "Date", "description": "Description", "amount": "Amount"}},
    {"name": "DISCOVER", "charges_negative": 0,
     "mapping": {"date": "Trans. Date", "description": "Description", "amount": "Amount"}},
]
# Filename fragments used to pre-select the bank on import.
BANK_FILENAME_HINTS = {
    "FIDELITY": ["fidelity"],
    "C1": ["capital one", "capitalone", "capital_one", "c1"],
    "AMEX": ["amex", "american express", "american_express"],
    "DISCOVER": ["discover"],
}

# Budget categories, in sheet order. is_savings marks money moved to savings rather than spent.
DEFAULT_CATEGORIES = [
    "Rent", "Utilities", "Groceries", "Fast Food", "Restaurants", "Bars", "Gym", "Car Insurance",
    "Health Insurance", "Dental Insurance", "Renters Insurance", "Student Loans", "Gas", "Phone",
    "Internet", "Car Payment", "Travel", "Golf", "Superlatives", "Laundry", "Investing", "Savings",
]
# Roth IRA isn't seeded; it's added (and flagged) when budgets that use it are loaded from Numbers files.
SAVINGS_CATEGORIES = ["Investing", "Savings", "Roth IRA"]

# Seed budget for the current month (categories not listed are 0). Total = 5299.13.
DEFAULT_BUDGET = {
    "Rent": 1675, "Utilities": 45, "Groceries": 250, "Fast Food": 50, "Restaurants": 300, "Bars": 100,
    "Gym": 250, "Car Insurance": 100, "Health Insurance": 25, "Dental Insurance": 75,
    "Renters Insurance": 9.13, "Internet": 75, "Car Payment": 625, "Travel": 400, "Golf": 100,
    "Superlatives": 200, "Laundry": 20, "Investing": 1000,
}

# Paycheck lines (same for 'mid' and 'end'). "Income" lines add to take-home; every other
# category except EMPLOYER_CATEGORY is a deduction. Take-home = 4614.17 - 2278.96 = 2335.21.
PAYCHECK_PERIODS = ["mid", "end"]
INCOME_CATEGORY = "Income"
EMPLOYER_CATEGORY = "Employer"          # not part of take-home; only feeds "401k Saved"
RETIREMENT_SUBCATEGORY = "401K"
EMPLOYER_MATCH_SUBCATEGORY = "401K Match"
TAX_CATEGORIES = ["FED Taxes", "State Taxes", "Local"]
DEFAULT_PAYCHECK = [
    ("Income", "Salary", 4614.17),
    ("FED Taxes", "Withholding", 468.37),
    ("FED Taxes", "Social Security", 243.46),
    ("FED Taxes", "Medicare", 56.94),
    ("State Taxes", "Withholding", 188.24),
    ("Local", "Withholding", 126.35),
    ("Other", "UHC HDHP", 23.85),
    ("Other", "HSA", 160.42),
    ("Other", "Commuter", 34.80),
    ("Other", "UHC Dental", 7.55),
    ("Other", "401K", 968.98),
    # Assumption: the employer matches the employee 401K contribution 1:1 (makes 401k Saved = 4 x 968.98).
    ("Employer", "401K Match", 968.98),
]

# Card payments / transfers between my own accounts double-count against card purchases.
# Matched case-insensitively against the description; the importer asks whether to skip them.
TRANSFER_PATTERNS = [
    r"payment\s*-?\s*thank\s*you", r"autopay", r"auto\s*pay", r"transfer\s+from", r"transfer\s+to",
    r"online\s+payment", r"mobile\s+payment", r"credit\s+card\s+payment", r"e-?payment", r"\bach\s+pmt\b",
    r"directpay", r"payment\s+received",
]
# Legitimate moves that map straight to a savings category (checked before TRANSFER_PATTERNS).
SAVINGS_TRANSFER_RULES = {r"investing\s+transfer": "Investing", r"savings\s+transfer": "Savings"}
