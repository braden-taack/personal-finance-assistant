"""Categorization cascade: savings rules -> ML -> LLM (parallel, with my history as context) -> manual review."""
from __future__ import annotations

import pandas as pd

import classifier
import db
import importer
import llm
import retrieval
from config import LLM_MODEL, LLM_PARALLEL, LLM_THRESHOLD, ML_THRESHOLD


def categorize_frame(df: pd.DataFrame, use_llm: bool = True) -> pd.DataFrame:
    """Fill category / cat_source / cat_confidence / needs_review for rows lacking a category.

    Rows that already carry a category (e.g. from your Numbers sheet) are kept as
    source='import' and later serve as training data. Fidelity-style "Investing Transfer" /
    "Savings Transfer" rows are mapped by rule (source='rule'). Everything else goes
    ML -> LLM -> manual review.
    """
    df = df.copy()
    for col, default in [("cat_source", None), ("cat_confidence", None), ("needs_review", 0)]:
        df[col] = default
    df["category"] = df["category"].astype(object)
    df["cat_source"] = df["cat_source"].astype(object)

    has_cat = df["category"].notna() & (df["category"].astype(str).str.strip() != "")
    df.loc[has_cat, ["cat_source", "cat_confidence"]] = ["import", 1.0]

    cats = db.get_categories()
    for idx in df.index[~has_cat]:
        rule_cat = importer.savings_rule(str(df.at[idx, "description"]))
        if rule_cat and rule_cat in cats:
            df.loc[idx, ["category", "cat_source", "cat_confidence"]] = [rule_cat, "rule", 1.0]

    todo = df.index[df["category"].isna()]
    if len(todo) == 0:
        return df

    preds = classifier.predict(df.loc[todo, "description"].tolist(), classifier.load())
    unsure = []
    for idx, (cat, conf) in zip(todo, preds):
        if cat and conf >= ML_THRESHOLD:
            df.loc[idx, ["category", "cat_source", "cat_confidence"]] = [cat, "ml", conf]
        else:
            unsure.append((idx, cat, conf))

    # LLM pass for everything the ML model wasn't sure about, several requests at once.
    llm_answers = [("", 0.0)] * len(unsure)
    model = llm.resolve_model(llm_model()) if use_llm and unsure and cats else None
    if model:
        # Give the LLM my own history: similar past transactions and typical merchants per category.
        index = retrieval.ExampleIndex(db.fetch_labeled())
        items = [(df.at[idx, "description"], float(df.at[idx, "amount"]), cat or None,
                  index.similar(df.at[idx, "description"])) for idx, cat, _ in unsure]
        llm_answers = llm.categorize_many(items, cats, model=model, workers=llm_parallel(), glossary=index.glossary())
    df.attrs["llm_model"] = model

    for (idx, cat, conf), (lcat, lconf) in zip(unsure, llm_answers):
        if lcat and lconf >= LLM_THRESHOLD:
            df.loc[idx, ["category", "cat_source", "cat_confidence"]] = [lcat, "llm", lconf]
            continue
        # Final fallback: flag for a human, pre-filled with the LLM's answer if it gave one, else the ML guess.
        if lcat:
            df.loc[idx, ["category", "cat_source", "cat_confidence", "needs_review"]] = [lcat, "llm", lconf, 1]
        else:
            df.loc[idx, ["category", "cat_source", "cat_confidence", "needs_review"]] = [
                cat or None, "ml" if cat else None, conf if cat else None, 1,
            ]
    return df


def llm_model() -> str:
    """The model chosen in the app, else the configured default (availability is checked by llm.resolve_model)."""
    return db.get_setting("llm_model") or LLM_MODEL


def llm_parallel() -> int:
    return max(1, int(db.get_setting("llm_parallel") or LLM_PARALLEL))
