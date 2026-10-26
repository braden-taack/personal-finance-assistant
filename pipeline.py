"""Categorization cascade: savings rules -> ML -> LLM -> manual review."""
from __future__ import annotations

import pandas as pd

import classifier
import db
import importer
import llm
from config import LLM_THRESHOLD, ML_THRESHOLD


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

    model = classifier.load()
    llm_up = use_llm and llm.is_available()

    preds = classifier.predict(df.loc[todo, "description"].tolist(), model)
    for idx, (cat, conf) in zip(todo, preds):
        if cat and conf >= ML_THRESHOLD:
            df.loc[idx, ["category", "cat_source", "cat_confidence"]] = [cat, "ml", conf]
            continue

        if llm_up and cats:
            lcat, lconf = llm.categorize(
                df.at[idx, "description"], float(df.at[idx, "amount"]), cats, hint=cat or None
            )
            if lcat and lconf >= LLM_THRESHOLD:
                df.loc[idx, ["category", "cat_source", "cat_confidence"]] = [lcat, "llm", lconf]
                continue

        # Final fallback: park the best guess (if any) but flag for a human.
        df.loc[idx, ["category", "cat_source", "cat_confidence", "needs_review"]] = [
            cat or None, "ml" if cat else None, conf if cat else None, 1,
        ]
    return df
