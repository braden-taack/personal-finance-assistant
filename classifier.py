"""ML categorizer: TF-IDF (word + char n-grams) -> logistic regression with probabilities.

Retraining is just `train()`; it always uses whatever categories currently exist in the
labeled data, so adding/renaming categories and relabeling rows then retraining works.
"""
from __future__ import annotations

import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

import db
from config import MIN_TRAIN_ROWS, MODEL_PATH


def clean(desc: str) -> str:
    """Strip noise common in bank descriptions (store numbers, dates, long digit runs)."""
    d = desc.lower()
    d = re.sub(r"\d{2}/\d{2}(/\d{2,4})?", " ", d)
    d = re.sub(r"#?\d{3,}", " ", d)
    d = re.sub(r"[^a-z&' ]", " ", d)
    return re.sub(r"\s+", " ", d).strip()


def _build() -> Pipeline:
    feats = FeatureUnion(
        [
            ("word", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=1)),
            ("char", TfidfVectorizer(preprocessor=clean, analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
        ]
    )
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    return Pipeline([("feats", feats), ("clf", clf)])


def train() -> dict:
    """Train on all labeled data. Returns metrics dict. Raises ValueError if too little data."""
    df = db.fetch_labeled()
    active = set(db.get_categories())
    df = df[df["category"].isin(active)]
    if len(df) < MIN_TRAIN_ROWS or df["category"].nunique() < 2:
        raise ValueError(
            f"Need at least {MIN_TRAIN_ROWS} labeled rows across 2+ categories "
            f"(have {len(df)} rows, {df['category'].nunique()} categories)."
        )
    pipe = _build()
    cv_acc = None
    counts = df["category"].value_counts()
    folds = int(min(5, counts.min()))
    if folds >= 2:
        cv_acc = float(cross_val_score(_build(), df["description"], df["category"], cv=folds).mean())
    pipe.fit(df["description"], df["category"])
    joblib.dump(pipe, MODEL_PATH)
    return {
        "rows": int(len(df)),
        "categories": int(df["category"].nunique()),
        "cv_accuracy": cv_acc,
        "classes": list(pipe.classes_),
    }


def load():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def predict(descriptions: list[str], model=None) -> list[tuple[str, float]]:
    """Return [(category, confidence)] per description. Empty model -> ('', 0.0)."""
    model = model or load()
    if model is None:
        return [("", 0.0)] * len(descriptions)
    proba = model.predict_proba(descriptions)
    idx = np.argmax(proba, axis=1)
    return [(str(model.classes_[i]), float(proba[n, i])) for n, i in enumerate(idx)]
