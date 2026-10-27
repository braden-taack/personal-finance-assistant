"""Similar labeled transactions for the LLM prompt, so it follows my categorization conventions.

Built from the labeled history (manual + imported labels). Descriptions are normalized with
classifier.clean and compared with character n-gram TF-IDF (cosine similarity), which copes with
store numbers, truncation and bank-specific suffixes.
"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import classifier


class ExampleIndex:
    def __init__(self, labeled: pd.DataFrame):
        """`labeled` needs description and category columns."""
        df = labeled.dropna(subset=["description", "category"]).copy()
        df["key"] = df["description"].astype(str).map(classifier.clean)
        df = df[df["key"] != ""]
        # One entry per (merchant, category), remembering how often I used it.
        self.rows = (df.groupby(["key", "category"])
                     .agg(description=("description", "first"), n=("description", "size"))
                     .reset_index())
        self.vec = self.matrix = None
        if len(self.rows):
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            self.matrix = self.vec.fit_transform(self.rows["key"])

    def similar(self, description: str, k: int = 8, min_score: float = 0.25) -> list[tuple[str, str, int]]:
        """Up to k past (description, category, times used), most similar first."""
        if self.matrix is None:
            return []
        scores = linear_kernel(self.vec.transform([classifier.clean(str(description))]), self.matrix).ravel()
        top = scores.argsort()[::-1][:k]
        return [(self.rows.at[i, "description"], self.rows.at[i, "category"], int(self.rows.at[i, "n"]))
                for i in top if scores[i] >= min_score]

    def glossary(self, per_category: int = 4) -> dict[str, list[str]]:
        """Most-used merchants per category, showing what each (personal) category means."""
        top = self.rows.sort_values("n", ascending=False).groupby("category").head(per_category)
        return top.groupby("category")["description"].apply(list).to_dict()
