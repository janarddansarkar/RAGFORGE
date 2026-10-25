"""Retriever metrics: Recall@K, Precision@K, HitRate@K and MRR.

Definitions follow specs_md/retriever_evaluation_metrics.md. These classes only do
the math; deciding what counts as relevant is up to the caller (see eval/retriever_eval.py).
"""


class RetrieverMetrics:
    """The metrics for ONE query, from its ranked list of retrieved results.

    matches[i] says what the i-th result (rank i+1) matched: the id of the relevant
    item it found (e.g. a page number), or None if it is not relevant.
    total_relevant is how many relevant items the query has in all.
    """

    def __init__(self, matches: list, total_relevant: int):
        self.matches = matches
        self.total_relevant = total_relevant

    def recall(self, k: int) -> float:
        """Share of the relevant items found in the top K."""
        found = {m for m in self.matches[:k] if m is not None}
        return len(found) / self.total_relevant

    def precision(self, k: int) -> float:
        """Share of the top K results that are relevant."""
        return sum(m is not None for m in self.matches[:k]) / k

    def hit(self, k: int) -> float:
        """1 if at least one relevant result is in the top K, else 0."""
        return float(any(m is not None for m in self.matches[:k]))

    def reciprocal_rank(self) -> float:
        """1 / rank of the first relevant result (0 if there is none)."""
        for rank, m in enumerate(self.matches, start=1):
            if m is not None:
                return 1 / rank
        return 0.0
