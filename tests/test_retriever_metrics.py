"""RetrieverMetrics: matches[i] is the golden page the result at rank i+1 hit, or None."""

import pytest

from metrics.retriever_metrics import RetrieverMetrics


def test_recall_counts_distinct_pages_once():
    # two chunks from page 4, none from page 5
    metrics = RetrieverMetrics([4, 4, None], total_relevant=2)
    assert metrics.recall(3) == 0.5


def test_recall_only_looks_at_the_top_k():
    metrics = RetrieverMetrics([None, None, 4], total_relevant=1)
    assert metrics.recall(2) == 0.0
    assert metrics.recall(3) == 1.0


def test_precision_divides_by_k_even_with_fewer_results():
    metrics = RetrieverMetrics([4, None], total_relevant=1)  # only 2 results came back
    assert metrics.precision(5) == pytest.approx(1 / 5)


def test_precision_counts_every_relevant_chunk():
    metrics = RetrieverMetrics([4, 4, None, 5], total_relevant=2)
    assert metrics.precision(4) == 0.75


def test_hit_is_one_when_any_relevant_chunk_is_in_the_top_k():
    metrics = RetrieverMetrics([None, 4], total_relevant=1)
    assert metrics.hit(1) == 0.0
    assert metrics.hit(2) == 1.0


@pytest.mark.parametrize(
    ("matches", "expected"),
    [([4], 1.0), ([None, 4], 0.5), ([None, None, None, 4], 0.25), ([None, None], 0.0)],
)
def test_reciprocal_rank(matches, expected):
    assert RetrieverMetrics(matches, total_relevant=1).reciprocal_rank() == expected


def test_nothing_found_scores_zero_everywhere():
    metrics = RetrieverMetrics([None, None, None], total_relevant=2)
    assert metrics.recall(3) == 0.0
    assert metrics.precision(3) == 0.0
    assert metrics.hit(3) == 0.0
    assert metrics.reciprocal_rank() == 0.0
