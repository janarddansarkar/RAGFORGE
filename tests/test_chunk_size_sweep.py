"""The pure parts of the chunk-size sweep: overlap, the question sample and the choice."""

from eval.chunk_size_sweep import ChunkSizeSweep


def entry(size, recall5, mrr, rel3, rel5, recall3=None):
    return {
        "chunk_size": size,
        "chunk_overlap": ChunkSizeSweep.overlap(size),
        "retrieval": {
            "Recall@5": recall5,
            "Recall@3": recall5 if recall3 is None else recall3,
            "MRR": mrr,
        },
        "context_relevance": {"top3": rel3, "top5": rel5},
    }


def test_overlap_is_fifteen_percent():
    assert ChunkSizeSweep.overlap(1000) == 150
    assert ChunkSizeSweep.overlap(400) == 60


def test_sample_is_spread_and_never_larger_than_the_set():
    picked = ChunkSizeSweep.sample_indices(45, 5)
    assert picked == {0, 9, 18, 27, 36}
    assert ChunkSizeSweep.sample_indices(3, 8) == {0, 1, 2}


def test_choice_takes_the_most_relevant_size_that_keeps_recall():
    results = {
        "1000": entry(1000, 0.90, 0.80, 0.30, 0.30),
        "600": entry(600, 0.88, 0.79, 0.45, 0.45),
        "250": entry(250, 0.70, 0.60, 0.80, 0.80),  # most relevant, but loses recall
    }
    choice = ChunkSizeSweep.choose(results, baseline=1000)
    assert choice["chunk_size"] == 600
    assert choice["chunk_overlap"] == 90


def test_choice_keeps_the_baseline_when_nothing_beats_it():
    results = {
        "1000": entry(1000, 0.90, 0.80, 0.50, 0.50),
        "600": entry(600, 0.90, 0.80, 0.40, 0.40),
    }
    assert ChunkSizeSweep.choose(results, baseline=1000)["chunk_size"] == 1000


def test_top_k_three_is_chosen_only_if_recall_holds_and_relevance_rises():
    base = {"1000": entry(1000, 0.90, 0.80, 0.60, 0.40, recall3=0.88)}
    assert ChunkSizeSweep.choose(base, 1000)["top_k"] == 3
    lost_recall = {"1000": entry(1000, 0.90, 0.80, 0.60, 0.40, recall3=0.70)}
    assert ChunkSizeSweep.choose(lost_recall, 1000)["top_k"] == 5
    no_gain = {"1000": entry(1000, 0.90, 0.80, 0.40, 0.50, recall3=0.90)}
    assert ChunkSizeSweep.choose(no_gain, 1000)["top_k"] == 5


def test_a_failed_judge_never_wins():
    results = {
        "1000": entry(1000, 0.90, 0.80, 0.30, 0.30),
        "600": entry(600, 0.90, 0.80, None, None),
    }
    assert ChunkSizeSweep.choose(results, baseline=1000)["chunk_size"] == 1000
