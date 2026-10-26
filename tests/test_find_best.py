"""find_best.py's plumbing with stub stages: which models are needed, and how stages combine."""

from eval.find_best import STAGES, BestConfigSearch, drop_missing, models_needed

CONFIG = {
    "embedding_model": "bge-m3",
    "llm_model": "qwen2.5:3b",
    "reranker_model": "qwen2.5:3b",
    "judge_model": "mistral:7b",
    "embedders": ["bge-m3", "nomic-embed-text"],
    "rerankers": ["qwen2.5:1.5b", "llama3.2:3b"],
    "generators": ["qwen2.5:3b", "gemma3:4b"],
}


def test_each_stage_asks_only_for_the_models_it_uses():
    only_method = models_needed(CONFIG, ("method",))
    assert only_method == {"bge-m3", "qwen2.5:3b"}
    assert "nomic-embed-text" in models_needed(CONFIG, ("embedder",))
    assert "mistral:7b" in models_needed(CONFIG, ("chunk_size",))
    everything = models_needed(CONFIG, STAGES)
    assert {"nomic-embed-text", "llama3.2:3b", "gemma3:4b", "mistral:7b"} <= everything


def test_missing_candidates_can_be_left_out():
    pulled = {"bge-m3", "qwen2.5:3b", "qwen2.5:1.5b", "mistral:7b"}
    cut = drop_missing(CONFIG, pulled)
    assert cut["embedders"] == ["bge-m3"]
    assert cut["rerankers"] == ["qwen2.5:1.5b"]
    assert cut["generators"] == ["qwen2.5:3b"]


def make_search(better: bool) -> BestConfigSearch:
    """A search whose chunk-size stage picks 400, and whose held-out check is stubbed."""
    config = {
        **CONFIG,
        "chunking_strategy": "recursive",
        "contextual": False,
        "default_method": "hybrid_rerank",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "max_new_tokens": 256,
    }
    search = BestConfigSearch.__new__(BestConfigSearch)
    search.config = config
    search.primary, search.tie_break, search.report = "MRR", "Recall@5", {}
    search.stage_chunk_size = lambda c: {
        **c,
        "chunk_size": 400,
        "chunk_overlap": 60,
        "top_k": 3,
    }

    def evaluate(cfg, method, holdout=False):
        mrr = 0.7  # today's settings
        if cfg["chunk_size"] == 400:
            mrr = 0.9 if better else 0.5
        return {"MRR": mrr, "Recall@5": 0.8, "Recall@1": 0.5, "sec/query": 1.0}

    search.evaluate = evaluate
    return search


def test_a_stage_that_is_not_run_keeps_todays_settings():
    result = make_search(better=True).run(("chunk_size",))
    assert result["settings"]["chunk_size"] == 400
    assert result["settings"]["embedding_model"] == "bge-m3"  # untouched
    assert result["settings"]["llm_model"] == "qwen2.5:3b"  # generator stage not run
    assert result["retrieval"] == "updated"


def test_a_pick_that_loses_on_held_out_questions_is_not_kept():
    result = make_search(better=False).run(("chunk_size",))
    assert result["retrieval"] == "kept current settings"
    assert result["settings"]["chunk_size"] == 1000
    assert result["settings"]["top_k"] == 5
