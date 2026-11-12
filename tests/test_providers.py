"""The provider choices: every component gets its own provider and model, and the user's keys."""

from providers import COMPONENTS, OFF, needed_keys, providers_for, settings_for

DEFAULTS = {c.key: (c.provider, c.model) for c in COMPONENTS}


def test_each_component_can_use_a_different_provider():
    choices = {
        **DEFAULTS,
        "answer": ("Anthropic", "claude-sonnet-5-5"),
        "judge": ("Google Gemini", "gemini-3.5-flash"),
    }
    settings = settings_for(
        choices, {"OpenAI": "o", "Anthropic": "a", "Google Gemini": "g"}
    )
    assert settings["llm_model"] == "anthropic/claude-sonnet-5-5"
    assert settings["judge_model"] == "gemini/gemini-3.5-flash"
    assert settings["guard_model"] == "openai/gpt-4o-mini"
    assert settings["embedding_model"] == "openai/text-embedding-3-small"
    assert settings["api_keys"] == {"openai": "o", "anthropic": "a", "gemini": "g"}


def test_anthropic_is_offered_for_chat_only():
    assert "Anthropic" in providers_for("chat")
    assert "Anthropic" not in providers_for("embeddings")
    assert "Anthropic" not in providers_for("speech")


def test_only_the_providers_in_use_need_a_key():
    assert needed_keys(DEFAULTS) == ["OpenAI"]
    choices = {
        **DEFAULTS,
        "answer": ("Anthropic", "x"),
        "embedding": ("Google Gemini", "y"),
    }
    assert needed_keys(choices) == ["Anthropic", "OpenAI", "Google Gemini"]


def test_voice_input_can_be_switched_off():
    settings = settings_for({**DEFAULTS, "speech": (OFF, "")}, {"OpenAI": "o"})
    assert settings["stt_model"] == ""
    assert needed_keys({**DEFAULTS, "speech": (OFF, "")}) == ["OpenAI"]
    assert settings_for(DEFAULTS, {"OpenAI": "o"})["stt_model"] == "openai/whisper-1"


def test_config_toml_has_a_default_for_every_model_the_code_asks_for():
    from vectorstore import load_config

    config = load_config()
    for key in (
        "embedding_model",
        "llm_model",
        "reranker_model",
        "guard_model",
        "judge_model",
        "stt_model",
    ):
        assert config[key], key
    assert "retry_model" in config  # empty = use llm_model


def test_cohere_is_offered_for_reranking_only_and_makes_the_reranker_a_cross_encoder():
    from providers import PROVIDERS, is_cross_encoder, models_of

    reranker = next(c for c in COMPONENTS if c.key == "reranker")
    assert "Cohere" in providers_for(reranker.kind, reranker.also)
    assert "Cohere" not in providers_for("chat")  # not for answers, guard or judge
    assert models_of(reranker, "Cohere")[0] == "rerank-v3.5"
    assert models_of(reranker, "OpenAI") == PROVIDERS["OpenAI"].chat
    assert is_cross_encoder("Cohere") and not is_cross_encoder("OpenAI")

    settings = settings_for(
        {**DEFAULTS, "reranker": ("Cohere", "rerank-v3.5")},
        {"OpenAI": "o", "Cohere": "c"},
    )
    assert settings["reranker_type"] == "cross_encoder"
    assert settings["cross_encoder_model"] == "cohere/rerank-v3.5"
    assert "cross_encoder_model" not in settings_for(
        DEFAULTS, {"OpenAI": "o"}
    )  # config.toml's default stays
    assert settings["api_keys"] == {"openai": "o", "cohere": "c"}
    assert needed_keys({**DEFAULTS, "reranker": ("Cohere", "rerank-v3.5")}) == [
        "OpenAI",
        "Cohere",
    ]


def test_the_default_reranker_is_still_the_llm_one():
    settings = settings_for(DEFAULTS, {"OpenAI": "o"})
    assert settings["reranker_type"] == "llm"
    assert settings["reranker_model"] == "openai/gpt-4o-mini"
