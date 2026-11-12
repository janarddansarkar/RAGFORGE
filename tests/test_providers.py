"""The provider choices: every component gets its own provider and model, and the user's keys."""

from providers import COMPONENTS, OFF, needed_keys, providers_for, settings_for

DEFAULTS = {c.key: (c.provider, c.model) for c in COMPONENTS}


def test_each_component_can_use_a_different_provider():
    choices = {
        **DEFAULTS,
        "answer": ("Anthropic", "claude-sonnet-5-5"),
        "judge": ("Google Gemini", "gemini-2.0-flash"),
    }
    settings = settings_for(
        choices, {"OpenAI": "o", "Anthropic": "a", "Google Gemini": "g"}
    )
    assert settings["llm_model"] == "anthropic/claude-sonnet-5-5"
    assert settings["judge_model"] == "gemini/gemini-2.0-flash"
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
