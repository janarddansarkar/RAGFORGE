"""Which Ollama models does RAGForge need? Used by setup.sh to pull the missing ones.

Run:  uv run python src/needed_models.py      # one model name per line

It reads the settings the app really uses (config.toml with best_config.json laid over it),
so the list is always right, even after you change a model in either file.
"""

from vectorstore import load_config


class NeededModels:
    """The Ollama models the current settings use."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

    def names(self) -> list[str]:
        wanted = [
            self.config["embedding_model"],
            self.config["llm_model"],
            self.config["guard_model"],
            self.config.get("retry_model"),  # "" means: use llm_model
            self.config["judge_model"],
        ]
        if self.config.get("reranker_type") == "llm":  # a cross-encoder comes from Hugging Face
            wanted.append(self.config["reranker_model"])
        names: list[str] = []
        for name in wanted:
            if name and name not in names:  # no blanks, no repeats, same order
                names.append(name)
        return names


if __name__ == "__main__":
    print("\n".join(NeededModels().names()))
