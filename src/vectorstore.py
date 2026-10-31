"""Vector store: build the Chroma store in ./vs from the PDFs in ./documents.

Run once (and again after adding PDFs or changing the embedding model):
    uv run python src/vectorstore.py            # build if empty
    uv run python src/vectorstore.py --rebuild  # wipe and build again
"""

import json
import re
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

from chunking import ChunkingPipeline
from models import make_embeddings

# The project folder (RAGForge/), so paths work from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


BEST_CONFIG_FILE = PROJECT_ROOT / "best_config.json"


def load_config() -> dict:
    """Read config.toml, then lay the settings eval/find_best.py found best on top of it.
    Folder paths become absolute, relative to the project. Also reads API keys from ./.env."""
    load_dotenv(PROJECT_ROOT / ".env")
    with open(PROJECT_ROOT / "config.toml", "rb") as f:
        config = tomllib.load(f)
    with open(PROJECT_ROOT / "providers.toml", "rb") as f:
        config["providers"] = tomllib.load(
            f
        )  # the models and prices of the cloud providers
    if BEST_CONFIG_FILE.exists():
        config.update(json.loads(BEST_CONFIG_FILE.read_text())["settings"])
    config["data_dir"] = PROJECT_ROOT / config["data_dir"]
    config["vs_dir"] = PROJECT_ROOT / config["vs_dir"]
    return config


class VectorStore:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

        # The chunking strategy decides which collection (and so which chunks) we use.
        self.pipeline = ChunkingPipeline.from_config(self.config)
        # and the embedding model, because vectors from different models can't be mixed.
        embedder = re.sub(r"[^A-Za-z0-9]+", "-", self.config["embedding_model"]).strip(
            "-"
        )
        collection = f"{self.config['collection_name']}_{self.pipeline.name}_{embedder}"

        # The embedding model comes from config.toml.
        embeddings = make_embeddings(self.config)
        # Opening Chroma with persist_directory reads/writes ./vs on disk.
        # The model name is saved with the collection when it is first created,
        # so the Retriever can check it later.
        self.store = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=str(self.config["vs_dir"]),
            collection_metadata={"embedding_model": self.config["embedding_model"]},
        )

    def count(self) -> int:
        return self.store._collection.count()

    def built_with(self) -> str | None:
        """The embedding model the stored vectors were made with."""
        metadata = self.store._collection.metadata or {}
        model = metadata.get("embedding_model")
        return str(model) if model else None

    # BUILD ---- chunk the PDFs (chunking.py) and store their embeddings in Chroma
    def build(self, rebuild: bool = False) -> None:
        if self.count() > 0 and not rebuild:
            print(f"Store already has {self.count()} chunks. Use --rebuild to redo it.")
            return
        if rebuild:
            self.store.reset_collection()

        chunks = self.pipeline.chunks()
        if not chunks:
            print(f"No PDFs found in '{self.config['data_dir']}' - store is empty.")
            return

        self.store.add_documents(chunks, ids=[str(c.id) for c in chunks])
        print(
            f"Indexed {len(chunks)} '{self.pipeline.name}' chunks "
            f"with '{self.config['embedding_model']}' into {self.config['vs_dir']}"
        )


# BUILD IT ---- python src/vectorstore.py [--rebuild]
if __name__ == "__main__":
    VectorStore().build(rebuild="--rebuild" in sys.argv)
