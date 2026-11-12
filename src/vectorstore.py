"""Vector store: chunk the PDFs in data_dir and keep their embeddings in a Chroma store (vs_dir).

The collection name includes the chunking settings and the embedding model, so changing either
builds a new store instead of mixing vectors.
"""

import re
import tomllib
from pathlib import Path

from langchain_chroma import Chroma

from chunking import ChunkingPipeline
from models import make_embeddings

# The project folder (RAGForge/), so paths work from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """Read config.toml. DocumentRAG adds the folders (data_dir, vs_dir) for each set of documents."""
    with open(PROJECT_ROOT / "config.toml", "rb") as f:
        config = tomllib.load(f)
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
    def build(self) -> None:
        if self.count() > 0:
            print(f"Store already has {self.count()} chunks.")
            return

        chunks = self.pipeline.chunks()
        if not chunks:
            print(f"No PDFs found in '{self.config['data_dir']}' - store is empty.")
            return

        self.store.add_documents(chunks, ids=[str(c.id) for c in chunks])
        print(
            f"Indexed {len(chunks)} '{self.pipeline.name}' chunks "
            f"with '{self.config['embedding_model']}' into {self.config['vs_dir']}"
        )
