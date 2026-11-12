"""Answer a question from given chunks with the generator's own prompt (no guard, cache or chat memory).

The evals use it so they measure the generator on its own: retrieval, then prompt, then the model.
"""

from langchain_core.documents import Document


def answer_from(generator, question: str, chunks: list[Document]) -> str:
    prompt = generator.build_prompt(question, chunks)
    return str(generator.retriever.llm.invoke(prompt).content).strip()
