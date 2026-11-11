# RAGForge: RAG Evaluation

`main` has one evaluation: the **RAG triad**. It needs no golden questions and no reference answers. A judge model scores each answer from the question, the retrieved chunks and the answer, so it works on any document a user uploads.

## The three scores

| Triad score | Link it checks | What it measures |
| --- | --- | --- |
| Context Relevance | question to retrieved chunks | How much of the retrieved text is relevant to the question (points at the retriever and re-ranker) |
| Groundedness | chunks to answer | Whether the answer's claims contradict the retrieved text (points at the generator) |
| Answer Relevance | question to answer | Whether the answer addresses the question |

Each score is 0 to 1, with the judge's reasons. They are computed with DeepEval's `ContextualRelevancyMetric`, `FaithfulnessMetric` and `AnswerRelevancyMetric`.

## Where it runs

```
 question -> input guard -> retriever -> re-ranker -> generator -> answer + citations
                                                                      |
                                       RAG TRIAD (judge model) <------+  scores the answer
```

- **In the web app:** every answer is scored in the background. The three scores appear under the answer with the reasons, and the sidebar shows the latest and the session average.
- **Command line:** `uv run python eval/rag_triad_eval.py "What rank r does LoRA use?"`

## Reading the scores

The triad is a signal to investigate, not a verdict on correctness. A judge is noisy, so compare setups under one judge. The judge should be a different model from the answer model, so a model never grades its own answers. The judge is chosen in the app's sidebar (OpenAI, Anthropic or Gemini, with your own key).
