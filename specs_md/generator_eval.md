# Generator Evaluation (`eval/generator_eval.py`)

This document explains what the generator evaluation measures, how each number is
calculated, and how to read it. The retriever has its own document, `retriever_eval.md`.

Run it (the runner builds the index first; a full run also does the other stages):

```bash
uv run python eval/run_provider_eval.py openai --stages generator          # retrieve, then answer (the real pipeline)
uv run python eval/run_provider_eval.py openai --stages generator,oracle   # also answer from the golden pages
uv run python eval/run_provider_eval.py openai --stages generator --limit 5   # first 5 questions (quick test)
```

The numbers are saved in `eval/results/<provider>_<date>_<time>.json`. `eval/generator_eval.py` can also be run on its
own (`--oracle`, `--limit`, `--generator ID`, `--judge ID`) once the vector store for the configured models exists;
it then writes `eval/generator_runs/<date>_<time>.json` and never overwrites a file. Set `OPENAI_API_KEY` (or the key of
the provider you use) first.

---

## 1. What is being evaluated

The **generator** (`src/generator.py`) takes a question and a set of text passages, and
writes an answer using only those passages. The evaluation asks:

> Is the answer true to the passages, on topic, and correct?

For every golden question the evaluation:

1. gets the passages (see section 2),
2. asks the generator to answer,
3. scores the answer on five metrics (section 4).

## 2. Two modes: realistic and oracle

| mode | passages given to the generator | what a low score means |
|---|---|---|
| **retrieved** (default) | the top `top_k` (5) chunks from the retriever | the retriever, the generator, or both |
| **oracle** (`--oracle`) | chunks from the golden pages, up to `top_k` | the generator (the retriever can't be the cause) |

Compare the two for the same generator:

- **Oracle is much better than retrieved:** retrieval is losing answers, so improve the
  retriever.
- **Oracle is no better than retrieved:** the generator is the bottleneck, so improve
  the generator (a better model or prompt).

## 3. The judge

Three of the metrics need a model to judge answers. This is the **judge**
(`judge_model` in `config.toml`, or `--judge`). Rules:

- The judge must be **a different model from the generator**, ideally a different family,
  so a model does not grade its own answers.
- It is a hosted model called through LiteLLM with your key, so it costs money. The default is `gpt-4o-mini`; the
  generator is `gpt-4o`.
- Judge replies are one word (`YES`, `NO`, `CORRECT`, `PARTLY`, `WRONG`). If the reply
  is unclear, it counts as the worst option for that question.
- Judges are **noisy**. Compare runs that used the **same judge**; scores from
  different judges are not comparable.

## 4. The metrics

Five metrics, each between 0 and 1 (higher is better). Each is computed **per question**
and then **averaged over all questions**.

### 4.1 Faithfulness

**What it means:** is every statement in the answer backed by the passages? Low
faithfulness means the model is making things up (hallucinating), even if the claim
happens to be true.

**How it is calculated:**

1. Split the answer into sentences.
2. For each sentence, ask the judge: *"Is the statement supported by the passages?
   YES or NO."*
3. Faithfulness = (number of YES) / (number of sentences).

**Special case:** if the answer is the refusal *"I can't find that in the documents"*,
faithfulness is 1.0, because it claims nothing.

### 4.2 Answer relevancy

**What it means:** does the answer stay on the topic of the question? A relevant answer
addresses what was asked, even if it is wrong or incomplete.

**How it is calculated** (the RAGAS approach):

1. Ask the judge to write the one question the answer responds to.
2. Embed that question and the real question with the embedding model (`embedding_model`).
3. Relevancy = cosine similarity between the two, with negative values set to 0.

**Reading it:** an answer that is off topic or a refusal produces an unrelated question
and scores low. A confident but **wrong** answer on the right topic still scores high,
so read relevancy together with correctness and faithfulness.

### 4.3 Answer correctness

**What it means:** does the answer agree with the golden (reference) answer?

**How it is calculated:** the judge sees the question, the reference and the answer,
and picks one word:

| verdict | score |
|---|---|
| `CORRECT` | 1.0 |
| `PARTLY` | 0.5 |
| `WRONG` | 0.0 |

This is the main quality number. It is the one to optimise when choosing a generator.

### 4.4 Semantic similarity

**What it means:** how close the answer is in meaning to the golden answer, without a
judge.

**How it is calculated:** embed the answer and the golden answer, and take the cosine
similarity of the two vectors. It needs no judge, so it is the only metric that stays
the same when you change the judge.

**Reading it:** it rewards similar wording and topic, so it moves less than correctness.
It cannot tell a correct answer from a fluent wrong one. Use it as a steady background
signal, not a verdict.

### 4.5 Context recall

**What it means:** do the passages contain what is needed for the golden answer? It
measures the **context**, not the generator. If it is low, the generator did not have
the information, so a bad answer is not its fault.

**How it is calculated:**

1. Split the golden answer into sentences.
2. For each, ask the judge: *"Is the statement supported by the passages?"*
3. Context recall = (number of YES) / (number of sentences).

**Reading it:** in **oracle** mode it is capped by how well the golden pages cover the
golden answer and how strict the judge is; it will not be 1.0.

### Also recorded

| field | meaning |
|---|---|
| `seconds_per_answer` | average time per question. In retrieved mode this includes retrieval and the generator; in oracle mode only the generator |

## 5. What is saved in each run file

```text
generator, judge        the models used
context                 "retrieved" or "oracle (golden pages)"
settings                embedding_model, reranker_model, default_method, top_k,
                        contextual, max_new_tokens
questions               number of questions scored
average                 the five metrics plus seconds_per_answer
rows                    per question: id, question, answer, reference, seconds, 5 scores
```

`rows` lets you read the actual weak answers, and `settings` shows exactly what
produced the numbers, so runs can be compared later.

## 6. Settings

In `config.toml`:

| setting | meaning |
|---|---|
| `llm_model` | the generator (a LiteLLM name such as `openai/gpt-4o`) |
| `judge_model` | the judge (a different model from the generator, e.g. `openai/gpt-4o-mini`) |
| `max_new_tokens` | longest answer the generator writes |
| `top_k` | how many chunks are given to the generator |
| `golden_file` | the question set (needs `golden_ans` for each question) |

## 7. How to read the results

- **Correctness is the headline;** faithfulness tells you about hallucination.
- **Compare oracle against retrieved** to see whether to fix retrieval or the generator.
- **Gaps of about 0.05 or less are noise.** There are 45 questions and a small judge.
- **Check the rows,** not just the averages. Open a few low-scoring answers and confirm
  the judge is right.
- **Compare only runs with the same judge and the same settings.**

## 8. Results

The saved OpenAI run (`gpt-4o` generator, `gpt-4o-mini` judge, 45 questions) is in `docs/EVALUATION_RESULTS.md`:

| context | faithfulness | answer relevancy | answer correctness | semantic similarity | context recall |
|---|---|---|---|---|---|
| retrieved chunks | 0.71 | 0.84 | 0.98 | 0.71 | 0.92 |
| golden pages (oracle) | 0.94 | 0.75 | 0.91 | 0.63 | 0.90 |

Correctness is already high, and faithfulness is much better when the generator is handed the right pages, so most of
the remaining gap comes from retrieval, not the answer model. Absolute scores depend on the judge, so only compare runs
with the same judge.

## 9. Limitations

- **Judges can be lenient on easy questions.** Faithfulness and context recall can come out near
  1.00 for every generator, so they separate weak models from strong ones poorly. Correctness
  and the judge-free semantic similarity are the more useful columns.
- **A small judge makes mistakes.** A weak judge can answer "NO" to every relevancy check, which is why relevancy
  uses the question-rewriting method instead of a yes/no verdict.
- **Sentence-level checks.** Faithfulness and context recall split text on `.`, `!` and
  `?`, so abbreviations and equations can split wrongly.
- **Golden answers are single references.** A correct answer worded differently can
  score `PARTLY`.
- **Small test set.** 45 questions from three papers.
- **Refusal handling.** A refusal gets faithfulness 1.0 but low correctness and
  relevancy, which is intended.

## 10. Related files

| file | role |
|---|---|
| `eval/run_provider_eval.py` | runs this evaluation (and the others) with one provider and saves the results |
| `eval/generator_eval.py` | the evaluation itself |
| `eval/answering.py` | answers a question from given chunks with the generator's own prompt |
| `src/generator.py` | the generator (prompt, answer from chunks) |
| `src/models.py` | the LiteLLM wrappers for the generator, judge and embedding models |
| `eval/results/` | one JSON file per saved run |
| `golden_data/rag_retriever_golden_dataset_45.json` | questions and golden answers |
