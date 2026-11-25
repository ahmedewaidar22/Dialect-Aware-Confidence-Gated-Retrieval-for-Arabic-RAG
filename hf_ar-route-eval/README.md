---
license: cc-by-4.0
language:
- ar
task_categories:
- question-answering
- text-retrieval
tags:
- arabic
- egyptian-arabic
- dialect
- retrieval-augmented-generation
- rag
- adaptive-retrieval
pretty_name: AR-Route Evaluation Data (Egyptian/MSA Arabic RAG)
size_categories:
- 1K<n<10K
configs:
- config_name: test_egy_msa
  data_files: test_egy_msa/egy_test.jsonl
- config_name: confirmatory
  data_files: confirmatory/confirm_test.jsonl
---

# AR-Route Evaluation Data

Evaluation data for the paper **"Dialect-Aware, Confidence-Gated Retrieval for Arabic RAG: When Learned Complexity Routing Fails to Transfer"** (Ahmed Dewaidar, Mohamed Shaheen, Eman AbdelAziz; Arab Academy for Science, Technology and Maritime Transport).

It contains parallel Egyptian Arabic / Modern Standard Arabic (MSA) questions with reference answers and gold passages, the frozen confirmatory test set, per-question outcomes for every compared system, and the split-integrity audit.

- Code: https://github.com/ahmedewaidar22/Dialect-Aware-Confidence-Gated-Retrieval-for-Arabic-RAG
- Models and retrieval corpus: https://huggingface.co/ahmedewaidar22/ar-route

## Contents

| Folder | File | Rows | Description |
|---|---|---|---|
| `test_egy_msa/` | `egy_test.jsonl` | 1,667 | Parallel Egyptian/MSA question pairs: the 1,000-question end-to-end test set (500 native ArDQA, 200 pilot, 300 held-out) and 667 native ArDQA questions used as a second held-out check. The `set` field gives the part. |
| | `heldout_ids.json` | | IDs of the 300 held-out questions removed from router training. |
| | `review.csv`, `review_map.json` | 130 | Native-speaker review of flagged Egyptian questions (keep / fix decisions). |
| | `summary.txt` | | Construction summary. |
| `confirmatory/` | `confirm_test.jsonl` | 1,027 | Frozen confirmatory test: unused ArabicaQA questions rewritten into Egyptian Arabic, each with its MSA original. |
| | `protocol.txt` | | Hypotheses and decision rules fixed before the run. |
| | `review.csv`, `confirm_test.jsonl.audit.csv` | | Quality review and automatic audit of the rewritten questions. |
| `results/` | `per_question.csv` | 1,000 | Correctness (1/0) and generator cost (US$) per question for every system on the end-to-end test. |
| | `confirmatory_results.jsonl` | 1,027 | Per-question outputs of the three confirmatory systems. |
| | `training_outcome_labels.jsonl` | 1,955 | Outcome of every retrieval strategy (C0/C1/C2) on every router training question. |
| `audit/` | `provenance_flags.csv` | 839 | Every evaluation question flagged as similar to a question used to build a system, with a manual same-question (y/n) decision. |
| | `provenance_report.txt` | | Summary of the split-integrity audit. |
| | `freeze_manifest.json` | | SHA-256 hashes of code, models, thresholds and prompts at the freeze. |

### Main fields

`egy_test.jsonl`: `id`, `set`, `q_egy` (Egyptian question), `q_msa` (MSA question), `answer`, `refs` (accepted reference answers), `gold_pids` (gold passage IDs in the corpus), `source`, `domain`, `qtype`, `label_msa`, `flags`, `origin_id`, `egy_markers`, `review`.

`confirm_test.jsonl`: `id`, `set`, `q_egy`, `q_msa`, `answer`, `refs`, `gold_pids`, `origin_id`, `flags`, `tries`.

Passage IDs refer to the retrieval corpus in the `ar-route` model repository.

## Headline result (frozen confirmatory test, 1,027 questions, Egyptian input, gpt-4.1-mini)

| System | Accuracy | US$ per 1,000 questions |
|---|---|---|
| Standard RAG | 74.3% | 0.606 |
| Always iterative retrieval (C2) | 78.9% | 0.963 |
| **Proposed model** | **76.8%** | **0.566** |

The proposed model beat standard RAG by +2.5 points (McNemar p = 0.012) and stayed within the 5-point non-inferiority margin of always-C2 at 41% lower generator cost. Results on the end-to-end test set are partly exploratory; see the paper.

## How the data were made

- **ArDQA** questions are natively written Egyptian/MSA pairs.
- **Pilot** questions were written with LLM help; **held-out** and **confirmatory** questions were rewritten from MSA into Egyptian Arabic by gpt-4.1-mini. Each non-native question was checked by one native speaker.
- Correctness was graded by an LLM judge (gpt-4.1-mini) after a normalized string match; the judge was validated against human labels on a 200-question sample (see paper).

## Limitations

Only Egyptian Arabic is covered. Most non-ArDQA Egyptian questions are machine-rewritten rather than natively authored, and each was reviewed by a single annotator. The confirmatory judge verdicts were not human-checked. Some evaluation questions resemble questions in training data; all such cases are listed in `audit/provenance_flags.csv`, and the paper reports results with them removed.

## Sources and licenses

This dataset builds on:
- **ArDQA**: M. J. Althobaiti, "ArDQA: A parallel multidomain benchmark for cross-dialectal Arabic question answering," Research Square preprint, 2025.
- **ArabicaQA** (MIT license): A. Abdallah et al., "ArabicaQA: A comprehensive dataset for Arabic question answering," SIGIR 2024.

The authors' additions (Egyptian rewrites, labels, reviews, results and audit files) are released under CC BY 4.0. Content taken from the source datasets remains under the terms of those datasets; please cite them when you use it.

## Citation

```bibtex
@misc{dewaidar2026arroute,
  title  = {Dialect-Aware, Confidence-Gated Retrieval for Arabic RAG: When Learned Complexity Routing Fails to Transfer},
  author = {Dewaidar, Ahmed and Shaheen, Mohamed and AbdelAziz, Eman},
  year   = {2026},
  note   = {Under review}
}
```
