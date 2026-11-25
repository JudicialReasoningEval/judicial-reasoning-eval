# Evaluating LLM-Generated Judicial Reasoning: Evidence from Italian Supreme Court Decisions

Code for the EACL 2027 paper by Simone Adobati, Federico Lorenzo Bruno and
Lorenzo Malandri (University of Milano-Bicocca).

Eleven open-weight LLMs draft the *Considerato in diritto* (the court's legal
reasoning) and the dispositive of Italian Supreme Court (Corte di Cassazione)
judgments in corruption cases (Art. 318–322 c.p.), and five metrics compare each
draft with what the court wrote:

| | Metric | Role |
|---|---|---|
| M1 | article-citation agreement (weighted and macro F1) | composite |
| M2 | premise composition (major premise, minor premise, conclusion) | composite |
| M3 | semantic alignment per premise (soft-cosine F1, Qwen3-Embedding-8B) | composite |
| M4 | dispositive agreement (Cohen's κ, gate κ ≥ 0.40) | gate |
| M5 | faithfulness to the case record (panel groundedness, gate ≥ 0.90) | gate |

This repository holds everything needed to run the study end to end: corpus
acquisition and conversion, the split, generation, the three-model labelling
and judging panel, the metrics, the statistics and the auxiliary analyses. The
judgments and the model outputs are **not** redistributed (see below).

---

## Why the data is not included

The 181 judgments are public, but the Court publishes them **unredacted**: they
name the parties, their counsel and the deciding judges, and 134 of the 181 give
a party's full date *and* place of birth alongside the offences charged. Nothing
is masked at source.

Because each model is given the *Ritenuto in fatto*, those names recur in the
generated drafts as well, so releasing the model outputs would disclose the same
personal data as releasing the judgments. This repository therefore ships the
code and a manifest, and leaves the documents with the Court that publishes them.

[`data/corpus_manifest.json`](data/corpus_manifest.json) identifies all 181
judgments: each one's entry in the Court's register, its retrieval URL, and the
SHA-256 of both the source PDF and the extracted Markdown.

```bash
python tools/fetch_corpus.py                    # download + verify the PDFs
python -m corpus.convert_sentenze               # convert  (GPU: see below)
python -m corpus.filter_sentenze                # -> sentenze_md_filtered/
python tools/fetch_corpus.py --verify-markdown  # check against the manifest
```

**Converting needs a GPU.** 149 of the 181 judgments are image-scanned: a text
extractor recovers only their ~130-character header line, and the corpus was
built by OCR-ing them with a vision model (GLM-OCR via vLLM). That is why
conversion is `corpus/convert_sentenze.py`, which runs text extraction first and
then OCRs anything under 1,000 characters.

Do not skip `--verify-markdown`. It compares every converted file against the
manifest and exits non-zero on any mismatch, distinguishing files that are still
thin (OCR has not run) from files that differ in content. **OCR is not
bit-reproducible** across model versions and decoding settings, so a rebuilt
corpus can legitimately differ from the one the reported numbers were computed
on; metrics computed on differing text are not comparable to the published
figures.

## Layout

```
pipeline/         split, prompts and generation
  cassazione_split.py   stratified 100/81 test/pool split (seed 42)
  case_sections.py      facts / reasoning / dispositive split, companion judgments, prompt check
  fewshot_selector.py   article-matched k=2 exemplar selection
  model_backends.py     model profiles (checkpoint, quantisation, context, decoding) + prompt rendering
  inference_judge.py    generate Considerato + P.Q.M. for the evaluated cases
  baselines.py          random / majority / keyword article baselines
  run_panel_*.py        panel drivers (premise labels, M5)
  run_frontier_validation.py  reference-panel check (proprietary judges via OpenRouter)
metrics/          the five metrics and the statistics
  score_articles.py     M1  article-citation F1
  m2_structural.py      M2  1-JSD of premise proportions + presence F1
  m3_semantic.py        M3  soft-cosine F1 per premise
  embeddings.py         embedding client (local or OpenAI-compatible endpoint)
  m4_dispositive.py     M4  deterministic P.Q.M. extractor + Cohen's kappa
  m5_faithfulness.py    M5  groundedness of citations and facts
  premise_labeler.py    sentence-level premise labelling (one labeller)
  consensus_labels.py   2-of-3 panel merge + Fleiss kappa
  consensus_m5.py       2-of-3 merge of the M5 judges + Fleiss kappa
  evaluate.py           per-model orchestrator -> metrics.json
  compare_models.py     paired permutation tests, BH-FDR, metric orthogonality
  seed_variance.py      across-seed stability (seeds 42-46)
  sensitivity.py        composite-weight simplex + M2-split sweep
  frontier_agreement.py open-weight panel vs reference panel
  stats_utils.py        bootstrap CIs, paired permutation, BH-FDR
corpus/           acquisition and preprocessing
  scraper.py            query the Court's SOLR API, download PDFs
  convert_sentenze.py   PDF -> Markdown, with GLM-OCR fallback for scans
  filter_sentenze.py    keep genuine Art. 318-322 c.p. criminal judgments
  benchmark_ocr.py      OCR benchmark against digital-text PDFs
analysis/         auxiliary analyses reported in the paper
  contamination_audit.py  word-12-gram overlap, reasoning trace vs published Considerato
  release_date_split.py   scores on judgments published before vs after each model's release
  fewshot_leakage.py      citation provenance: input, exemplars, court, or nowhere
  premise_order.py        premise order in the court's own Considerato
  reasoning_language.py   language of the reasoning traces (Italian vs English)
  negation_probe.py       M3 polarity stress test + paired controls
  m3_polarity_audit.py    annotation sheet for real M3-selected pairs
cost_tracking/    per-call API metering against public list prices
splits/           test_100, fewshot_pool_81, evaluated_91, frontier_validation_20
data/             corpus manifest (identity, URLs, checksums)
tools/            corpus fetcher, manifest builder
tests/            regression tests
```

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

M1, M2, M4 and all statistics run on CPU. M3 needs a GPU for
Qwen3-Embedding-8B (or an OpenAI-compatible embedding endpoint, below).
Generation and the panel need vLLM and a GPU sized for the model profile.

## Running the study

All commands run from the repository root and write under `results/`.

### 1. Corpus and split

Fetch and convert the judgments (above), then rebuild the split. It is seeded,
so it reproduces `splits/test_100.json` and `splits/fewshot_pool_81.json`
exactly:

```bash
python -m pipeline.cassazione_split
```

### 2. Generation

The model receives the court header and the *Ritenuto in fatto* only; the
*Considerato* and the *P.Q.M.* are withheld as the gold standard.
[`pipeline/case_sections.py`](pipeline/case_sections.py) defines the split:

- test judgments whose facts cannot be separated from the reasoning are set
  aside, leaving **91 evaluated cases** out of 100;
- exemplars without a separable reasoning section are dropped (79 of 81 remain),
  and a companion judgment of the same proceedings is never offered as an
  exemplar for its test case;
- every prompt is checked before the model is loaded for passages of the test
  court's own reasoning: two or more of three probe passages abort the run, a
  single one (shared legal boilerplate) is logged as a warning.

[`splits/evaluated_91.json`](splits/evaluated_91.json) lists the evaluated
cases, the usable pool and the companion pairs; a test checks that the code
reproduces it once the corpus is fetched.

```bash
python -m pipeline.inference_judge --profile gpt-oss-20b --dry-run        # build + check every prompt
python -m pipeline.inference_judge --profile gpt-oss-20b --n-samples 5    # seeds 42-46
python -m pipeline.inference_judge --profile gpt-oss-20b --n-samples 5 --k 0 \
    --results-dir results/_zeroshot-gpt-oss-20b                            # zero-shot control
```

`PROFILES` in [`pipeline/model_backends.py`](pipeline/model_backends.py) holds
the exact checkpoint, quantisation, tensor-parallel size, context length,
generation budget and reasoning settings of each reported model (temperature
1.0 throughout). The eleven reported profiles are `qwen3.5-397b-a17b`,
`qwen3.5-122b-a10b`, `minimax-m2.7-nvfp4`, `nemotron-3-super-120b`,
`gpt-oss-120b`, `mistral-medium-3.5-128b`, `nemotron-3-nano-30b-a3b`,
`qwen3.5-9b`, `gpt-oss-20b`, `qwen3.6-35b-a3b` and `minicpm5-1b`. Each run
writes `inference_results.jsonl` (455 records for 91 cases × 5 seeds),
`fewshot_used.jsonl` and `run_manifest.json`. Nine of the eleven models ran on
rented Blackwell-class GPUs: one seed of those nine takes about 5.4 GPU-hours.
Qwen3.5-9B and GPT-OSS-20B fit a single workstation GPU.

### 3. Premise labels (three-model panel)

Each sentence of the court's *Considerato* and of each model's is labelled with
the premises it carries (major premise, minor premise, conclusion) by three
open-weight labellers: `gpt-oss-120b`, `Nemotron-3-Super-120B` and
`Qwen3.6-27B`, merged by strict majority (2 of 3). The exact serving arguments
and reasoning settings of each labeller are in `LABELERS` in
[`pipeline/run_panel_labeling.py`](pipeline/run_panel_labeling.py). Serve each
labeller on any OpenAI-compatible endpoint and label both sides into
`results/panel_labels/`:

```bash
vllm serve openai/gpt-oss-120b --max-model-len 32768 --port 8000
export LABELER_BASE_URL=http://localhost:8000/v1 LABELER_MODEL=openai/gpt-oss-120b
export REASONING_EFFORT=low                     # Nemotron / Qwen: REASONING_EFFORT= and
                                                # LABELER_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
TAG=gpt-oss-120b
python -m metrics.premise_labeler --source judge \
    --out results/panel_labels/sillogismo_judge.$TAG.jsonl
python -m metrics.premise_labeler --source llm --results-dir results/<model> \
    --out results/panel_labels/<model>.sillogismo_labels.$TAG.jsonl
```

After all three labellers:

```bash
python -m metrics.consensus_labels      # majority labels + Fleiss kappa -> results/_panel_agreement.json
```

### 4. Faithfulness (M5)

The same three models judge each seed-42 output against the complete facts
section of its case (where no facts heading is recognised, two of the 91
cases, the full judgment text). Serve each judge with `--max-model-len 65536` (complete
facts plus a complete *Considerato*), then:

```bash
python -m metrics.m5_faithfulness --results-dir results/<model> \
    --out results/panel_m5/<model>.m5.$TAG.jsonl
python -m metrics.consensus_m5          # after all three judges
```

### 5. Metrics and statistics

```bash
python -m metrics.evaluate --results-dir results/<model>   # M1-M4, reads M5 -> metrics.json
python -m metrics.seed_variance
python -m metrics.compare_models --models <model> <model> ...
python -m metrics.sensitivity
```

M3 loads Qwen3-Embedding-8B locally, or uses an OpenAI-compatible
`/v1/embeddings` endpoint when `EMBEDDING_BASE_URL` is set. The embedding model
and a served judge do not fit together on one 32 GB GPU: label and judge first,
stop the server, then evaluate. `--skip-m3` gives a GPU-free pass.

### 6. Auxiliary analyses

```bash
python -m pipeline.baselines --baseline keyword    # on the 91 cases, from the facts the models receive
python -m analysis.contamination_audit             # word-12-gram overlap with the published Considerato
python -m analysis.release_date_split              # needs metrics.evaluate outputs for the 11 models
python -m analysis.fewshot_leakage                 # citation provenance
python -m analysis.premise_order                   # court side only
python -m analysis.reasoning_language
python -m analysis.negation_probe                  # M3 polarity stress test (needs the embedding model)
```

### 7. Reference panel

A 20-case subset stratified by charged article
([`splits/frontier_validation_20.json`](splits/frontier_validation_20.json)) is
relabelled by a reference panel of three proprietary LLMs (Claude Sonnet 5,
Gemini 3.6 Flash, GLM 5.2) through OpenRouter, under the same rubric, prompt,
chunking and 2-of-3 majority. This checks whether the premise labels depend on
the open-weight panel that produced them; both panels are LLM judges, so their
agreement shows convergence, not correctness. The models and their reasoning
settings are in `JUDGES` in
[`pipeline/run_frontier_validation.py`](pipeline/run_frontier_validation.py).
This step needs an OpenRouter account (`OPENROUTER_API_KEY`) and costs money;
`--dry-run` prices it first, and `--budget` stops the run once OpenRouter's
reported charges reach the limit.

```bash
python -m pipeline.run_frontier_validation --task A --dry-run
python -m pipeline.run_frontier_validation --task A --budget <USD>
python -m metrics.frontier_agreement --premise-only
```

## A note on GPU provisioning

The reported runs used rented GPUs from a commodity marketplace. That
provisioning code is account-specific and is not included. The
`pipeline/run_panel_*.py` drivers are shipped so that their roster, serving
arguments, budgets and resume logic can be audited; they exit with an
explanation if asked to rent. Serve the models yourself and drive the metric
modules directly, as in steps 3 and 4.

## Tests

```bash
python -m pytest tests/ -q
```

Tests that re-derive the split from the judgments need the corpus and are
skipped until it is fetched. Everything else runs without data.

## Citation

```bibtex
@inproceedings{adobati2027evaluating,
  title     = {Evaluating {LLM}-Generated Judicial Reasoning: Evidence from {I}talian Supreme Court Decisions},
  author    = {Adobati, Simone and Bruno, Federico Lorenzo and Malandri, Lorenzo},
  booktitle = {Proceedings of the 2027 Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year      = {2027}
}
```

## Licence

Code: MIT ([`LICENSE`](LICENSE)). The licence covers the source code only. The
judgments are public acts of the Italian State, published by the Corte di
Cassazione at italgiure.giustizia.it; they are not covered by it and are not
redistributed here (see [`data/corpus_manifest.json`](data/corpus_manifest.json)).
