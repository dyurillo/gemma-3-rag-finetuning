# gemma-3-rag-finetuning — LoRA fine-tune of Gemma 3 1B for RAG with verbatim citations

A rank-16 LoRA adapter, fine-tuned in **PyTorch/PEFT**, that teaches
`google/gemma-3-1b-it` to (1) answer a question from the four retrieved
chunks, (2) back every answer with a verbatim
`Quote: "<sentence>" [doc-id]` line, and (3) abstain with a fixed line
when the chunks don't contain the answer. The adapter is then loaded onto
a vLLM server as a low-rank weight update, and this repo measures the
serving overhead.

Answer *quality* is scored by the companion
[rag-eval-harness repository](https://github.com/dyurillo/rag-eval-harness):
a 48-question RAG dataset with BM25 retrieval and four scorers (exact
match, LLM judge, retrieval, diagnostic quote format). This repository
trains and serves; it does not judge answers. Adapter weights and
training/serving run artifacts are generated outside the checkout and are
not committed here. Answer-quality results are produced by the companion
evaluation harness.

## Results

Numbers below are from the final run batch of the harness (n = 48
questions, judge `Qwen/Qwen3.8-27B-FP8` passing at ≥ 4/5, temperature 0,
0 errors). The full matrix, quote-compliance detail and failure analysis
live in the harness repo's README.

**Quality — overall pass rate, 48 questions**

| model | no retrieval | k = 1 | k = 2 | k = 4 |
|---|---|---|---|---|
| `google/gemma-3-1b-it` (base) | 7 | 25 | 28 | 30 |
| + LoRA `rag-lora-1b` (sentence targets) | 6 | 26 | 25 | 27 |

Per-run outputs: [rag-eval-harness/results](https://github.com/dyurillo/rag-eval-harness/tree/main/results).

The adapter's pass rate trails the base in three of four cells and is
one case ahead at k = 1 (26 vs 25): k = 4 is 27 vs 30, k = 2 is 25 vs
28, and no retrieval is 6 vs 7. The ±1–2 run-to-run drift is about the
size of the 3-case gap at k = 4, and since this is a small, paired
48-question evaluation, treat the difference as descriptive; no
inferential significance claim is made here. What it *did* change
(k = 4):

| metric | base | + LoRA |
|---|---|---|
| verbatim quote compliance (all 48 cases) | 14/48 (29%) | 45/48 (94%) |
| verbatim quote compliance (answered questions only) | 3/37 (8%) | 32/35 (91%) |
| unanswerable abstention (n = 6) | 6/6 | 6/6 |

- **Format** — the base model skips the quote line, quotes a sentence it
  omits the doc id for, or emits literal `[doc-id]`/`<sentence>`
  placeholders; the adapter produces the required quote line in
  essentially every answer (45/48 overall, 32/35 of its answered
  questions, at k = 4).
- **Abstention** — 6/6 with retrieval at k = 4 for both models. With less
  context both fall for near-miss traps (the adapter 4/6 at k = 2);
  closed-book, both deterministically answer "AWS" to the cloud-provider
  question — a parametric-knowledge leak the 20% abstention mix doesn't
  fully suppress.
- **Accuracy** — what the adapter loses is reading comprehension, not
  format: it recites the first sentence or first table row of a chunk
  instead of localizing the answer fact.

**Iteration (span → sentence targets).** The first adapter was trained on
bare SQuAD answer spans and learned to answer in fragments. Its targets
were changed to the full sentence containing the span — plus the next
sentence for why/how/and questions, so two-part answers come out complete —
and the retrained adapter is the `rag-lora-1b` row above. The span-target
adapter itself was not run through the harness, so it has no row in the
table. That loop (train → grade → inspect failures → change the targets →
retrain) is the point of the project.

**Training loss** — curves and run metadata are generated locally during
training and are not committed to this repository.

## Failure taxonomy (from the harness)

- **Recite-from-top** — the 1B model can't always do the second pass of
  "find the relevant line": it recites the chunk's opening — the
  rotation-structure sentences instead of "Tuesdays 10:00 Central", the
  wrong row of the SLA table, the whole pipeline instead of the two
  carrier names.
- **Near-miss hallucination** — when the context mentions an adjacent
  fact, it answers with that (region name → cloud provider, IC rotation →
  IC stipend, 2020 pilot shipment → 2025 shipment count). More frequent
  with less context: 4/6 abstentions at k = 2, 6/6 at k = 4.
- **Over-abstention** — declines an answer that is in the retrieved
  context (7 of the 42 answerables at k = 4, versus 5 for the base).
- **Irreducible (base fails too)** — two-hop multi-document questions,
  derived arithmetic (excursion timing = 3 readings × 90 s), the
  wrong-service trap, and retrieval misses at low k.

## Training data

- **500 SQuAD v2 examples** from `rajpurkar/squad_v2` (CC-BY-SA-4.0):
  **400 answerable, 75 unanswerable, 25 no-context** (~20% of targets
  abstain).
- Each prompt is the **gold passage + 3 distractor chunks** (K=4),
  rendered with the harness's *exact* RAG prompt (copied verbatim by
  `scripts/prepare_data.py`).
- Answerable targets are two lines: the full sentence containing the
  answer span (plus the next sentence for why/how/and questions, so
  two-part answers come out complete — bare SQuAD spans trained the model
  to answer in fragments), then the verbatim quote line
  `Quote: "<sentence>" [doc-id]` (the span's sentence, tagged with the
  gold chunk's doc id). Unanswerable and no-context targets are the
  single abstention line, with no quote.
- Train on **public** SQuAD v2; the harness tests on an **unseen corpus**,
  so the results measure generalization, not memorization.
- The **last 50 rows** are held out for validation loss and for the
  serving workload; the model trains on the other 450.

## Method

- **Model:** `google/gemma-3-1b-it`, bf16, one GPU, gradient checkpointing
  (`use_reentrant=False`).
- **Adapter:** LoRA **r=16** (≤ the server's max rank 16), **alpha=32**,
  **dropout=0.05**. `train.py` uses a regex to target `q_proj`, `k_proj`,
  `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, excluding
  vision/audio encoder modules; it does not target every linear layer.
- **Training:** 3 epochs, effective batch **16** (4 × 4 grad-accum), lr
  **2e-4**, **cosine** schedule with **3% warmup**, max length 2048, seed
  42 — TRL `SFTTrainer` on conversational prompt/completion rows,
  **loss on the completion only**.
- **Serving benchmark:** every request is forced to a **fixed 64-token
  output** (`temperature=0`, `max_tokens=64`, `ignore_eos=True`), so base
  vs. adapter differ only by LoRA serving overhead. Measured at
  **concurrency 1 and 16**, 200 requests each, with a short warmup per
  level. `--base-model` selects the base (default
  `google/gemma-3-1b-it`).

## Reproduce

For the optional Nix/direnv environment, copy `.envrc.example` to `.envrc`,
review the defaults, then run `direnv allow`. Keep machine-specific values
and credentials in the ignored `.env` file, never in the example. You can
also set the environment variables manually.

Full training requires a CUDA GPU with bf16 support and about 24 GB of free
memory, plus access to download the base model. Before running, set
`DATASET_DIR`, `CHECKPOINT_DIR`, and `BENCHMARK_DIR` to writable locations;
`prepare_data.py` writes the generated dataset under `DATASET_DIR`.

`[admin]` = needs the admin account (starts/stops the vLLM containers).
`[gpu]` = needs a free GPU (the training user sets `CUDA_VISIBLE_DEVICES`).
The `systemctl` service names below are specific to the author's local
NixOS/Podman setup and are not provided by this repository. On another
machine, start equivalent vLLM servers yourself; the benchmark script expects
a compatible server at `http://localhost:8001` and the adapter mounted under
`/adapters`.

```bash
# 1. [admin] free the training GPU by stopping the vLLM container on it
sudo systemctl stop podman-vllm

# 2. [gpu] install dependencies, prepare data, sanity-check, then train
uv sync --locked
uv run python scripts/prepare_data.py  # downloads SQuAD v2 and writes training data
uv run python train.py --dry-run       # CPU-only data + masking check
uv run python train.py                 # add --overwrite to retrain over it

# 3. [admin] start the LoRA-enabled bench server on :8001
sudo systemctl start podman-vllm-bench

# 4. benchmark base, then the adapter (loads /adapters/rag-lora-1b)
uv run python serve_and_bench.py --tag base
uv run python serve_and_bench.py --tag lora --adapter-name rag-lora-1b
uv run python serve_and_bench.py --compare <base>.json <lora>.json

# 5. keep the adapter loaded for the eval harness, then run the harness
uv run python serve_and_bench.py --adapter-name rag-lora-1b --load-only
#    -> point rag-eval-harness at http://localhost:8001/v1, model=rag-lora-1b

# 6. [admin] stop the bench server when done
sudo systemctl stop podman-vllm-bench
```

## Limitations

- The eval set is only **48 questions**, so each cell has wide error bars
  (±1–2 between runs of the same cell).
- The judge is a single `Qwen/Qwen3.8-27B-FP8` model; rubric noise is the
  dominant variance source.
- SQuAD v2's "unanswerable" labels contain **some noise**.
- Closed-book, both base and adapter deterministically answer "AWS" to
  the near-miss cloud-provider question — parametric knowledge the
  abstention targets don't fully suppress.
- Fine-tuning a small model on a 20% abstention mix can push it toward
  over-abstention on answerable questions; the harness results track this
  (answerable cases where the model declines to answer).

## Data & credit

- Source data: **`rajpurkar/squad_v2`** (train split), licensed
  **CC-BY-SA-4.0**.
