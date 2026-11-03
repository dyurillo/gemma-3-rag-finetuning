"""LoRA fine-tune google/gemma-3-1b-it to answer from retrieved context (and abstain).

Trains a rank-16 LoRA adapter on the SQuAD-v2 RAG dataset written by
scripts/prepare_data.py, using TRL's conversational prompt/completion format
(loss on the completion only). The adapter is saved to $CHECKPOINT_DIR/rag-lora-1b
for serving on the vLLM stack (see serve_and_bench.py). Answer *quality* is
measured by a separate eval-harness repo, not here.

Run (as the training user with direnv loaded, on a GPU with ~24 GB free):
    uv run python train.py --dry-run   # CPU-only: data + tokenizer + masking check
    uv run python train.py             # full training run
    uv run python train.py --overwrite # allow replacing an existing adapter

Env: DATASET_DIR, CHECKPOINT_DIR, BENCHMARK_DIR (see .envrc).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets
import peft
import torch
import transformers
import trl
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerBase,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-3-1b-it"
DATA_SUBDIR = "squad-v2-rag"  # written by scripts/prepare_data.py

ADAPTER_NAME = "rag-lora-1b"
LORA_R = 16  # must stay <= 16, the vLLM server's max LoRA rank
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# LoRA sits on the attention/MLP projections only. The 1B base is text-only, so
# there are no encoders or per-layer embedding projections to avoid; the regex
# still excludes encoder-like modules if the base model ever changes.
# scripts/smoke_test.py imports these from here.
LORA_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
ENCODER_MARKERS = ("vision", "audio")
LORA_TARGET_REGEX = (
    rf"^(?!.*(?:{'|'.join(ENCODER_MARKERS)})).*\.(?:{'|'.join(LORA_PROJECTIONS)})$"
)

NUM_EPOCHS = 3
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4  # effective batch = 4 * 4 = 16
LEARNING_RATE = 2e-4
SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
MAX_LENGTH = 2048
SEED = 42
VAL_SIZE = 50  # the last 50 rows are the held-out validation split

MIN_FREE_GB = 24  # generous headroom for the training run; the 1B base itself is small
KINDS = ("answerable", "unanswerable", "no_context")

# Single source of truth for the chat-template/masking helpers and the LoRA
# target: scripts/smoke_test.py imports these from train.
Message = dict[str, str]
Example = dict[str, Any]


def special_tokens(tok: PreTrainedTokenizerBase) -> tuple[str, int]:
    """EOS string and pad id, narrowed from transformers' loose types for the type checker."""
    eos = tok.eos_token
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    assert isinstance(eos, str) and isinstance(pad_id, int), (
        "tokenizer lacks an EOS/pad token"
    )
    return eos, pad_id


def render(
    tok: PreTrainedTokenizerBase, messages: list[Message], add_generation_prompt: bool
) -> str:
    """The chat template as text: what vLLM builds from the harness's chat request."""
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    assert isinstance(text, str)
    return text


def encode_example(
    tok: PreTrainedTokenizerBase, ex: Example
) -> tuple[list[int], list[int]]:
    """Prompt ids (user turn + assistant header) and completion ids (answer + end-of-turn)."""
    prompt_text = render(tok, ex["prompt"], add_generation_prompt=True)
    full_text = render(tok, ex["prompt"] + ex["completion"], add_generation_prompt=False)
    assert full_text.startswith(prompt_text), (
        f"{ex['id']}: prompt is not a prefix of prompt+completion"
    )
    prompt_ids: list[int] = tok(prompt_text, add_special_tokens=False).input_ids
    completion_ids: list[int] = tok(
        full_text[len(prompt_text) :], add_special_tokens=False
    ).input_ids
    joint: list[int] = tok(full_text, add_special_tokens=False).input_ids
    assert joint == prompt_ids + completion_ids, (
        f"{ex['id']}: token boundary differs from joint tokenization"
    )
    return prompt_ids, completion_ids


def end_of_turn_id(tok: PreTrainedTokenizerBase) -> int:
    """The special token the chat template puts right after a model reply.

    Found from the template rather than assumed: for chat models it is often not
    tokenizer.eos_token, and it's the token the adapter must learn to emit.
    """
    marker = "SMOKE_PROBE_REPLY"
    probe = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": marker},
    ]
    text = render(tok, probe, add_generation_prompt=False)
    assert marker in text, "chat template dropped the assistant message"
    suffix_ids: list[int] = tok(
        text.split(marker, 1)[1], add_special_tokens=False
    ).input_ids
    special_ids = set(tok.all_special_ids) | {
        i for i, t in tok.added_tokens_decoder.items() if t.special
    }
    assert suffix_ids and suffix_ids[0] in special_ids, (
        f"no special token right after the reply: {tok.convert_ids_to_tokens(suffix_ids[:3])}"
    )
    return suffix_ids[0]


def stop_token_ids() -> set[int]:
    """Token ids that end generation, per the model's generation config (vLLM uses these too)."""
    eos = GenerationConfig.from_pretrained(MODEL_ID).eos_token_id
    if eos is None:
        return set()
    return {eos} if isinstance(eos, int) else set(eos)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def require_env(name: str) -> str:
    """Read a required env var, failing fast with a clear message."""
    val = os.environ.get(name)
    assert val, f"{name} is not set (is direnv loaded in this shell?)"
    return val


def load_jsonl(path: Path) -> list[Example]:
    """Read the training jsonl written by scripts/prepare_data.py."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_rows(rows: list[Example]) -> tuple[list[Example], list[Example]]:
    """The last VAL_SIZE rows validate; the rest train (the file is pre-shuffled)."""
    assert len(rows) > VAL_SIZE, f"need more than {VAL_SIZE} rows, got {len(rows)}"
    return rows[:-VAL_SIZE], rows[-VAL_SIZE:]


def example_length(tok: PreTrainedTokenizerBase, ex: Example) -> int:
    """Total tokens for one example (prompt + completion), via the chat template."""
    prompt_ids, completion_ids = encode_example(tok, ex)
    return len(prompt_ids) + len(completion_ids)


def check_lengths(
    tok: PreTrainedTokenizerBase, rows: list[Example]
) -> dict[str, list[int]]:
    """Assert every example fits MAX_LENGTH (never truncate silently); group by kind."""
    by_kind: dict[str, list[int]] = {k: [] for k in KINDS}
    for ex in rows:
        kind = ex["kind"]
        assert kind in by_kind, f"{ex['id']}: unknown kind {kind!r}"
        n = example_length(tok, ex)
        by_kind[kind].append(n)
        assert n <= MAX_LENGTH, (
            f"{ex['id']} ({kind}) is {n} tokens > MAX_LENGTH={MAX_LENGTH}; "
            "regenerate the data with a shorter context"
        )
    return by_kind


def print_length_stats(by_kind: dict[str, list[int]]) -> None:
    """Print per-kind token-length stats (used by --dry-run and the full run)."""
    for kind in KINDS:
        lens = by_kind[kind]
        if lens:
            print(
                f"{kind:<14} n={len(lens):>3}  max={max(lens):>4}  "
                f"mean={sum(lens) / len(lens):.0f}"
            )


# ---------------------------------------------------------------------------
# Label-masking checks
# ---------------------------------------------------------------------------
def verify_labels(
    tok: PreTrainedTokenizerBase, ex: Example, label_ids: list[int], tag: str
) -> str:
    """Assert the unmasked label ids are exactly this row's completion; return decoded text.

    The unmasked labels must start with the completion text, contain the
    end-of-turn token exactly once, and contain no prompt text such as
    "Question:".
    """
    eot = end_of_turn_id(tok)
    count = sum(1 for t in label_ids if t == eot)
    assert count == 1, (
        f"{tag}: expected exactly one end-of-turn token in the labels, found {count}"
    )
    text = tok.decode(label_ids, skip_special_tokens=True)
    assert isinstance(text, str)
    completion_text = ex["completion"][0]["content"]
    assert isinstance(completion_text, str)
    assert text.startswith(completion_text), (
        f"{tag}: labels do not start with the completion text {completion_text!r}"
    )
    assert "Question:" not in text, f"{tag}: prompt text leaked into the labels"
    print(f"{tag}decoded labels: {text!r}")
    return text


def build_batch_lists(
    tok: PreTrainedTokenizerBase, rows: list[Example]
) -> dict[str, list[list[int]]]:
    """Hand-built batch (plain lists, no torch) mirroring the completion-only masking."""
    _, pad_id = special_tokens(tok)
    seqs: list[list[int]] = []
    labels: list[list[int]] = []
    for ex in rows:
        prompt_ids, completion_ids = encode_example(tok, ex)
        seqs.append(prompt_ids + completion_ids)
        labels.append([-100] * len(prompt_ids) + completion_ids)
    width = max(len(s) for s in seqs)
    return {
        "input_ids": [s + [pad_id] * (width - len(s)) for s in seqs],
        "attention_mask": [[1] * len(s) + [0] * (width - len(s)) for s in seqs],
        "labels": [y + [-100] * (width - len(y)) for y in labels],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune google/gemma-3-1b-it on the SQuAD-v2 RAG data."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load data + tokenizer, print length stats, run the masking check, and exit "
        "(CPU-only, no model)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing adapter directory",
    )
    args = parser.parse_args()

    dataset_dir = require_env("DATASET_DIR")
    checkpoint_dir = require_env("CHECKPOINT_DIR")
    benchmark_dir = require_env("BENCHMARK_DIR")
    data_path = Path(dataset_dir) / DATA_SUBDIR / "train.jsonl"
    manifest_path = Path(dataset_dir) / DATA_SUBDIR / "manifest.json"
    assert data_path.exists(), f"{data_path} is missing; run scripts/prepare_data.py first"
    assert manifest_path.exists(), (
        f"{manifest_path} is missing; run scripts/prepare_data.py first"
    )

    print(
        f"torch {torch.__version__}  transformers {transformers.__version__}  "
        f"peft {peft.__version__}  trl {trl.__version__}  datasets {datasets.__version__}"
    )

    rows = load_jsonl(data_path)
    counts = {k: sum(r["kind"] == k for r in rows) for k in KINDS}
    print(f"{len(rows)} rows: {counts}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    special_tokens(tok)  # fail fast if the tokenizer lacks an end-of-turn / pad token

    by_kind = check_lengths(tok, rows)
    print("token lengths (prompt + completion):")
    print_length_stats(by_kind)

    train_rows, val_rows = split_rows(rows)
    print(f"train={len(train_rows)}  val={len(val_rows)}")

    if args.dry_run:
        # One row per kind, hand-built batch, no model loaded.
        mix = [next(r for r in rows if r["kind"] == k) for k in KINDS]
        batch = build_batch_lists(tok, mix)
        label_ids = [t for t in batch["labels"][0] if t != -100]
        verify_labels(tok, mix[0], label_ids, tag="[dry-run] ")
        print("\ndry-run OK (no model loaded)")
        return

    # --- GPU check, before loading the model -----------------------------
    assert torch.cuda.is_available(), (
        "CUDA unavailable: check nvidia-smi as this user and LD_LIBRARY_PATH from shell.nix"
    )
    free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
    print(f"cuda:0 free {free_gb:.1f} GB")
    assert free_gb >= MIN_FREE_GB, (
        f"cuda:0 has {free_gb:.1f} GB free, need {MIN_FREE_GB} GB. "
        "Stop the vLLM server on it, or pick another GPU with CUDA_VISIBLE_DEVICES=<n>"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}
    )
    # Gradient checkpointing is enabled manually (matching scripts/smoke_test.py)
    # rather than via SFTConfig, so the known-good order is: use_cache off, then GC.
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_REGEX,
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=str(Path(checkpoint_dir) / "sft-run"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=SCHEDULER,
        max_length=MAX_LENGTH,
        bf16=True,
        report_to="none",
        logging_steps=1,
        eval_strategy="epoch",  # transformers 5.x kwarg name (evaluation_strategy is gone)
        save_strategy="no",
        packing=False,  # trl 1.15 SFTConfig kwarg (default False); explicit for clarity
        seed=SEED,
    )
    # 3% warmup. SFTConfig is a TrainingArguments subclass at runtime and has
    # warmup_ratio, but basedpyright can't see that inherited field in this
    # transformers 5.19 build, so set it after init and silence just this access.
    # The LR scheduler is built from the args object at trainer.train() time,
    # so the value assigned here is the one the warmup ramp uses.
    config.warmup_ratio = WARMUP_RATIO  # pyright: ignore[reportAttributeAccessIssue]

    train_sft = [{"prompt": r["prompt"], "completion": r["completion"]} for r in train_rows]
    val_sft = [{"prompt": r["prompt"], "completion": r["completion"]} for r in val_rows]
    train_ds = datasets.Dataset.from_list(train_sft)
    val_ds = datasets.Dataset.from_list(val_sft)
    # trl 1.15: "prompt"/"completion" (lists of messages) is the conversational
    # prompt/completion format; loss is computed on the completion only.
    # trl 1.15 SFTTrainer kwargs: "processing_class" (not "tokenizer") and
    # "peft_config" (an SFTTrainer arg, not an SFTConfig one).
    trainer = SFTTrainer(
        model=base_model,
        processing_class=tok,
        args=config,
        peft_config=lora,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # --- Pre-train: pull one batch from the trainer's own collator --------
    # trainer.train_dataset is the tokenized dataset (input_ids /
    # attention_mask / labels), in the same order as train_rows.
    tds = trainer.train_dataset
    assert tds is not None, "trainer.train_dataset is None after init"
    assert isinstance(tds, datasets.Dataset), (
        "expected a regular (non-streaming) HF Dataset for trainer.train_dataset"
    )
    dc = trainer.data_collator
    assert dc is not None, "trainer.data_collator is None after init"
    feats = [tds[i] for i in range(min(PER_DEVICE_BATCH, len(tds)))]
    batch = dc(feats)
    assert isinstance(batch, dict) and "labels" in batch, (
        "trainer data collator did not return a labels field"
    )
    labels = batch["labels"]
    assert isinstance(labels, torch.Tensor), "collator labels is not a tensor"
    label_ids = [int(t) for t in labels[0].tolist() if t != -100]
    verify_labels(tok, train_rows[0], label_ids, tag="[collator] ")

    print(
        f"\ntraining: {NUM_EPOCHS} epochs, effective batch "
        f"{PER_DEVICE_BATCH * GRAD_ACCUM}, lr {LEARNING_RATE}, "
        f"{SCHEDULER} + {WARMUP_RATIO:.0%} warmup"
    )
    start = time.time()
    torch.cuda.reset_peak_memory_stats(0)
    trainer.train()
    wall_time = time.time() - start
    peak_gb = torch.cuda.max_memory_allocated(0) / 1e9

    log_hist = trainer.state.log_history
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_path = Path(benchmark_dir) / f"train_{ts}.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for entry in log_hist:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"wrote per-step log to {log_path}")

    final_train_loss: float | None = None
    final_eval_loss: float | None = None
    for entry in log_hist:
        if "loss" in entry:
            final_train_loss = entry["loss"]
        if "eval_loss" in entry:
            final_eval_loss = entry["eval_loss"]
    print(
        f"final train loss {final_train_loss}  "
        f"final eval loss {final_eval_loss}"
    )

    # --- Save only the adapter -------------------------------------------
    model = trainer.model
    assert isinstance(model, PeftModel), (
        f"trainer.model is {type(model).__name__}, expected a PeftModel"
    )
    adapter_dir = Path(checkpoint_dir) / ADAPTER_NAME
    if adapter_dir.exists() and any(adapter_dir.iterdir()) and not args.overwrite:
        sys.exit(f"{adapter_dir} already exists; pass --overwrite to replace it")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))  # PeftModel: adapter weights only
    shutil.copy2(str(manifest_path), str(adapter_dir / "manifest.json"))

    run_info = {
        "constants": {
            "model": MODEL_ID,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_REGEX,
            "num_epochs": NUM_EPOCHS,
            "per_device_batch": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM,
            "effective_batch": PER_DEVICE_BATCH * GRAD_ACCUM,
            "lr": LEARNING_RATE,
            "scheduler": SCHEDULER,
            "warmup_ratio": WARMUP_RATIO,
            "max_length": MAX_LENGTH,
            "seed": SEED,
            "val_size": VAL_SIZE,
            "adapter_name": ADAPTER_NAME,
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
            "datasets": datasets.__version__,
        },
        "row_counts": {
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            **counts,
        },
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "wall_time_s": round(wall_time, 1),
        "peak_gpu_memory_gb": round(peak_gb, 2),
    }
    (adapter_dir / "training_run.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"saved adapter to {adapter_dir} (+ manifest.json, training_run.json)")

    # --- Post-training greedy generation, one validation row per kind -----
    print("\n--- post-training greedy generation (one val row per kind) ---")
    model.eval()
    _, pad_id = special_tokens(tok)
    eos_ids = sorted(stop_token_ids() | {end_of_turn_id(tok)})
    for kind in KINDS:
        ex = next((e for e in val_rows if e["kind"] == kind), None)
        assert ex is not None, f"no validation row of kind {kind!r}"
        prompt_text = render(tok, ex["prompt"], add_generation_prompt=True)
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(
            "cuda:0"
        )
        prompt_len = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=eos_ids,
            )
        assert isinstance(out, torch.Tensor)
        gen_text = tok.decode(out[0, prompt_len :], skip_special_tokens=True)
        assert isinstance(gen_text, str)
        print(f"[{kind}] expected : {ex['completion'][0]['content']}")
        print(f"[{kind}] generated: {gen_text.strip()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
