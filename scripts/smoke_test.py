"""Smoke test for the LoRA training stack (Gemma 3 1B-it, chat-format RAG data).

Two modes:

  Training stack, on a GPU with ~24 GB free (stop vllm-bench first if it holds that GPU):
      CUDA_VISIBLE_DEVICES=1 uv run python scripts/smoke_test.py

  Serving round trip, after the first mode has saved smoke-lora and vllm-bench is
  running the same base model (this process needs no GPU):
      uv run python scripts/smoke_test.py --serve-check

The first mode checks env vars and paths, the GPU, the training data, the chat
template (end-of-turn token and completion-only masking), model load, the LoRA
wrap (language-model layers only, never the vision or audio encoders), a tiny
overfit run, generation, and adapter save. The second loads that adapter into
vLLM and confirms vLLM reproduces what it memorized, which proves the adapter's
weight names match the modules vLLM serves.
"""

import argparse
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import datasets
import peft
import torch
import transformers
import trl
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from train import (
    MODEL_ID,
    LORA_PROJECTIONS,
    LORA_TARGET_REGEX,
    ENCODER_MARKERS,
    end_of_turn_id,
    render,
    stop_token_ids,
)

DATA_SUBDIR = "squad-v2-rag"  # written by scripts/prepare_data.py
ADAPTER_NAME = "smoke-lora"
MAX_TOKENS = 1536  # prepare_data.py aims for ~1,000 tokens per example
MIN_FREE_GB = 24  # generous headroom for the overfit run; the 1B base itself is small
KINDS = ("answerable", "unanswerable", "no_context")

Example = dict[str, Any]
Message = dict[str, str]


def section(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def load_jsonl(path: Path) -> list[Example]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pad_token_id(tok: PreTrainedTokenizerBase) -> int:
    """The tokenizer's pad id, narrowed from transformers' loose types."""
    pad_id = tok.pad_token_id
    assert isinstance(pad_id, int), "tokenizer has no pad token"
    return pad_id


def encode_example(
    tok: PreTrainedTokenizerBase, ex: Example
) -> tuple[list[int], list[int]]:
    """Token ids for the prompt and for the completion, as completion-only training splits them.

    The prompt is the user turn plus the model-turn header (the generation prompt).
    The completion is whatever the full conversation adds after that: the answer
    and the end-of-turn token. Completion-only loss only works if the prompt's
    tokens are an exact prefix of the full conversation's tokens, so check that.
    """
    prompt_text = render(tok, ex["prompt"], add_generation_prompt=True)
    full_text = render(
        tok, ex["prompt"] + ex["completion"], add_generation_prompt=False
    )
    assert full_text.startswith(prompt_text), (
        f"{ex['id']}: chat template's prompt is not a prefix of prompt+completion"
    )
    prompt_ids: list[int] = tok(prompt_text, add_special_tokens=False).input_ids
    completion_ids: list[int] = tok(
        full_text[len(prompt_text) :], add_special_tokens=False
    ).input_ids
    joint: list[int] = tok(full_text, add_special_tokens=False).input_ids
    assert joint == prompt_ids + completion_ids, (
        f"{ex['id']}: prompt/completion token boundary differs from joint tokenization"
    )
    return prompt_ids, completion_ids


def build_batch(
    tok: PreTrainedTokenizerBase, rows: list[Example], with_token_type_ids: bool = False
) -> dict[str, torch.Tensor]:
    pad_id = pad_token_id(tok)
    seqs: list[list[int]] = []
    labels: list[list[int]] = []
    for ex in rows:
        prompt_ids, completion_ids = encode_example(tok, ex)
        seqs.append(prompt_ids + completion_ids)
        labels.append(
            [-100] * len(prompt_ids) + completion_ids
        )  # loss on the answer only
    width = max(len(s) for s in seqs)
    batch = {
        "input_ids": torch.tensor([s + [pad_id] * (width - len(s)) for s in seqs]),
        "attention_mask": torch.tensor(
            [[1] * len(s) + [0] * (width - len(s)) for s in seqs]
        ),
        "labels": torch.tensor([y + [-100] * (width - len(y)) for y in labels]),
    }
    if with_token_type_ids:  # multimodal Gemma forward may require it; 0 = text token
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
    return batch


def pick_mix(rows: list[Example]) -> list[Example]:
    """Four rows covering every kind, so the overfit test exercises abstention too."""
    by_kind = {k: [r for r in rows if r["kind"] == k] for k in KINDS}
    return [
        by_kind["answerable"][0],
        by_kind["unanswerable"][0],
        by_kind["no_context"][0],
        by_kind["answerable"][1],
    ]


def check_env() -> None:
    section("environment")
    for var in ("DATASET_DIR", "CHECKPOINT_DIR", "HF_HOME"):
        val = os.environ.get(var)
        assert val, f"{var} is not set (is direnv loaded in this shell?)"
        assert os.access(val, os.W_OK), f"{var}={val} is not writable by this user"
        print(f"{var}={val}")
    free_gb = shutil.disk_usage(os.environ["HF_HOME"]).free / 1e9
    print(f"free disk at HF_HOME: {free_gb:.0f} GB")
    assert free_gb > 25, "need ~25 GB free for the model"
    assert ".venv" in sys.executable, f"running {sys.executable}, not the project venv"
    for mod in (torch, transformers, peft, trl, datasets):
        print(f"{mod.__name__:<13} {mod.__version__}")
    print(f"{'torch CUDA':<13} {torch.version.cuda}")


def check_gpu() -> None:
    section("gpu")
    assert torch.cuda.is_available(), (
        "CUDA unavailable: check LD_LIBRARY_PATH from shell.nix and run nvidia-smi as this user"
    )
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        name = torch.cuda.get_device_name(i)
        print(f"cuda:{i} {name}  free {free / 1e9:.1f} / {total / 1e9:.1f} GB")
    # Only cuda:0 is used. Pick the physical GPU with CUDA_VISIBLE_DEVICES=<n>.
    free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
    assert free_gb >= MIN_FREE_GB, (
        f"cuda:0 has {free_gb:.1f} GB free, need ~{MIN_FREE_GB} GB. Stop the vLLM server on it, "
        "or run on another GPU with CUDA_VISIBLE_DEVICES=<n>"
    )
    assert torch.cuda.is_bf16_supported(), "GPU lacks bf16 support"


def check_data() -> list[Example]:
    section("data")
    path = Path(os.environ["DATASET_DIR"]) / DATA_SUBDIR / "train.jsonl"
    assert path.exists(), f"{path} is missing; run scripts/prepare_data.py first"
    rows = load_jsonl(path)
    counts = {k: sum(r["kind"] == k for r in rows) for k in KINDS}
    print(f"{len(rows)} rows: {counts}")
    assert sum(counts.values()) == len(rows), "rows with an unknown 'kind'"
    for r in rows:
        assert [m["role"] for m in r["prompt"]] == ["user"], (
            f"{r['id']}: prompt must be one user message"
        )
        assert [m["role"] for m in r["completion"]] == ["assistant"], (
            f"{r['id']}: completion must be one assistant message"
        )
        assert r["prompt"][0]["content"].endswith("Answer:\n"), (
            f"{r['id']}: prompt doesn't end like the harness template"
        )
    abstentions = {
        r["completion"][0]["content"] for r in rows if r["kind"] != "answerable"
    }
    assert len(abstentions) == 1, f"abstention targets differ: {abstentions}"
    print(f"abstention target: {next(iter(abstentions))!r}")
    return rows


def check_tokenizer(rows: list[Example]) -> tuple[PreTrainedTokenizerBase, int]:
    section("chat template + label masking")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    eot = end_of_turn_id(tok)
    pad_id = pad_token_id(tok)
    stops = stop_token_ids()
    print(
        f"end-of-turn id={eot} {tok.convert_ids_to_tokens(eot)!r}  "
        f"pad id={pad_id} {tok.pad_token!r}  stop ids={sorted(stops)}"
    )
    assert eot in stops, (
        "the end-of-turn token doesn't stop generation (check generation_config.json)"
    )
    assert pad_id != eot, (
        "pad and end-of-turn are the same token; masking pad would hide every stop"
    )

    lens = [sum(map(len, encode_example(tok, r))) for r in rows]
    print(f"sequence tokens: max={max(lens)}  mean={sum(lens) / len(lens):.0f}")
    assert max(lens) <= MAX_TOKENS, (
        f"longest example is {max(lens)} tokens; raise MAX_TOKENS"
    )

    ex = next(r for r in rows if r["kind"] == "answerable")
    print(
        "rendered prompt tail:",
        repr(render(tok, ex["prompt"], add_generation_prompt=True)[-80:]),
    )
    label_ids = [t for t in build_batch(tok, [ex])["labels"][0].tolist() if t != -100]
    print(f"loss is computed on: {tok.decode(label_ids)!r}")
    text = tok.decode(label_ids, skip_special_tokens=True)
    assert text.startswith(ex["completion"][0]["content"]), (
        "the answer isn't at the start of the labels"
    )
    assert label_ids.count(eot) == 1, (
        "labels need exactly one end-of-turn token, or the model won't learn to stop"
    )
    assert "Question:" not in text, "prompt tokens are leaking into the loss"
    return tok, eot


def check_lora_targets(model: torch.nn.Module) -> None:
    """LoRA must sit on language-model projections only, never on the encoders."""
    names = [n for n, _ in model.named_modules()]
    targeted = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
    assert targeted, (
        "LoRA matched no modules; compare LORA_TARGET_REGEX with the module names"
    )
    leaked = [n for n in targeted if any(k in n for k in ENCODER_MARKERS)]
    assert not leaked, f"LoRA landed on encoder layers: {leaked[:3]}"
    if any("language_model" in n for n in names):
        outside = [n for n in targeted if "language_model" not in n]
        assert not outside, f"LoRA landed outside the language model: {outside[:3]}"
    counts = {p: sum(n.endswith("." + p) for n in targeted) for p in LORA_PROJECTIONS}
    print(f"LoRA on {len(targeted)} modules: {counts}")
    print(f"first: {targeted[0]}")


def check_lora_overfit(
    tok: PreTrainedTokenizerBase, eot: int, train: list[Example]
) -> None:
    section("model + LoRA")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}
    )
    print(
        f"loaded {type(model).__name__} in {time.time() - t0:.0f}s, "
        f"{torch.cuda.memory_allocated(0) / 1e9:.1f} GB on cuda:0"
    )
    with_tt = "token_type_ids" in inspect.signature(model.forward).parameters
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=LORA_TARGET_REGEX,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    check_lora_targets(model)

    section("overfit 4 examples (loss should collapse)")
    rows = pick_mix(train)
    batch = {k: v.to("cuda:0") for k, v in build_batch(tok, rows, with_tt).items()}
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-4)
    torch.cuda.reset_peak_memory_stats(0)
    model.train()
    losses: list[float] = []
    for step in range(40):
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        if step % 10 == 0 or step == 39:
            print(f"step {step:>2}  loss {losses[-1]:.4f}")
    print(f"peak memory {torch.cuda.max_memory_allocated(0) / 1e9:.1f} GB")
    assert losses[-1] < 0.5 * losses[0], (
        "loss barely moved: check requires_grad, masking, lr"
    )

    section("generate from overfit adapter (should reproduce each target)")
    model.eval()
    for ex in rows[:3]:
        prompt_text = render(tok, ex["prompt"], add_generation_prompt=True)
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(
            "cuda:0"
        )
        prompt_ids = enc["input_ids"]
        assert isinstance(prompt_ids, torch.Tensor)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id(tok),
                eos_token_id=sorted(stop_token_ids() | {eot}),
            )
        assert isinstance(
            out, torch.Tensor
        )  # plain tensor unless return_dict_in_generate=True
        text = tok.decode(out[0, prompt_ids.shape[1] :], skip_special_tokens=True)
        assert isinstance(text, str)
        print(f"[{ex['kind']}] expected : {ex['completion'][0]['content']}")
        print(f"[{ex['kind']}] generated: {text.strip()}")

    section("save adapter")
    out_dir = Path(os.environ["CHECKPOINT_DIR"]) / ADAPTER_NAME
    model.save_pretrained(str(out_dir))
    files = sorted(p.name for p in out_dir.iterdir())
    print(f"{out_dir}: {files}")
    assert {"adapter_config.json", "adapter_model.safetensors"} <= set(files)
    with safe_open(str(out_dir / "adapter_model.safetensors"), framework="pt") as f:
        keys = sorted(f.keys())
    print(f"{len(keys)} adapter tensors, e.g. {keys[0]}")


def serve_check(base_url: str, train: list[Example]) -> None:
    """Load smoke-lora into vllm-bench and check vLLM reproduces what it memorized."""
    import httpx  # installed with openai; only this mode needs it

    section("serving round trip (vLLM)")
    deadline = time.monotonic() + 300
    while True:
        try:
            r = httpx.get(f"{base_url}/v1/models", timeout=5)
            if r.is_success:
                break
        except httpx.TransportError:
            pass
        assert time.monotonic() < deadline, (
            f"no vLLM server at {base_url}; start podman-vllm-bench"
        )
        time.sleep(5)
    served = [m["id"] for m in r.json()["data"]]
    assert MODEL_ID in served, (
        f"server is serving {served}, not {MODEL_ID}; set vllm-bench's --model to {MODEL_ID}"
    )

    adapter = {"lora_name": ADAPTER_NAME}
    httpx.post(
        f"{base_url}/v1/unload_lora_adapter", json=adapter, timeout=30
    )  # may not be loaded
    r = httpx.post(
        f"{base_url}/v1/load_lora_adapter",
        json={**adapter, "lora_path": f"/adapters/{ADAPTER_NAME}"},
        timeout=120,
    )
    assert r.is_success, f"vLLM refused the adapter: {r.status_code} {r.text[:500]}"
    print(f"loaded {ADAPTER_NAME} from /adapters/{ADAPTER_NAME}")

    def ask(model: str, ex: Example) -> str:
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": ex["prompt"],
                "temperature": 0,
                "max_tokens": 128,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"]).strip()

    results: list[tuple[str, str, str]] = []
    for ex in pick_mix(train)[:3]:
        target = ex["completion"][0]["content"]
        lora_out, base_out = ask(ADAPTER_NAME, ex), ask(MODEL_ID, ex)
        print(
            f"[{ex['kind']}] expected: {target}\n    adapter: {lora_out}\n    base   : {base_out}"
        )
        results.append((target, lora_out, base_out))
    httpx.post(f"{base_url}/v1/unload_lora_adapter", json=adapter, timeout=30)

    if all(lora_out == base_out for _, lora_out, base_out in results):
        raise AssertionError(
            "the adapter loaded but changed nothing: its weight names probably don't match "
            "the modules vLLM serves. Compare the adapter tensor names printed by the "
            "training mode with vLLM's log."
        )
    misses = [t for t, lora_out, _ in results if lora_out != t]
    assert not misses, (
        f"vLLM's adapter output differs from what it memorized for: {misses}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--serve-check",
        action="store_true",
        help="load smoke-lora into vllm-bench and test it",
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("BENCH_URL", "http://localhost:8001")
    )
    args = parser.parse_args()

    if args.serve_check:
        serve_check(str(args.base_url).rstrip("/"), check_data())
        print("\nServing round trip passed.")
        return

    check_env()
    check_gpu()
    train = check_data()
    tok, eot = check_tokenizer(train)
    check_lora_overfit(tok, eot, train)
    print(
        "\nAll checks passed. Next: start vllm-bench and run this script with --serve-check."
    )


if __name__ == "__main__":
    main()
