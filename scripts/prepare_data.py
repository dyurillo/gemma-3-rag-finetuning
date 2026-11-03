"""Build chat-format RAG training data from SQuAD v2 for the LoRA adapter.

Every example is laid out exactly the way the eval harness
(rag-eval-harness/harness.py) prompts the model: K retrieved chunks, each
labelled with a doc_id, then the question, sent as a single user message. The
adapter learns this format, so the template and context layout below are copied
verbatim from the harness. If the harness prompt ever changes, regenerate this
data and retrain.

The mix (N_TRAIN rows; train.py holds out the last rows for validation loss):
  answerable    gold paragraph + K-1 distractors -> the full sentence around the
                answer span (plus the next sentence for why/how/and questions),
                then a verbatim quote line: Quote: "<sentence>" [doc-id]
  unanswerable  SQuAD v2's deliberately unanswerable questions + distractors
                -> the harness's abstention sentence
  no_context    the harness's "nothing retrieved" placeholder -> abstention

About a fifth of all targets are abstentions. More than that risks teaching
the model to refuse answerable questions.

Source: rajpurkar/squad_v2 (CC-BY-SA-4.0). Credit it in the README.

Run:    uv run python scripts/prepare_data.py
Writes: $DATASET_DIR/squad-v2-rag/train.jsonl and manifest.json
"""

import json
import os
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SEED = 42
K = 4  # chunks per prompt; matches the harness's default --k
N_TRAIN = 500
N_UNANSWERABLE = 75  # SQuAD v2 unanswerable questions (with no_context, ~20% of targets abstain)
N_NO_CONTEXT = 25  # nothing retrieved
MAX_PARAGRAPH_CHARS = 900  # keeps K chunks plus instructions under ~1,024 tokens

ABSTAIN = "The provided context does not contain the answer."

# --- Copied verbatim from rag-eval-harness/harness.py. Keep in sync. ---------
ANSWER_PROMPT_WITH_CONTEXT = """\
Answer the question using only the context below. Do not use outside
knowledge.

If the context contains the answer, reply with the answer in one or two
short sentences, then on the next line a supporting quote in exactly this
form:
Quote: "<sentence>" [doc-id]
where <sentence> is a verbatim copy of a sentence from the context that
supports your answer, and doc-id is the id shown in square brackets at the
start of the context section it came from.

If the context does not contain the answer, do not answer and do not quote.
Reply with exactly one line:
The provided context does not contain the answer.

Context:
{context}

Question: {question}

Answer:
"""


def format_context(chunks: Sequence[tuple[str, str]]) -> str:
    """Label each chunk with its doc_id, as harness.format_context does."""
    if not chunks:
        return "(no relevant context was retrieved)"
    return "\n\n".join(f"[{doc_id}]\n{text}" for doc_id, text in chunks)


# --- end of copied code -------------------------------------------------------

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def sentence_spans(paragraph: str) -> list[tuple[int, int]]:
    """(start, end) char spans of the sentences in paragraph.

    Sentences end at [.!?] followed by whitespace, so decimals ("3.2.x") and
    list numbering don't split mid-sentence.
    """
    starts = [0] + [m.end() for m in _SENTENCE_BOUNDARY_RE.finditer(paragraph)]
    return list(zip(starts, starts[1:] + [len(paragraph)]))


def sentence_around(paragraph: str, char_index: int) -> tuple[str, str]:
    """The sentence containing paragraph[char_index], and the one after it.

    Returns (sentence, next_sentence); next_sentence is "" for the last
    sentence of the paragraph.
    """
    spans = sentence_spans(paragraph)
    for i, (start, end) in enumerate(spans):
        if start <= char_index < end:
            nxt = (
                paragraph[spans[i + 1][0] : spans[i + 1][1]].strip()
                if i + 1 < len(spans)
                else ""
            )
            return paragraph[start:end].strip(), nxt
    raise AssertionError(f"char index {char_index} out of range")


_COMPOUND_QUESTION_RE = re.compile(r"\b(why|how|and)\b", re.IGNORECASE)


def answerable_target(gold: str, answer: str, doc_id: str, question: str) -> str:
    """Two-line target for an answerable row: sentence answer, then the quote.

    Line 1 is the full sentence of the gold passage that contains the answer
    span - not the bare span. The eval's reference answers are complete
    sentences and the judge scores completeness, while a bare SQuAD span
    trains the model to answer in fragments ("48" instead of "48, because").
    For compound questions (why/how/and) the following sentence is appended
    so two-part references come out complete. Line 2 quotes the span's
    sentence verbatim, tagged with the gold chunk's doc id (its label in the
    prompt).
    """
    idx = gold.lower().find(answer.strip().lower())
    if idx == -1:
        raise AssertionError(f"answer {answer!r} not found in gold passage")
    span_sentence, next_sentence = sentence_around(gold, idx)
    if next_sentence and _COMPOUND_QUESTION_RE.search(question):
        answer_line = f"{span_sentence} {next_sentence}"
    else:
        answer_line = span_sentence
    return f'{answer_line}\nQuote: "{span_sentence}" [{doc_id}]'


Chunk = tuple[str, str]  # (doc_id, text)
Example = dict[str, Any]


def doc_id_for(title: str) -> str:
    """Kebab-case ASCII id, like the eval corpus's file names.

    'Super_Bowl_50' -> 'super-bowl-50', 'Beyoncé' -> 'beyonce'.
    """
    ascii_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    )
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-") or "untitled"


def index_paragraphs(
    titles: Sequence[str], contexts: Sequence[str]
) -> dict[str, list[str]]:
    """Unique paragraphs per article, skipping ones too long to use as chunks."""
    paragraphs: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for title, ctx in zip(titles, contexts):
        if len(ctx) <= MAX_PARAGRAPH_CHARS and (title, ctx) not in seen:
            seen.add((title, ctx))
            paragraphs[title].append(ctx)
    return dict(paragraphs)


def pick_distractors(
    rng: random.Random,
    paragraphs: dict[str, list[str]],
    title: str,
    gold: str,
    answer: str | None,
    n: int,
    same_article: bool,
) -> list[Chunk]:
    """Choose n distractor chunks that must not contain the answer.

    Answerable questions get distractors from the same article: on-topic but
    wrong, like real retrieval. Any paragraph containing the answer string is
    excluded. Unanswerable questions only get distractors from other articles,
    because SQuAD only guarantees the question is unanswerable from its own
    paragraph; another paragraph of the same article might answer it.
    """
    pool: list[tuple[str, str]] = []
    if same_article:
        pool = [
            (title, p)
            for p in paragraphs.get(title, [])
            if p != gold and (answer is None or answer.lower() not in p.lower())
        ]
        rng.shuffle(pool)
    picked = pool[:n]

    other_titles = [t for t in paragraphs if t != title]
    assert other_titles, "need at least two articles to draw distractors from"
    while len(picked) < n:
        other = rng.choice(other_titles)
        candidate = (other, rng.choice(paragraphs[other]))
        if candidate not in picked:
            picked.append(candidate)
    return [(doc_id_for(t), p) for t, p in picked]


def make_example(
    example_id: str, kind: str, question: str, chunks: list[Chunk], target: str
) -> Example:
    """One TRL conversational prompt-completion row, matching the harness's request."""
    prompt = ANSWER_PROMPT_WITH_CONTEXT.format(
        context=format_context(chunks), question=question
    )
    return {
        "id": example_id,
        "kind": kind,
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": target}],
    }


def build_examples(
    ids: Sequence[str],
    titles: Sequence[str],
    contexts: Sequence[str],
    questions: Sequence[str],
    answers: Sequence[Sequence[str]],
    rng: random.Random,
) -> list[Example]:
    """Sample the answerable / unanswerable / no-context mix from SQuAD v2 rows."""
    paragraphs = index_paragraphs(titles, contexts)
    want = {
        "no_context": N_NO_CONTEXT,
        "unanswerable": N_UNANSWERABLE,
        "answerable": N_TRAIN - N_UNANSWERABLE - N_NO_CONTEXT,
    }
    got = dict.fromkeys(want, 0)
    examples: list[Example] = []
    used_questions: set[str] = set()

    order = list(range(len(ids)))
    rng.shuffle(order)
    for i in order:
        if got == want:
            break
        question, title, gold = questions[i].strip(), titles[i], contexts[i]
        if question in used_questions or len(gold) > MAX_PARAGRAPH_CHARS:
            continue

        has_answer = len(answers[i]) > 0
        if has_answer and got["no_context"] < want["no_context"]:
            kind = "no_context"
        elif has_answer and got["answerable"] < want["answerable"]:
            kind = "answerable"
        elif not has_answer and got["unanswerable"] < want["unanswerable"]:
            kind = "unanswerable"
        else:
            continue

        if kind == "no_context":
            chunks: list[Chunk] = []
            target = ABSTAIN
        else:
            answer = answers[i][0].strip() if kind == "answerable" else None
            assert answer is None or answer.lower() in gold.lower(), (
                f"answer not in paragraph: {ids[i]}"
            )
            distractors = pick_distractors(
                rng,
                paragraphs,
                title,
                gold,
                answer,
                K - 1,
                same_article=kind == "answerable",
            )
            gold_id = doc_id_for(title)
            chunks = [(gold_id, gold), *distractors]
            rng.shuffle(chunks)  # the gold chunk can appear in any position
            target = (
                answerable_target(gold, answer, gold_id, question)
                if answer is not None
                else ABSTAIN
            )

        examples.append(make_example(ids[i], kind, question, chunks, target))
        got[kind] += 1
        used_questions.add(question)

    assert got == want, f"not enough source rows: wanted {want}, got {got}"
    rng.shuffle(examples)  # mix kinds, so train.py's held-out tail is representative
    return examples


def write_jsonl(path: Path, rows: Sequence[Example]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    from datasets import load_dataset  # imported here so the helpers above import fast

    out_dir = Path(os.environ["DATASET_DIR"]) / "squad-v2-rag"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("rajpurkar/squad_v2", split="train")
    print(f"downloaded {len(ds)} rows, columns={ds.column_names}")

    examples = build_examples(
        ids=list(ds["id"]),
        titles=list(ds["title"]),
        contexts=list(ds["context"]),
        questions=list(ds["question"]),
        answers=[a["text"] for a in ds["answers"]],
        rng=random.Random(SEED),
    )
    write_jsonl(out_dir / "train.jsonl", examples)

    counts = {
        kind: sum(e["kind"] == kind for e in examples)
        for kind in ("answerable", "unanswerable", "no_context")
    }
    prompt_chars = [len(e["prompt"][0]["content"]) for e in examples]
    manifest = {
        "source": "rajpurkar/squad_v2 (train split), CC-BY-SA-4.0",
        "prompt_source": "copied verbatim from rag-eval-harness/harness.py",
        "seed": SEED,
        "k": K,
        "max_paragraph_chars": MAX_PARAGRAPH_CHARS,
        "abstain_target": ABSTAIN,
        "answerable_target_format": (
            "<sentence containing the span; + the next sentence for "
            "why/how/and questions>\\nQuote: \"<the span's sentence>\" [doc-id]"
        ),
        "counts": counts,
        "prompt_chars": {
            "max": max(prompt_chars),
            "mean": round(sum(prompt_chars) / len(prompt_chars)),
        },
        "answer_prompt_template": ANSWER_PROMPT_WITH_CONTEXT,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"wrote {len(examples)} examples to {out_dir}: {counts}")
    print(
        f"prompt length: max {max(prompt_chars)} chars, mean {manifest['prompt_chars']['mean']} (~4 chars/token)"
    )
    for kind in ("answerable", "unanswerable"):
        sample = next(e for e in examples if e["kind"] == kind)
        print(
            f"\n--- example ({kind}) ---\n{sample['prompt'][0]['content']}{sample['completion'][0]['content']}"
        )


if __name__ == "__main__":
    main()
