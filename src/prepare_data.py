"""Build the SFT training file from NuminaMath-CoT.

The one thing that matters here: NuminaMath-CoT contains 7,345 rows sourced from
`gsm8k`. We evaluate on GSM8K, so keeping them would make every number in the
README a lie. `--drop-sources` removes them before sampling.

    python src/prepare_data.py --out data/sft_train.jsonl --n 20000
"""

import argparse
import os

from prompts import training_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="project-numina/NuminaMath-CoT")
    p.add_argument("--out", default="data/sft_train.jsonl")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--max-chars", type=int, default=8000,
                   help="drop very long solutions; they dominate a 2048-token budget")
    p.add_argument("--min-chars", type=int, default=200,
                   help="drop stubs that teach the model nothing")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop-sources", nargs="*", default=["gsm8k"],
                   help="sources that overlap the evaluation benchmark")
    return p.parse_args()


def to_text(row):
    return {"text": training_text(row["problem"], row["solution"])}


def main():
    # Imported here, not at module scope, so the pure helpers stay testable
    # on a machine with no `datasets` install.
    from datasets import load_dataset

    a = parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    ds = load_dataset(a.dataset, split="train")
    n_raw = len(ds)
    print(f"[load] {n_raw} rows from {a.dataset}")

    if a.drop_sources:
        ds = ds.filter(lambda r: r["source"] not in a.drop_sources, num_proc=4)
        print(f"[decontaminate] dropped {a.drop_sources}: {n_raw} -> {len(ds)}")

    ds = ds.map(to_text, remove_columns=ds.column_names, num_proc=4)
    ds = ds.filter(
        lambda r: a.min_chars <= len(r["text"]) <= a.max_chars,
        num_proc=4,
    )
    print(f"[length] kept {len(ds)} rows in [{a.min_chars}, {a.max_chars}] chars")

    if a.n < len(ds):
        ds = ds.shuffle(seed=a.seed).select(range(a.n))
    print(f"[sample] {len(ds)} rows -> {a.out}")

    ds.to_json(a.out)

    # Sanity check: a leaked instruction or an empty solution would show up here.
    print(f"[preview] {ds[0]['text'][:300]!r}")


if __name__ == "__main__":
    main()
