<div align="center">

# 📦 boxed

**A 1.5B model that shows its work and puts the answer in a box.**

QLoRA fine-tune of Qwen2.5-1.5B on a leak-free slice of NuminaMath-CoT, evaluated before
and after on the same 250 GSM8K questions — on a single free Kaggle T4.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Base model](https://img.shields.io/badge/base-Qwen2.5--1.5B-5B21B6?style=flat-square)](https://huggingface.co/Qwen/Qwen2.5-1.5B)
[![Method](https://img.shields.io/badge/method-QLoRA%20%2B%20TRL-0f766e?style=flat-square)](configs/boxed-1.5b.yaml)
[![Hardware](https://img.shields.io/badge/hardware-1%C3%97%20T4%2016GB-76B900?style=flat-square)](RUNBOOK.md)
[![Last commit](https://img.shields.io/github/last-commit/Raufnarejo505/boxed?style=flat-square)](https://github.com/Raufnarejo505/boxed/commits/master)

[Results](#results) · [The filter](#the-filter-that-everyone-skips) · [Quick start](#quick-start) · [Repository map](#what-is-in-here) · [Tests](#tests) · [Limitations](#what-it-cannot-do)

</div>

---

I fine-tuned Qwen2.5-1.5B — the base checkpoint, not the chat one — on a filtered slice of
NuminaMath-CoT, then put the same 250 GSM8K questions to it before and after training.

Making that comparison honest took longer than the training did, which is most of what this
repo is about.

Everything ran on a single free Kaggle T4. No A100, no rented cluster, no API budget.

---

## Results

| Model | GSM8K (n=250) | What it is |
| --- | --- | --- |
| `Qwen/Qwen2.5-1.5B` (base) | _TBD_ | run `eval_gsm8k.py` before training |
| `boxed-1.5b-lora` | _TBD_ | the adapter from this run |
| `boxed-1.5b-merged` | _TBD_ | merged weights, what the eval loads |

Both fine-tuned rows come from `src/eval_gsm8k.py` with the same prompt, greedy decoding and
a single seed. No rerunning with different seeds until the number looked good.

The cells are `TBD` rather than filled with a number that cannot be reproduced from this
repo. `results/` is reserved for the eval JSON — `.gitignore` keeps `results/*.json` for
exactly that reason — and it stays empty until the run is published.

If the delta turns out to be small, that is the result, and it gets reported.

---

## The filter that everyone skips

NuminaMath-CoT has 859,494 rows. **7,345 of them were scraped from GSM8K**, which is the
benchmark this project evaluates on.

Train on the whole file, evaluate on GSM8K, and you are not measuring reasoning. You are
measuring whether the model remembers problems it has already read. The number goes up and
it means nothing.

`prepare_data.py` drops every row whose `source` is `gsm8k` before sampling:

```yaml
# configs/boxed-1.5b.yaml
data:
  drop_sources: ["gsm8k"]
  max_chars: 8000
  min_chars: 200
```

One line of filtering, and it is the difference between a result and a demo.

Rows longer than 8,000 characters get dropped too. Those solutions do not fit in a
2,048-token window, and letting them through means the model spends its whole budget
reading.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. build the training file (~20k rows after filtering)
python src/prepare_data.py --out data/sft_train.jsonl --n 20000

# 2. baseline FIRST
python src/eval_gsm8k.py --model Qwen/Qwen2.5-1.5B --limit 250 --out results/base.json

# 3. train, about 1-1.5h on one T4
python src/train_sft.py --config configs/boxed-1.5b.yaml

# 4. evaluate the adapter
python src/eval_gsm8k.py --model Qwen/Qwen2.5-1.5B \
    --adapter outputs/boxed-1.5b --limit 250 --out results/lora.json

# 5. merge and push to the Hub
python src/push_to_hub.py --config configs/boxed-1.5b.yaml --adapter outputs/boxed-1.5b
```

> [!IMPORTANT]
> Step 2 is not a warm-up. Train first and evaluate later and you never get a clean
> baseline — a repo with a fine-tuned model and no comparison asks the reader to take the
> improvement on faith.

[RUNBOOK.md](RUNBOOK.md) has the Kaggle-specific version of this, including the things that
quietly lose your work.

---

## What is in here

```text
configs/boxed-1.5b.yaml   every hyperparameter; nothing is hardcoded in the code
src/prompts.py            prompt framing and grading, stdlib only so it can be tested
src/prepare_data.py       drop the GSM8K rows, filter by length, format the turns
src/train_sft.py          QLoRA via trl + peft + bitsandbytes
src/eval_gsm8k.py         batched greedy generation and answer extraction
src/push_to_hub.py        merge into fp16 and push both repos
tests/                    CPU-only checks, no GPU or model download
results/                  eval JSON behind the table above (gitignored except *.json)
notebooks/                the Kaggle run, cell by cell
RUNBOOK.md                Kaggle specifics, including what loses your work
COMMITS.md                suggested commit sequence, for whoever reads the log
model_card.md             model card and citation block
```

---

## Tests

Two suites, neither needs a GPU or a download:

```bash
python tests/test_smoke.py       # prompt framing, extraction, grading
python tests/test_api_compat.py  # our arguments vs the installed libraries
```

The second one exists because trl 0.13 renamed the trainer's `tokenizer` argument to
`processing_class` and dropped the old name without a deprecation shim. My trainer call
raised `TypeError` on it. The test now checks our arguments against whatever trl is actually
installed, so that failure shows up as a failing test in five seconds instead of a stack
trace forty minutes into a GPU run.

It also pins the assumption the eval loop depends on: Qwen's `<|im_end|>` is a different
token from its `eos_token`. Miss that and every generation runs to the token limit and
parses to nothing — which looks exactly like a model that learned nothing.

---

## Training configuration

| | |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B`, Apache-2.0 |
| GPU | 1× Kaggle T4, 16 GB |
| Quantization | nf4 with double quant, fp16 compute |
| LoRA | r=16, alpha=32, dropout 0.05, all linear layers |
| Sequence length | 2048 |
| Effective batch | 16 (8 × 2 accumulation) |
| LR | 2e-4, cosine, 3% warmup |
| Epochs | 2 |
| Wall clock | ~1.5 h training, ~10 min per eval |
| Seed | 42 |

---

## What it cannot do

- **It is 1.5B.** This teaches the shape of a solution. It does not add knowledge.
- **GSM8K only.** Nothing here says anything about MATH, code, or instruction following.
- **One seed, one run.** A delta under about 2 points is noise.
- **The prompt is rigid.** It was trained on one phrasing and degrades on free-form chat.
- **Filtered training set, on purpose.** Not competitive with a full-data run — that was the
  tradeoff for being able to check the result.
- **Not evaluated for safety or factuality.** Do not put it in front of users.

---

## License and credits

Repo code is MIT. The weights inherit Apache-2.0 from Qwen2.5-1.5B.

Built on [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) and
[NuminaMath-CoT](https://huggingface.co/datasets/project-numina/NuminaMath-CoT), both
Apache-2.0. See [model_card.md](model_card.md) for the citation block.

<div align="center">
<sub>Built by <a href="https://github.com/Raufnarejo505">Abdul Rauf</a> · a ⭐ helps if this was useful.</sub>
</div>
