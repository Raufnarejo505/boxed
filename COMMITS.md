# Suggested commit history

You have the whole repo built in one go, so you can commit it as a single blob and
nobody will complain. But a reviewer reading `git log` learns more from a handful of
commits that each explain *why* something exists. Below is a sequence that holds
together, in an order that actually builds.

**The style**, if you want to follow it:

- Subject line: imperative mood, lowercase, no trailing period, aim for 50 characters.
  "add" not "added", "fix" not "fixed".
- Blank line, then a body that explains the reason, not the mechanics. The diff already
  shows *what* changed; the body is for *why*.
- Wrap the body at 72 characters.
- Prefixes (`feat:`, `test:`, `docs:`) are Conventional Commits. Useful, common, and
  easy to drop if you find them noisy.

---

### 1.

```
chore: scaffold repo with config, license and gitignore
```

Everything that varies between runs lives in `configs/boxed-1.5b.yaml`, so the scripts
stay readable and you can ablate a hyperparameter without touching Python. Code is MIT;
the weights stay Apache-2.0 because they are derived from Qwen2.5-1.5B.

Files: `configs/`, `LICENSE`, `.gitignore`, `requirements.txt`

### 2.

```
feat: add prompt framing and grading helpers
```

Training and evaluation have to build the same prompt, or the comparison means
nothing. Both use `src/prompts.py`, which imports only the standard library, so the
framing and the answer check can be tested on a laptop with no GPU and no model
download. It is also the only place the `\boxed{}` convention is written down.

Files: `src/prompts.py`

### 3.

```
feat: add decontaminated data preparation
```

NuminaMath-CoT ships 7,345 rows sourced from GSM8K, which is the benchmark this project
evaluates on. Training on those would turn a reasoning result into a recall result. Drop
them before sampling, then drop solutions over 8,000 characters, which cannot fit the
2,048-token window and crowd out the problems that can.

Files: `src/prepare_data.py`

### 4.

```
feat: add QLoRA training script
```

4-bit nf4 with LoRA on every linear projection fits a 1.5B model into 16 GB. The T4 has
no bf16, so compute stays in fp16 and the optimizer is paged AdamW 8-bit. The script
exits early without CUDA instead of failing three frames deep inside bitsandbytes.

Files: `src/train_sft.py`

### 5.

```
feat: add GSM8K evaluation with batched generation
```

Batched greedy decoding with left padding so the answers line up. The `\boxed{}` extractor
matches braces instead of using a regex, because `\boxed{\frac{1}{2}}` truncates under a
regex and silently scores zero. Generation stops on a second eos token, `<|im_end|>`,
because Qwen's actual `eos_token` is `<|endoftext|>` and the model will happily run to the
token limit without it.

Files: `src/eval_gsm8k.py`

### 6.

```
feat: add merge and Hub publishing
```

Push the adapter first. It is about 20 MB, it takes seconds, and it means a dead session
cannot cost the whole run. The merge runs in fp32 and only then casts to fp16 for
storage; merging into half precision throws away accuracy for no benefit.

Files: `src/push_to_hub.py`

### 7.

```
test: add CPU-only checks and API drift detection
```

The smoke suite covers prompt framing and grading. The API suite exists because trl 0.13
renamed the trainer's `tokenizer` argument to `processing_class` and removed the old name
without a shim, which turned my trainer call into a `TypeError`. Comparing our arguments
against whatever version is installed makes that a failed test in five seconds instead of
a wasted GPU session forty minutes in.

Files: `tests/`

### 8.

```
docs: add Kaggle runbook and model card
```

The runbook records the things that cost real time: sessions wipe `/kaggle/working`, the
internet is off by default, and a stale `transformers` stays in memory after `pip`
upgrades it. The model card states the limitations plainly, including that the training
set was deliberately filtered.

Files: `README.md`, `RUNBOOK.md`, `model_card.md`, `notebooks/`

### 9. After you run it

```
docs: record baseline and adapter accuracy
```

Fill the results table from `results/base.json` and `results/lora.json`. Commit both JSON
files too, so the numbers in the README can be checked against the raw predictions.

Files: `README.md`, `model_card.md`, `results/`

---

## Commands

`git` is not installed on the machine this repo was built on, so this part is yours.

```bash
git init
git config user.name  "Your Name"
git config user.email "you@example.com"
git add -A
git commit -m "chore: scaffold repo with config, license and gitignore"
```

For a commit with a body, repeat `-m`:

```bash
git add src/prompts.py
git commit -m "feat: add prompt framing and grading helpers" \
           -m "Training and evaluation have to build the same prompt, or the comparison means nothing. Both use src/prompts.py, which imports only the standard library, so the framing and answer check can be tested with no GPU and no model download. It is also the only place the \boxed{} convention is defined."
```

Then create the empty repo on GitHub and push:

```bash
git remote add origin https://github.com/Raufnarejo505/boxed.git
git branch -M master
git push -u origin master
```

## One commit instead

If the history is not worth the effort, use this and move on:

```
feat: post-train Qwen2.5-1.5B on decontaminated NuminaMath with QLoRA

QLoRA run on a filtered slice of NuminaMath-CoT with the 7,345 GSM8K-derived
rows removed, evaluated on GSM8K against the untrained baseline. Ships the data
prep, training, eval and merge scripts plus CPU-only tests, so the reported
delta can be re-run instead of taken on trust.
```

## What must not be committed

`.gitignore` already covers it, but in case you override it:

- `.venv/` and `__pycache__/`
- `data/`, `outputs/`, anything ending in `.safetensors`
- `results/` is the exception: those JSON files belong in the repo, because they are the
  evidence behind the README table.
