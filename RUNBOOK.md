# RUNBOOK — running this on Kaggle

Kaggle is an unusual environment: sessions are capped, the disk is wiped, and the internet is
off by default. Read this before your first run, or you will lose a checkpoint.

## 0. One-time setup

1. **Verify your phone** on Kaggle, or you cannot enable internet in notebooks.
2. Create a [Hugging Face](https://huggingface.co) account and a **write** token:
   Settings → Access Tokens → New token → *Write*.
3. In Kaggle: **Add-ons → Secrets** → add `HF_TOKEN` with your HF token.
   Notebooks can then read it via `UserSecretsClient` — never paste a token into a cell.
4. Create a GitHub repo (empty, no README) and a HF model repo `abdulrauf5121/boxed-1.5b-lora`.

## 1. Create the notebook

- New Notebook → **File → Import** the files from this repo (or clone from GitHub inside the
  notebook once internet is on).
- **Settings → Accelerator:** `GPU T4 x2`. You only need one, but verify which you got:

```python
import torch
print(torch.cuda.get_device_name(0))   # should say Tesla T4
print(torch.cuda.is_available())       # must be True
```

- **Settings → Internet: On.** Without this, `pip install` and `push_to_hub` both fail.
- **Settings → Persistence:** "Files only" is enough.

## 2. Install

```python
!pip install -q -r requirements.txt
```

Pin exact versions — `transformers` on Kaggle's image drifts and `trl`'s trainer API changed
across 0.12 → 0.13. If you skip the pins and get `unexpected keyword argument 'max_seq_length'`
or `'dataset_text_field'`, that is the version drift, not your bug.

**After installing, restart the kernel** (Run → Restart). A stale `transformers` in memory is
the second most common cause of weird import errors.

## 2.5 Verify before you spend quota

```python
!python tests/test_smoke.py
!python tests/test_api_compat.py
```

Both must exit 0. These cost seconds and catch the problems that otherwise surface
20 minutes into a run. If `test_api_compat` complains about `processing_class`, the
installed `trl` is not the pinned version — reinstall from `requirements.txt` and
restart the kernel.

## 3. Secrets + login

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

login(token=UserSecretsClient().get_secret("HF_TOKEN"))
```

## 4. Order of operations

```python
!python src/prepare_data.py --out data/sft_train.jsonl --n 20000
!python src/eval_gsm8k.py --model Qwen/Qwen2.5-1.5B --limit 250 --out results/base.json
!python src/train_sft.py --config configs/boxed-1.5b.yaml
!python src/eval_gsm8k.py --model Qwen/Qwen2.5-1.5B --adapter outputs/boxed-1.5b \
    --limit 250 --out results/lora.json
!python src/push_to_hub.py --config configs/boxed-1.5b.yaml --adapter outputs/boxed-1.5b
```

Run the **baseline eval first**. If you train first, you will "forget", and a repo with no
baseline number is a repo nobody believes.

## 5. Do not lose your work

Kaggle wipes `/kaggle/working` unless you save it. Three safety nets, use at least two:

- **Push to the Hub as you go.** `train_sft.py` writes checkpoints every 200 steps; after
  training, run `push_to_hub.py` immediately.
- **Save Version → "Save & Run All (Commit)"** at the end. That persists output files.
- Set `save_steps` low enough that a 12 h timeout never costs more than ~10 minutes of work.

The 12-hour session cap is real. A 1.5B LoRA run of this size finishes in 1–2 hours, so you
have headroom — but a 7B run at these settings would not.

## 6. Quota math

Free tier is roughly **30 GPU-hours/week**, reset weekly. Budget for this project:

| Step | Time on one T4 |
|---|---|
| prepare data | 2–4 min (cpu-bound; run with GPU off to save quota) |
| baseline eval, n=250 | ~10 min |
| training, 2 epochs @ 20k rows | ~1.0–1.5 h |
| adapter eval, n=250 | ~10 min |
| merge + push | ~10 min |

That leaves ~90% of the weekly quota for experiments — ablate learning rate, r, or data size.

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | batch too big for 16 GB | set `per_device_train_batch_size: 4` and double `gradient_accumulation_steps` |
| loss → `nan` after a few steps | fp16 instability | lower LR to 1e-4, keep `optim: paged_adamw_8bit` |
| eval score ≈ 0 | answer extraction mismatch | inspect `results/*.json` raw predictions |
| `push_to_hub` 401 | token lacks write scope | regenerate as *Write*, re-add the secret |
| model outputs are one long run-on | trained on over-long sequences | the data filter caps length at 8000 chars; check it ran |
| session killed at 9h on TPU | TPU cap is 9h, not 12h | use GPU for this repo |
