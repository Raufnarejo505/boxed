"""GSM8K evaluation: batched greedy generation + robust answer extraction.

    # baseline (must run this before training)
    python src/eval_gsm8k.py --model Qwen/Qwen2.5-1.5B --limit 250 --out results/base.json

    # adapter
    python src/eval_gsm8k.py --model Qwen/Qwen2.5-1.5B \
        --adapter outputs/boxed-1.5b --limit 250 --out results/lora.json

Greedy decoding, one seed, exact match on the final numeric answer. Deliberately
simple: a complicated grader makes the result harder to defend, not easier.

Prompting and grading live in `prompts.py` so they can be tested without a GPU.
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import eval_prefix, extract_prediction, gold_of, is_correct


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir to attach")
    p.add_argument("--dataset", default="openai/gsm8k")
    p.add_argument("--config", default="main")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out", default="results/eval.json")
    p.add_argument("--save-preds", action="store_true",
                   help="store raw generations too (bigger file, easier to debug)")
    return p.parse_args()


def main():
    from datasets import load_dataset

    a = parse_args()
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve the device instead of hardcoding GPU 0, so the script also runs on
    # a CPU-only machine for smoke tests. On Kaggle this picks the T4.
    if torch.cuda.is_available():
        device_map, dtype = {"": 0}, torch.float16
    else:
        device_map, dtype = "cpu", torch.float32
        print("[warn] no CUDA device detected; running on CPU in fp32 (slow)")

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dtype, device_map=device_map
    )
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        model = model.merge_and_unload()
    model.eval()

    # Qwen's eos_token is <|endoftext|>, but generation ends at <|im_end|>.
    # Without both, the model rambles to max_new_tokens and nothing parses.
    eos_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        eos_ids.append(im_end)

    ds = load_dataset(a.dataset, a.config, split=a.split)
    n = min(a.limit, len(ds))

    records, n_correct = [], 0
    for start in range(0, n, a.batch_size):
        batch = ds.select(range(start, min(start + a.batch_size, n)))
        prompts = [eval_prefix(q) for q in batch["question"]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=a.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        for prompt, ids, gold_raw in zip(prompts, out[:, prompt_len:], batch["answer"]):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            gold = gold_of(gold_raw)
            pred = extract_prediction(text)
            ok = is_correct(pred, gold)
            n_correct += int(ok)
            rec = {"question": prompt, "gold": gold, "pred": pred, "correct": ok}
            if a.save_preds:
                rec["generation"] = text
            records.append(rec)

        done = min(start + a.batch_size, n)
        print(f"[{done}/{n}] running acc = {n_correct / done:.3f}", flush=True)

    acc = n_correct / n
    print(f"\n=== GSM8K accuracy: {acc:.4f} ({n_correct}/{n}) ===")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    payload = {
        "model": a.model,
        "adapter": a.adapter,
        "dataset": f"{a.dataset}:{a.config}:{a.split}",
        "n": n,
        "accuracy": acc,
        "n_correct": n_correct,
        "decoding": "greedy",
        "max_new_tokens": a.max_new_tokens,
        "records": records,
    }
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {a.out}")

    # Show a few failures: they tell you whether the model can't reason or
    # your extractor can't parse.
    for r in [r for r in records if not r["correct"]][:3]:
        print(f"\nFAIL gold={r['gold']!r} pred={r['pred']!r}")


if __name__ == "__main__":
    main()
