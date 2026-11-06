---
base_model: Qwen/Qwen2.5-1.5B
library_name: peft
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
datasets:
  - project-numina/NuminaMath-CoT
tags:
  - qlora
  - lora
  - math
  - reasoning
  - sft
---

# boxed

A 1.5B model that shows its work and puts the final answer in `\boxed{}`.

This is Qwen2.5-1.5B (base) after a short QLoRA run on a filtered slice of
NuminaMath-CoT. It was built as a small, reproducible demonstration of
post-training on free hardware, not as a competitive math model. A single Kaggle
T4 was enough.

## Results

GSM8K test set, greedy decoding, exact match on the final answer, 250 questions.

| Model | Accuracy |
|---|---|
| `Qwen/Qwen2.5-1.5B` (base) | _TBD_ |
| **this model** | _TBD_ |

Both numbers come from the same script with the same prompt and one seed. The
eval script ships with the repo, so the comparison can be re-run rather than
taken on trust.

## Why the dataset is filtered

NuminaMath-CoT contains 859,494 rows, and 7,345 of them were sourced from GSM8K
itself. Training on those and then evaluating on GSM8K would measure recall of
the benchmark instead of reasoning. Every row with `source == "gsm8k"` was
removed before sampling.

That leaves the model trained on a deliberately smaller and slightly biased
slice. It costs absolute score and buys an honest delta. This is not aiming to be
a state of the art math model.

Rows longer than 8,000 characters were dropped as well, since they do not fit a
2,048-token window. The final 20,000 rows were sampled with seed 42.

## Training

| | |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B` (Apache-2.0) |
| Method | QLoRA, 4-bit nf4 with double quant, fp16 compute |
| LoRA | r=16, alpha=32, dropout 0.05, applied to all linear projections |
| Sequence length | 2,048 |
| Batch | 8 per device, 2 accumulation steps, 16 effective |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Epochs | 2 |
| Optimizer | paged AdamW 8-bit |
| Hardware | one T4 with 16 GB, roughly 1.5 hours |

If you hit out-of-memory on a 16 GB card, drop the batch size to 4 and raise
accumulation to 4. The effective batch stays the same.

## Using it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "abdulrauf5121/boxed-1.5b-merged"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="float16", device_map="auto")

prompt = ("Solve the following math problem step by step. "
          "Put the final answer in \\boxed{}.\n\n"
          "A train travels 60 km/h for 2.5 hours. How far does it go?")
text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
    eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")],
)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The `<|im_end|>` entry in `eos_token_id` matters. Qwen's `eos_token` is
`<|endoftext|>`, but this model stops on `<|im_end|>`. Pass only the first one
and it runs to the token limit every time.

The adapter-only repo loads on top of the base model with `peft.PeftModel` if you
would rather not pull the merged weights.

## Limitations

- 1.5B parameters. It makes solutions look plausible; that is not the same as
  being right.
- Evaluated on GSM8K only. Nothing here supports claims about MATH, code or
  general chat.
- One seed, one run. Differences under roughly 2 points are noise.
- The prompt format is rigid. It was trained on one phrasing and degrades on
  free-form instructions.
- Trained on a filtered subset by design, so it is weaker than a full-data run.
- Not checked for safety or factuality. Do not serve it to users.

## Attribution

Built on [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B) and
[`project-numina/NuminaMath-CoT`](https://huggingface.co/datasets/project-numina/NuminaMath-CoT),
both Apache-2.0. This model is distributed under Apache-2.0.
