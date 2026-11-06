"""QLoRA SFT for Qwen2.5-1.5B on a single 16 GB T4.

    python src/train_sft.py --config configs/boxed-1.5b.yaml

Everything is driven by the config; nothing here is tuned in code.
"""

import argparse
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/boxed-1.5b.yaml")
    p.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    return p.parse_args()


def build_model(base_model, lora_cfg):
    # T4 (sm_75) has no bf16 support, so fp16 everywhere.
    dtype = torch.float16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        torch_dtype=dtype,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, peft_config


def main():
    a = parse_args()

    # 4-bit QLoRA needs CUDA. Fail with something readable instead of a
    # bitsandbytes stack trace three frames deep.
    if not torch.cuda.is_available():
        raise SystemExit(
            "QLoRA training needs a CUDA GPU; this repo targets Kaggle's T4.\n"
            "Run the CPU-safe checks instead: python tests/test_smoke.py"
        )

    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg, lora_cfg, t = cfg["data"], cfg["lora"], cfg["train"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, peft_config = build_model(cfg["base_model"], lora_cfg)

    ds = load_dataset("json", data_files=data_cfg["train_file"], split="train")
    print(f"[data] {len(ds)} training rows")

    args = SFTConfig(
        output_dir=t["output_dir"],
        max_seq_length=t["max_seq_length"],
        dataset_text_field="text",
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        seed=t["seed"],
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,  # keep samples separate; packing hides truncation bugs
        report_to=[],
    )

    # trl 0.12.2 renamed `tokenizer` -> `processing_class` and does NOT accept
    # **kwargs, so passing `tokenizer=` here raises TypeError, not a warning.
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=a.resume)

    out = t["output_dir"]
    os.makedirs(out, exist_ok=True)
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"[done] adapter saved to {out}")


if __name__ == "__main__":
    main()
