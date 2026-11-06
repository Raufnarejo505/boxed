"""Push the LoRA adapter and the merged fp16 model to the Hugging Face Hub.

    python src/push_to_hub.py --config configs/boxed-1.5b.yaml \
        --adapter outputs/boxed-1.5b --adapter-only

Requires `huggingface-cli login` (or HF_TOKEN in the environment).
"""

import argparse
import os

import torch
import yaml
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/boxed-1.5b.yaml")
    p.add_argument("--adapter", default="outputs/boxed-1.5b")
    p.add_argument("--merged-out", default="outputs/merged")
    p.add_argument("--adapter-only", action="store_true",
                   help="skip the merge; useful as an early checkpoint backup")
    return p.parse_args()


def main():
    a = parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    hub, api = cfg["hub"], HfApi()

    # 1. Adapter repo: small (~20 MB), push this first so work is never lost.
    api.create_repo(hub["lora_repo"], private=hub["private"], exist_ok=True)
    api.upload_folder(
        folder_path=a.adapter,
        repo_id=hub["lora_repo"],
        commit_message="Add LoRA adapter (boxed-1.5b)",
    )
    print(f"[hub] adapter -> https://huggingface.co/{hub['lora_repo']}")

    if a.adapter_only:
        return

    # 2. Merged repo. Merge in fp32 -- accumulating fp16 deltas loses precision,
    #    and merging into 4-bit weights is lossy -- then cast down for storage.
    device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(a.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.float32, device_map=device_map
    )
    model = PeftModel.from_pretrained(base, a.adapter)
    model = model.merge_and_unload().half()

    os.makedirs(a.merged_out, exist_ok=True)
    model.save_pretrained(a.merged_out, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(a.merged_out)
    print(f"[merge] wrote {a.merged_out}")

    api.create_repo(hub["merged_repo"], private=hub["private"], exist_ok=True)
    api.upload_folder(
        folder_path=a.merged_out,
        repo_id=hub["merged_repo"],
        commit_message="Add merged fp16 model (boxed-1.5b)",
    )
    print(f"[hub] merged -> https://huggingface.co/{hub['merged_repo']}")


if __name__ == "__main__":
    main()
