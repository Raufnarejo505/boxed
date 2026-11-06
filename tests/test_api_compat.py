"""Verify our code against the *installed* library APIs, not against assumption.

    python tests/test_api_compat.py

This suite exists because of a real bug: trl 0.12.2's ``SFTTrainer.__init__``
has no ``tokenizer`` parameter and no ``**kwargs``, so ``tokenizer=tokenizer``
raised TypeError at build time. Introspecting the signatures is the only way to
catch this class of drift before a 40-minute training run does.

Skips cleanly (exit 0) when the ML stack is not installed.
"""

import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
CONFIG = os.path.join(HERE, "..", "configs", "boxed-1.5b.yaml")
sys.path.insert(0, SRC)

failures = []
skipped = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except ImportError as e:
        skipped.append(name)
        print(f"  SKIP  {name} ({e})")
    except AssertionError as e:
        failures.append(name)
        print(f"  FAIL  {name}: {e}")
    except Exception as e:  # noqa: BLE001
        failures.append(name)
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


def _params(obj):
    return set(inspect.signature(obj.__init__).parameters)


# ------------------------------------------------------------------- trl
def test_sfttrainer_accepts_what_we_pass():
    from trl import SFTTrainer

    params = _params(SFTTrainer)
    required = {"model", "args", "train_dataset", "peft_config", "processing_class"}
    missing = required - params
    assert not missing, f"SFTTrainer is missing {sorted(missing)}; signature drifted"

    # If `tokenizer` is absent, it must not be sneaked through **kwargs.
    if "tokenizer" not in params:
        assert "kwargs" not in params, (
            "SFTTrainer has **kwargs but no tokenizer param; verify manually"
        )


def test_sftconfig_has_our_fields():
    from trl import SFTConfig

    fields = set(SFTConfig.__dataclass_fields__)
    for key in ("max_seq_length", "dataset_text_field", "packing"):
        assert key in fields, f"SFTConfig has no {key!r} (renamed?)"


def test_every_yaml_train_key_is_valid():
    import yaml
    from transformers import TrainingArguments
    from trl import SFTConfig

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    valid = set(SFTConfig.__dataclass_fields__) | set(
        TrainingArguments.__dataclass_fields__
    )
    unknown = [k for k in cfg["train"] if k not in valid]
    assert not unknown, f"configs/boxed-1.5b.yaml has unknown keys: {unknown}"


# ------------------------------------------------------------------ peft
def test_lora_keys_are_valid():
    from peft import LoraConfig

    fields = set(LoraConfig.__dataclass_fields__)
    for key in ("r", "lora_alpha", "lora_dropout", "target_modules", "bias",
                "task_type"):
        assert key in fields, f"LoraConfig has no {key!r}"


def test_peft_helpers_exist():
    from peft import PeftModel, prepare_model_for_kbit_training

    assert callable(prepare_model_for_kbit_training)
    # merge_and_unload is defined on the tuner (LoraModel) and reached from a
    # PeftModel *instance* through __getattr__ delegation. Checking it on the
    # class gives a false negative -- a bug in an earlier version of this test.
    from peft.tuners.lora import LoraModel

    assert hasattr(LoraModel, "merge_and_unload"), "LoraModel lost merge_and_unload"
    assert hasattr(PeftModel, "__getattr__"), "PeftModel no longer delegates to the tuner"


# ------------------------------------------------------------ transformers
def test_bnb_config_keys_are_valid():
    # Inspect __init__ rather than __dataclass_fields__: the dataclass only
    # declares `quant_method`, so the field check silently reports every real
    # argument as missing. Do not construct it -- that requires bitsandbytes,
    # which is intentionally absent on a CPU-only box.
    from transformers import BitsAndBytesConfig

    params = set(inspect.signature(BitsAndBytesConfig.__init__).parameters)
    for key in ("load_in_4bit", "bnb_4bit_quant_type",
                "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant"):
        assert key in params, f"BitsAndBytesConfig has no {key!r}"


def test_qwen_tokenizer_has_chat_tokens():
    # No download when the tokenizer is already cached; otherwise skip.
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception as e:  # noqa: BLE001
        raise ImportError(f"tokenizer not cached: {type(e).__name__}") from None

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert im_end is not None and im_end != tok.unk_token_id, (
        "<|im_end|> is not a known token; eval will never stop generating"
    )
    # The thing this whole repo depends on: eos and im_end are DIFFERENT.
    assert im_end != tok.eos_token_id, (
        "im_end equals eos; the dual-eos handling in eval_gsm8k.py is redundant"
    )


# ------------------------------------------------------------------ static
def test_train_script_uses_processing_class():
    import re

    with open(os.path.join(SRC, "train_sft.py")) as f:
        src = f.read()
    # Strip comments so the explanatory note about the old name does not trip this.
    code = re.sub(r"#.*", "", src)
    assert "processing_class=tokenizer" in code, "SFTTrainer call lost processing_class"
    assert "tokenizer=tokenizer" not in code, (
        "train_sft.py passes tokenizer= to SFTTrainer; trl 0.12.2 rejects it"
    )


def test_eval_passes_both_eos_ids():
    import re

    with open(os.path.join(SRC, "eval_gsm8k.py")) as f:
        code = re.sub(r"#.*", "", f.read())
    assert "eos_token_id=eos_ids" in code, "eval no longer passes the eos list"
    assert "im_end" in code, "eval lost the <|im_end|> handling"


if __name__ == "__main__":
    print("checking installed APIs\n")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name[5:], fn)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    if skipped:
        print(f"all checks passed ({len(skipped)} skipped: missing deps)")
    else:
        print("all checks passed")
