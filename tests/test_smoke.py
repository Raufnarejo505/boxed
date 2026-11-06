"""Fast checks that need no GPU, no network, and no model download.

    python tests/test_smoke.py

Run this anywhere with Python 3.9+. If PyYAML is missing the config check is
skipped rather than failed, so the suite still runs on a bare interpreter.

It catches the failure modes that otherwise only surface 40 minutes into a run:
a train/eval prompt mismatch, and an answer extractor that silently scores 0.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import prompts  # noqa: E402

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        failures.append(name)
        print(f"  FAIL  {name}: {e}")
    except Exception as e:  # noqa: BLE001
        failures.append(name)
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- framing
# The single most expensive bug: train on one prompt, evaluate on another.
def test_training_text_starts_with_eval_prefix():
    train = prompts.training_text("What is 1+1?", "It is 2. \\boxed{2}")
    prefix = prompts.eval_prefix("What is 1+1?")
    assert train.startswith(prefix), (
        "train/eval framing diverged\n"
        f"  train={train[:120]!r}\n  eval ={prefix[:120]!r}"
    )


def test_answer_convention_renders_once():
    text = prompts.eval_prefix("Q")
    assert "\\boxed{}" in text, "prompt lost its \\boxed{} convention"
    assert "{{" not in text and "}}" not in text, "double braces leaked through .format"


def test_problem_reaches_the_model():
    assert "UniqueMarker42" in prompts.eval_prefix("UniqueMarker42")


def test_training_text_is_terminated():
    t = prompts.training_text("q", "a")
    assert t.endswith("<|im_end|>\n"), repr(t[-20:])
    assert t.count("<|im_end|>") == 2, "expected one terminator per turn"


# -------------------------------------------------------------- extraction
def test_boxed_simple():
    assert prompts.extract_boxed("blah \\boxed{42}") == "42"


def test_boxed_takes_last():
    assert prompts.extract_boxed("\\boxed{1} then \\boxed{2}") == "2"


def test_boxed_nested_braces():
    assert prompts.extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_boxed_unbalanced_does_not_hang():
    assert prompts.extract_boxed("\\boxed{12") == "12"


def test_boxed_absent():
    assert prompts.extract_boxed("no answer here") is None


def test_prediction_falls_back_to_last_number():
    assert prompts.extract_prediction("so the answer is 18") == "18"
    assert prompts.extract_prediction("first 3 then 7") == "7"
    assert prompts.extract_prediction("nothing") is None


def test_prediction_prefers_boxed_over_trailing_number():
    assert prompts.extract_prediction("\\boxed{5} and also 9") == "5"


# -------------------------------------------------------------- comparison
def test_normalization():
    assert prompts.is_correct("1,000", "1000") is True
    assert prompts.is_correct("12.", "12") is True
    assert prompts.is_correct("$5$", "5") is True
    assert prompts.is_correct(" 42 ", "42") is True
    assert prompts.is_correct("0.50", "0.5") is True
    assert prompts.is_correct(None, "5") is False
    assert prompts.is_correct("7", "8") is False


def test_known_limitation_is_documented():
    # LaTeX fractions are NOT graded equivalent to decimals. Documented
    # limitation, not an accident: widening the grader risks false positives.
    assert prompts.is_correct("\\frac{1}{2}", "0.5") is False


def test_gold_extraction():
    assert prompts.gold_of("reasoning\n#### 18") == "18"
    assert prompts.gold_of("#### 1,234") == "1,234"


# ------------------------------------------------------------------ config
def test_config_loads():
    try:
        import yaml
    except ImportError:
        raise AssertionError("skipped: PyYAML not installed") from None
    path = os.path.join(HERE, "..", "configs", "boxed-1.5b.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["base_model"] == "Qwen/Qwen2.5-1.5B"
    assert "gsm8k" in cfg["data"]["drop_sources"], "decontamination source was removed"
    t = cfg["train"]
    assert t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] == 16


if __name__ == "__main__":
    print("running smoke checks\n")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name[5:], fn)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all checks passed")
