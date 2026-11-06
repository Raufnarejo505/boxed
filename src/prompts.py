"""Prompt framing and answer grading — the parts worth testing without a GPU.

Deliberately imports nothing but the standard library. Both the training script
and the eval script build their prompts here, so "train on one format, evaluate
on another" cannot happen silently.

The escaping that matters: in the source, the instruction ends with ``{{}}``
because ``str.format`` collapses it to a literal ``\\boxed{}``. A prompt that
says ``{{}}`` would still run, and would quietly teach the model the wrong
output convention.
"""

import re

PROMPT = (
    "Solve the following math problem step by step. "
    "Put the final answer in \\boxed{{}}.\n\n{problem}"
)

# Full training example: framing, answer, and the turn terminator.
CHAT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"

# Inference prompt: identical framing, but open-ended for generation.
PREFIX = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def instruction(problem):
    """The user turn, with the answer convention spelled out."""
    return PROMPT.format(problem=problem.strip())


def training_text(problem, solution):
    """One complete SFT example."""
    return CHAT.format(prompt=instruction(problem), answer=solution.strip())


def eval_prefix(problem):
    """The exact context the model sees at evaluation time."""
    return PREFIX.format(prompt=instruction(problem))


def extract_boxed(text):
    """Content of the LAST \\boxed{...}, matching nested braces.

    Brace matching (rather than a regex) is required because answers like
    \\boxed{\\frac{1}{2}} are common and a naive regex truncates them.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth, buf = 1, []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf)
        buf.append(ch)
        i += 1
    return "".join(buf)  # unbalanced: take what is there


def extract_prediction(text):
    """Prefer \\boxed{}, else the last number in the text."""
    boxed = extract_boxed(text)
    if boxed is not None and boxed.strip():
        return boxed
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def normalize(s):
    """Strip presentation noise that should not affect correctness."""
    s = s.strip()
    s = re.sub(r"\\text\{.*?\}", "", s)
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    s = s.replace("\\%", "").replace("%", "")
    return s.rstrip(".")


def is_correct(pred, gold):
    """Exact match after normalization, with a numeric fallback."""
    if pred is None:
        return False
    a, b = normalize(pred), normalize(gold)
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def gold_of(answer_field):
    """GSM8K stores the final answer after '####'."""
    return answer_field.split("####")[-1].strip()
