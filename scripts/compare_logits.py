#!/usr/bin/env python3
"""Compare Forge and Hugging Face logit exports and emit a JSON report."""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = load(args.reference)
    candidate = load(args.candidate)
    if reference["prompt"] != candidate["prompt"]:
        raise SystemExit("prompt mismatch")
    if reference["token_ids"] != candidate["token_ids"]:
        raise SystemExit("token ID mismatch")
    expected = reference["last_logits"]
    actual = candidate["last_logits"]
    if len(expected) != len(actual):
        raise SystemExit("logit length mismatch")

    errors = [abs(a - b) for a, b in zip(actual, expected)]
    ref_argmax = max(range(len(expected)), key=expected.__getitem__)
    candidate_argmax = max(range(len(actual)), key=actual.__getitem__)
    report = {
        "reference": reference["implementation"],
        "candidate": candidate["implementation"],
        "prompt": reference["prompt"],
        "tokens": len(reference["token_ids"]),
        "logits": len(expected),
        "max_absolute_error": max(errors),
        "mean_absolute_error": sum(errors) / len(errors),
        "reference_argmax": ref_argmax,
        "candidate_argmax": candidate_argmax,
        "argmax_match": ref_argmax == candidate_argmax,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
