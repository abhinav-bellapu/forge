#!/usr/bin/env python3
"""Export Hugging Face GPT-2 last-position logits for Forge parity checks."""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    model = GPT2LMHeadModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_safetensors=True,
    )
    model.eval()
    with torch.inference_mode():
        logits = model(torch.tensor([token_ids], dtype=torch.long)).logits[0, -1]

    payload = {
        "implementation": "huggingface-pytorch",
        "prompt": args.prompt,
        "token_ids": token_ids,
        "last_logits": logits.float().cpu().tolist(),
    }
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {len(payload['last_logits'])} logits to {args.output}")


if __name__ == "__main__":
    main()
