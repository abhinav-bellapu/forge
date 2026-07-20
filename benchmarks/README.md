# GPT-2 validation results

These reports were produced on an Apple M5 CPU with 10 cores using the public
`openai-community/gpt2` 124M-parameter SafeTensors checkpoint. Model files and
raw 50,257-element logit exports are gitignored; the compact reports are kept
for reproducibility.

Artifact SHA-256 values used for these measurements:

- `config.json`: `0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4`
- `tokenizer.json`: `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`
- `model.safetensors`: `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`

## Numerical parity

Forge FP32 logits were compared with `transformers.GPT2LMHeadModel` across three
prompts and all 150,771 resulting vocabulary logits. Maximum absolute error was
`1.876e-4`; all three argmax tokens matched. See `gpt2_fp32_parity*.json` and
`gpt2_parity_summary.json`.

Reproduce one case:

```bash
cargo run --release -- gpt2-logits \
  --model-dir models/gpt2 \
  --prompt "Hello, my name is" \
  --output /tmp/forge_logits.json

python -m venv /tmp/forge-parity
/tmp/forge-parity/bin/pip install -r scripts/requirements-parity.txt
/tmp/forge-parity/bin/python scripts/export_hf_logits.py \
  --model-dir models/gpt2 \
  --prompt "Hello, my name is" \
  --output /tmp/hf_logits.json
/tmp/forge-parity/bin/python scripts/compare_logits.py \
  --reference /tmp/hf_logits.json \
  --candidate /tmp/forge_logits.json
```

## Performance and memory

Five-run generation benchmarks used an eight-token output and excluded model
loading. The checked-in reports record raw timings:

- FP32: 13.82 tokens/s on one thread and 46.71 tokens/s on 10 threads (`3.38x`).
- INT8: 21.92 tokens/s on one thread and 65.34 tokens/s on 10 threads (`2.98x`).
- INT8 weight storage: 125,340,740 bytes versus 497,759,232 bytes FP32 (`74.82%` reduction).

Performance varies with load and hardware. Resume language intentionally uses
the more conservative reproduced figure of `2.6x` parallel speedup.
