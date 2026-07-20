# Forge

A Rust inference runtime for transformer language models, from educational
building blocks through Hugging Face GPT-2 compatibility.

## Current status

work in progress, most recently: GPT-2 SafeTensors inference, INT8 quantization,
and parallel CPU kernels

## Implemented

- Project skeleton and CLI (`generate`, `train`, `eval`, `bench`, `inspect`)
- Character-level tokenizer (`vocab.json`, encode/decode)
- Tensor engine, causal self-attention, minimal `TinyModel` forward pass
- Autoregressive decoding (greedy, temperature, top-k, top-p, seeded sampling)
- JSON checkpoint save/load for model weights
- KV-cached decoding and incremental generation (optimized autoregressive inference)
- Multi-head causal attention and multi-head KV cache
- Learned per-layer projection after concatenated attention heads
- Transformer residual pathways and layer normalization
- Feed-forward network (MLP) with GELU activation
- Second residual pathway
- Second LayerNorm
- Multi-layer transformer architecture
- Per-layer KV cache
- Configurable depth (`n_layers`)
- Optional tied input/output embeddings (`tie_embeddings`)
- Cross-entropy loss and local text dataset loader
- Educational output-layer training (embeddings + `w_o`)
- Hidden-state gradients into prefix token embeddings
- Input-side gradients into positional embeddings
- Sliding training windows capped to the model context length
- Read-only checkpoint evaluation with loss and perplexity (`forge eval`)
- Optional global gradient clipping for stable local training
- Active/stored parameter accounting and model inspection (`forge inspect`)
- Finite-difference gradient checking for trained parameters
- Local inference benchmarking (`forge bench`, no network or file output by default)
- Hugging Face GPT-2 `config.json` + `model.safetensors` loading
- GPT-2 BPE tokenization from `tokenizer.json`
- GPT-2 pre-LayerNorm blocks, fused QKV projections, final LayerNorm, and tied logits
- Symmetric per-channel INT8 weight quantization for embeddings and linear layers
- Cache-friendly tiled matrix multiplication with Rayon CPU parallelism
- Full-vocabulary logit parity tooling against Hugging Face/PyTorch
- Machine-readable FP32/INT8 throughput and memory benchmarks

## GPT-2 inference

Download the public GPT-2 artifacts (about 524 MiB, gitignored):

```bash
bash scripts/download_gpt2.sh
```

Generate with FP32 weights:

```bash
cargo run --release -- gpt2-generate \
  --model-dir models/gpt2 \
  --prompt "Hello, my name is" \
  --max-new-tokens 20
```

Quantize all matrix weights to per-channel INT8 before generation:

```bash
cargo run --release -- gpt2-generate \
  --model-dir models/gpt2 \
  --prompt "Hello, my name is" \
  --max-new-tokens 20 \
  --int8
```

### Reproduced GPT-2 results

On an Apple M5 CPU (10 cores), the checked-in five-run reports measured:

- `2.98x` INT8 parallel speedup (65.34 vs 21.92 tokens/s)
- `74.82%` model-weight memory reduction (474.7 MiB to 119.5 MiB)
- `1.876e-4` maximum FP32 logit error versus Hugging Face across 150,771 logits
- matching argmax tokens on all three validation prompts

See [`benchmarks/`](benchmarks/) for the reports and exact reproduction commands.

Run an FP32 or INT8 benchmark and optionally save JSON:

```bash
cargo run --release -- gpt2-bench \
  --model-dir models/gpt2 \
  --max-new-tokens 8 \
  --runs 5 \
  --int8 \
  --json-output /tmp/gpt2-int8.json
```

## Gradient checking

Forge compares **analytic gradients** (hand-derived backprop through the output layer and
embedding inputs) against **numerical gradients** (central finite differences on tiny
deterministic models). This validates the training update path before adding deeper
backprop through transformer layers. Numerical checks also verify that positional and
untied token embeddings have the same input-side effect for a unique token position.
Transformer attention, FFN, and LayerNorm weights remain frozen during training.

## Quick start

```bash
cargo build
cargo test
```

### Random generation (default)

```bash
cargo run -- generate --prompt "hello"
cargo run -- generate --prompt "hello" --temperature 0 --seed 42
cargo run -- generate --prompt "hello" --temperature 0.8 --top-k 10 --seed 42
cargo run -- generate --prompt "hello" --temperature 0.8 --top-p 0.9 --seed 42
```

`--top-k` and `--top-p` are alternative sampling filters and cannot be combined.

### Checkpoints

Save a random model to JSON:

```bash
cargo run -- save-random-checkpoint --output model.json --seed 42
```

Generate using saved weights:

```bash
cargo run -- generate --prompt "hello" --checkpoint model.json --temperature 0 --seed 42
```

Current checkpoints use format v3. Compatible v1/v2 stacked-model checkpoints load with
identity attention-output projections, preserving their original forward behavior.

### Model inspection

Inspect the default architecture, including active and stored parameter counts:

```bash
cargo run -- inspect --seed 42
```

Inspect a checkpoint instead:

```bash
cargo run -- inspect --checkpoint trained.json
```

For tied embeddings, Forge reports the serialized but inactive standalone output matrix
separately from the active parameter total.

### Benchmarking (local timing only)

Measure generation throughput with repeated runs (stdout only, no files written):

```bash
cargo run -- bench --prompt "hello" --max-new-tokens 20 --runs 5 --seed 42
```

Optional checkpoint:

```bash
cargo run -- bench --prompt "hello" --max-new-tokens 20 --runs 5 --seed 42 --checkpoint model.json
```

### Training (local text only)

Train on a local UTF-8 `.txt` file and save weights:

```bash
cargo run -- train --input tiny.txt --epochs 5 --learning-rate 0.01 --output trained.json
```

Cap the averaged batch gradient's global L2 norm when experimenting with larger
learning rates:

```bash
cargo run -- train --input tiny.txt --epochs 5 --learning-rate 0.05 \
  --max-grad-norm 1.0 --output trained.json
```

Longer corpora are split into sliding next-token examples whose prefixes never exceed
the loaded model's context length.

### Evaluation

Evaluate a checkpoint on local text without updating its weights:

```bash
cargo run -- eval --input validation.txt --checkpoint trained.json
```

Omit `--checkpoint` to evaluate a reproducible randomly initialized model instead.

Checkpoints are local JSON only (no cloud APIs, no external model formats).

## Ignored artifacts

Generated checkpoints are gitignored (`*.checkpoint.json`, `checkpoints/`, `models/`). Do not commit large weight files unless intentional.
