# Forge

A tiny Rust inference runtime for transformer language models.

## Current status

work in progress, most recently: loss and perplexity evaluation

## Implemented

- Project skeleton and CLI (`forge generate`, `forge train`)
- Character-level tokenizer (`vocab.json`, encode/decode)
- Tensor engine, causal self-attention, minimal `TinyModel` forward pass
- Autoregressive decoding (greedy, temperature, top-k, seeded sampling)
- JSON checkpoint save/load for model weights
- KV-cached decoding and incremental generation (optimized autoregressive inference)
- Multi-head causal attention and multi-head KV cache
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
- Finite-difference gradient checking for trained parameters
- Local inference benchmarking (`forge bench`, no network or file output by default)

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
```

### Checkpoints

Save a random model to JSON:

```bash
cargo run -- save-random-checkpoint --output model.json --seed 42
```

Generate using saved weights:

```bash
cargo run -- generate --prompt "hello" --checkpoint model.json --temperature 0 --seed 42
```

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
