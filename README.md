# Forge

A tiny Rust inference runtime for transformer language models.

## Current status

Sprint 16 — minimal local training loop

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

### Training (local text only)

Train on a local UTF-8 `.txt` file and save weights:

```bash
cargo run -- train --input tiny.txt --epochs 5 --learning-rate 0.01 --output trained.json
```

Checkpoints are local JSON only (no cloud APIs, no external model formats).

## Ignored artifacts

Generated checkpoints are gitignored (`*.checkpoint.json`, `checkpoints/`, `models/`). Do not commit large weight files unless intentional.
