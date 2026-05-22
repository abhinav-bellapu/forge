# Forge

A tiny Rust inference runtime for transformer language models.

## Current status

Sprint 8 — improved sampling

## Implemented

- Project skeleton and CLI (`forge generate`)
- Character-level tokenizer (`vocab.json`, encode/decode)
- Tensor engine:
  - Core: shapes, indexing, `add`, `matmul`, 1D `softmax`, `last_row`
  - Attention prep: `transpose_2d`, `scalar_mul`, `scalar_div`, `softmax_rows`, `row`
- Self-attention: scaled dot-product (`Attention::scaled_dot_product`)
- Minimal model: embeddings, Q/K/V projections, forward pass (`TinyModel::forward`)
- Autoregressive decoding with configurable sampling:
  - Greedy (`--temperature 0`)
  - Full-vocab temperature sampling
  - Top-k sampling (`--top-k N`)
  - Seeded generation (`--seed`)

## Quick start

```bash
cargo build
cargo test
cargo run -- generate --prompt "hello"
cargo run -- generate --prompt "hello" --temperature 0 --seed 42
cargo run -- generate --prompt "hello" --temperature 0.8 --top-k 10 --seed 42
```
