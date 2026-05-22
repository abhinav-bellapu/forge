# Forge

A tiny Rust inference runtime for transformer language models.

## Current status

Sprint 5 — self-attention

## Implemented

- Project skeleton and CLI (`forge generate`)
- Character-level tokenizer (`vocab.json`, encode/decode)
- Tensor engine:
  - Core: shapes, indexing, `add`, `matmul`, 1D `softmax`
  - Attention prep: `transpose_2d`, `scalar_mul`, `scalar_div`, `softmax_rows`, `row`
- Self-attention: scaled dot-product (`Attention::scaled_dot_product`)

## Quick start

```bash
cargo build
cargo test
cargo run -- generate --prompt "hello"
```
