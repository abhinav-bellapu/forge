# Forge

A tiny Rust inference runtime for transformer language models.

## Current status

Sprint 3 — tensor engine

## Implemented

- Project skeleton and CLI (`forge generate`)
- Character-level tokenizer (`vocab.json`, encode/decode)
- Tensor engine (`Tensor`: shapes, indexing, add, matmul, softmax)

## Quick start

```bash
cargo build
cargo test
cargo run -- generate --prompt "hello"
```
