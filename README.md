# Forge

A tiny Rust inference runtime for transformer language models.

## Current status

Sprint 1 — project skeleton and CLI

## Roadmap

- tokenizer
- tensor engine
- transformer block
- autoregressive generation
- sampling
- KV cache
- checkpoint loading
- benchmarks

## Quick start

```bash
cargo build
cargo test
cargo run -- generate --prompt "hello"
```
