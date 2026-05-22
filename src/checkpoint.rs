//! Checkpoint loading (placeholder for a future sprint).

#![allow(dead_code)]

use serde::Deserialize;

/// Minimal config shape for future checkpoint / config loading.
#[derive(Debug, Deserialize)]
pub struct CheckpointMeta {
    pub name: Option<String>,
}
