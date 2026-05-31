//! JSON checkpoint save/load for [`TinyModel`].

use crate::model::{ModelConfig, TinyModel};
use crate::tokenizer::{self, Tokenizer};
use anyhow::bail;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Current on-disk checkpoint format version.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 1;

/// Forge model checkpoint (pretty-printed JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub format_version: u32,
    pub model: TinyModel,
}

/// Save a model checkpoint to a JSON file.
pub fn save_checkpoint(model: &TinyModel, path: impl AsRef<Path>) -> anyhow::Result<()> {
    let path = path.as_ref();
    model.config.validate()?;
    model.validate_shapes()?;

    let checkpoint = Checkpoint {
        format_version: CHECKPOINT_FORMAT_VERSION,
        model: model.clone(),
    };

    let file = File::create(path)
        .map_err(|e| anyhow::anyhow!("failed to create checkpoint {}: {e}", path.display()))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &checkpoint)
        .map_err(|e| anyhow::anyhow!("failed to write checkpoint JSON: {e}"))?;

    Ok(())
}

/// Load a model checkpoint from a JSON file.
pub fn load_checkpoint(path: impl AsRef<Path>) -> anyhow::Result<TinyModel> {
    let path = path.as_ref();
    let file = File::open(path)
        .map_err(|e| anyhow::anyhow!("failed to open checkpoint {}: {e}", path.display()))?;
    let checkpoint: Checkpoint = serde_json::from_reader(file)
        .map_err(|e| anyhow::anyhow!("failed to parse checkpoint {}: {e}", path.display()))?;

    if checkpoint.format_version != CHECKPOINT_FORMAT_VERSION {
        bail!(
            "unsupported checkpoint format version {} (expected {CHECKPOINT_FORMAT_VERSION})",
            checkpoint.format_version
        );
    }

    checkpoint.model.config.validate()?;
    checkpoint.model.validate_shapes()?;

    Ok(checkpoint.model)
}

/// Create a random model from the default tokenizer vocab and save it.
pub fn save_random_checkpoint(output: impl AsRef<Path>, seed: u64) -> anyhow::Result<()> {
    let tokenizer = Tokenizer::from_file(tokenizer::default_vocab_path())?;
    let config = ModelConfig::for_vocab(tokenizer.vocab_size());
    let model = TinyModel::new_random(config, seed)?;
    save_checkpoint(&model, output)
}

/// CLI entry point for `forge save-random-checkpoint`.
pub fn run_save_random_checkpoint(output: &Path, seed: u64) -> anyhow::Result<()> {
    save_random_checkpoint(output, seed)?;
    println!("Saved checkpoint to {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{generate, GenerateRequest};
    use crate::model::ModelConfig;
    use std::path::PathBuf;

    fn test_model() -> TinyModel {
        let config = ModelConfig {
            vocab_size: 16,
            max_seq_len: 8,
            d_model: 4,
            n_heads: 4,
            n_layers: 2,
        };
        TinyModel::new_random(config, 42).unwrap()
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "forge_checkpoint_{}_{name}.json",
            std::process::id()
        ))
    }

    #[test]
    fn checkpoint_roundtrip_preserves_weights() {
        let model = test_model();
        let path = temp_path("roundtrip");

        save_checkpoint(&model, &path).unwrap();
        let loaded = load_checkpoint(&path).unwrap();

        assert_eq!(model.config, loaded.config);
        assert_eq!(model.token_embeddings.data, loaded.token_embeddings.data);
        assert_eq!(model.layers[0].w_q.data, loaded.layers[0].w_q.data);
        assert_eq!(model.w_o.data, loaded.w_o.data);
        assert_eq!(
            model.layers[0].attn_norm.gamma.data,
            loaded.layers[0].attn_norm.gamma.data
        );
        assert_eq!(
            model.layers[0].attn_norm.beta.data,
            loaded.layers[0].attn_norm.beta.data
        );
        assert_eq!(model.layers[0].ffn.w1.data, loaded.layers[0].ffn.w1.data);
        assert_eq!(model.layers[0].ffn.w2.data, loaded.layers[0].ffn.w2.data);
        assert_eq!(
            model.layers[0].ffn_norm.gamma.data,
            loaded.layers[0].ffn_norm.gamma.data
        );
        assert_eq!(
            model.layers[0].ffn_norm.beta.data,
            loaded.layers[0].ffn_norm.beta.data
        );
        assert_eq!(model.layers.len(), loaded.layers.len());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn loaded_model_validates_shapes() {
        let model = test_model();
        let path = temp_path("validate");

        save_checkpoint(&model, &path).unwrap();
        let loaded = load_checkpoint(&path).unwrap();
        assert!(loaded.validate_shapes().is_ok());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn invalid_checkpoint_path_errors() {
        let err = load_checkpoint("/nonexistent/forge_model.json").unwrap_err();
        assert!(err.to_string().contains("failed to open"));
    }

    #[test]
    fn generation_with_loaded_checkpoint_matches_original() {
        let model = test_model();
        let path = temp_path("generate");

        save_checkpoint(&model, &path).unwrap();
        let loaded = load_checkpoint(&path).unwrap();

        let req = GenerateRequest {
            prompt: "ab".to_string(),
            max_new_tokens: 3,
            temperature: 0.0,
            seed: Some(1),
            top_k: None,
        };

        let tokenizer = Tokenizer::from_file(tokenizer::default_vocab_path()).unwrap();

        let out_original = generate(&req, &tokenizer, &model).unwrap();
        let out_loaded = generate(&req, &tokenizer, &loaded).unwrap();

        assert_eq!(out_original.generated_tokens, out_loaded.generated_tokens);
        assert_eq!(out_original.output_text, out_loaded.output_text);

        let _ = std::fs::remove_file(path);
    }
}
