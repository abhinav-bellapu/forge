//! JSON checkpoint save/load for [`TinyModel`].

use crate::model::{ModelConfig, TinyModel};
use crate::tokenizer::{self, Tokenizer};
use anyhow::bail;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Current on-disk checkpoint format version.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 2;
const COMPATIBLE_LEGACY_FORMAT_VERSION: u32 = 1;

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
    let value: serde_json::Value = serde_json::from_reader(file)
        .map_err(|e| anyhow::anyhow!("failed to parse checkpoint {}: {e}", path.display()))?;

    let format_version = value
        .get("format_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "checkpoint {} is missing a valid format_version",
                path.display()
            )
        })?;

    if format_version != CHECKPOINT_FORMAT_VERSION
        && format_version != COMPATIBLE_LEGACY_FORMAT_VERSION
    {
        bail!(
            "unsupported checkpoint format version {} (expected {CHECKPOINT_FORMAT_VERSION})",
            format_version
        );
    }

    let checkpoint: Checkpoint = serde_json::from_value(value).map_err(|e| {
        if format_version == COMPATIBLE_LEGACY_FORMAT_VERSION {
            anyhow::anyhow!(
                "checkpoint {} uses an incompatible legacy v1 model schema; regenerate or migrate it: {e}",
                path.display()
            )
        } else {
            anyhow::anyhow!("failed to parse checkpoint {}: {e}", path.display())
        }
    })?;

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
            tie_embeddings: true,
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
        assert_eq!(model.config.tie_embeddings, loaded.config.tie_embeddings);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn checkpoint_roundtrip_preserves_tie_embeddings_false() {
        let config = ModelConfig {
            vocab_size: 16,
            max_seq_len: 8,
            d_model: 4,
            n_heads: 4,
            n_layers: 2,
            tie_embeddings: false,
        };
        let model = TinyModel::new_random(config.clone(), 43).unwrap();
        let path = temp_path("tie_false");

        save_checkpoint(&model, &path).unwrap();
        let loaded = load_checkpoint(&path).unwrap();

        assert!(!loaded.config.tie_embeddings);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn tied_embedding_model_reloads_successfully() {
        let model = test_model();
        assert!(model.config.tie_embeddings);
        let path = temp_path("tied_reload");

        save_checkpoint(&model, &path).unwrap();
        let loaded = load_checkpoint(&path).unwrap();
        assert!(loaded.validate_shapes().is_ok());
        assert!(loaded.config.tie_embeddings);

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
    fn compatible_v1_checkpoint_still_loads() {
        let model = test_model();
        let path = temp_path("compatible_v1");
        let checkpoint = Checkpoint {
            format_version: COMPATIBLE_LEGACY_FORMAT_VERSION,
            model: model.clone(),
        };
        std::fs::write(&path, serde_json::to_vec(&checkpoint).unwrap()).unwrap();

        let loaded = load_checkpoint(&path).unwrap();
        assert_eq!(loaded.config, model.config);
        assert_eq!(loaded.token_embeddings.data, model.token_embeddings.data);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn incompatible_v1_checkpoint_has_actionable_error() {
        let path = temp_path("incompatible_v1");
        std::fs::write(&path, r#"{"format_version":1,"model":{}}"#).unwrap();

        let err = load_checkpoint(&path).unwrap_err();
        assert!(err.to_string().contains("incompatible legacy v1"));
        assert!(err.to_string().contains("regenerate or migrate"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn unknown_version_is_rejected_before_model_deserialization() {
        let path = temp_path("unknown_version");
        std::fs::write(&path, r#"{"format_version":999,"model":{}}"#).unwrap();

        let err = load_checkpoint(&path).unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported checkpoint format version 999"));

        let _ = std::fs::remove_file(path);
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
