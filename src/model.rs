//! Minimal transformer-style forward pass (embeddings + attention + logits).

use crate::attention::Attention;
use crate::tensor::Tensor;
use anyhow::bail;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Hyperparameters for [`TinyModel`].
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
}

impl ModelConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.vocab_size == 0 {
            bail!("vocab_size must be greater than 0");
        }
        if self.max_seq_len == 0 {
            bail!("max_seq_len must be greater than 0");
        }
        if self.d_model == 0 {
            bail!("d_model must be greater than 0");
        }
        Ok(())
    }
}

/// Tiny language model: embeddings, single-head attention, vocab logits.
#[derive(Debug, Clone)]
pub struct TinyModel {
    pub config: ModelConfig,

    /// `[vocab_size, d_model]`
    pub token_embeddings: Tensor,
    /// `[max_seq_len, d_model]`
    pub positional_embeddings: Tensor,

    /// `[d_model, d_model]`
    pub w_q: Tensor,
    /// `[d_model, d_model]`
    pub w_k: Tensor,
    /// `[d_model, d_model]`
    pub w_v: Tensor,

    /// `[d_model, vocab_size]`
    pub w_o: Tensor,
}

impl TinyModel {
    /// Create a model with small random weights in `[-0.1, 0.1]`.
    pub fn new_random(config: ModelConfig, seed: u64) -> anyhow::Result<Self> {
        config.validate()?;

        let vocab_size = config.vocab_size;
        let max_seq_len = config.max_seq_len;
        let d_model = config.d_model;

        let mut rng = StdRng::seed_from_u64(seed);

        Ok(Self {
            config: config.clone(),
            token_embeddings: random_tensor(vocab_size, d_model, &mut rng)?,
            positional_embeddings: random_tensor(max_seq_len, d_model, &mut rng)?,
            w_q: random_tensor(d_model, d_model, &mut rng)?,
            w_k: random_tensor(d_model, d_model, &mut rng)?,
            w_v: random_tensor(d_model, d_model, &mut rng)?,
            w_o: random_tensor(d_model, vocab_size, &mut rng)?,
        })
    }

    /// Embed token IDs with learned token and positional vectors.
    ///
    /// Output shape: `[seq_len, d_model]`.
    pub fn embed_tokens(&self, token_ids: &[usize]) -> anyhow::Result<Tensor> {
        self.validate_token_ids(token_ids)?;

        let seq_len = token_ids.len();
        let d_model = self.config.d_model;
        let mut data = vec![0.0; seq_len * d_model];

        for (pos, &token_id) in token_ids.iter().enumerate() {
            for d in 0..d_model {
                let token_vec = self.token_embeddings.get2d(token_id, d)?;
                let pos_vec = self.positional_embeddings.get2d(pos, d)?;
                data[pos * d_model + d] = token_vec + pos_vec;
            }
        }

        Tensor::new(data, vec![seq_len, d_model])
    }

    /// Full forward pass: embeddings → Q/K/V → attention → logits.
    ///
    /// Output shape: `[seq_len, vocab_size]`.
    pub fn forward(&self, token_ids: &[usize]) -> anyhow::Result<Tensor> {
        let x = self.embed_tokens(token_ids)?;
        let q = x.matmul(&self.w_q)?;
        let k = x.matmul(&self.w_k)?;
        let v = x.matmul(&self.w_v)?;
        let context = Attention::scaled_dot_product(&q, &k, &v)?;
        let logits = context.matmul(&self.w_o)?;
        Ok(logits)
    }

    fn validate_token_ids(&self, token_ids: &[usize]) -> anyhow::Result<()> {
        if token_ids.is_empty() {
            bail!("token_ids cannot be empty");
        }
        if token_ids.len() > self.config.max_seq_len {
            bail!(
                "sequence length {} exceeds max_seq_len {}",
                token_ids.len(),
                self.config.max_seq_len
            );
        }
        for (i, &id) in token_ids.iter().enumerate() {
            if id >= self.config.vocab_size {
                bail!("token id {id} at position {i} is out of vocab range");
            }
        }
        Ok(())
    }
}

fn random_tensor(rows: usize, cols: usize, rng: &mut StdRng) -> anyhow::Result<Tensor> {
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        data.push(rng.gen_range(-0.1..0.1));
    }
    Tensor::new(data, vec![rows, cols])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 16,
            max_seq_len: 8,
            d_model: 4,
        }
    }

    #[test]
    fn valid_config_initializes_model() {
        let model = TinyModel::new_random(test_config(), 42).unwrap();
        assert_eq!(model.token_embeddings.shape(), &[16, 4]);
        assert_eq!(model.positional_embeddings.shape(), &[8, 4]);
        assert_eq!(model.w_o.shape(), &[4, 16]);
    }

    #[test]
    fn invalid_config_errors() {
        let mut cfg = test_config();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());

        cfg = test_config();
        cfg.max_seq_len = 0;
        assert!(cfg.validate().is_err());

        cfg = test_config();
        cfg.d_model = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn embedding_output_shape() {
        let model = TinyModel::new_random(test_config(), 1).unwrap();
        let emb = model.embed_tokens(&[1, 2, 3]).unwrap();
        assert_eq!(emb.shape(), &[3, 4]);
    }

    #[test]
    fn embedding_rejects_invalid_token_ids() {
        let model = TinyModel::new_random(test_config(), 1).unwrap();
        assert!(model.embed_tokens(&[16]).is_err());
    }

    #[test]
    fn embedding_rejects_long_sequences() {
        let model = TinyModel::new_random(test_config(), 1).unwrap();
        assert!(model.embed_tokens(&[0; 9]).is_err());
    }

    #[test]
    fn embedding_rejects_empty_input() {
        let model = TinyModel::new_random(test_config(), 1).unwrap();
        assert!(model.embed_tokens(&[]).is_err());
    }

    #[test]
    fn forward_output_shape() {
        let model = TinyModel::new_random(test_config(), 7).unwrap();
        let logits = model.forward(&[2, 5, 7]).unwrap();
        assert_eq!(logits.shape(), &[3, 16]);
    }

    #[test]
    fn same_seed_produces_identical_weights() {
        let cfg = test_config();
        let a = TinyModel::new_random(cfg.clone(), 123).unwrap();
        let b = TinyModel::new_random(cfg, 123).unwrap();
        assert_eq!(a.token_embeddings.data, b.token_embeddings.data);
        assert_eq!(a.w_q.data, b.w_q.data);
        assert_eq!(a.w_o.data, b.w_o.data);
    }

    #[test]
    fn different_seeds_produce_different_weights() {
        let cfg = test_config();
        let a = TinyModel::new_random(cfg.clone(), 1).unwrap();
        let b = TinyModel::new_random(cfg, 2).unwrap();
        assert_ne!(a.token_embeddings.data, b.token_embeddings.data);
    }
}
