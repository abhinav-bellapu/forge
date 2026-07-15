//! Minimal transformer-style forward pass (embeddings + stacked layers + logits).

use crate::attention::{Attention, ModelKvCache, MultiHeadKvCache};
use crate::tensor::Tensor;
use anyhow::bail;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Hyperparameters for [`TinyModel`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    #[serde(default = "default_tie_embeddings")]
    pub tie_embeddings: bool,
}

fn default_tie_embeddings() -> bool {
    true
}

impl ModelConfig {
    /// Default inference hyperparameters for a given vocabulary size.
    pub fn for_vocab(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len: 64,
            d_model: 16,
            n_heads: 4,
            n_layers: 2,
            tie_embeddings: true,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

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
        if self.n_heads == 0 {
            bail!("n_heads must be greater than 0");
        }
        if self.n_layers == 0 {
            bail!("n_layers must be greater than 0");
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            bail!(
                "d_model {} must be divisible by n_heads {}",
                self.d_model,
                self.n_heads
            );
        }
        Ok(())
    }
}

/// Position-wise feed-forward network (two linear layers + GELU).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForward {
    /// `[d_model, d_ff]`
    pub w1: Tensor,
    /// `[d_ff, d_model]`
    pub w2: Tensor,
}

impl FeedForward {
    pub fn new_random(d_model: usize, seed: u64) -> anyhow::Result<Self> {
        if d_model == 0 {
            bail!("d_model must be greater than 0");
        }

        let d_ff = 4 * d_model;
        let mut rng = StdRng::seed_from_u64(seed);

        Ok(Self {
            w1: random_tensor(d_model, d_ff, &mut rng)?,
            w2: random_tensor(d_ff, d_model, &mut rng)?,
        })
    }

    /// `x @ w1 → GELU → @ w2`; output shape `[rows, d_model]`.
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let hidden = x.matmul(&self.w1)?;
        let hidden = hidden.gelu()?;
        hidden.matmul(&self.w2)
    }
}

/// Per-token layer normalization (`gamma`, `beta` shape `[1, d_model]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub epsilon: f32,
}

impl LayerNorm {
    pub fn new(d_model: usize) -> anyhow::Result<Self> {
        if d_model == 0 {
            bail!("d_model must be greater than 0");
        }

        let gamma = Tensor::new(vec![1.0; d_model], vec![1, d_model])?;
        let beta = Tensor::zeros(vec![1, d_model])?;

        Ok(Self {
            gamma,
            beta,
            epsilon: 1e-5,
        })
    }

    /// Normalize each row, then apply affine transform `normalized * gamma + beta`.
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        if x.ndim() != 2 {
            bail!("layer norm expects 2D input, got {}D", x.ndim());
        }
        if x.shape()[1] != self.gamma.shape()[1] {
            bail!(
                "layer norm feature mismatch: {} vs {}",
                x.shape()[1],
                self.gamma.shape()[1]
            );
        }

        let normalized = x.normalize_last_dim(self.epsilon)?;
        let rows = normalized.shape()[0];
        let cols = normalized.shape()[1];
        let mut data = Vec::with_capacity(rows * cols);

        for r in 0..rows {
            for c in 0..cols {
                let n = normalized.get2d(r, c)?;
                let g = self.gamma.get2d(0, c)?;
                data.push(n * g);
            }
        }

        let scaled = Tensor::new(data, normalized.shape().to_vec())?;
        scaled.add_broadcast_row(&self.beta)
    }
}

/// One transformer block: attention + FFN, each with residual and layer norm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerLayer {
    /// `[d_model, d_model]`
    pub w_q: Tensor,
    /// `[d_model, d_model]`
    pub w_k: Tensor,
    /// `[d_model, d_model]`
    pub w_v: Tensor,
    /// `[d_model, d_model]`
    pub w_attn_out: Tensor,

    pub attn_norm: LayerNorm,

    pub ffn: FeedForward,
    pub ffn_norm: LayerNorm,
}

impl TransformerLayer {
    pub fn new_random(d_model: usize, seed: u64) -> anyhow::Result<Self> {
        if d_model == 0 {
            bail!("d_model must be greater than 0");
        }

        let mut rng = StdRng::seed_from_u64(seed);

        Ok(Self {
            w_q: random_tensor(d_model, d_model, &mut rng)?,
            w_k: random_tensor(d_model, d_model, &mut rng)?,
            w_v: random_tensor(d_model, d_model, &mut rng)?,
            w_attn_out: random_tensor(d_model, d_model, &mut rng)?,
            attn_norm: LayerNorm::new(d_model)?,
            ffn: FeedForward::new_random(d_model, seed.wrapping_add(1))?,
            ffn_norm: LayerNorm::new(d_model)?,
        })
    }

    /// Full forward: attention → residual → norm → FFN → residual → norm.
    pub fn forward(&self, x: &Tensor, n_heads: usize) -> anyhow::Result<Tensor> {
        let q = x.matmul(&self.w_q)?;
        let k = x.matmul(&self.w_k)?;
        let v = x.matmul(&self.w_v)?;
        let attention_output = Attention::multi_head_causal(&q, &k, &v, n_heads)?;
        let attention_output = attention_output.matmul(&self.w_attn_out)?;
        let norm1 = self.attn_norm.forward(&x.add(&attention_output)?)?;
        let ffn_out = self.ffn.forward(&norm1)?;
        let residual2 = norm1.add(&ffn_out)?;
        self.ffn_norm.forward(&residual2)
    }

    /// Incremental forward for one new token row using a per-layer KV cache.
    pub fn forward_incremental(
        &self,
        x: &Tensor,
        cache: &mut MultiHeadKvCache,
        _n_heads: usize,
    ) -> anyhow::Result<Tensor> {
        let q = x.matmul(&self.w_q)?;
        let k = x.matmul(&self.w_k)?;
        let v = x.matmul(&self.w_v)?;
        let attention_output = Attention::multi_head_cached(&q, &k, &v, cache)?;
        let attention_output = attention_output.matmul(&self.w_attn_out)?;
        let norm1 = self.attn_norm.forward(&x.add(&attention_output)?)?;
        let ffn_out = self.ffn.forward(&norm1)?;
        let residual2 = norm1.add(&ffn_out)?;
        self.ffn_norm.forward(&residual2)
    }

    pub fn validate_shapes(&self, d_model: usize, prefix: &str) -> anyhow::Result<()> {
        let d_ff = 4 * d_model;
        expect_shape(&self.w_q, &[d_model, d_model], &format!("{prefix}.w_q"))?;
        expect_shape(&self.w_k, &[d_model, d_model], &format!("{prefix}.w_k"))?;
        expect_shape(&self.w_v, &[d_model, d_model], &format!("{prefix}.w_v"))?;
        expect_shape(
            &self.w_attn_out,
            &[d_model, d_model],
            &format!("{prefix}.w_attn_out"),
        )?;
        expect_shape(
            &self.attn_norm.gamma,
            &[1, d_model],
            &format!("{prefix}.attn_norm.gamma"),
        )?;
        expect_shape(
            &self.attn_norm.beta,
            &[1, d_model],
            &format!("{prefix}.attn_norm.beta"),
        )?;
        expect_shape(&self.ffn.w1, &[d_model, d_ff], &format!("{prefix}.ffn.w1"))?;
        expect_shape(&self.ffn.w2, &[d_ff, d_model], &format!("{prefix}.ffn.w2"))?;
        expect_shape(
            &self.ffn_norm.gamma,
            &[1, d_model],
            &format!("{prefix}.ffn_norm.gamma"),
        )?;
        expect_shape(
            &self.ffn_norm.beta,
            &[1, d_model],
            &format!("{prefix}.ffn_norm.beta"),
        )?;
        Ok(())
    }
}

/// Tiny language model: embeddings, stacked transformer layers, logits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinyModel {
    pub config: ModelConfig,

    /// `[vocab_size, d_model]`
    pub token_embeddings: Tensor,
    /// `[max_seq_len, d_model]`
    pub positional_embeddings: Tensor,

    pub layers: Vec<TransformerLayer>,

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
        let n_layers = config.n_layers;

        let mut rng = StdRng::seed_from_u64(seed);

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let layer_seed = seed.wrapping_add((i as u64).wrapping_mul(1000));
            layers.push(TransformerLayer::new_random(d_model, layer_seed)?);
        }

        Ok(Self {
            config: config.clone(),
            token_embeddings: random_tensor(vocab_size, d_model, &mut rng)?,
            positional_embeddings: random_tensor(max_seq_len, d_model, &mut rng)?,
            layers,
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

    /// Embed a single token at a sequence position.
    ///
    /// Output shape: `[1, d_model]`.
    pub fn embed_token(&self, token_id: usize, position: usize) -> anyhow::Result<Tensor> {
        if token_id >= self.config.vocab_size {
            bail!("token id {token_id} is out of vocab range");
        }
        if position >= self.config.max_seq_len {
            bail!(
                "position {position} exceeds max_seq_len {}",
                self.config.max_seq_len
            );
        }

        let d_model = self.config.d_model;
        let mut data = vec![0.0; d_model];
        for (d, value) in data.iter_mut().enumerate() {
            let token_vec = self.token_embeddings.get2d(token_id, d)?;
            let pos_vec = self.positional_embeddings.get2d(position, d)?;
            *value = token_vec + pos_vec;
        }

        Tensor::new(data, vec![1, d_model])
    }

    /// Hidden states after all transformer layers (before logits projection).
    ///
    /// Output shape: `[seq_len, d_model]`.
    pub fn forward_hidden(&self, token_ids: &[usize]) -> anyhow::Result<Tensor> {
        let mut x = self.embed_tokens(token_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, self.config.n_heads)?;
        }
        Ok(x)
    }

    /// Full forward pass: embeddings → stacked layers → logits.
    ///
    /// Recomputes attention over the entire sequence each call (no KV cache).
    /// Uses causal masking so logits match incremental KV-cache decoding.
    /// Output shape: `[seq_len, vocab_size]`.
    pub fn forward(&self, token_ids: &[usize]) -> anyhow::Result<Tensor> {
        let hidden = self.forward_hidden(token_ids)?;
        self.project_logits(&hidden)
    }

    /// Incremental forward for one new token using per-layer KV caches.
    ///
    /// Embeds only the new token, runs each layer with its cache, and returns
    /// logits for that position.
    ///
    /// `position` must equal the cache length before append.
    ///
    /// Output shape: `[1, vocab_size]`.
    pub fn forward_incremental(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut ModelKvCache,
    ) -> anyhow::Result<Tensor> {
        if cache.layers.len() != self.layers.len() {
            bail!(
                "cache has {} layers but model has {}",
                cache.layers.len(),
                self.layers.len()
            );
        }
        if position != cache.len() {
            bail!(
                "position {position} does not match cache length {}",
                cache.len()
            );
        }

        let mut x = self.embed_token(token_id, position)?;
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            x = layer.forward_incremental(&x, layer_cache, self.config.n_heads)?;
        }
        self.project_logits(&x)
    }

    /// Map hidden states to vocabulary logits (tied or untied output projection).
    pub fn project_logits(&self, hidden: &Tensor) -> anyhow::Result<Tensor> {
        if self.config.tie_embeddings {
            hidden.matmul(&self.token_embeddings.transpose_2d()?)
        } else {
            hidden.matmul(&self.w_o)
        }
    }

    /// Verify that weight tensor shapes match [`ModelConfig`].
    pub fn validate_shapes(&self) -> anyhow::Result<()> {
        self.config.validate()?;

        let vs = self.config.vocab_size;
        let msl = self.config.max_seq_len;
        let dm = self.config.d_model;

        expect_shape(&self.token_embeddings, &[vs, dm], "token_embeddings")?;
        expect_shape(
            &self.positional_embeddings,
            &[msl, dm],
            "positional_embeddings",
        )?;
        expect_shape(&self.w_o, &[dm, vs], "w_o")?;

        if self.layers.len() != self.config.n_layers {
            bail!(
                "layers length {} does not match n_layers {}",
                self.layers.len(),
                self.config.n_layers
            );
        }

        for (i, layer) in self.layers.iter().enumerate() {
            layer.validate_shapes(dm, &format!("layers[{i}]"))?;
        }

        Ok(())
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

fn expect_shape(tensor: &Tensor, expected: &[usize], name: &str) -> anyhow::Result<()> {
    if tensor.shape() != expected {
        bail!(
            "{name} shape mismatch: expected {expected:?}, got {:?}",
            tensor.shape()
        );
    }
    Ok(())
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
    use crate::attention::ModelKvCache;

    fn test_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 16,
            max_seq_len: 8,
            d_model: 4,
            n_heads: 4,
            n_layers: 2,
            tie_embeddings: true,
        }
    }

    #[test]
    fn valid_config_initializes_model() {
        let model = TinyModel::new_random(test_config(), 42).unwrap();
        assert_eq!(model.token_embeddings.shape(), &[16, 4]);
        assert_eq!(model.positional_embeddings.shape(), &[8, 4]);
        assert_eq!(model.w_o.shape(), &[4, 16]);
        assert_eq!(model.layers.len(), 2);
        assert!(model
            .layers
            .iter()
            .all(|layer| layer.w_attn_out.shape() == [4, 4]));
    }

    #[test]
    fn multiple_layers_initialize() {
        let mut cfg = test_config();
        cfg.n_layers = 3;
        let model = TinyModel::new_random(cfg, 1).unwrap();
        assert_eq!(model.layers.len(), 3);
        assert!(model.validate_shapes().is_ok());
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

        cfg = test_config();
        cfg.n_heads = 0;
        assert!(cfg.validate().is_err());

        cfg = test_config();
        cfg.n_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_d_model_not_divisible_by_n_heads() {
        let mut cfg = test_config();
        cfg.n_heads = 3;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("divisible"));
    }

    #[test]
    fn head_dim_helper() {
        let cfg = ModelConfig::for_vocab(16);
        assert_eq!(cfg.head_dim(), 4);
        assert_eq!(cfg.n_layers, 2);
        assert!(cfg.tie_embeddings);
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
        assert_eq!(a.layers[0].w_q.data, b.layers[0].w_q.data);
        assert_eq!(a.w_o.data, b.w_o.data);
    }

    #[test]
    fn different_seeds_produce_different_weights() {
        let cfg = test_config();
        let a = TinyModel::new_random(cfg.clone(), 1).unwrap();
        let b = TinyModel::new_random(cfg, 2).unwrap();
        assert_ne!(a.token_embeddings.data, b.token_embeddings.data);
    }

    #[test]
    fn validate_shapes_accepts_correct_model() {
        let model = TinyModel::new_random(test_config(), 1).unwrap();
        assert!(model.validate_shapes().is_ok());
    }

    #[test]
    fn incremental_forward_output_shape() {
        let model = TinyModel::new_random(test_config(), 7).unwrap();
        let mut cache = ModelKvCache::new(
            model.config.n_layers,
            model.config.n_heads,
            model.config.head_dim(),
        )
        .unwrap();
        let logits = model.forward_incremental(2, 0, &mut cache).unwrap();
        assert_eq!(logits.shape(), &[1, 16]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.layers.len(), 2);
    }

    #[test]
    fn checkpoint_preserves_layer_norm_params() {
        let model = TinyModel::new_random(test_config(), 99).unwrap();
        let path =
            std::env::temp_dir().join(format!("forge_ln_checkpoint_{}.json", std::process::id()));

        crate::checkpoint::save_checkpoint(&model, &path).unwrap();
        let loaded = crate::checkpoint::load_checkpoint(&path).unwrap();

        assert_eq!(
            model.layers[0].attn_norm.gamma.data,
            loaded.layers[0].attn_norm.gamma.data
        );
        assert_eq!(
            model.layers[0].attn_norm.beta.data,
            loaded.layers[0].attn_norm.beta.data
        );
        assert_eq!(
            model.layers[0].attn_norm.epsilon,
            loaded.layers[0].attn_norm.epsilon
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn feed_forward_output_shape() {
        let ffn = FeedForward::new_random(4, 42).unwrap();
        let x = Tensor::new(vec![1.0; 8], vec![2, 4]).unwrap();
        let out = ffn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
    }

    #[test]
    fn feed_forward_gelu_behavior() {
        let ffn = FeedForward {
            w1: Tensor::new(vec![-1.0; 16], vec![4, 4]).unwrap(),
            w2: Tensor::new(vec![1.0; 16], vec![4, 4]).unwrap(),
        };
        let x = Tensor::new(vec![1.0; 4], vec![1, 4]).unwrap();
        let out = ffn.forward(&x).unwrap();
        let relu_out = x
            .matmul(&ffn.w1)
            .unwrap()
            .relu()
            .unwrap()
            .matmul(&ffn.w2)
            .unwrap();
        assert_ne!(out.data, relu_out.data);
        assert!(out.data.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn feed_forward_deterministic_same_seed() {
        let a = FeedForward::new_random(4, 77).unwrap();
        let b = FeedForward::new_random(4, 77).unwrap();
        assert_eq!(a.w1.data, b.w1.data);
        assert_eq!(a.w2.data, b.w2.data);
    }

    #[test]
    fn transformer_layer_output_shape() {
        let layer = TransformerLayer::new_random(4, 3).unwrap();
        let x = Tensor::new(vec![0.1; 12], vec![3, 4]).unwrap();
        let out = layer.forward(&x, 4).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn attention_output_projection_changes_layer_output() {
        let projected = TransformerLayer::new_random(4, 3).unwrap();
        let mut zero_projection = projected.clone();
        zero_projection.w_attn_out = Tensor::zeros(vec![4, 4]).unwrap();
        let x = Tensor::new(vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.5, -0.3], vec![2, 4]).unwrap();

        let with_projection = projected.forward(&x, 2).unwrap();
        let without_attention = zero_projection.forward(&x, 2).unwrap();
        assert_ne!(with_projection.data, without_attention.data);
    }

    #[test]
    fn transformer_layer_incremental_matches_full_forward() {
        let layer = TransformerLayer::new_random(4, 11).unwrap();
        let token_rows = Tensor::new(
            vec![0.2, -0.1, 0.3, 0.0, 0.5, 0.1, -0.2, 0.4, 0.3, 0.0, 0.1, 0.2],
            vec![3, 4],
        )
        .unwrap();

        let full_out = layer.forward(&token_rows, 4).unwrap();

        let mut cache = crate::attention::MultiHeadKvCache::empty(4, 1).unwrap();
        let mut incremental_out = None;

        for row in 0..3 {
            let x = token_rows.row(row).unwrap();
            let x = Tensor::new(x.data, vec![1, 4]).unwrap();
            incremental_out = Some(layer.forward_incremental(&x, &mut cache, 4).unwrap());
        }

        let inc = incremental_out.unwrap();
        let full_last: Vec<f32> = (0..4).map(|c| full_out.get2d(2, c).unwrap()).collect();
        let inc_row = inc.last_row().unwrap();

        for (a, b) in inc_row.iter().zip(full_last.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "layer incremental mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn incremental_forward_matches_full_forward() {
        let model = TinyModel::new_random(test_config(), 11).unwrap();
        let token_ids = [2usize, 5, 7];

        let full_logits = model.forward(&token_ids).unwrap();
        let mut cache = ModelKvCache::new(
            model.config.n_layers,
            model.config.n_heads,
            model.config.head_dim(),
        )
        .unwrap();

        for (pos, &tid) in token_ids.iter().enumerate() {
            let incremental = model.forward_incremental(tid, pos, &mut cache).unwrap();
            let inc_row = incremental.last_row().unwrap();
            let full_row: Vec<f32> = (0..full_logits.shape()[1])
                .map(|c| full_logits.get2d(pos, c).unwrap())
                .collect();

            assert_eq!(inc_row.len(), full_row.len());
            for (a, b) in inc_row.iter().zip(full_row.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "logit mismatch at position {pos}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn checkpoint_preserves_ffn_weights() {
        let model = TinyModel::new_random(test_config(), 88).unwrap();
        let path =
            std::env::temp_dir().join(format!("forge_ffn_checkpoint_{}.json", std::process::id()));

        crate::checkpoint::save_checkpoint(&model, &path).unwrap();
        let loaded = crate::checkpoint::load_checkpoint(&path).unwrap();

        assert_eq!(model.layers[0].ffn.w1.data, loaded.layers[0].ffn.w1.data);
        assert_eq!(model.layers[0].ffn.w2.data, loaded.layers[0].ffn.w2.data);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn checkpoint_preserves_ffn_norm() {
        let model = TinyModel::new_random(test_config(), 89).unwrap();
        let path = std::env::temp_dir().join(format!(
            "forge_ffn_norm_checkpoint_{}.json",
            std::process::id()
        ));

        crate::checkpoint::save_checkpoint(&model, &path).unwrap();
        let loaded = crate::checkpoint::load_checkpoint(&path).unwrap();

        assert_eq!(
            model.layers[0].ffn_norm.gamma.data,
            loaded.layers[0].ffn_norm.gamma.data
        );
        assert_eq!(
            model.layers[0].ffn_norm.beta.data,
            loaded.layers[0].ffn_norm.beta.data
        );
        assert_eq!(
            model.layers[0].ffn_norm.epsilon,
            loaded.layers[0].ffn_norm.epsilon
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn checkpoint_roundtrip_preserves_layers() {
        let model = TinyModel::new_random(test_config(), 55).unwrap();
        let path = std::env::temp_dir().join(format!(
            "forge_layers_checkpoint_{}.json",
            std::process::id()
        ));

        crate::checkpoint::save_checkpoint(&model, &path).unwrap();
        let loaded = crate::checkpoint::load_checkpoint(&path).unwrap();

        assert_eq!(model.config, loaded.config);
        assert_eq!(model.layers.len(), loaded.layers.len());
        for (i, layer) in model.layers.iter().enumerate() {
            assert_eq!(layer.w_q.data, loaded.layers[i].w_q.data);
            assert_eq!(layer.w_k.data, loaded.layers[i].w_k.data);
            assert_eq!(layer.w_v.data, loaded.layers[i].w_v.data);
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn incremental_forward_rejects_invalid_position() {
        let model = TinyModel::new_random(test_config(), 1).unwrap();
        let mut cache = ModelKvCache::new(
            model.config.n_layers,
            model.config.n_heads,
            model.config.head_dim(),
        )
        .unwrap();
        let err = model.forward_incremental(1, 1, &mut cache).unwrap_err();
        assert!(err.to_string().contains("position"));
    }

    #[test]
    fn validate_shapes_rejects_mismatched_weights() {
        let mut model = TinyModel::new_random(test_config(), 1).unwrap();
        model.w_o = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let err = model.validate_shapes().unwrap_err();
        assert!(err.to_string().contains("w_o"));
    }

    #[test]
    fn validate_shapes_rejects_mismatched_attention_output_projection() {
        let mut model = TinyModel::new_random(test_config(), 1).unwrap();
        model.layers[0].w_attn_out = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let err = model.validate_shapes().unwrap_err();
        assert!(err.to_string().contains("w_attn_out"));
    }

    #[test]
    fn layer_norm_output_shape() {
        let ln = LayerNorm::new(4).unwrap();
        let x = Tensor::new(vec![1.0; 8], vec![2, 4]).unwrap();
        let out = ln.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
    }

    #[test]
    fn layer_norm_normalized_rows_near_zero_mean() {
        let ln = LayerNorm::new(3).unwrap();
        let x = Tensor::new(vec![1.0, 4.0, 9.0, 2.0, 8.0, 18.0], vec![2, 3]).unwrap();
        let out = ln.forward(&x).unwrap();
        let m = out.mean_last_dim().unwrap();
        for r in 0..2 {
            assert!(m.get2d(r, 0).unwrap().abs() < 1e-3);
        }
    }

    #[test]
    fn layer_norm_gamma_beta_application() {
        let mut ln = LayerNorm::new(2).unwrap();
        ln.gamma = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        ln.beta = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let x = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
        let out = ln.forward(&x).unwrap();
        assert_eq!(out.get2d(0, 0).unwrap(), 1.0);
        assert_eq!(out.get2d(0, 1).unwrap(), -1.0);
    }

    #[test]
    fn residual_connection_changes_output() {
        let model = TinyModel::new_random(test_config(), 5).unwrap();
        let token_ids = [1usize, 2];

        let logits_with_residual = model.forward(&token_ids).unwrap();

        let x = model.embed_tokens(&token_ids).unwrap();
        let layer = &model.layers[0];
        let q = x.matmul(&layer.w_q).unwrap();
        let k = x.matmul(&layer.w_k).unwrap();
        let v = x.matmul(&layer.w_v).unwrap();
        let attention_only =
            Attention::multi_head_causal(&q, &k, &v, model.config.n_heads).unwrap();
        let normalized = layer.attn_norm.forward(&attention_only).unwrap();
        let logits_no_residual = model.project_logits(&normalized).unwrap();

        assert_ne!(logits_with_residual.data, logits_no_residual.data);
    }

    #[test]
    fn tied_projection_output_shape() {
        let model = TinyModel::new_random(test_config(), 20).unwrap();
        assert!(model.config.tie_embeddings);
        let hidden = Tensor::new(vec![0.1; 12], vec![3, 4]).unwrap();
        let logits = model.project_logits(&hidden).unwrap();
        assert_eq!(logits.shape(), &[3, 16]);
    }

    #[test]
    fn untied_projection_output_shape() {
        let mut cfg = test_config();
        cfg.tie_embeddings = false;
        let model = TinyModel::new_random(cfg, 21).unwrap();
        let hidden = Tensor::new(vec![0.1; 12], vec![3, 4]).unwrap();
        let logits = model.project_logits(&hidden).unwrap();
        assert_eq!(logits.shape(), &[3, 16]);
    }

    #[test]
    fn tied_projection_vocab_dimension_correctness() {
        let mut cfg = test_config();
        cfg.vocab_size = 8;
        let model = TinyModel::new_random(cfg, 22).unwrap();
        let hidden = Tensor::new(vec![0.2; 4], vec![1, 4]).unwrap();
        let tied = model.project_logits(&hidden).unwrap();
        let manual = hidden
            .matmul(&model.token_embeddings.transpose_2d().unwrap())
            .unwrap();
        assert_eq!(tied.data, manual.data);
        assert_eq!(tied.shape()[1], 8);
    }

    #[test]
    fn untied_projection_uses_w_o() {
        let mut cfg = test_config();
        cfg.tie_embeddings = false;
        let model = TinyModel::new_random(cfg, 23).unwrap();
        let hidden = Tensor::new(vec![0.2; 4], vec![1, 4]).unwrap();
        let untied = model.project_logits(&hidden).unwrap();
        let manual = hidden.matmul(&model.w_o).unwrap();
        assert_eq!(untied.data, manual.data);
    }

    #[test]
    fn tied_and_untied_logits_differ() {
        let mut cfg_tied = test_config();
        cfg_tied.tie_embeddings = true;
        let mut cfg_untied = test_config();
        cfg_untied.tie_embeddings = false;
        let tied = TinyModel::new_random(cfg_tied, 24).unwrap();
        let untied = TinyModel::new_random(cfg_untied, 24).unwrap();
        let token_ids = [1usize, 2];
        assert_ne!(
            tied.forward(&token_ids).unwrap().data,
            untied.forward(&token_ids).unwrap().data
        );
    }

    #[test]
    fn checkpoint_preserves_tie_embeddings() {
        let mut cfg = test_config();
        cfg.tie_embeddings = false;
        let model = TinyModel::new_random(cfg, 90).unwrap();
        let path =
            std::env::temp_dir().join(format!("forge_tie_checkpoint_{}.json", std::process::id()));

        crate::checkpoint::save_checkpoint(&model, &path).unwrap();
        let loaded = crate::checkpoint::load_checkpoint(&path).unwrap();

        assert_eq!(model.config.tie_embeddings, loaded.config.tie_embeddings);
        assert!(!loaded.config.tie_embeddings);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn generation_works_with_tied_embeddings() {
        let model = TinyModel::new_random(test_config(), 91).unwrap();
        assert!(model.config.tie_embeddings);
        let logits = model.forward(&[1, 2]).unwrap();
        assert_eq!(logits.shape(), &[2, 16]);
    }

    #[test]
    fn generation_works_with_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_embeddings = false;
        let model = TinyModel::new_random(cfg, 92).unwrap();
        let logits = model.forward(&[1, 2]).unwrap();
        assert_eq!(logits.shape(), &[2, 16]);
    }

    #[test]
    fn model_kv_cache_correct_layer_count() {
        let cache = ModelKvCache::new(2, 4, 1).unwrap();
        assert_eq!(cache.layers.len(), 2);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn model_kv_cache_grows_with_incremental_forward() {
        let model = TinyModel::new_random(test_config(), 12).unwrap();
        let mut cache = ModelKvCache::new(
            model.config.n_layers,
            model.config.n_heads,
            model.config.head_dim(),
        )
        .unwrap();

        model.forward_incremental(2, 0, &mut cache).unwrap();
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
        for layer_cache in &cache.layers {
            assert_eq!(layer_cache.len(), 1);
        }

        model.forward_incremental(5, 1, &mut cache).unwrap();
        assert_eq!(cache.len(), 2);
        for layer_cache in &cache.layers {
            assert_eq!(layer_cache.len(), 2);
        }
    }
}
