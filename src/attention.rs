//! Scaled dot-product self-attention, multi-head attention, and KV caches.

use crate::tensor::Tensor;
use anyhow::bail;

/// Cached key/value rows from prior tokens (`[cached_seq_len, d_model]` each).
#[derive(Debug, Clone)]
pub struct KvCache {
    pub keys: Tensor,
    pub values: Tensor,
}

impl KvCache {
    pub fn new(keys: Tensor, values: Tensor) -> anyhow::Result<Self> {
        Self::validate_pair(&keys, &values)?;
        Ok(Self { keys, values })
    }

    /// Empty cache (`[0, d_model]` keys/values) before the first token is processed.
    pub fn empty(d_model: usize) -> anyhow::Result<Self> {
        let keys = Tensor::empty_rows(d_model)?;
        let values = Tensor::empty_rows(d_model)?;
        Self::new(keys, values)
    }

    /// Number of cached token positions.
    pub fn len(&self) -> usize {
        self.keys.shape()[0]
    }

    pub fn append(&mut self, new_keys: &Tensor, new_values: &Tensor) -> anyhow::Result<()> {
        Self::validate_pair(new_keys, new_values)?;

        let d_model = self.keys.shape()[1];
        if new_keys.shape()[1] != d_model {
            bail!(
                "cache append feature dimension mismatch: {} vs {}",
                new_keys.shape()[1],
                d_model
            );
        }

        self.keys = self.keys.concat_rows(new_keys)?;
        self.values = self.values.concat_rows(new_values)?;
        Ok(())
    }

    fn validate_pair(keys: &Tensor, values: &Tensor) -> anyhow::Result<()> {
        if keys.ndim() != 2 {
            bail!("cache keys must be 2D, got {}D", keys.ndim());
        }
        if values.ndim() != 2 {
            bail!("cache values must be 2D, got {}D", values.ndim());
        }
        if keys.shape() != values.shape() {
            bail!(
                "cache keys/values shape mismatch: {:?} vs {:?}",
                keys.shape(),
                values.shape()
            );
        }
        Ok(())
    }
}

/// Per-head KV caches for incremental multi-head decoding.
#[derive(Debug, Clone)]
pub struct MultiHeadKvCache {
    pub heads: Vec<KvCache>,
}

impl MultiHeadKvCache {
    pub fn empty(n_heads: usize, head_dim: usize) -> anyhow::Result<Self> {
        if n_heads == 0 {
            bail!("n_heads must be greater than 0");
        }
        if head_dim == 0 {
            bail!("head_dim must be greater than 0");
        }

        let mut heads = Vec::with_capacity(n_heads);
        for _ in 0..n_heads {
            heads.push(KvCache::empty(head_dim)?);
        }
        Ok(Self { heads })
    }

    /// Cached sequence length (same across all heads).
    pub fn len(&self) -> usize {
        self.heads.first().map(|c| c.len()).unwrap_or(0)
    }
}

/// Per-layer KV caches for multi-layer incremental decoding.
#[derive(Debug, Clone)]
pub struct ModelKvCache {
    pub layers: Vec<MultiHeadKvCache>,
}

impl ModelKvCache {
    pub fn new(n_layers: usize, n_heads: usize, head_dim: usize) -> anyhow::Result<Self> {
        if n_layers == 0 {
            bail!("n_layers must be greater than 0");
        }

        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(MultiHeadKvCache::empty(n_heads, head_dim)?);
        }
        Ok(Self { layers })
    }

    /// Cached sequence length (same across all layers).
    pub fn len(&self) -> usize {
        self.layers.first().map(|c| c.len()).unwrap_or(0)
    }
}

/// Single-head scaled dot-product attention.
#[derive(Debug, Default)]
pub struct Attention;

impl Attention {
    /// Causal scaled dot-product attention for decoder-only models.
    ///
    /// Computes `softmax_rows(mask((Q @ K^T) / sqrt(d_k))) @ V` where the mask
    /// zeroes out scores for future key positions (`col > row`).
    ///
    /// Matches incremental KV-cache decoding, which only sees past and current K/V.
    ///
    /// - `Q`: `[seq_len, d_k]`
    /// - `K`: `[seq_len, d_k]`
    /// - `V`: `[seq_len, d_v]`
    /// - output: `[seq_len, d_v]`
    pub fn scaled_dot_product(q: &Tensor, k: &Tensor, v: &Tensor) -> anyhow::Result<Tensor> {
        Self::validate_inputs(q, k, v)?;

        let d_k = q.shape()[1];
        let scale = (d_k as f32).sqrt();

        let k_t = k.transpose_2d()?;
        let mut scores = q.matmul(&k_t)?;
        Self::apply_causal_mask(&mut scores)?;
        let scores = scores.scalar_div(scale)?;
        let weights = scores.softmax_rows()?;
        let output = weights.matmul(v)?;

        Ok(output)
    }

    /// Set score entries above the causal diagonal to `-inf` (no future keys).
    fn apply_causal_mask(scores: &mut Tensor) -> anyhow::Result<()> {
        let rows = scores.shape()[0];
        let cols = scores.shape()[1];
        if rows != cols {
            bail!(
                "causal mask requires square scores [{rows}, {cols}], got {:?}",
                scores.shape()
            );
        }

        for r in 0..rows {
            for c in (r + 1)..cols {
                scores.set2d(r, c, f32::NEG_INFINITY)?;
            }
        }

        Ok(())
    }

    fn validate_inputs(q: &Tensor, k: &Tensor, v: &Tensor) -> anyhow::Result<()> {
        if q.ndim() != 2 {
            bail!("q must be a 2D tensor, got {}D", q.ndim());
        }
        if k.ndim() != 2 {
            bail!("k must be a 2D tensor, got {}D", k.ndim());
        }
        if v.ndim() != 2 {
            bail!("v must be a 2D tensor, got {}D", v.ndim());
        }

        let d_k = q.shape()[1];
        if d_k == 0 {
            bail!("d_k must be greater than 0");
        }

        if q.shape()[1] != k.shape()[1] {
            bail!(
                "q and k feature dimension mismatch: {} vs {}",
                q.shape()[1],
                k.shape()[1]
            );
        }

        if k.shape()[0] != v.shape()[0] {
            bail!(
                "k and v sequence length mismatch: {} vs {}",
                k.shape()[0],
                v.shape()[0]
            );
        }

        if q.shape()[0] != k.shape()[0] {
            bail!(
                "q and k sequence length mismatch: {} vs {}",
                q.shape()[0],
                k.shape()[0]
            );
        }

        Ok(())
    }

    /// Attention for one new query against cached keys/values.
    ///
    /// - `q_new`: `[1, d_model]`
    /// - `cache.keys` / `cache.values`: `[seq, d_model]`
    /// - output: `[1, d_model]`
    ///
    /// Reuses prior K/V instead of recomputing attention over the full sequence.
    pub fn scaled_dot_product_cached(q_new: &Tensor, cache: &KvCache) -> anyhow::Result<Tensor> {
        if q_new.ndim() != 2 {
            bail!("q_new must be a 2D tensor, got {}D", q_new.ndim());
        }
        if q_new.shape()[0] != 1 {
            bail!("q_new must have one row, got {}", q_new.shape()[0]);
        }

        let d_k = q_new.shape()[1];
        if d_k == 0 {
            bail!("d_k must be greater than 0");
        }

        if cache.keys.shape()[1] != d_k {
            bail!(
                "q_new and cache keys feature dimension mismatch: {} vs {}",
                d_k,
                cache.keys.shape()[1]
            );
        }

        if cache.len() == 0 {
            bail!("cache must contain at least one key/value row");
        }

        let scale = (d_k as f32).sqrt();
        let k_t = cache.keys.transpose_2d()?;
        let scores = q_new.matmul(&k_t)?;
        let scores = scores.scalar_div(scale)?;
        let weights = scores.softmax_rows()?;
        let output = weights.matmul(&cache.values)?;

        Ok(output)
    }

    /// Multi-head causal attention over column slices of Q/K/V.
    ///
    /// - `q` / `k` / `v`: `[seq_len, d_model]`
    /// - output: `[seq_len, d_model]` (head outputs concatenated along columns)
    pub fn multi_head_causal(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_heads: usize,
    ) -> anyhow::Result<Tensor> {
        if n_heads == 0 {
            bail!("n_heads must be greater than 0");
        }

        if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 {
            bail!("multi_head_causal requires 2D q/k/v tensors");
        }

        let d_model = q.shape()[1];
        if d_model % n_heads != 0 {
            bail!("d_model {d_model} must be divisible by n_heads {n_heads}");
        }

        if k.shape() != q.shape() || v.shape() != q.shape() {
            bail!(
                "q/k/v shape mismatch: {:?} vs {:?} vs {:?}",
                q.shape(),
                k.shape(),
                v.shape()
            );
        }

        let head_dim = d_model / n_heads;
        let mut head_outputs = Vec::with_capacity(n_heads);

        for h in 0..n_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            let q_h = q.slice_cols(start, end)?;
            let k_h = k.slice_cols(start, end)?;
            let v_h = v.slice_cols(start, end)?;
            let out_h = Self::scaled_dot_product(&q_h, &k_h, &v_h)?;
            head_outputs.push(out_h);
        }

        Tensor::concat_cols(&head_outputs)
    }

    /// Incremental multi-head attention for one new token.
    ///
    /// - `q_new` / `k_new` / `v_new`: `[1, d_model]`
    /// - output: `[1, d_model]`
    pub fn multi_head_cached(
        q_new: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        cache: &mut MultiHeadKvCache,
    ) -> anyhow::Result<Tensor> {
        let n_heads = cache.heads.len();
        if n_heads == 0 {
            bail!("multi-head cache has no heads");
        }

        if q_new.shape()[0] != 1 || k_new.shape()[0] != 1 || v_new.shape()[0] != 1 {
            bail!("multi_head_cached expects one-row q/k/v tensors");
        }

        let d_model = q_new.shape()[1];
        if d_model % n_heads != 0 {
            bail!("d_model {d_model} must be divisible by n_heads {n_heads}");
        }

        if k_new.shape() != q_new.shape() || v_new.shape() != q_new.shape() {
            bail!("q_new/k_new/v_new shape mismatch");
        }

        let head_dim = d_model / n_heads;
        let mut head_outputs = Vec::with_capacity(n_heads);

        for (h, head_cache) in cache.heads.iter_mut().enumerate() {
            let start = h * head_dim;
            let end = start + head_dim;
            let q_h = q_new.slice_cols(start, end)?;
            let k_h = k_new.slice_cols(start, end)?;
            let v_h = v_new.slice_cols(start, end)?;

            if head_cache.len() == 0 {
                *head_cache = KvCache::new(k_h, v_h)?;
            } else {
                head_cache.append(&k_h, &v_h)?;
            }

            let out_h = Self::scaled_dot_product_cached(&q_h, head_cache)?;
            head_outputs.push(out_h);
        }

        Tensor::concat_cols(&head_outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn output_shape_is_correct() {
        let q = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let k = q.clone();
        let v = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let out = Attention::scaled_dot_product(&q, &k, &v).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
    }

    #[test]
    fn uniform_queries_mix_value_rows_causally() {
        // Identical query/key rows; causal mask limits what each position sees.
        let q = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let k = q.clone();
        let v = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let out = Attention::scaled_dot_product(&q, &k, &v).unwrap();
        assert_eq!(out.get2d(0, 0).unwrap(), 1.0);
        assert_eq!(out.get2d(0, 1).unwrap(), 0.0);
        assert_eq!(out.get2d(1, 0).unwrap(), 0.5);
        assert_eq!(out.get2d(1, 1).unwrap(), 0.5);
    }

    #[test]
    fn causal_mask_blocks_future_keys() {
        let q = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).unwrap();
        let k = Tensor::new(vec![10.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let v = Tensor::new(vec![100.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let out = Attention::scaled_dot_product(&q, &k, &v).unwrap();
        // Position 0 only sees key 0, which points at value row 0.
        assert_eq!(out.get2d(0, 0).unwrap(), 100.0);
        assert_eq!(out.get2d(0, 1).unwrap(), 0.0);
    }

    #[test]
    fn single_position_attention_passes_value() {
        let q = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let k = q.clone();
        let v = Tensor::new(vec![7.0], vec![1, 1]).unwrap();

        let out = Attention::scaled_dot_product(&q, &k, &v).unwrap();
        assert_eq!(out.shape(), &[1, 1]);
        assert_eq!(out.get2d(0, 0).unwrap(), 7.0);
    }

    #[test]
    fn qk_feature_dimension_mismatch_errors() {
        let q = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
        let k = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let v = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let err = Attention::scaled_dot_product(&q, &k, &v).unwrap_err();
        assert!(err.to_string().contains("feature dimension"));
    }

    #[test]
    fn kv_sequence_length_mismatch_errors() {
        let q = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let k = q.clone();
        let v = Tensor::new(vec![1.0; 6], vec![3, 2]).unwrap();
        let err = Attention::scaled_dot_product(&q, &k, &v).unwrap_err();
        assert!(err.to_string().contains("sequence length"));
    }

    #[test]
    fn non_2d_tensors_error() {
        let one_d = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let two_d = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();
        assert!(Attention::scaled_dot_product(&one_d, &two_d, &two_d)
            .unwrap_err()
            .to_string()
            .contains("2D"));
    }

    #[test]
    fn cache_append_correctness() {
        let k0 = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let v0 = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let mut cache = KvCache::new(k0, v0).unwrap();

        let k1 = Tensor::new(vec![5.0, 6.0], vec![1, 2]).unwrap();
        let v1 = Tensor::new(vec![7.0, 8.0], vec![1, 2]).unwrap();
        cache.append(&k1, &v1).unwrap();

        assert_eq!(cache.keys.shape(), &[2, 2]);
        assert_eq!(cache.keys.get2d(0, 0).unwrap(), 1.0);
        assert_eq!(cache.keys.get2d(1, 0).unwrap(), 5.0);
        assert_eq!(cache.values.get2d(1, 1).unwrap(), 8.0);
    }

    #[test]
    fn cache_append_preserves_ordering() {
        let k0 = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let v0 = k0.clone();
        let mut cache = KvCache::new(k0, v0).unwrap();

        for row in 2..5 {
            let k = Tensor::new(vec![row as f32, 0.0], vec![1, 2]).unwrap();
            let v = k.clone();
            cache.append(&k, &v).unwrap();
        }

        assert_eq!(cache.len(), 4);
        for (i, expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert_eq!(cache.keys.get2d(i, 0).unwrap(), *expected);
        }
    }

    #[test]
    fn cache_append_rejects_shape_mismatch() {
        let k0 = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let v0 = k0.clone();
        let mut cache = KvCache::new(k0, v0).unwrap();
        let bad_k = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
        let bad_v = bad_k.clone();
        assert!(cache.append(&bad_k, &bad_v).is_err());
    }

    #[test]
    fn incremental_attention_output_shape() {
        let q_new = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let keys = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let values = Tensor::new(vec![10.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let cache = KvCache::new(keys, values).unwrap();

        let out = Attention::scaled_dot_product_cached(&q_new, &cache).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
    }

    #[test]
    fn multi_head_causal_output_shape() {
        let q = Tensor::new(vec![1.0; 8], vec![2, 4]).unwrap();
        let k = q.clone();
        let v = q.clone();

        let out = Attention::multi_head_causal(&q, &k, &v, 2).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
    }

    #[test]
    fn multi_head_causal_rejects_invalid_n_heads() {
        let q = Tensor::new(vec![1.0; 8], vec![2, 4]).unwrap();
        let k = q.clone();
        let v = q.clone();

        assert!(Attention::multi_head_causal(&q, &k, &v, 0).is_err());
        assert!(Attention::multi_head_causal(&q, &k, &v, 3).is_err());
    }

    #[test]
    fn multi_head_causal_with_one_head_matches_scaled_dot_product() {
        let q = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let k = q.clone();
        let v = Tensor::new(vec![10.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let single = Attention::scaled_dot_product(&q, &k, &v).unwrap();
        let multi = Attention::multi_head_causal(&q, &k, &v, 1).unwrap();

        assert_eq!(single.shape(), multi.shape());
        for r in 0..single.shape()[0] {
            for c in 0..single.shape()[1] {
                assert!((single.get2d(r, c).unwrap() - multi.get2d(r, c).unwrap()).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn multi_head_kv_cache_initializes_heads() {
        let cache = MultiHeadKvCache::empty(4, 2).unwrap();
        assert_eq!(cache.heads.len(), 4);
        assert_eq!(cache.len(), 0);
        for head in &cache.heads {
            assert_eq!(head.keys.shape(), &[0, 2]);
        }
    }

    #[test]
    fn multi_head_cached_output_shape() {
        let q = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 4]).unwrap();
        let k = q.clone();
        let v = q.clone();
        let mut cache = MultiHeadKvCache::empty(2, 2).unwrap();

        let out = Attention::multi_head_cached(&q, &k, &v, &mut cache).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn multi_head_cached_grows_cache_length() {
        let q1 = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let k1 = q1.clone();
        let v1 = q1.clone();
        let mut cache = MultiHeadKvCache::empty(2, 1).unwrap();

        Attention::multi_head_cached(&q1, &k1, &v1, &mut cache).unwrap();
        assert_eq!(cache.len(), 1);

        let q2 = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let k2 = q2.clone();
        let v2 = q2.clone();
        Attention::multi_head_cached(&q2, &k2, &v2, &mut cache).unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn peaked_attention_favors_matching_value_row() {
        // Query 0 aligns strongly with key 0.
        let q = Tensor::new(vec![3.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let k = q.clone();
        let v = Tensor::new(vec![10.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let out = Attention::scaled_dot_product(&q, &k, &v).unwrap();
        assert!(out.get2d(0, 0).unwrap() > 5.0);
        assert!(out.get2d(0, 0).unwrap() > out.get2d(0, 1).unwrap());
    }
}
