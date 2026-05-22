//! Scaled dot-product self-attention.

#![allow(dead_code)] // used by unit tests; model will wire this in Sprint 6+

use crate::tensor::Tensor;
use anyhow::bail;

/// Single-head scaled dot-product attention.
#[derive(Debug, Default)]
pub struct Attention;

impl Attention {
    /// Compute `softmax_rows((Q @ K^T) / sqrt(d_k)) @ V`.
    ///
    /// - `Q`: `[seq_len, d_k]`
    /// - `K`: `[seq_len, d_k]`
    /// - `V`: `[seq_len, d_v]`
    /// - output: `[seq_len, d_v]`
    pub fn scaled_dot_product(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> anyhow::Result<Tensor> {
        Self::validate_inputs(q, k, v)?;

        let d_k = q.shape()[1];
        let scale = (d_k as f32).sqrt();

        let k_t = k.transpose_2d()?;
        let scores = q.matmul(&k_t)?;
        let scores = scores.scalar_div(scale)?;
        let weights = scores.softmax_rows()?;
        let output = weights.matmul(v)?;

        Ok(output)
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
    fn uniform_queries_mix_value_rows() {
        // Identical query/key rows -> uniform attention over both positions.
        let q = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let k = q.clone();
        let v = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let out = Attention::scaled_dot_product(&q, &k, &v).unwrap();
        assert_eq!(out.get2d(0, 0).unwrap(), 0.5);
        assert_eq!(out.get2d(0, 1).unwrap(), 0.5);
        assert_eq!(out.get2d(1, 0).unwrap(), 0.5);
        assert_eq!(out.get2d(1, 1).unwrap(), 0.5);
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
