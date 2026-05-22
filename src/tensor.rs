//! Tensor operations for Forge inference (row-major `f32` storage).

#![allow(dead_code)] // full API surface; not every method used by the binary yet

use anyhow::bail;
use serde::{Deserialize, Serialize};

/// Dense tensor stored in row-major order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a tensor, validating that `data.len()` matches the product of `shape`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> anyhow::Result<Self> {
        Self::validate_shape(&shape, data.len())?;
        Ok(Self { data, shape })
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> anyhow::Result<Self> {
        if shape.is_empty() {
            bail!("shape cannot be empty");
        }
        if shape.iter().any(|&d| d == 0) {
            bail!("shape dimensions cannot be zero");
        }
        let n: usize = shape.iter().product();
        Ok(Self {
            data: vec![0.0; n],
            shape,
        })
    }

    fn validate_shape(shape: &[usize], data_len: usize) -> anyhow::Result<()> {
        if shape.is_empty() {
            bail!("shape cannot be empty");
        }
        if shape.iter().any(|&d| d == 0) {
            bail!("shape dimensions cannot be zero");
        }
        let expected: usize = shape.iter().product();
        if expected != data_len {
            bail!(
                "data length {data_len} does not match shape product {expected}"
            );
        }
        Ok(())
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Read one element from a 1D tensor.
    pub fn get1d(&self, index: usize) -> anyhow::Result<f32> {
        if self.ndim() != 1 {
            bail!("expected 1D tensor, got {}D", self.ndim());
        }
        if index >= self.shape[0] {
            bail!(
                "index {index} out of bounds for length {}",
                self.shape[0]
            );
        }
        Ok(self.data[index])
    }

    /// Read one element from a 2D tensor (`row`, `col`).
    pub fn get2d(&self, row: usize, col: usize) -> anyhow::Result<f32> {
        let idx = self.index2d(row, col)?;
        Ok(self.data[idx])
    }

    /// Write one element in a 2D tensor.
    pub fn set2d(&mut self, row: usize, col: usize, value: f32) -> anyhow::Result<()> {
        let idx = self.index2d(row, col)?;
        self.data[idx] = value;
        Ok(())
    }

    fn index2d(&self, row: usize, col: usize) -> anyhow::Result<usize> {
        if self.ndim() != 2 {
            bail!("expected 2D tensor, got {}D", self.ndim());
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        if row >= rows || col >= cols {
            bail!("index ({row}, {col}) out of bounds for shape [{rows}, {cols}]");
        }
        Ok(row * cols + col)
    }

    /// Elementwise add; shapes must match exactly.
    pub fn add(&self, other: &Tensor) -> anyhow::Result<Tensor> {
        if self.shape != other.shape {
            bail!(
                "shape mismatch: {:?} vs {:?}",
                self.shape,
                other.shape
            );
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Matrix multiply for 2D tensors: `[m, k] x [k, n] -> [m, n]`.
    pub fn matmul(&self, other: &Tensor) -> anyhow::Result<Tensor> {
        if self.ndim() != 2 {
            bail!("matmul requires 2D left operand, got {}D", self.ndim());
        }
        if other.ndim() != 2 {
            bail!("matmul requires 2D right operand, got {}D", other.ndim());
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let k_other = other.shape[0];
        let n = other.shape[1];

        if k != k_other {
            bail!("matmul shape mismatch: [{m}, {k}] x [{k_other}, {n}]");
        }

        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for t in 0..k {
                    let a = self.data[i * k + t];
                    let b = other.data[t * n + j];
                    sum += a * b;
                }
                out[i * n + j] = sum;
            }
        }

        Self::new(out, vec![m, n])
    }

    /// Softmax over a 1D tensor (numerically stable).
    pub fn softmax(&self) -> anyhow::Result<Tensor> {
        if self.ndim() != 1 {
            bail!("softmax only supports 1D tensors, got {}D", self.ndim());
        }
        if self.data.is_empty() {
            bail!("cannot softmax empty tensor");
        }

        let max = self
            .data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let exps: Vec<f32> = self.data.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 {
            bail!("softmax denominator is zero");
        }

        let data: Vec<f32> = exps.into_iter().map(|e| e / sum).collect();
        Self::new(data, self.shape.clone())
    }

    /// Transpose a 2D tensor: `[rows, cols]` → `[cols, rows]`.
    pub fn transpose_2d(&self) -> anyhow::Result<Tensor> {
        if self.ndim() != 2 {
            bail!("transpose_2d requires 2D tensor, got {}D", self.ndim());
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut out = vec![0.0; rows * cols];

        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = self.data[r * cols + c];
            }
        }

        Self::new(out, vec![cols, rows])
    }

    /// Multiply every element by a scalar.
    pub fn scalar_mul(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x * scalar).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Divide every element by a scalar.
    pub fn scalar_div(&self, scalar: f32) -> anyhow::Result<Tensor> {
        if scalar == 0.0 {
            bail!("scalar division by zero");
        }
        let data: Vec<f32> = self.data.iter().map(|x| x / scalar).collect();
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
        })
    }

    /// Extract one row from a 2D tensor as a 1D tensor.
    pub fn row(&self, row_index: usize) -> anyhow::Result<Tensor> {
        if self.ndim() != 2 {
            bail!("row requires 2D tensor, got {}D", self.ndim());
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        if row_index >= rows {
            bail!("row index {row_index} out of bounds for {rows} rows");
        }

        let mut data = Vec::with_capacity(cols);
        for c in 0..cols {
            data.push(self.data[row_index * cols + c]);
        }

        Self::new(data, vec![cols])
    }

    /// Return the final row of a 2D tensor as a `Vec<f32>`.
    pub fn last_row(&self) -> anyhow::Result<Vec<f32>> {
        if self.ndim() != 2 {
            bail!("last_row requires 2D tensor, got {}D", self.ndim());
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        if rows == 0 {
            bail!("cannot take last row of tensor with zero rows");
        }

        let start = (rows - 1) * cols;
        Ok(self.data[start..start + cols].to_vec())
    }

    /// Apply softmax independently to each row of a 2D tensor.
    pub fn softmax_rows(&self) -> anyhow::Result<Tensor> {
        if self.ndim() != 2 {
            bail!("softmax_rows only supports 2D tensors, got {}D", self.ndim());
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut out = Vec::with_capacity(rows * cols);

        for r in 0..rows {
            let row = self.row(r)?;
            let sm = row.softmax()?;
            for c in 0..cols {
                out.push(sm.data[c]);
            }
        }

        Self::new(out, self.shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_empty_shape() {
        let err = Tensor::new(vec![1.0], vec![]).unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn invalid_zero_dimension() {
        let err = Tensor::zeros(vec![2, 0]).unwrap_err();
        assert!(err.to_string().contains("zero"));
    }

    #[test]
    fn invalid_data_length() {
        let err = Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 2]).unwrap_err();
        assert!(err.to_string().contains("does not match"));
    }

    #[test]
    fn zeros_constructor() {
        let t = Tensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn numel_and_ndim() {
        let t = Tensor::new(vec![1.0; 12], vec![3, 4]).unwrap();
        assert_eq!(t.numel(), 12);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn indexing_1d_and_2d() {
        // row-major layout for [[1, 2], [3, 4]]
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.get1d(0).unwrap_err().to_string(), "expected 1D tensor, got 2D");

        let one_d = Tensor::new(vec![9.0, 8.0, 7.0], vec![3]).unwrap();
        assert_eq!(one_d.get1d(1).unwrap(), 8.0);

        assert_eq!(t.get2d(0, 1).unwrap(), 2.0);
        assert_eq!(t.get2d(1, 0).unwrap(), 3.0);
    }

    #[test]
    fn out_of_bounds_indexing() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert!(t.get2d(2, 0).is_err());
        assert!(t.get2d(0, 2).is_err());

        let one_d = Tensor::new(vec![1.0], vec![1]).unwrap();
        assert!(one_d.get1d(1).is_err());
    }

    #[test]
    fn set2d_correctness() {
        let mut t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        t.set2d(1, 1, 99.0).unwrap();
        assert_eq!(t.get2d(1, 1).unwrap(), 99.0);
        assert_eq!(t.get2d(0, 0).unwrap(), 1.0);
    }

    #[test]
    fn add_correctness() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn add_shape_mismatch() {
        let a = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn matmul_correctness() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get2d(0, 0).unwrap(), 19.0);
        assert_eq!(c.get2d(0, 1).unwrap(), 22.0);
        assert_eq!(c.get2d(1, 0).unwrap(), 43.0);
        assert_eq!(c.get2d(1, 1).unwrap(), 50.0);
    }

    #[test]
    fn matmul_shape_mismatch() {
        let a = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();
        // inner dims 2 vs 3 do not match
        let b = Tensor::new(vec![1.0; 6], vec![3, 2]).unwrap();
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn softmax_sums_to_one() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let s = t.softmax().unwrap();
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_preserves_ordering() {
        let t = Tensor::new(vec![3.0, 1.0, 2.0], vec![3]).unwrap();
        let s = t.softmax().unwrap();
        assert!(s.get1d(0).unwrap() > s.get1d(2).unwrap());
        assert!(s.get1d(2).unwrap() > s.get1d(1).unwrap());
    }

    #[test]
    fn softmax_rejects_2d() {
        let t = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();
        assert!(t.softmax().is_err());
    }

    #[test]
    fn transpose_2d_correctness() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let tt = t.transpose_2d().unwrap();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.get2d(0, 0).unwrap(), 1.0);
        assert_eq!(tt.get2d(0, 1).unwrap(), 4.0);
        assert_eq!(tt.get2d(1, 0).unwrap(), 2.0);
        assert_eq!(tt.get2d(1, 1).unwrap(), 5.0);
        assert_eq!(tt.get2d(2, 0).unwrap(), 3.0);
        assert_eq!(tt.get2d(2, 1).unwrap(), 6.0);
    }

    #[test]
    fn transpose_2d_rejects_non_2d() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(t.transpose_2d().is_err());
    }

    #[test]
    fn scalar_mul_correctness() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let out = t.scalar_mul(2.0);
        assert_eq!(out.data, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn scalar_div_correctness() {
        let t = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let out = t.scalar_div(2.0).unwrap();
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn scalar_div_rejects_zero() {
        let t = Tensor::new(vec![1.0], vec![1]).unwrap();
        assert!(t.scalar_div(0.0).is_err());
    }

    #[test]
    fn softmax_rows_keeps_shape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = t.softmax_rows().unwrap();
        assert_eq!(s.shape(), &[2, 3]);
    }

    #[test]
    fn softmax_rows_each_row_sums_to_one() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = t.softmax_rows().unwrap();
        for r in 0..2 {
            let mut sum = 0.0f32;
            for c in 0..3 {
                sum += s.get2d(r, c).unwrap();
            }
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_rows_preserves_ordering_within_row() {
        let t = Tensor::new(vec![3.0, 1.0, 2.0, 0.5, 2.5, 1.5], vec![2, 3]).unwrap();
        let s = t.softmax_rows().unwrap();
        assert!(s.get2d(0, 0).unwrap() > s.get2d(0, 2).unwrap());
        assert!(s.get2d(0, 2).unwrap() > s.get2d(0, 1).unwrap());
        assert!(s.get2d(1, 1).unwrap() > s.get2d(1, 2).unwrap());
        assert!(s.get2d(1, 2).unwrap() > s.get2d(1, 0).unwrap());
    }

    #[test]
    fn softmax_rows_rejects_non_2d() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(t.softmax_rows().is_err());
    }

    #[test]
    fn last_row_correctness() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(t.last_row().unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn last_row_rejects_non_2d() {
        let t = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        assert!(t.last_row().is_err());
    }

    #[test]
    fn tensor_serde_roundtrip() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let json = serde_json::to_string(&t).unwrap();
        let restored: Tensor = serde_json::from_str(&json).unwrap();
        assert_eq!(t, restored);
    }
}
