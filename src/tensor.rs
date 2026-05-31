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

    /// Create an empty 2D tensor with shape `[0, cols]` (used for KV cache warm-up).
    pub fn empty_rows(cols: usize) -> anyhow::Result<Self> {
        if cols == 0 {
            bail!("cols must be greater than 0");
        }
        Ok(Self {
            data: vec![],
            shape: vec![0, cols],
        })
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
            bail!("data length {data_len} does not match shape product {expected}");
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
            bail!("index {index} out of bounds for length {}", self.shape[0]);
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
            bail!("shape mismatch: {:?} vs {:?}", self.shape, other.shape);
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Elementwise add `other` into `self`; shapes must match exactly.
    pub fn add_inplace(&mut self, other: &Tensor) -> anyhow::Result<()> {
        if self.shape != other.shape {
            bail!("shape mismatch: {:?} vs {:?}", self.shape, other.shape);
        }
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
        Ok(())
    }

    /// Elementwise ReLU: `max(0, x)`, preserving shape.
    pub fn relu(&self) -> anyhow::Result<Tensor> {
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(0.0)).collect();
        Self::new(data, self.shape.clone())
    }

    /// Elementwise GELU (GPT approximation), preserving shape.
    ///
    /// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
    pub fn gelu(&self) -> anyhow::Result<Tensor> {
        const SQRT_2_OVER_PI: f32 = 0.797_884_560_802_865_4;
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Elementwise multiply; shapes must match exactly.
    pub fn mul(&self, other: &Tensor) -> anyhow::Result<Tensor> {
        if self.shape != other.shape {
            bail!("shape mismatch: {:?} vs {:?}", self.shape, other.shape);
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Add a `[1, cols]` row broadcast across all rows of a `[rows, cols]` tensor.
    pub fn add_broadcast_row(&self, row: &Tensor) -> anyhow::Result<Tensor> {
        if self.ndim() != 2 {
            bail!(
                "add_broadcast_row requires 2D self tensor, got {}D",
                self.ndim()
            );
        }
        if row.ndim() != 2 {
            bail!(
                "add_broadcast_row requires 2D row tensor, got {}D",
                row.ndim()
            );
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        if row.shape() != &[1, cols] {
            bail!("row shape must be [1, {cols}], got {:?}", row.shape());
        }

        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(self.data[r * cols + c] + row.data[c]);
            }
        }

        Self::new(data, vec![rows, cols])
    }

    /// Mean across columns for each row: `[rows, cols]` → `[rows, 1]`.
    pub fn mean_last_dim(&self) -> anyhow::Result<Tensor> {
        if self.ndim() != 2 {
            bail!("mean_last_dim requires 2D tensor, got {}D", self.ndim());
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = Vec::with_capacity(rows);

        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += self.data[r * cols + c];
            }
            data.push(sum / cols as f32);
        }

        Self::new(data, vec![rows, 1])
    }

    /// Variance across columns for each row: `[rows, cols]` → `[rows, 1]`.
    pub fn variance_last_dim(&self) -> anyhow::Result<Tensor> {
        if self.ndim() != 2 {
            bail!("variance_last_dim requires 2D tensor, got {}D", self.ndim());
        }

        let mean = self.mean_last_dim()?;
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = Vec::with_capacity(rows);

        for r in 0..rows {
            let m = mean.get2d(r, 0)?;
            let mut var_sum = 0.0f32;
            for c in 0..cols {
                let diff = self.data[r * cols + c] - m;
                var_sum += diff * diff;
            }
            data.push(var_sum / cols as f32);
        }

        Self::new(data, vec![rows, 1])
    }

    /// Per-row normalization: `(x - mean) / sqrt(var + epsilon)`.
    pub fn normalize_last_dim(&self, epsilon: f32) -> anyhow::Result<Tensor> {
        if self.ndim() != 2 {
            bail!(
                "normalize_last_dim requires 2D tensor, got {}D",
                self.ndim()
            );
        }

        let mean = self.mean_last_dim()?;
        let var = self.variance_last_dim()?;
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = Vec::with_capacity(rows * cols);

        for r in 0..rows {
            let m = mean.get2d(r, 0)?;
            let v = var.get2d(r, 0)?;
            let denom = (v + epsilon).sqrt();
            for c in 0..cols {
                let x = self.data[r * cols + c];
                data.push((x - m) / denom);
            }
        }

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

        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

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

    /// Concatenate two 2D tensors along rows: `[m, n] + [k, n] -> [m + k, n]`.
    pub fn concat_rows(&self, other: &Tensor) -> anyhow::Result<Tensor> {
        if self.ndim() != 2 {
            bail!("concat_rows requires 2D left tensor, got {}D", self.ndim());
        }
        if other.ndim() != 2 {
            bail!(
                "concat_rows requires 2D right tensor, got {}D",
                other.ndim()
            );
        }

        let rows_a = self.shape[0];
        let cols_a = self.shape[1];
        let rows_b = other.shape[0];
        let cols_b = other.shape[1];

        if cols_a != cols_b {
            bail!("concat_rows column mismatch: {} vs {}", cols_a, cols_b);
        }

        let mut data = Vec::with_capacity(self.data.len() + other.data.len());
        data.extend_from_slice(&self.data);
        data.extend_from_slice(&other.data);

        Tensor::new(data, vec![rows_a + rows_b, cols_a])
    }

    /// Slice column range from a 2D tensor: `[rows, cols]` → `[rows, end_col - start_col]`.
    pub fn slice_cols(&self, start_col: usize, end_col: usize) -> anyhow::Result<Tensor> {
        if self.ndim() != 2 {
            bail!("slice_cols requires 2D tensor, got {}D", self.ndim());
        }
        if end_col <= start_col {
            bail!("slice_cols requires end_col > start_col");
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        if end_col > cols {
            bail!("slice_cols end_col {end_col} out of bounds for {cols} columns");
        }

        let out_cols = end_col - start_col;
        let mut data = Vec::with_capacity(rows * out_cols);
        for r in 0..rows {
            for c in start_col..end_col {
                data.push(self.data[r * cols + c]);
            }
        }

        Tensor::new(data, vec![rows, out_cols])
    }

    /// Concatenate 2D tensors along columns: `[rows, a] + [rows, b] + ... -> [rows, sum]`.
    pub fn concat_cols(tensors: &[Tensor]) -> anyhow::Result<Tensor> {
        if tensors.is_empty() {
            bail!("concat_cols requires at least one tensor");
        }

        let rows = tensors[0].shape()[0];
        let mut total_cols = 0;

        for (i, t) in tensors.iter().enumerate() {
            if t.ndim() != 2 {
                bail!("concat_cols tensor {i} must be 2D, got {}D", t.ndim());
            }
            if t.shape()[0] != rows {
                bail!(
                    "concat_cols row mismatch at tensor {i}: {} vs {rows}",
                    t.shape()[0]
                );
            }
            total_cols += t.shape()[1];
        }

        let mut data = Vec::with_capacity(rows * total_cols);
        for r in 0..rows {
            for t in tensors {
                let cols = t.shape()[1];
                let start = r * cols;
                data.extend_from_slice(&t.data[start..start + cols]);
            }
        }

        Tensor::new(data, vec![rows, total_cols])
    }

    /// Apply softmax independently to each row of a 2D tensor.
    pub fn softmax_rows(&self) -> anyhow::Result<Tensor> {
        if self.ndim() != 2 {
            bail!(
                "softmax_rows only supports 2D tensors, got {}D",
                self.ndim()
            );
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
        assert_eq!(
            t.get1d(0).unwrap_err().to_string(),
            "expected 1D tensor, got 2D"
        );

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
    fn add_inplace_correctness() {
        let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        a.add_inplace(&b).unwrap();
        assert_eq!(a.data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn mul_correctness() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.data, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn add_broadcast_row_correctness() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap();
        let out = t.add_broadcast_row(&row).unwrap();
        assert_eq!(out.get2d(0, 0).unwrap(), 11.0);
        assert_eq!(out.get2d(1, 1).unwrap(), 24.0);
    }

    #[test]
    fn mean_last_dim_correctness() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let m = t.mean_last_dim().unwrap();
        assert_eq!(m.shape(), &[2, 1]);
        assert_eq!(m.get2d(0, 0).unwrap(), 2.0);
        assert_eq!(m.get2d(1, 0).unwrap(), 5.0);
    }

    #[test]
    fn variance_last_dim_correctness() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let v = t.variance_last_dim().unwrap();
        assert_eq!(v.shape(), &[1, 1]);
        assert!((v.get2d(0, 0).unwrap() - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn normalize_last_dim_near_zero_mean() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![2, 3]).unwrap();
        let n = t.normalize_last_dim(1e-5).unwrap();
        let m = n.mean_last_dim().unwrap();
        for r in 0..2 {
            assert!(m.get2d(r, 0).unwrap().abs() < 1e-4);
        }
    }

    #[test]
    fn normalize_last_dim_near_unit_variance() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![2, 3]).unwrap();
        let n = t.normalize_last_dim(1e-5).unwrap();
        let v = n.variance_last_dim().unwrap();
        for r in 0..2 {
            assert!((v.get2d(r, 0).unwrap() - 1.0).abs() < 1e-4);
        }
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
    fn slice_cols_correctness() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = t.slice_cols(1, 3).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(s.get2d(0, 0).unwrap(), 2.0);
        assert_eq!(s.get2d(0, 1).unwrap(), 3.0);
        assert_eq!(s.get2d(1, 0).unwrap(), 5.0);
        assert_eq!(s.get2d(1, 1).unwrap(), 6.0);
    }

    #[test]
    fn slice_cols_rejects_bad_bounds() {
        let t = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
        assert!(t.slice_cols(2, 2).is_err());
        assert!(t.slice_cols(0, 4).is_err());
        let one_d = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        assert!(one_d.slice_cols(0, 1).is_err());
    }

    #[test]
    fn concat_cols_correctness() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = Tensor::concat_cols(&[a, b]).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
        assert_eq!(c.get2d(0, 0).unwrap(), 1.0);
        assert_eq!(c.get2d(0, 2).unwrap(), 5.0);
        assert_eq!(c.get2d(1, 3).unwrap(), 8.0);
    }

    #[test]
    fn concat_cols_rejects_mismatched_rows() {
        let a = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0; 2], vec![1, 2]).unwrap();
        assert!(Tensor::concat_cols(&[a, b]).is_err());
    }

    #[test]
    fn concat_cols_rejects_empty_input() {
        assert!(Tensor::concat_cols(&[]).is_err());
    }

    #[test]
    fn concat_rows_correctness() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0], vec![1, 2]).unwrap();
        let c = a.concat_rows(&b).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn concat_rows_rejects_mismatched_cols() {
        let a = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
        let err = a.concat_rows(&b).unwrap_err();
        assert!(err.to_string().contains("column"));
    }

    #[test]
    fn concat_rows_rejects_non_2d() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap();
        assert!(a.concat_rows(&b).is_err());
    }

    #[test]
    fn relu_correctness() {
        let t = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![1, 5]).unwrap();
        let out = t.relu().unwrap();
        assert_eq!(out.data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn relu_preserves_shape() {
        let t = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]).unwrap();
        let out = t.relu().unwrap();
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn gelu_preserves_shape() {
        let t = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let out = t.gelu().unwrap();
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn gelu_zero_is_near_zero() {
        let t = Tensor::new(vec![0.0], vec![1]).unwrap();
        let out = t.gelu().unwrap();
        assert!(out.get1d(0).unwrap().abs() < 1e-6);
    }

    #[test]
    fn gelu_nonzero_for_negative_inputs() {
        let t = Tensor::new(vec![-1.0], vec![1]).unwrap();
        let out = t.gelu().unwrap();
        assert!(out.get1d(0).unwrap() < 0.0);
        assert!(out.get1d(0).unwrap().abs() > 1e-3);
    }

    #[test]
    fn gelu_differs_from_relu() {
        let t = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![1, 5]).unwrap();
        let gelu = t.gelu().unwrap();
        let relu = t.relu().unwrap();
        assert_ne!(gelu.data, relu.data);
    }

    #[test]
    fn gelu_deterministic() {
        let t = Tensor::new(vec![0.5, -0.3, 1.2], vec![1, 3]).unwrap();
        let a = t.gelu().unwrap();
        let b = t.gelu().unwrap();
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn gelu_correctness_known_value() {
        // gelu(1.0) ≈ 0.841192 with the GPT tanh approximation
        let t = Tensor::new(vec![1.0], vec![1]).unwrap();
        let out = t.gelu().unwrap();
        assert!((out.get1d(0).unwrap() - 0.841_192).abs() < 1e-4);
    }

    #[test]
    fn tensor_serde_roundtrip() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let json = serde_json::to_string(&t).unwrap();
        let restored: Tensor = serde_json::from_str(&json).unwrap();
        assert_eq!(t, restored);
    }
}
