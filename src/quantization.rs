//! Symmetric per-channel INT8 weight quantization.
//!
//! Activations remain `f32`; model weights are stored as signed bytes plus one
//! scale per row or column. Row scales are useful for embedding tables and
//! tied output projections, while column scales match dense linear layers.

use crate::tensor::Tensor;
use anyhow::bail;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationAxis {
    Row,
    Column,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedMatrix {
    pub data: Vec<i8>,
    pub rows: usize,
    pub cols: usize,
    pub scales: Vec<f32>,
    pub axis: QuantizationAxis,
}

impl QuantizedMatrix {
    pub fn quantize(tensor: &Tensor, axis: QuantizationAxis) -> anyhow::Result<Self> {
        if tensor.ndim() != 2 {
            bail!("INT8 quantization requires a 2D tensor");
        }
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        let channels = match axis {
            QuantizationAxis::Row => rows,
            QuantizationAxis::Column => cols,
        };
        let mut scales = vec![1.0; channels];

        for (channel, scale) in scales.iter_mut().enumerate() {
            let mut max_abs = 0.0f32;
            match axis {
                QuantizationAxis::Row => {
                    for col in 0..cols {
                        max_abs = max_abs.max(tensor.data[channel * cols + col].abs());
                    }
                }
                QuantizationAxis::Column => {
                    for row in 0..rows {
                        max_abs = max_abs.max(tensor.data[row * cols + channel].abs());
                    }
                }
            }
            *scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        }

        let mut data = vec![0i8; tensor.numel()];
        for row in 0..rows {
            for col in 0..cols {
                let scale = match axis {
                    QuantizationAxis::Row => scales[row],
                    QuantizationAxis::Column => scales[col],
                };
                data[row * cols + col] = (tensor.data[row * cols + col] / scale)
                    .round()
                    .clamp(-127.0, 127.0) as i8;
            }
        }

        Ok(Self {
            data,
            rows,
            cols,
            scales,
            axis,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<i8>() + self.scales.len() * std::mem::size_of::<f32>()
    }

    pub fn dequantize(&self) -> anyhow::Result<Tensor> {
        let mut data = vec![0.0f32; self.data.len()];
        for row in 0..self.rows {
            for col in 0..self.cols {
                let scale = match self.axis {
                    QuantizationAxis::Row => self.scales[row],
                    QuantizationAxis::Column => self.scales[col],
                };
                data[row * self.cols + col] = self.data[row * self.cols + col] as f32 * scale;
            }
        }
        Tensor::new(data, vec![self.rows, self.cols])
    }

    /// Dequantize one embedding row into a `[1, cols]` tensor.
    pub fn row(&self, row: usize) -> anyhow::Result<Tensor> {
        if self.axis != QuantizationAxis::Row {
            bail!("row lookup requires row-wise quantization");
        }
        if row >= self.rows {
            bail!("row {row} is out of bounds for {} rows", self.rows);
        }
        let scale = self.scales[row];
        let start = row * self.cols;
        let data = self.data[start..start + self.cols]
            .iter()
            .map(|&value| value as f32 * scale)
            .collect();
        Tensor::new(data, vec![1, self.cols])
    }

    /// Multiply `input [m, rows]` by this column-quantized matrix `[rows, cols]`.
    pub fn matmul_rhs(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        if self.axis != QuantizationAxis::Column {
            bail!("linear matmul requires column-wise quantization");
        }
        if input.ndim() != 2 || input.shape()[1] != self.rows {
            bail!(
                "quantized matmul shape mismatch: {:?} x [{}, {}]",
                input.shape(),
                self.rows,
                self.cols
            );
        }
        let m = input.shape()[0];
        let mut output = vec![0.0f32; m * self.cols];
        let compute_row = |(row, out): (usize, &mut [f32])| {
            let input_row = &input.data[row * self.rows..(row + 1) * self.rows];
            for (inner, &activation) in input_row.iter().enumerate() {
                let weight_row = &self.data[inner * self.cols..(inner + 1) * self.cols];
                for col in 0..self.cols {
                    out[col] += activation * weight_row[col] as f32 * self.scales[col];
                }
            }
        };
        let work = m.saturating_mul(self.rows).saturating_mul(self.cols);
        if work >= 32 * 1024 && m == 1 {
            const COLUMN_TILE: usize = 128;
            output
                .par_chunks_mut(COLUMN_TILE)
                .enumerate()
                .for_each(|(tile, out)| {
                    let start = tile * COLUMN_TILE;
                    for (inner, &activation) in input.data[..self.rows].iter().enumerate() {
                        let width = out.len();
                        let weights = &self.data
                            [inner * self.cols + start..inner * self.cols + start + width];
                        let scales = &self.scales[start..start + width];
                        for ((dst, &weight), &scale) in
                            out.iter_mut().zip(weights.iter()).zip(scales.iter())
                        {
                            *dst += activation * weight as f32 * scale;
                        }
                    }
                });
        } else if work >= 32 * 1024 && m < rayon::current_num_threads() {
            output.par_iter_mut().enumerate().for_each(|(index, dst)| {
                let row = index / self.cols;
                let col = index % self.cols;
                let input_row = &input.data[row * self.rows..(row + 1) * self.rows];
                let mut sum = 0.0f32;
                for (inner, &activation) in input_row.iter().enumerate() {
                    sum +=
                        activation * self.data[inner * self.cols + col] as f32 * self.scales[col];
                }
                *dst = sum;
            });
        } else if work >= 32 * 1024 {
            output
                .par_chunks_mut(self.cols)
                .enumerate()
                .for_each(compute_row);
        } else {
            output
                .chunks_mut(self.cols)
                .enumerate()
                .for_each(compute_row);
        }
        Tensor::new(output, vec![m, self.cols])
    }

    /// Tied embedding projection: `hidden [m, cols] @ embedding^T`.
    pub fn project_rows(&self, hidden: &Tensor) -> anyhow::Result<Tensor> {
        if self.axis != QuantizationAxis::Row {
            bail!("tied projection requires row-wise quantization");
        }
        if hidden.ndim() != 2 || hidden.shape()[1] != self.cols {
            bail!(
                "quantized projection shape mismatch: {:?} x [{}, {}]^T",
                hidden.shape(),
                self.rows,
                self.cols
            );
        }
        let m = hidden.shape()[0];
        let mut output = vec![0.0f32; m * self.rows];
        let work = m.saturating_mul(self.rows).saturating_mul(self.cols);
        if work >= 32 * 1024 && m == 1 {
            let x = &hidden.data[..self.cols];
            const ROW_TILE: usize = 128;
            output
                .par_chunks_mut(ROW_TILE)
                .enumerate()
                .for_each(|(tile, out)| {
                    let first_row = tile * ROW_TILE;
                    for (offset, dst) in out.iter_mut().enumerate() {
                        let weight_row = first_row + offset;
                        let q = &self.data[weight_row * self.cols..(weight_row + 1) * self.cols];
                        let dot: f32 = x.iter().zip(q.iter()).map(|(&a, &b)| a * b as f32).sum();
                        *dst = dot * self.scales[weight_row];
                    }
                });
        } else {
            output
                .chunks_mut(self.rows)
                .enumerate()
                .for_each(|(input_row, out)| {
                    let x = &hidden.data[input_row * self.cols..(input_row + 1) * self.cols];
                    for (weight_row, dst) in out.iter_mut().enumerate() {
                        let q = &self.data[weight_row * self.cols..(weight_row + 1) * self.cols];
                        let dot: f32 = x.iter().zip(q.iter()).map(|(&a, &b)| a * b as f32).sum();
                        *dst = dot * self.scales[weight_row];
                    }
                });
        }
        Tensor::new(output, vec![m, self.rows])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_quantization_roundtrip_is_close() {
        let tensor = Tensor::new(vec![1.0, -2.0, 0.5, 4.0, -1.0, 2.0], vec![2, 3]).unwrap();
        let quantized = QuantizedMatrix::quantize(&tensor, QuantizationAxis::Row).unwrap();
        let restored = quantized.dequantize().unwrap();
        for (&actual, &expected) in restored.data.iter().zip(tensor.data.iter()) {
            assert!((actual - expected).abs() < 0.02);
        }
        assert!(quantized.memory_bytes() < tensor.numel() * 4);
    }

    #[test]
    fn quantized_linear_matmul_tracks_fp32() {
        let input = Tensor::new(vec![1.0, -0.5, 2.0, 0.25], vec![2, 2]).unwrap();
        let weight = Tensor::new(vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5], vec![2, 3]).unwrap();
        let expected = input.matmul(&weight).unwrap();
        let quantized = QuantizedMatrix::quantize(&weight, QuantizationAxis::Column).unwrap();
        let actual = quantized.matmul_rhs(&input).unwrap();
        for (&actual, &expected) in actual.data.iter().zip(expected.data.iter()) {
            assert!((actual - expected).abs() < 0.03, "{actual} vs {expected}");
        }
    }

    #[test]
    fn quantized_tied_projection_tracks_fp32() {
        let embedding = Tensor::new(vec![1.0, -2.0, 0.5, 0.25, -0.75, 1.5], vec![3, 2]).unwrap();
        let hidden = Tensor::new(vec![0.25, 2.0], vec![1, 2]).unwrap();
        let expected = hidden.matmul(&embedding.transpose_2d().unwrap()).unwrap();
        let quantized = QuantizedMatrix::quantize(&embedding, QuantizationAxis::Row).unwrap();
        let actual = quantized.project_rows(&hidden).unwrap();
        for (&actual, &expected) in actual.data.iter().zip(expected.data.iter()) {
            assert!((actual - expected).abs() < 0.03, "{actual} vs {expected}");
        }
    }
}
