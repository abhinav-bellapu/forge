//! Minimal local training loop with output-layer and token/position input gradients.
//!
//! Sprint 18 adds finite-difference gradient checking and validated gradient helpers.

use crate::checkpoint::{load_checkpoint, save_checkpoint};
use crate::cli::TrainArgs;
use crate::generation::model_config_for_tokenizer;
use crate::model::TinyModel;
use crate::tensor::Tensor;
use crate::tokenizer::{self, Tokenizer};
use anyhow::bail;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::fs;
use std::path::Path;

/// Hyperparameters for [`train`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
}

impl TrainingConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.learning_rate <= 0.0 {
            bail!("learning_rate must be greater than 0");
        }
        if self.epochs == 0 {
            bail!("epochs must be greater than 0");
        }
        if self.batch_size == 0 {
            bail!("batch_size must be greater than 0");
        }
        Ok(())
    }
}

/// One next-token training example: prefix token IDs → target token ID.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingExample {
    pub prefix: Vec<usize>,
    pub target: usize,
}

/// Local UTF-8 text corpus as next-token prediction pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct TextDataset {
    pub examples: Vec<TrainingExample>,
}

impl TextDataset {
    /// Load a local `.txt` file and build prefix → next-token examples.
    pub fn from_file(path: impl AsRef<Path>, tokenizer: &Tokenizer) -> anyhow::Result<Self> {
        Self::from_file_with_context(path, tokenizer, usize::MAX)
    }

    /// Load a local `.txt` file using prefixes capped at `max_seq_len` tokens.
    pub fn from_file_with_context(
        path: impl AsRef<Path>,
        tokenizer: &Tokenizer,
        max_seq_len: usize,
    ) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
        if text.trim().is_empty() {
            bail!("training file {} is empty", path.display());
        }
        let tokens = tokenizer.encode(&text, false, false);
        Self::from_tokens_with_context(&tokens, max_seq_len)
    }

    /// Build examples from an already-tokenized sequence.
    pub fn from_tokens(tokens: &[usize]) -> anyhow::Result<Self> {
        Self::from_tokens_with_context(tokens, usize::MAX)
    }

    /// Build examples with a sliding prefix capped at `max_seq_len` tokens.
    pub fn from_tokens_with_context(tokens: &[usize], max_seq_len: usize) -> anyhow::Result<Self> {
        if max_seq_len == 0 {
            bail!("max_seq_len must be greater than 0");
        }
        if tokens.len() < 2 {
            bail!(
                "need at least 2 tokens to form training pairs, got {}",
                tokens.len()
            );
        }

        let mut examples = Vec::with_capacity(tokens.len() - 1);
        for target_position in 1..tokens.len() {
            let prefix_start = target_position.saturating_sub(max_seq_len);
            examples.push(TrainingExample {
                prefix: tokens[prefix_start..target_position].to_vec(),
                target: tokens[target_position],
            });
        }
        Ok(Self { examples })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

/// Model dimensions used by training gradient buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrainDims {
    d_model: usize,
    vocab_size: usize,
    max_seq_len: usize,
}

impl TrainDims {
    fn from_model(model: &TinyModel) -> Self {
        Self {
            d_model: model.config.d_model,
            vocab_size: model.config.vocab_size,
            max_seq_len: model.config.max_seq_len,
        }
    }

    fn embedding_len(&self) -> usize {
        self.vocab_size * self.d_model
    }

    fn w_o_len(&self) -> usize {
        self.d_model * self.vocab_size
    }

    fn positional_embedding_len(&self) -> usize {
        self.max_seq_len * self.d_model
    }
}

/// Accumulated analytic gradients for one batch (transformer weights stay frozen).
#[derive(Debug, Clone, PartialEq)]
struct BatchGradients {
    token_embeddings: Vec<f32>,
    positional_embeddings: Vec<f32>,
    w_o: Vec<f32>,
}

impl BatchGradients {
    fn new(dims: TrainDims, tied: bool) -> Self {
        Self {
            token_embeddings: vec![0.0; dims.embedding_len()],
            positional_embeddings: vec![0.0; dims.positional_embedding_len()],
            w_o: if tied {
                Vec::new()
            } else {
                vec![0.0; dims.w_o_len()]
            },
        }
    }

    fn validate(&self, dims: TrainDims, tied: bool) -> anyhow::Result<()> {
        validate_embedding_grad_buffer(&self.token_embeddings, dims)?;
        validate_positional_embedding_grad_buffer(&self.positional_embeddings, dims)?;
        if tied {
            if !self.w_o.is_empty() {
                bail!("tied model must not accumulate w_o gradients");
            }
        } else {
            validate_w_o_grad_buffer(&self.w_o, dims)?;
        }
        Ok(())
    }
}

/// Analytic gradients for a single training example.
#[derive(Debug, Clone, PartialEq)]
pub struct ExampleGradients {
    pub loss: f32,
    pub grad_token_embeddings: Vec<f32>,
    pub grad_positional_embeddings: Vec<f32>,
    pub grad_w_o: Vec<f32>,
}

/// Average cross-entropy over sequence positions (numerically stable softmax per row).
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> anyhow::Result<f32> {
    if logits.ndim() != 2 {
        bail!(
            "cross_entropy_loss expects 2D logits, got {}D",
            logits.ndim()
        );
    }

    let seq_len = logits.shape()[0];
    let vocab_size = logits.shape()[1];
    if targets.len() != seq_len {
        bail!(
            "targets length {} does not match logits rows {}",
            targets.len(),
            seq_len
        );
    }
    if seq_len == 0 {
        bail!("cross_entropy_loss requires at least one row");
    }

    let mut total = 0.0f32;
    for (row, &target) in targets.iter().enumerate().take(seq_len) {
        if target >= vocab_size {
            bail!("target {target} at row {row} is out of vocab range {vocab_size}");
        }
        let row_logits = logits.row(row)?;
        let probs = row_logits.softmax()?;
        let p = probs.get1d(target)?;
        if p <= 0.0 {
            bail!("softmax probability for target {target} is non-positive");
        }
        total -= p.ln();
    }

    Ok(total / seq_len as f32)
}

/// Gradient of cross-entropy loss w.r.t. logits at the final position.
pub fn logits_gradient(logits: &[f32], target: usize) -> anyhow::Result<Vec<f32>> {
    let row = Tensor::new(logits.to_vec(), vec![logits.len()])?;
    let probs = row.softmax()?;
    let mut grad = probs.data;
    if target >= grad.len() {
        bail!("target {target} out of logits range {}", grad.len());
    }
    grad[target] -= 1.0;
    Ok(grad)
}

/// Gradient of cross-entropy loss w.r.t. the final hidden state (length `d_model`).
///
/// Untied: `d_hidden = grad_logits @ w_o^T`
/// Tied:   `d_hidden = grad_logits @ token_embeddings`
pub fn hidden_gradient(grad_logits: &[f32], model: &TinyModel) -> anyhow::Result<Vec<f32>> {
    let dims = TrainDims::from_model(model);
    validate_logits_grad_buffer(grad_logits, dims)?;

    let mut d_hidden = vec![0.0f32; dims.d_model];

    if model.config.tie_embeddings {
        for (d, value) in d_hidden.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (v, &logit_grad) in grad_logits.iter().enumerate() {
                sum += logit_grad * model.token_embeddings.get2d(v, d)?;
            }
            *value = sum;
        }
    } else {
        for (d, value) in d_hidden.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (v, &logit_grad) in grad_logits.iter().enumerate() {
                sum += logit_grad * model.w_o.get2d(d, v)?;
            }
            *value = sum;
        }
    }

    validate_hidden_grad_buffer(&d_hidden, dims)?;
    Ok(d_hidden)
}

/// Per-epoch training summary.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochMetrics {
    pub loss: f32,
    pub examples: usize,
}

/// Full training run summary.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingResult {
    pub epochs: Vec<EpochMetrics>,
}

/// Finite-difference gradient checking utilities (compiled for tests).
#[cfg(test)]
pub mod gradcheck {
    use super::*;

    pub const DEFAULT_EPS: f32 = 1e-4;
    pub const DEFAULT_TOLERANCE: f32 = 2e-2;

    /// Cross-entropy loss for one example (transformer forward, last-position logits).
    pub fn example_loss(model: &TinyModel, example: &TrainingExample) -> anyhow::Result<f32> {
        validate_training_example(model, example)?;
        let hidden = model.forward_hidden(&example.prefix)?;
        let logits = model.project_logits(&hidden)?;
        let logits_last = logits.last_row()?;
        cross_entropy_single(&logits_last, example.target)
    }

    /// Loss using a fixed hidden row and the model output projection (for output-only grad checks).
    pub fn example_loss_fixed_hidden(
        model: &TinyModel,
        h_last: &[f32],
        target: usize,
    ) -> anyhow::Result<f32> {
        let dims = TrainDims::from_model(model);
        validate_hidden_grad_buffer(h_last, dims)?;
        validate_token_id(target, dims)?;

        let hidden = Tensor::new(h_last.to_vec(), vec![1, dims.d_model])?;
        let logits = model.project_logits(&hidden)?;
        cross_entropy_single(&logits.last_row()?, target)
    }

    /// Central-difference gradient for tied output projection with fixed hidden states.
    pub fn numerical_tied_output_grad_fixed_hidden(
        model: &TinyModel,
        h_last: &[f32],
        target: usize,
        token_id: usize,
        dim: usize,
        eps: f32,
    ) -> anyhow::Result<f32> {
        let dims = TrainDims::from_model(model);
        validate_token_id(token_id, dims)?;
        if dim >= dims.d_model {
            bail!("dim {dim} out of range for d_model {}", dims.d_model);
        }

        let idx = token_id * dims.d_model + dim;
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.token_embeddings.data[idx] += eps;
        minus.token_embeddings.data[idx] -= eps;

        let loss_plus = example_loss_fixed_hidden(&plus, h_last, target)?;
        let loss_minus = example_loss_fixed_hidden(&minus, h_last, target)?;
        Ok((loss_plus - loss_minus) / (2.0 * eps))
    }

    /// Central-difference gradient for one `token_embeddings` element (full forward).
    pub fn numerical_token_embedding_grad(
        model: &TinyModel,
        example: &TrainingExample,
        token_id: usize,
        dim: usize,
        eps: f32,
    ) -> anyhow::Result<f32> {
        let dims = TrainDims::from_model(model);
        validate_token_id(token_id, dims)?;
        if dim >= dims.d_model {
            bail!("dim {dim} out of range for d_model {}", dims.d_model);
        }

        let idx = token_id * dims.d_model + dim;
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.token_embeddings.data[idx] += eps;
        minus.token_embeddings.data[idx] -= eps;

        let loss_plus = example_loss(&plus, example)?;
        let loss_minus = example_loss(&minus, example)?;
        Ok((loss_plus - loss_minus) / (2.0 * eps))
    }

    /// Central-difference gradient for one `positional_embeddings` element.
    pub fn numerical_positional_embedding_grad(
        model: &TinyModel,
        example: &TrainingExample,
        position: usize,
        dim: usize,
        eps: f32,
    ) -> anyhow::Result<f32> {
        let dims = TrainDims::from_model(model);
        if position >= dims.max_seq_len {
            bail!(
                "position {position} out of range for max_seq_len {}",
                dims.max_seq_len
            );
        }
        if dim >= dims.d_model {
            bail!("dim {dim} out of range for d_model {}", dims.d_model);
        }

        let idx = position * dims.d_model + dim;
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.positional_embeddings.data[idx] += eps;
        minus.positional_embeddings.data[idx] -= eps;

        let loss_plus = example_loss(&plus, example)?;
        let loss_minus = example_loss(&minus, example)?;
        Ok((loss_plus - loss_minus) / (2.0 * eps))
    }

    /// Central-difference gradient for one `w_o` element (untied models).
    pub fn numerical_w_o_grad(
        model: &TinyModel,
        example: &TrainingExample,
        row: usize,
        col: usize,
        eps: f32,
    ) -> anyhow::Result<f32> {
        let dims = TrainDims::from_model(model);
        if row >= dims.d_model || col >= dims.vocab_size {
            bail!("w_o index ({row}, {col}) out of bounds");
        }

        let idx = row * dims.vocab_size + col;
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.w_o.data[idx] += eps;
        minus.w_o.data[idx] -= eps;

        let loss_plus = example_loss(&plus, example)?;
        let loss_minus = example_loss(&minus, example)?;
        Ok((loss_plus - loss_minus) / (2.0 * eps))
    }

    /// Relative error between analytic and numerical gradient values.
    pub fn relative_error(analytic: f32, numerical: f32) -> f32 {
        let scale = analytic.abs().max(numerical.abs()).max(1e-6);
        (analytic - numerical).abs() / scale
    }

    /// Returns true when analytic and numerical gradients agree within tolerance.
    pub fn gradients_close(analytic: f32, numerical: f32, tolerance: f32) -> bool {
        relative_error(analytic, numerical) <= tolerance
            || (analytic - numerical).abs() <= tolerance
    }
}

/// Educational training loop: forward through the full model, backprop through the
/// output projection into `token_embeddings` / `w_o`, and one step into prefix
/// token and positional embeddings via `hidden_gradient`.
pub fn train(
    model: &mut TinyModel,
    dataset: &TextDataset,
    config: &TrainingConfig,
    seed: u64,
) -> anyhow::Result<TrainingResult> {
    config.validate()?;
    if dataset.is_empty() {
        bail!("dataset has no training examples");
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut epochs = Vec::with_capacity(config.epochs);

    for _ in 0..config.epochs {
        let mut shuffled = dataset.examples.clone();
        shuffled.shuffle(&mut rng);

        let mut epoch_loss = 0.0f32;
        let mut num_examples = 0usize;

        for batch in shuffled.chunks(config.batch_size) {
            let batch_loss = train_batch(model, batch, config.learning_rate)?;
            epoch_loss += batch_loss * batch.len() as f32;
            num_examples += batch.len();
        }

        epochs.push(EpochMetrics {
            loss: epoch_loss / num_examples as f32,
            examples: num_examples,
        });
    }

    Ok(TrainingResult { epochs })
}

/// Compute analytic gradients for one example without modifying the model.
pub fn compute_example_gradients(
    model: &TinyModel,
    example: &TrainingExample,
) -> anyhow::Result<ExampleGradients> {
    validate_training_example(model, example)?;

    let dims = TrainDims::from_model(model);
    let tied = model.config.tie_embeddings;

    let hidden = model.forward_hidden(&example.prefix)?;
    let logits = model.project_logits(&hidden)?;
    let h_last = hidden.last_row()?;
    let logits_last = logits.last_row()?;
    let grad_logits = logits_gradient(&logits_last, example.target)?;
    let d_hidden = hidden_gradient(&grad_logits, model)?;
    let loss = cross_entropy_single(&logits_last, example.target)?;

    let mut grad_token_embeddings = vec![0.0; dims.embedding_len()];
    let mut grad_positional_embeddings = vec![0.0; dims.positional_embedding_len()];
    let mut grad_w_o = if tied {
        Vec::new()
    } else {
        vec![0.0; dims.w_o_len()]
    };

    if tied {
        accumulate_output_tied_grad(&mut grad_token_embeddings, &grad_logits, &h_last, dims)?;
    } else {
        accumulate_output_w_o_grad(&mut grad_w_o, &grad_logits, &h_last, dims)?;
    }

    for (position, &token_id) in example.prefix.iter().enumerate() {
        accumulate_input_embedding_grad(&mut grad_token_embeddings, token_id, &d_hidden, dims)?;
        accumulate_positional_embedding_grad(
            &mut grad_positional_embeddings,
            position,
            &d_hidden,
            dims,
        )?;
    }

    validate_embedding_grad_buffer(&grad_token_embeddings, dims)?;
    validate_positional_embedding_grad_buffer(&grad_positional_embeddings, dims)?;
    if !tied {
        validate_w_o_grad_buffer(&grad_w_o, dims)?;
    }

    Ok(ExampleGradients {
        loss,
        grad_token_embeddings,
        grad_positional_embeddings,
        grad_w_o,
    })
}

fn train_batch(
    model: &mut TinyModel,
    batch: &[TrainingExample],
    learning_rate: f32,
) -> anyhow::Result<f32> {
    if batch.is_empty() {
        bail!("train_batch received empty batch");
    }

    let dims = TrainDims::from_model(model);
    let tied = model.config.tie_embeddings;
    let batch_len = batch.len() as f32;

    let mut batch_grads = BatchGradients::new(dims, tied);
    let mut total_loss = 0.0f32;

    for example in batch {
        let example_grads = compute_example_gradients(model, example)?;
        total_loss += example_grads.loss;
        merge_example_gradients(&mut batch_grads, &example_grads, tied);
    }

    batch_grads.validate(dims, tied)?;
    apply_batch_gradients(model, &batch_grads, dims, tied, learning_rate / batch_len)?;

    Ok(total_loss / batch_len)
}

fn merge_example_gradients(batch: &mut BatchGradients, example: &ExampleGradients, tied: bool) {
    for (acc, &g) in batch
        .token_embeddings
        .iter_mut()
        .zip(example.grad_token_embeddings.iter())
    {
        *acc += g;
    }
    for (acc, &g) in batch
        .positional_embeddings
        .iter_mut()
        .zip(example.grad_positional_embeddings.iter())
    {
        *acc += g;
    }
    if !tied {
        for (acc, &g) in batch.w_o.iter_mut().zip(example.grad_w_o.iter()) {
            *acc += g;
        }
    }
}

fn apply_batch_gradients(
    model: &mut TinyModel,
    batch: &BatchGradients,
    dims: TrainDims,
    tied: bool,
    scale: f32,
) -> anyhow::Result<()> {
    batch.validate(dims, tied)?;
    apply_embedding_grad(
        &mut model.token_embeddings,
        &batch.token_embeddings,
        scale,
        dims,
    )?;
    apply_positional_embedding_grad(
        &mut model.positional_embeddings,
        &batch.positional_embeddings,
        scale,
        dims,
    )?;
    if !tied {
        apply_w_o_grad(&mut model.w_o, &batch.w_o, scale, dims)?;
    }
    Ok(())
}

fn validate_training_example(model: &TinyModel, example: &TrainingExample) -> anyhow::Result<()> {
    if example.prefix.is_empty() {
        bail!("training example prefix must not be empty");
    }
    if example.prefix.len() > model.config.max_seq_len {
        bail!(
            "training prefix length {} exceeds max_seq_len {}",
            example.prefix.len(),
            model.config.max_seq_len
        );
    }
    validate_token_ids(model, &example.prefix)?;
    validate_token_id(example.target, TrainDims::from_model(model))?;
    Ok(())
}

fn validate_token_ids(model: &TinyModel, token_ids: &[usize]) -> anyhow::Result<()> {
    let dims = TrainDims::from_model(model);
    for (pos, &token_id) in token_ids.iter().enumerate() {
        if token_id >= dims.vocab_size {
            bail!("token id {token_id} at position {pos} is out of vocab range");
        }
    }
    Ok(())
}

fn validate_token_id(token_id: usize, dims: TrainDims) -> anyhow::Result<()> {
    if token_id >= dims.vocab_size {
        bail!("token id {token_id} is out of vocab range");
    }
    Ok(())
}

fn validate_logits_grad_buffer(grad_logits: &[f32], dims: TrainDims) -> anyhow::Result<()> {
    if grad_logits.len() != dims.vocab_size {
        bail!(
            "grad_logits length {} does not match vocab_size {}",
            grad_logits.len(),
            dims.vocab_size
        );
    }
    Ok(())
}

fn validate_hidden_grad_buffer(d_hidden: &[f32], dims: TrainDims) -> anyhow::Result<()> {
    if d_hidden.len() != dims.d_model {
        bail!(
            "d_hidden length {} does not match d_model {}",
            d_hidden.len(),
            dims.d_model
        );
    }
    Ok(())
}

fn validate_embedding_grad_buffer(grad: &[f32], dims: TrainDims) -> anyhow::Result<()> {
    if grad.len() != dims.embedding_len() {
        bail!(
            "token embedding grad length {} does not match expected {}",
            grad.len(),
            dims.embedding_len()
        );
    }
    Ok(())
}

fn validate_positional_embedding_grad_buffer(grad: &[f32], dims: TrainDims) -> anyhow::Result<()> {
    if grad.len() != dims.positional_embedding_len() {
        bail!(
            "positional embedding grad length {} does not match expected {}",
            grad.len(),
            dims.positional_embedding_len()
        );
    }
    Ok(())
}

fn validate_w_o_grad_buffer(grad: &[f32], dims: TrainDims) -> anyhow::Result<()> {
    if grad.len() != dims.w_o_len() {
        bail!(
            "w_o grad length {} does not match expected {}",
            grad.len(),
            dims.w_o_len()
        );
    }
    Ok(())
}

fn cross_entropy_single(logits: &[f32], target: usize) -> anyhow::Result<f32> {
    let row = Tensor::new(logits.to_vec(), vec![logits.len()])?;
    let probs = row.softmax()?;
    let p = probs.get1d(target)?;
    Ok(-p.ln())
}

fn accumulate_output_tied_grad(
    grad_e: &mut [f32],
    grad_logits: &[f32],
    h_last: &[f32],
    dims: TrainDims,
) -> anyhow::Result<()> {
    validate_embedding_grad_buffer(grad_e, dims)?;
    validate_logits_grad_buffer(grad_logits, dims)?;
    validate_hidden_grad_buffer(h_last, dims)?;

    for v in 0..dims.vocab_size {
        for d in 0..dims.d_model {
            grad_e[v * dims.d_model + d] += grad_logits[v] * h_last[d];
        }
    }
    Ok(())
}

fn accumulate_output_w_o_grad(
    grad_w_o: &mut [f32],
    grad_logits: &[f32],
    h_last: &[f32],
    dims: TrainDims,
) -> anyhow::Result<()> {
    validate_w_o_grad_buffer(grad_w_o, dims)?;
    validate_logits_grad_buffer(grad_logits, dims)?;
    validate_hidden_grad_buffer(h_last, dims)?;

    for d in 0..dims.d_model {
        for v in 0..dims.vocab_size {
            grad_w_o[d * dims.vocab_size + v] += h_last[d] * grad_logits[v];
        }
    }
    Ok(())
}

fn accumulate_input_embedding_grad(
    grad_e: &mut [f32],
    token_id: usize,
    d_hidden: &[f32],
    dims: TrainDims,
) -> anyhow::Result<()> {
    validate_embedding_grad_buffer(grad_e, dims)?;
    validate_hidden_grad_buffer(d_hidden, dims)?;
    validate_token_id(token_id, dims)?;

    for d in 0..dims.d_model {
        grad_e[token_id * dims.d_model + d] += d_hidden[d];
    }
    Ok(())
}

fn accumulate_positional_embedding_grad(
    grad: &mut [f32],
    position: usize,
    d_hidden: &[f32],
    dims: TrainDims,
) -> anyhow::Result<()> {
    validate_positional_embedding_grad_buffer(grad, dims)?;
    validate_hidden_grad_buffer(d_hidden, dims)?;
    if position >= dims.max_seq_len {
        bail!(
            "position {position} is out of range for max_seq_len {}",
            dims.max_seq_len
        );
    }

    for (d, &hidden_grad) in d_hidden.iter().enumerate() {
        grad[position * dims.d_model + d] += hidden_grad;
    }
    Ok(())
}

fn apply_embedding_grad(
    embeddings: &mut Tensor,
    grad: &[f32],
    scale: f32,
    dims: TrainDims,
) -> anyhow::Result<()> {
    validate_embedding_grad_buffer(grad, dims)?;
    for v in 0..dims.vocab_size {
        for d in 0..dims.d_model {
            let idx = v * dims.d_model + d;
            embeddings.data[idx] -= scale * grad[idx];
        }
    }
    Ok(())
}

fn apply_positional_embedding_grad(
    embeddings: &mut Tensor,
    grad: &[f32],
    scale: f32,
    dims: TrainDims,
) -> anyhow::Result<()> {
    validate_positional_embedding_grad_buffer(grad, dims)?;
    for (weight, &gradient) in embeddings.data.iter_mut().zip(grad.iter()) {
        *weight -= scale * gradient;
    }
    Ok(())
}

fn apply_w_o_grad(
    w_o: &mut Tensor,
    grad: &[f32],
    scale: f32,
    dims: TrainDims,
) -> anyhow::Result<()> {
    validate_w_o_grad_buffer(grad, dims)?;
    for d in 0..dims.d_model {
        for v in 0..dims.vocab_size {
            let idx = d * dims.vocab_size + v;
            w_o.data[idx] -= scale * grad[idx];
        }
    }
    Ok(())
}

/// CLI entry point for `forge train`.
pub fn run_train(args: &TrainArgs) -> anyhow::Result<()> {
    let config = TrainingConfig {
        learning_rate: args.learning_rate,
        epochs: args.epochs,
        batch_size: args.batch_size,
    };
    config.validate()?;

    let tokenizer = Tokenizer::from_file(tokenizer::default_vocab_path())?;

    let mut model = if let Some(path) = &args.checkpoint {
        load_checkpoint(path)?
    } else {
        let model_config = model_config_for_tokenizer(&tokenizer);
        TinyModel::new_random(model_config, args.seed)?
    };

    if model.config.vocab_size != tokenizer.vocab_size() {
        bail!(
            "model vocab_size {} does not match tokenizer {}",
            model.config.vocab_size,
            tokenizer.vocab_size()
        );
    }

    let dataset =
        TextDataset::from_file_with_context(&args.input, &tokenizer, model.config.max_seq_len)?;

    let result = train(&mut model, &dataset, &config, args.seed)?;

    for (i, metrics) in result.epochs.iter().enumerate() {
        println!(
            "Epoch {} Loss: {:.6} ({} examples)",
            i + 1,
            metrics.loss,
            metrics.examples
        );
    }

    save_checkpoint(&model, &args.output)?;
    println!("Saved trained checkpoint to {}", args.output.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::gradcheck::{
        gradients_close, numerical_token_embedding_grad, numerical_w_o_grad, DEFAULT_EPS,
        DEFAULT_TOLERANCE,
    };
    use super::*;
    use crate::checkpoint::{load_checkpoint, save_checkpoint};
    use crate::generation::{generate, GenerateRequest};
    use crate::model::ModelConfig;
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    struct TempFile(PathBuf);

    impl TempFile {
        fn txt(name: &str, contents: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "forge_s18_{}_{name}_{}.txt",
                std::process::id(),
                name
            ));
            fs::write(&path, contents).unwrap();
            Self(path)
        }

        fn json(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "forge_s18_{}_{name}_{}.json",
                std::process::id(),
                name
            ));
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn test_tokenizer() -> Tokenizer {
        Tokenizer::from_file(tokenizer::default_vocab_path()).unwrap()
    }

    fn micro_config(tie_embeddings: bool) -> ModelConfig {
        ModelConfig {
            vocab_size: 8,
            max_seq_len: 16,
            d_model: 4,
            n_heads: 2,
            n_layers: 1,
            tie_embeddings,
        }
    }

    fn micro_model(tie_embeddings: bool, seed: u64) -> TinyModel {
        TinyModel::new_random(micro_config(tie_embeddings), seed).unwrap()
    }

    fn tiny_model(seed: u64) -> TinyModel {
        let tok = test_tokenizer();
        let config = ModelConfig::for_vocab(tok.vocab_size());
        TinyModel::new_random(config, seed).unwrap()
    }

    fn example(prefix: &[usize], target: usize) -> TrainingExample {
        TrainingExample {
            prefix: prefix.to_vec(),
            target,
        }
    }

    fn embedding_row(model: &TinyModel, token_id: usize) -> Vec<f32> {
        let dm = model.config.d_model;
        (0..dm)
            .map(|d| model.token_embeddings.get2d(token_id, d).unwrap())
            .collect()
    }

    #[test]
    fn cross_entropy_valid_shapes() {
        let logits = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let loss = cross_entropy_loss(&logits, &[0, 2]).unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn cross_entropy_invalid_target_errors() {
        let logits = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let err = cross_entropy_loss(&logits, &[3]).unwrap_err();
        assert!(err.to_string().contains("out of vocab"));
    }

    #[test]
    fn cross_entropy_target_length_mismatch_errors() {
        let logits = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = cross_entropy_loss(&logits, &[0]).unwrap_err();
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn cross_entropy_lower_for_correct_predictions() {
        let confident = Tensor::new(vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0], vec![2, 3]).unwrap();
        let uncertain = Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap();
        let loss_confident = cross_entropy_loss(&confident, &[0, 1]).unwrap();
        let loss_uncertain = cross_entropy_loss(&uncertain, &[0, 1]).unwrap();
        assert!(loss_confident < loss_uncertain);
    }

    #[test]
    fn cross_entropy_deterministic() {
        let logits = Tensor::new(vec![0.5, -1.0, 2.0, 0.0, 1.5, -0.5], vec![2, 3]).unwrap();
        let a = cross_entropy_loss(&logits, &[1, 0]).unwrap();
        let b = cross_entropy_loss(&logits, &[1, 0]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn cross_entropy_rejects_non_2d_logits() {
        let logits = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(cross_entropy_loss(&logits, &[0, 1, 2]).is_err());
    }

    #[test]
    fn hidden_gradient_shape_equals_d_model() {
        let model = tiny_model(1);
        let grad_logits = vec![0.1f32; model.config.vocab_size];
        let d_hidden = hidden_gradient(&grad_logits, &model).unwrap();
        assert_eq!(d_hidden.len(), model.config.d_model);
    }

    #[test]
    fn hidden_gradient_rejects_invalid_grad_size() {
        let model = tiny_model(1);
        let err = hidden_gradient(&[0.1, 0.2], &model).unwrap_err();
        assert!(err.to_string().contains("grad_logits length"));
    }

    #[test]
    fn hidden_gradient_is_nonzero_for_random_model() {
        let model = tiny_model(2);
        let mut grad_logits = vec![0.01f32; model.config.vocab_size];
        grad_logits[3] = -0.02;
        let d_hidden = hidden_gradient(&grad_logits, &model).unwrap();
        assert!(d_hidden.iter().any(|&x| x.abs() > 1e-8));
    }

    #[test]
    fn hidden_gradient_untied_matches_manual_matmul() {
        let mut config = ModelConfig::for_vocab(test_tokenizer().vocab_size());
        config.tie_embeddings = false;
        let model = TinyModel::new_random(config, 3).unwrap();
        let grad_logits: Vec<f32> = (0..model.config.vocab_size)
            .map(|i| (i as f32 * 0.01) - 0.5)
            .collect();
        let d_hidden = hidden_gradient(&grad_logits, &model).unwrap();
        for (d, &actual) in d_hidden.iter().enumerate() {
            let mut expected = 0.0f32;
            for (v, &logit_grad) in grad_logits.iter().enumerate() {
                expected += logit_grad * model.w_o.get2d(d, v).unwrap();
            }
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn hidden_gradient_tied_matches_manual_matmul() {
        let model = tiny_model(4);
        let grad_logits: Vec<f32> = (0..model.config.vocab_size)
            .map(|i| (i as f32 * 0.01) - 0.5)
            .collect();
        let d_hidden = hidden_gradient(&grad_logits, &model).unwrap();
        for (d, &actual) in d_hidden.iter().enumerate() {
            let mut expected = 0.0f32;
            for (v, &logit_grad) in grad_logits.iter().enumerate() {
                expected += logit_grad * model.token_embeddings.get2d(v, d).unwrap();
            }
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn validate_training_example_rejects_invalid_token_id() {
        let model = micro_model(true, 1);
        let err = validate_training_example(&model, &example(&[99], 1)).unwrap_err();
        assert!(err.to_string().contains("out of vocab"));
    }

    #[test]
    fn validate_training_example_rejects_invalid_target() {
        let model = micro_model(true, 1);
        let err = validate_training_example(&model, &example(&[1], 99)).unwrap_err();
        assert!(err.to_string().contains("out of vocab"));
    }

    #[test]
    fn validate_training_example_rejects_prefix_beyond_model_context() {
        let model = micro_model(true, 1);
        let prefix = vec![1; model.config.max_seq_len + 1];
        let err = validate_training_example(&model, &example(&prefix, 2)).unwrap_err();
        assert!(err.to_string().contains("max_seq_len"));
    }

    #[test]
    fn accumulate_input_embedding_grad_rejects_bad_hidden_size() {
        let dims = TrainDims {
            d_model: 4,
            vocab_size: 8,
            max_seq_len: 16,
        };
        let mut grad = vec![0.0; dims.embedding_len()];
        let err = accumulate_input_embedding_grad(&mut grad, 1, &[0.1, 0.2], dims).unwrap_err();
        assert!(err.to_string().contains("d_hidden length"));
    }

    #[test]
    fn apply_embedding_grad_rejects_bad_buffer_size() {
        let dims = TrainDims {
            d_model: 4,
            vocab_size: 8,
            max_seq_len: 16,
        };
        let mut emb = Tensor::zeros(vec![8, 4]).unwrap();
        let err = apply_embedding_grad(&mut emb, &[0.0; 10], 0.01, dims).unwrap_err();
        assert!(err.to_string().contains("token embedding grad length"));
    }

    #[test]
    fn gradcheck_tied_output_projection_matches_numerical_fixed_hidden() {
        let model = micro_model(true, 21);
        let ex = example(&[1, 2, 3], 4);

        let hidden = model.forward_hidden(&ex.prefix).unwrap();
        let logits = model.project_logits(&hidden).unwrap();
        let h_last = hidden.last_row().unwrap();
        let logits_last = logits.last_row().unwrap();
        let grad_logits = logits_gradient(&logits_last, ex.target).unwrap();

        let checks = [(1usize, 0usize), (2, 1), (4, 2), (3, 3)];

        for &(token_id, dim) in &checks {
            let analytic = grad_logits[token_id] * h_last[dim];
            let numeric = gradcheck::numerical_tied_output_grad_fixed_hidden(
                &model,
                &h_last,
                ex.target,
                token_id,
                dim,
                DEFAULT_EPS,
            )
            .unwrap();
            assert!(
                gradients_close(analytic, numeric, DEFAULT_TOLERANCE),
                "tied output grad [{token_id},{dim}] analytic={analytic} numeric={numeric}"
            );
        }
    }

    #[test]
    fn gradcheck_untied_w_o_matches_numerical() {
        let model = micro_model(false, 22);
        let ex = example(&[1, 2], 3);

        let analytic = compute_example_gradients(&model, &ex).unwrap();
        let checks = [(0usize, 1usize), (1, 2), (2, 3), (3, 4)];

        for &(row, col) in &checks {
            let idx = row * model.config.vocab_size + col;
            let numeric = numerical_w_o_grad(&model, &ex, row, col, DEFAULT_EPS).unwrap();
            assert!(
                gradients_close(analytic.grad_w_o[idx], numeric, DEFAULT_TOLERANCE),
                "w_o[{row},{col}] analytic={} numeric={}",
                analytic.grad_w_o[idx],
                numeric
            );
        }
    }

    #[test]
    fn prefix_embedding_update_applies_d_hidden_to_each_token() {
        let model = micro_model(false, 23);
        let ex = example(&[2, 5, 2], 3);

        let hidden = model.forward_hidden(&ex.prefix).unwrap();
        let logits = model.project_logits(&hidden).unwrap();
        let logits_last = logits.last_row().unwrap();
        let grad_logits = logits_gradient(&logits_last, ex.target).unwrap();
        let d_hidden = hidden_gradient(&grad_logits, &model).unwrap();

        let dims = TrainDims::from_model(&model);
        let mut grad_e = vec![0.0; dims.embedding_len()];
        for &token_id in &ex.prefix {
            accumulate_input_embedding_grad(&mut grad_e, token_id, &d_hidden, dims).unwrap();
        }

        for &token_id in &[2usize, 5] {
            for (d, &hidden_grad) in d_hidden.iter().enumerate() {
                let idx = token_id * dims.d_model + d;
                let expected =
                    hidden_grad * ex.prefix.iter().filter(|&&t| t == token_id).count() as f32;
                assert!(
                    (grad_e[idx] - expected).abs() < 1e-6,
                    "token {token_id} dim {d}: got {} expected {expected}",
                    grad_e[idx]
                );
            }
        }
    }

    #[test]
    fn positional_embedding_update_applies_d_hidden_to_each_position() {
        let model = micro_model(false, 23);
        let d_hidden = vec![0.1, -0.2, 0.3, -0.4];
        let dims = TrainDims::from_model(&model);
        let mut grad = vec![0.0; dims.positional_embedding_len()];

        for position in 0..3 {
            accumulate_positional_embedding_grad(&mut grad, position, &d_hidden, dims).unwrap();
        }

        for position in 0..3 {
            let start = position * dims.d_model;
            assert_eq!(&grad[start..start + dims.d_model], d_hidden.as_slice());
        }
        assert!(grad[3 * dims.d_model..].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn compute_example_gradients_includes_output_and_prefix_paths() {
        let model = micro_model(true, 24);
        let ex = example(&[2, 5], 3);
        let grads = compute_example_gradients(&model, &ex).unwrap();

        let hidden = model.forward_hidden(&ex.prefix).unwrap();
        let logits = model.project_logits(&hidden).unwrap();
        let h_last = hidden.last_row().unwrap();
        let logits_last = logits.last_row().unwrap();
        let grad_logits = logits_gradient(&logits_last, ex.target).unwrap();
        let d_hidden = hidden_gradient(&grad_logits, &model).unwrap();

        let dm = model.config.d_model;
        let output_only = grad_logits[2] * h_last[1];
        let prefix_only = d_hidden[1];
        let combined = grads.grad_token_embeddings[2 * dm + 1];
        assert!(
            (combined - (output_only + prefix_only)).abs() < 1e-5,
            "combined={combined} output={output_only} prefix={prefix_only}"
        );
        assert_eq!(
            grads.grad_positional_embeddings[dm + 1],
            prefix_only,
            "position 1 should receive the input-side hidden gradient"
        );
    }

    #[test]
    fn numerical_token_embedding_grad_is_finite_on_full_forward() {
        let model = micro_model(true, 30);
        let ex = example(&[1, 2], 3);
        let grad = numerical_token_embedding_grad(&model, &ex, 2, 0, DEFAULT_EPS).unwrap();
        assert!(grad.is_finite());
    }

    #[test]
    fn numerical_positional_grad_matches_unique_untied_token_input_grad() {
        let model = micro_model(false, 31);
        let ex = example(&[1, 2], 3);
        let token_grad = numerical_token_embedding_grad(&model, &ex, 2, 0, DEFAULT_EPS).unwrap();
        let position_grad =
            gradcheck::numerical_positional_embedding_grad(&model, &ex, 1, 0, DEFAULT_EPS).unwrap();

        assert!(gradients_close(
            token_grad,
            position_grad,
            DEFAULT_TOLERANCE
        ));
    }

    #[test]
    fn dataset_builds_next_token_pairs() {
        let tok = test_tokenizer();
        let tokens = tok.encode("hello", false, false);
        let dataset = TextDataset::from_tokens(&tokens).unwrap();
        assert_eq!(dataset.len(), tokens.len() - 1);
        assert_eq!(dataset.examples[0].prefix, vec![tokens[0]]);
        assert_eq!(dataset.examples[0].target, tokens[1]);
    }

    #[test]
    fn dataset_caps_prefixes_with_sliding_context_window() {
        let dataset = TextDataset::from_tokens_with_context(&[1, 2, 3, 4, 5], 2).unwrap();

        assert_eq!(dataset.examples.len(), 4);
        assert_eq!(dataset.examples[0], example(&[1], 2));
        assert_eq!(dataset.examples[1], example(&[1, 2], 3));
        assert_eq!(dataset.examples[2], example(&[2, 3], 4));
        assert_eq!(dataset.examples[3], example(&[3, 4], 5));
        assert!(dataset.examples.iter().all(|item| item.prefix.len() <= 2));
    }

    #[test]
    fn dataset_rejects_zero_context_window() {
        let err = TextDataset::from_tokens_with_context(&[1, 2], 0).unwrap_err();
        assert!(err.to_string().contains("max_seq_len"));
    }

    #[test]
    fn dataset_from_file_loads_utf8_text() {
        let tok = test_tokenizer();
        let path = TempFile::txt("corpus", "hello hello");
        let dataset = TextDataset::from_file(path.path(), &tok).unwrap();
        assert!(dataset.len() >= 10);
    }

    #[test]
    fn train_cli_handles_corpus_larger_than_model_context() {
        let contents = "a".repeat(70);
        let input = TempFile::txt("long_context", &contents);
        let output = TempFile::json("long_context");
        let args = TrainArgs {
            input: input.path().to_path_buf(),
            epochs: 1,
            learning_rate: 0.01,
            output: output.path().to_path_buf(),
            batch_size: 8,
            seed: 42,
            checkpoint: None,
        };

        run_train(&args).unwrap();
        assert!(output.path().exists());
    }

    #[test]
    fn dataset_rejects_empty_file() {
        let tok = test_tokenizer();
        let path = TempFile::txt("empty", "   ");
        let err = TextDataset::from_file(path.path(), &tok).unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn dataset_rejects_single_token_sequence() {
        let tok = test_tokenizer();
        let tokens = tok.encode("a", false, false);
        let err = TextDataset::from_tokens(&tokens).unwrap_err();
        assert!(err.to_string().contains("at least 2 tokens"));
    }

    #[test]
    fn training_config_validation() {
        let mut cfg = TrainingConfig {
            learning_rate: 0.01,
            epochs: 1,
            batch_size: 4,
        };
        assert!(cfg.validate().is_ok());

        cfg.learning_rate = 0.0;
        assert!(cfg.validate().is_err());
        cfg.learning_rate = 0.01;
        cfg.epochs = 0;
        assert!(cfg.validate().is_err());
        cfg.epochs = 1;
        cfg.batch_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn training_reduces_loss_on_repeated_corpus() {
        let tok = test_tokenizer();
        let path = TempFile::txt("repeat", "hello hello hello hello hello hello");
        let dataset = TextDataset::from_file(path.path(), &tok).unwrap();
        let mut model = tiny_model(99);
        let config = TrainingConfig {
            learning_rate: 0.05,
            epochs: 8,
            batch_size: 4,
        };

        let result = train(&mut model, &dataset, &config, 7).unwrap();
        assert_eq!(result.epochs.len(), 8);
        assert!(
            result.epochs.last().unwrap().loss < result.epochs.first().unwrap().loss,
            "first={:?} last={:?}",
            result.epochs.first(),
            result.epochs.last()
        );
    }

    #[test]
    fn training_converges_on_repeated_corpus() {
        let tok = test_tokenizer();
        let path = TempFile::txt(
            "converge",
            "hello hello hello hello hello hello hello hello",
        );
        let dataset = TextDataset::from_file(path.path(), &tok).unwrap();
        let mut model = tiny_model(101);
        let config = TrainingConfig {
            learning_rate: 0.05,
            epochs: 20,
            batch_size: 8,
        };

        let result = train(&mut model, &dataset, &config, 11).unwrap();
        let initial = result.epochs.first().unwrap().loss;
        let final_loss = result.epochs.last().unwrap().loss;

        assert!(final_loss < initial);
        assert!(
            final_loss <= initial * 0.95,
            "initial={initial} final={final_loss}"
        );
        assert_eq!(
            result.epochs.iter().map(|m| m.examples).collect::<Vec<_>>(),
            vec![dataset.len(); 20]
        );
    }

    #[test]
    fn multiple_token_embeddings_change_after_training() {
        let tok = test_tokenizer();
        let tokens = tok.encode("hello", false, false);
        let dataset = TextDataset::from_tokens(&tokens).unwrap();
        let mut model = tiny_model(55);

        let distinct: Vec<usize> = tokens
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let before: Vec<Vec<f32>> = distinct
            .iter()
            .map(|&id| embedding_row(&model, id))
            .collect();

        train(
            &mut model,
            &dataset,
            &TrainingConfig {
                learning_rate: 0.05,
                epochs: 6,
                batch_size: 2,
            },
            9,
        )
        .unwrap();

        let mut changed = 0usize;
        for (&id, row_before) in distinct.iter().zip(before.iter()) {
            let row_after = embedding_row(&model, id);
            if row_before != &row_after {
                changed += 1;
            }
        }
        assert!(
            changed >= 3,
            "expected multiple prefix token embeddings to change, got {changed}"
        );
    }

    #[test]
    fn training_is_deterministic_with_fixed_seed() {
        let tok = test_tokenizer();
        let path = TempFile::txt("det", "hi hi hi hi hi");
        let dataset = TextDataset::from_file(path.path(), &tok).unwrap();
        let config = TrainingConfig {
            learning_rate: 0.02,
            epochs: 3,
            batch_size: 2,
        };

        let mut a = tiny_model(50);
        let mut b = tiny_model(50);
        let ra = train(&mut a, &dataset, &config, 123).unwrap();
        let rb = train(&mut b, &dataset, &config, 123).unwrap();

        assert_eq!(ra.epochs, rb.epochs);
        assert_eq!(a.token_embeddings.data, b.token_embeddings.data);
    }

    #[test]
    fn epoch_metrics_track_example_count() {
        let tok = test_tokenizer();
        let path = TempFile::txt("metrics", "hello hello hello");
        let dataset = TextDataset::from_file(path.path(), &tok).unwrap();
        let mut model = tiny_model(66);
        let result = train(
            &mut model,
            &dataset,
            &TrainingConfig {
                learning_rate: 0.03,
                epochs: 2,
                batch_size: 4,
            },
            1,
        )
        .unwrap();

        assert_eq!(result.epochs.len(), 2);
        assert_eq!(result.epochs[0].examples, dataset.len());
        assert_eq!(result.epochs[1].examples, dataset.len());
    }

    #[test]
    fn checkpoint_save_load_after_training() {
        let tok = test_tokenizer();
        let txt = TempFile::txt("ckpt", "hello hello hello");
        let dataset = TextDataset::from_file(txt.path(), &tok).unwrap();
        let mut model = tiny_model(33);
        let config = TrainingConfig {
            learning_rate: 0.03,
            epochs: 2,
            batch_size: 2,
        };
        train(&mut model, &dataset, &config, 1).unwrap();

        let path = TempFile::json("trained");
        save_checkpoint(&model, path.path()).unwrap();
        let loaded = load_checkpoint(path.path()).unwrap();

        assert_eq!(model.token_embeddings.data, loaded.token_embeddings.data);
        assert!(loaded.validate_shapes().is_ok());
    }

    #[test]
    fn trained_model_generates_after_checkpoint_reload() {
        let tok = test_tokenizer();
        let txt = TempFile::txt("gen", "hello hello hello hello hello");
        let dataset = TextDataset::from_file(txt.path(), &tok).unwrap();
        let mut model = tiny_model(44);
        train(
            &mut model,
            &dataset,
            &TrainingConfig {
                learning_rate: 0.05,
                epochs: 10,
                batch_size: 4,
            },
            2,
        )
        .unwrap();

        let path = TempFile::json("gen_ckpt");
        save_checkpoint(&model, path.path()).unwrap();
        let loaded = load_checkpoint(path.path()).unwrap();

        let req = GenerateRequest {
            prompt: "hel".to_string(),
            max_new_tokens: 2,
            temperature: 0.0,
            seed: Some(1),
            top_k: None,
        };
        let out = generate(&req, &tok, &loaded).unwrap();
        assert_eq!(out.generated_tokens.len(), 2);
    }

    #[test]
    fn training_updates_untied_w_o() {
        let tok = test_tokenizer();
        let mut config = ModelConfig::for_vocab(tok.vocab_size());
        config.tie_embeddings = false;
        let mut model = TinyModel::new_random(config, 77).unwrap();
        let path = TempFile::txt("untied", "hello hello hello hello");
        let dataset = TextDataset::from_file(path.path(), &tok).unwrap();
        let w_o_before = model.w_o.data.clone();

        train(
            &mut model,
            &dataset,
            &TrainingConfig {
                learning_rate: 0.05,
                epochs: 4,
                batch_size: 2,
            },
            5,
        )
        .unwrap();

        assert_ne!(model.w_o.data, w_o_before);
    }

    #[test]
    fn training_updates_tied_embeddings() {
        let tok = test_tokenizer();
        let mut model = tiny_model(88);
        assert!(model.config.tie_embeddings);
        let path = TempFile::txt("tied", "hello hello hello hello");
        let dataset = TextDataset::from_file(path.path(), &tok).unwrap();
        let emb_before = model.token_embeddings.data.clone();

        train(
            &mut model,
            &dataset,
            &TrainingConfig {
                learning_rate: 0.05,
                epochs: 4,
                batch_size: 2,
            },
            6,
        )
        .unwrap();

        assert_ne!(model.token_embeddings.data, emb_before);
    }

    #[test]
    fn training_updates_positional_embeddings() {
        let tok = test_tokenizer();
        let mut model = tiny_model(89);
        let path = TempFile::txt("positions", "hello hello hello hello");
        let dataset = TextDataset::from_file(path.path(), &tok).unwrap();
        let positions_before = model.positional_embeddings.data.clone();

        train(
            &mut model,
            &dataset,
            &TrainingConfig {
                learning_rate: 0.05,
                epochs: 2,
                batch_size: 2,
            },
            7,
        )
        .unwrap();

        assert_ne!(model.positional_embeddings.data, positions_before);
    }

    #[test]
    fn compute_example_gradients_rejects_empty_prefix() {
        let model = micro_model(true, 5);
        let err = compute_example_gradients(&model, &example(&[], 1)).unwrap_err();
        assert!(err.to_string().contains("prefix"));
    }
}
