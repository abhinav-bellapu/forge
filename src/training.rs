//! Minimal local training loop (output-layer gradients only, no full backprop).

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
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
        if text.trim().is_empty() {
            bail!("training file {} is empty", path.display());
        }
        let tokens = tokenizer.encode(&text, false, false);
        Self::from_tokens(&tokens)
    }

    /// Build examples from an already-tokenized sequence.
    pub fn from_tokens(tokens: &[usize]) -> anyhow::Result<Self> {
        if tokens.len() < 2 {
            bail!(
                "need at least 2 tokens to form training pairs, got {}",
                tokens.len()
            );
        }

        let mut examples = Vec::with_capacity(tokens.len() - 1);
        for i in 0..tokens.len() - 1 {
            examples.push(TrainingExample {
                prefix: tokens[..=i].to_vec(),
                target: tokens[i + 1],
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

/// Per-epoch loss summary.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingResult {
    pub epoch_losses: Vec<f32>,
}

/// Educational training loop: forward through the full model, backprop only through
/// the output projection into `token_embeddings` and (when untied) `w_o`.
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
    let mut epoch_losses = Vec::with_capacity(config.epochs);

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

        epoch_losses.push(epoch_loss / num_examples as f32);
    }

    Ok(TrainingResult { epoch_losses })
}

/// Run one batch; returns mean loss over examples in the batch.
fn train_batch(
    model: &mut TinyModel,
    batch: &[TrainingExample],
    learning_rate: f32,
) -> anyhow::Result<f32> {
    if batch.is_empty() {
        bail!("train_batch received empty batch");
    }

    let dm = model.config.d_model;
    let vs = model.config.vocab_size;
    let tied = model.config.tie_embeddings;
    let batch_len = batch.len() as f32;

    let mut grad_embeddings = vec![0.0f32; vs * dm];
    let mut grad_w_o = vec![0.0f32; dm * vs];
    let mut total_loss = 0.0f32;

    for example in batch {
        let hidden = model.forward_hidden(&example.prefix)?;
        let logits = model.project_logits(&hidden)?;
        let h_last = hidden.last_row()?;
        let logits_last = logits.last_row()?;
        let grad_logits = softmax_grad(&logits_last, example.target)?;

        total_loss += cross_entropy_single(&logits_last, example.target)?;

        if tied {
            accumulate_tied_embedding_grad(&mut grad_embeddings, &grad_logits, &h_last, dm, vs);
        } else {
            accumulate_w_o_grad(&mut grad_w_o, &grad_logits, &h_last, dm, vs);
            let last_token = *example
                .prefix
                .last()
                .ok_or_else(|| anyhow::anyhow!("prefix must not be empty"))?;
            accumulate_input_embedding_grad(
                &mut grad_embeddings,
                &grad_logits,
                &model.w_o,
                last_token,
                dm,
                vs,
            )?;
        }
    }

    let scale = learning_rate / batch_len;
    apply_embedding_grad(&mut model.token_embeddings, &grad_embeddings, scale, dm, vs);

    if !tied {
        apply_w_o_grad(&mut model.w_o, &grad_w_o, scale, dm, vs);
    }

    Ok(total_loss / batch_len)
}

fn cross_entropy_single(logits: &[f32], target: usize) -> anyhow::Result<f32> {
    let row = Tensor::new(logits.to_vec(), vec![logits.len()])?;
    let probs = row.softmax()?;
    let p = probs.get1d(target)?;
    Ok(-p.ln())
}

fn softmax_grad(logits: &[f32], target: usize) -> anyhow::Result<Vec<f32>> {
    let row = Tensor::new(logits.to_vec(), vec![logits.len()])?;
    let probs = row.softmax()?;
    let mut grad: Vec<f32> = probs.data;
    if target >= grad.len() {
        bail!("target {target} out of logits range {}", grad.len());
    }
    grad[target] -= 1.0;
    Ok(grad)
}

fn accumulate_tied_embedding_grad(
    grad_e: &mut [f32],
    grad_logits: &[f32],
    h_last: &[f32],
    d_model: usize,
    vocab_size: usize,
) {
    for v in 0..vocab_size {
        for d in 0..d_model {
            grad_e[v * d_model + d] += grad_logits[v] * h_last[d];
        }
    }
}

fn accumulate_w_o_grad(
    grad_w_o: &mut [f32],
    grad_logits: &[f32],
    h_last: &[f32],
    d_model: usize,
    vocab_size: usize,
) {
    for d in 0..d_model {
        for v in 0..vocab_size {
            grad_w_o[d * vocab_size + v] += h_last[d] * grad_logits[v];
        }
    }
}

fn accumulate_input_embedding_grad(
    grad_e: &mut [f32],
    grad_logits: &[f32],
    w_o: &Tensor,
    last_token: usize,
    d_model: usize,
    vocab_size: usize,
) -> anyhow::Result<()> {
    for d in 0..d_model {
        let mut d_h = 0.0f32;
        for v in 0..vocab_size {
            d_h += grad_logits[v] * w_o.get2d(d, v)?;
        }
        grad_e[last_token * d_model + d] += d_h;
    }
    Ok(())
}

fn apply_embedding_grad(
    embeddings: &mut Tensor,
    grad: &[f32],
    scale: f32,
    d_model: usize,
    vocab_size: usize,
) {
    for v in 0..vocab_size {
        for d in 0..d_model {
            let idx = v * d_model + d;
            embeddings.data[idx] -= scale * grad[idx];
        }
    }
}

fn apply_w_o_grad(w_o: &mut Tensor, grad: &[f32], scale: f32, d_model: usize, vocab_size: usize) {
    for d in 0..d_model {
        for v in 0..vocab_size {
            let idx = d * vocab_size + v;
            w_o.data[idx] -= scale * grad[idx];
        }
    }
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
    let dataset = TextDataset::from_file(&args.input, &tokenizer)?;

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

    let result = train(&mut model, &dataset, &config, args.seed)?;

    for (i, loss) in result.epoch_losses.iter().enumerate() {
        println!("Epoch {} Loss: {loss:.6}", i + 1);
    }

    save_checkpoint(&model, &args.output)?;
    println!("Saved trained checkpoint to {}", args.output.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{load_checkpoint, save_checkpoint};
    use crate::generation::{generate, GenerateRequest};
    use crate::model::ModelConfig;
    use std::path::PathBuf;

    fn test_tokenizer() -> Tokenizer {
        Tokenizer::from_file(tokenizer::default_vocab_path()).unwrap()
    }

    fn tiny_model(seed: u64) -> TinyModel {
        let tok = test_tokenizer();
        let config = ModelConfig::for_vocab(tok.vocab_size());
        TinyModel::new_random(config, seed).unwrap()
    }

    fn temp_txt(name: &str, contents: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("forge_train_{}_{name}.txt", std::process::id()));
        fs::write(&path, contents).unwrap();
        path
    }

    fn temp_json(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("forge_train_{}_{name}.json", std::process::id()))
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
        // row 0 peaks at index 0; row 1 peaks at index 1
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
    fn dataset_builds_next_token_pairs() {
        let tok = test_tokenizer();
        let tokens = tok.encode("hello", false, false);
        let dataset = TextDataset::from_tokens(&tokens).unwrap();
        assert_eq!(dataset.len(), tokens.len() - 1);
        assert_eq!(dataset.examples[0].prefix, vec![tokens[0]]);
        assert_eq!(dataset.examples[0].target, tokens[1]);
    }

    #[test]
    fn dataset_from_file_loads_utf8_text() {
        let tok = test_tokenizer();
        let path = temp_txt("corpus", "hello hello");
        let dataset = TextDataset::from_file(&path, &tok).unwrap();
        assert!(dataset.len() >= 10);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn dataset_rejects_empty_file() {
        let tok = test_tokenizer();
        let path = temp_txt("empty", "   ");
        let err = TextDataset::from_file(&path, &tok).unwrap_err();
        assert!(err.to_string().contains("empty"));
        let _ = fs::remove_file(path);
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
        let path = temp_txt("repeat", "hello hello hello hello hello hello");
        let dataset = TextDataset::from_file(&path, &tok).unwrap();
        let mut model = tiny_model(99);
        let config = TrainingConfig {
            learning_rate: 0.05,
            epochs: 8,
            batch_size: 4,
        };

        let result = train(&mut model, &dataset, &config, 7).unwrap();
        assert!(result.epoch_losses.len() == 8);
        assert!(
            result.epoch_losses.last().unwrap() < result.epoch_losses.first().unwrap(),
            "first={:?} last={:?}",
            result.epoch_losses.first(),
            result.epoch_losses.last()
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn training_is_deterministic_with_fixed_seed() {
        let tok = test_tokenizer();
        let path = temp_txt("det", "hi hi hi hi hi");
        let dataset = TextDataset::from_file(&path, &tok).unwrap();
        let config = TrainingConfig {
            learning_rate: 0.02,
            epochs: 3,
            batch_size: 2,
        };

        let mut a = tiny_model(50);
        let mut b = tiny_model(50);
        let ra = train(&mut a, &dataset, &config, 123).unwrap();
        let rb = train(&mut b, &dataset, &config, 123).unwrap();

        assert_eq!(ra.epoch_losses, rb.epoch_losses);
        assert_eq!(a.token_embeddings.data, b.token_embeddings.data);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn checkpoint_save_load_after_training() {
        let tok = test_tokenizer();
        let txt = temp_txt("ckpt", "hello hello hello");
        let dataset = TextDataset::from_file(&txt, &tok).unwrap();
        let mut model = tiny_model(33);
        let config = TrainingConfig {
            learning_rate: 0.03,
            epochs: 2,
            batch_size: 2,
        };
        train(&mut model, &dataset, &config, 1).unwrap();

        let path = temp_json("trained");
        save_checkpoint(&model, &path).unwrap();
        let loaded = load_checkpoint(&path).unwrap();

        assert_eq!(model.token_embeddings.data, loaded.token_embeddings.data);
        assert!(loaded.validate_shapes().is_ok());

        let _ = fs::remove_file(txt);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn trained_model_generates_after_checkpoint_reload() {
        let tok = test_tokenizer();
        let txt = temp_txt("gen", "hello hello hello hello hello");
        let dataset = TextDataset::from_file(&txt, &tok).unwrap();
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

        let path = temp_json("gen_ckpt");
        save_checkpoint(&model, &path).unwrap();
        let loaded = load_checkpoint(&path).unwrap();

        let req = GenerateRequest {
            prompt: "hel".to_string(),
            max_new_tokens: 2,
            temperature: 0.0,
            seed: Some(1),
            top_k: None,
        };
        let out = generate(&req, &tok, &loaded).unwrap();
        assert_eq!(out.generated_tokens.len(), 2);

        let _ = fs::remove_file(txt);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn training_updates_untied_w_o() {
        let tok = test_tokenizer();
        let mut config = ModelConfig::for_vocab(tok.vocab_size());
        config.tie_embeddings = false;
        let mut model = TinyModel::new_random(config, 77).unwrap();
        let path = temp_txt("untied", "hello hello hello hello");
        let dataset = TextDataset::from_file(&path, &tok).unwrap();
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

        let _ = fs::remove_file(path);
    }

    #[test]
    fn training_updates_tied_embeddings() {
        let tok = test_tokenizer();
        let mut model = tiny_model(88);
        assert!(model.config.tie_embeddings);
        let path = temp_txt("tied", "hello hello hello hello");
        let dataset = TextDataset::from_file(&path, &tok).unwrap();
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

        let _ = fs::remove_file(path);
    }
}
