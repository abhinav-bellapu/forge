//! Token sampling from logits.

use crate::tensor::Tensor;
use anyhow::bail;
use rand::rngs::StdRng;
use rand::Rng;

/// Sampling helpers for next-token prediction.
#[derive(Debug, Default)]
pub struct Sampler;

impl Sampler {
    /// Return the index of the largest logit.
    pub fn argmax(logits: &[f32]) -> anyhow::Result<usize> {
        if logits.is_empty() {
            bail!("logits cannot be empty");
        }

        let mut best_idx = 0usize;
        let mut best_val = logits[0];

        for (i, &val) in logits.iter().enumerate().skip(1) {
            if val > best_val {
                best_val = val;
                best_idx = i;
            }
        }

        Ok(best_idx)
    }

    /// Sample a token index using temperature-scaled softmax over the full vocabulary.
    pub fn sample_with_temperature(
        logits: &[f32],
        temperature: f32,
        rng: &mut StdRng,
    ) -> anyhow::Result<usize> {
        if logits.is_empty() {
            bail!("logits cannot be empty");
        }
        if temperature <= 0.0 {
            bail!("temperature must be greater than 0 for sampling");
        }

        let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();
        let probs = softmax_probabilities(&scaled)?;
        sample_from_probabilities(&probs, rng)
    }

    /// Sample from the top-`k` logits after temperature scaling.
    pub fn sample_top_k(
        logits: &[f32],
        temperature: f32,
        k: usize,
        rng: &mut StdRng,
    ) -> anyhow::Result<usize> {
        if logits.is_empty() {
            bail!("logits cannot be empty");
        }
        if temperature <= 0.0 {
            bail!("temperature must be greater than 0 for sampling");
        }
        if k == 0 {
            bail!("top-k k must be greater than 0");
        }

        let k = k.min(logits.len());
        let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();

        let mut ranked: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);

        let top_indices: Vec<usize> = ranked.iter().map(|(idx, _)| *idx).collect();
        let top_logits: Vec<f32> = ranked.iter().map(|(_, val)| *val).collect();

        let probs = softmax_probabilities(&top_logits)?;
        let local_idx = sample_from_probabilities(&probs, rng)?;
        Ok(top_indices[local_idx])
    }
}

/// Numerically stable softmax probabilities for a 1D logit vector.
fn softmax_probabilities(logits: &[f32]) -> anyhow::Result<Vec<f32>> {
    if logits.is_empty() {
        bail!("logits cannot be empty");
    }
    let tensor = Tensor::new(logits.to_vec(), vec![logits.len()])?;
    Ok(tensor.softmax()?.data)
}

/// Draw one index from a normalized probability vector.
fn sample_from_probabilities(probs: &[f32], rng: &mut StdRng) -> anyhow::Result<usize> {
    if probs.is_empty() {
        bail!("probabilities cannot be empty");
    }

    let r: f32 = rng.gen();
    let mut cumulative = 0.0f32;

    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if r <= cumulative {
            return Ok(i);
        }
    }

    Ok(probs.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::collections::HashSet;

    #[test]
    fn argmax_correctness() {
        assert_eq!(Sampler::argmax(&[1.0, 3.0, 2.0]).unwrap(), 1);
        assert_eq!(Sampler::argmax(&[5.0]).unwrap(), 0);
    }

    #[test]
    fn argmax_rejects_empty_logits() {
        assert!(Sampler::argmax(&[]).is_err());
    }

    #[test]
    fn temperature_sampling_rejects_invalid_temperature() {
        let mut rng = StdRng::seed_from_u64(1);
        assert!(Sampler::sample_with_temperature(&[1.0, 2.0], 0.0, &mut rng).is_err());
        assert!(Sampler::sample_with_temperature(&[1.0, 2.0], -1.0, &mut rng).is_err());
    }

    #[test]
    fn top_k_rejects_invalid_k() {
        let mut rng = StdRng::seed_from_u64(1);
        assert!(Sampler::sample_top_k(&[1.0, 2.0], 1.0, 0, &mut rng).is_err());
    }

    #[test]
    fn top_k_only_samples_from_top_set() {
        let logits = vec![0.1, 5.0, 3.0, 0.2, 4.0];
        let allowed: HashSet<usize> = [1usize, 4, 2].into_iter().collect();

        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..100 {
            let idx = Sampler::sample_top_k(&logits, 1.0, 3, &mut rng).unwrap();
            assert!(allowed.contains(&idx));
        }
    }

    #[test]
    fn top_k_with_k_one_matches_argmax() {
        let logits = [0.1, 5.0, 3.0, 0.2];
        let mut rng = StdRng::seed_from_u64(7);
        let top1 = Sampler::sample_top_k(&logits, 1.0, 1, &mut rng).unwrap();
        let greedy = Sampler::argmax(&logits).unwrap();
        assert_eq!(top1, greedy);
    }

    #[test]
    fn top_k_clamps_when_k_exceeds_vocab() {
        let logits = vec![1.0, 3.0, 2.0];
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..20 {
            let idx = Sampler::sample_top_k(&logits, 1.0, 100, &mut rng).unwrap();
            assert!(idx < logits.len());
        }
    }
}
