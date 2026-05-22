//! Token sampling from logits.

use crate::tensor::Tensor;
use anyhow::bail;
use rand::Rng;
use rand::rngs::StdRng;

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

    /// Sample a token index using temperature-scaled softmax probabilities.
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
        let probs = Tensor::new(scaled, vec![logits.len()])?.softmax()?;

        let r: f32 = rng.gen();
        let mut cumulative = 0.0f32;

        for (i, &prob) in probs.data.iter().enumerate() {
            cumulative += prob;
            if r <= cumulative {
                return Ok(i);
            }
        }

        Ok(logits.len() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
