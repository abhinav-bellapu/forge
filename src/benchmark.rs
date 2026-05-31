//! Inference benchmarking for autoregressive generation.

use crate::cli::BenchArgs;
use crate::generation::{generate, load_tokenizer_and_model, validate_request, GenerateRequest};
use anyhow::bail;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Configuration for a generation benchmark run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkConfig {
    pub prompt: String,
    pub max_new_tokens: u32,
    pub runs: u32,
    pub seed: u64,
    pub checkpoint: Option<PathBuf>,
}

impl From<&BenchArgs> for BenchmarkConfig {
    fn from(args: &BenchArgs) -> Self {
        Self {
            prompt: args.prompt.clone(),
            max_new_tokens: args.max_new_tokens,
            runs: args.runs,
            seed: args.seed,
            checkpoint: args.checkpoint.clone(),
        }
    }
}

/// Aggregated timing results from repeated generation runs.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkResult {
    pub runs: u32,
    pub tokens_generated: u32,
    pub avg_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub tokens_per_second: f64,
}

/// Validate benchmark parameters.
pub fn validate_config(config: &BenchmarkConfig) -> anyhow::Result<()> {
    if config.prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }
    if config.max_new_tokens == 0 {
        bail!("max_new_tokens must be greater than 0");
    }
    if config.runs == 0 {
        bail!("runs must be greater than 0");
    }
    Ok(())
}

/// Run generation repeatedly and collect timing statistics.
///
/// Model loading happens once; each timed run calls [`generate`] with greedy
/// decoding (`temperature = 0`) and the configured seed for reproducibility.
pub fn run_benchmark(config: &BenchmarkConfig) -> anyhow::Result<BenchmarkResult> {
    validate_config(config)?;

    let req = GenerateRequest {
        prompt: config.prompt.clone(),
        max_new_tokens: config.max_new_tokens,
        temperature: 0.0,
        seed: Some(config.seed),
        top_k: None,
    };
    validate_request(&req)?;

    let (tokenizer, model) = load_tokenizer_and_model(config.seed, config.checkpoint.as_deref())?;

    let mut durations = Vec::with_capacity(config.runs as usize);
    let mut tokens_generated = 0u32;

    for _ in 0..config.runs {
        let start = Instant::now();
        let result = generate(&req, &tokenizer, &model)?;
        durations.push(start.elapsed());
        tokens_generated = result.generated_tokens.len() as u32;
    }

    let total: Duration = durations.iter().copied().sum();
    let avg_duration = total / config.runs;
    let min_duration = durations.iter().copied().min().unwrap_or(Duration::ZERO);
    let max_duration = durations.iter().copied().max().unwrap_or(Duration::ZERO);
    let avg_secs = avg_duration.as_secs_f64();
    let tokens_per_second = if avg_secs > 0.0 {
        f64::from(tokens_generated) / avg_secs
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        runs: config.runs,
        tokens_generated,
        avg_duration,
        min_duration,
        max_duration,
        tokens_per_second,
    })
}

/// Run `forge bench` and print results.
pub fn run_bench(args: &BenchArgs) -> anyhow::Result<()> {
    let config = BenchmarkConfig::from(args);
    let result = run_benchmark(&config)?;

    println!(
        "Benchmark ({} runs, {} new tokens):",
        result.runs, config.max_new_tokens
    );
    println!("  Average: {:.4}s", result.avg_duration.as_secs_f64());
    println!("  Min:     {:.4}s", result.min_duration.as_secs_f64());
    println!("  Max:     {:.4}s", result.max_duration.as_secs_f64());
    println!("  Tokens:  {}", result.tokens_generated);
    println!("  Throughput: {:.1} tokens/sec", result.tokens_per_second);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::save_random_checkpoint;
    use std::path::PathBuf;

    fn sample_config() -> BenchmarkConfig {
        BenchmarkConfig {
            prompt: "hello".to_string(),
            max_new_tokens: 4,
            runs: 2,
            seed: 42,
            checkpoint: None,
        }
    }

    struct TempFile(PathBuf);

    impl TempFile {
        fn json(name: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("forge_bench_{}_{name}.json", std::process::id()));
            Self(path)
        }

        fn path(&self) -> &PathBuf {
            &self.0
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn assert_result_sane(result: &BenchmarkResult, expected_tokens: u32, runs: u32) {
        assert_eq!(result.runs, runs);
        assert_eq!(result.tokens_generated, expected_tokens);
        assert!(!result.avg_duration.is_zero());
        assert!(result.min_duration <= result.avg_duration);
        assert!(result.max_duration >= result.avg_duration);
        assert!(result.tokens_per_second > 0.0);
        assert!(result.tokens_per_second.is_finite());
    }

    #[test]
    fn validate_config_rejects_zero_runs() {
        let mut config = sample_config();
        config.runs = 0;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("runs"));
    }

    #[test]
    fn validate_config_rejects_zero_max_new_tokens() {
        let mut config = sample_config();
        config.max_new_tokens = 0;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("max_new_tokens"));
    }

    #[test]
    fn validate_config_rejects_empty_prompt() {
        let mut config = sample_config();
        config.prompt = "   ".to_string();
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("prompt"));
    }

    #[test]
    fn benchmark_with_random_model_produces_sane_results() {
        let config = sample_config();
        let result = run_benchmark(&config).unwrap();
        assert_result_sane(&result, config.max_new_tokens, config.runs);
    }

    #[test]
    fn benchmark_with_checkpoint_produces_sane_results() {
        let path = TempFile::json("ckpt");
        save_random_checkpoint(path.path(), 7).unwrap();

        let config = BenchmarkConfig {
            checkpoint: Some(path.path().clone()),
            ..sample_config()
        };
        let result = run_benchmark(&config).unwrap();
        assert_result_sane(&result, config.max_new_tokens, config.runs);
    }

    #[test]
    fn benchmark_token_count_is_deterministic_across_runs() {
        let config = BenchmarkConfig {
            runs: 5,
            max_new_tokens: 8,
            ..sample_config()
        };
        let result = run_benchmark(&config).unwrap();
        assert_eq!(result.tokens_generated, 8);
    }

    #[test]
    fn benchmark_result_fields_are_nonnegative() {
        let result = run_benchmark(&sample_config()).unwrap();
        assert!(result.avg_duration.as_secs_f64() >= 0.0);
        assert!(result.min_duration.as_secs_f64() >= 0.0);
        assert!(result.max_duration.as_secs_f64() >= 0.0);
        assert!(result.tokens_generated > 0);
        assert!(result.tokens_per_second >= 0.0);
    }
}
