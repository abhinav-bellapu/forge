use crate::attention::ModelKvCache;
use crate::checkpoint::load_checkpoint;
use crate::cli::GenerateArgs;
use crate::model::{ModelConfig, TinyModel};
use crate::sampling::Sampler;
use crate::tokenizer::{self, Tokenizer};
use anyhow::bail;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::Path;

/// Parameters for a single generation request.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub seed: Option<u64>,
    pub top_k: Option<usize>,
}

/// Result of autoregressive generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationResult {
    pub input_tokens: Vec<usize>,
    pub all_tokens: Vec<usize>,
    pub generated_tokens: Vec<usize>,
    pub output_text: String,
}

impl From<&GenerateArgs> for GenerateRequest {
    fn from(args: &GenerateArgs) -> Self {
        Self {
            prompt: args.prompt.clone(),
            max_new_tokens: args.max_new_tokens,
            temperature: args.temperature,
            seed: args.seed,
            top_k: args.top_k,
        }
    }
}

/// Validate generation parameters.
pub fn validate_request(req: &GenerateRequest) -> anyhow::Result<()> {
    if req.prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }
    if req.max_new_tokens == 0 {
        bail!("max_new_tokens must be greater than 0");
    }
    if req.temperature < 0.0 {
        bail!("temperature must be >= 0");
    }
    if let Some(k) = req.top_k {
        if k == 0 {
            bail!("top_k must be greater than 0");
        }
    }
    Ok(())
}

/// Load tokenizer and model (from checkpoint or random initialization).
pub fn load_tokenizer_and_model(
    seed: u64,
    checkpoint: Option<&Path>,
) -> anyhow::Result<(Tokenizer, TinyModel)> {
    let tokenizer = Tokenizer::from_file(tokenizer::default_vocab_path())?;

    let model = if let Some(path) = checkpoint {
        let loaded = load_checkpoint(path)?;
        if loaded.config.vocab_size != tokenizer.vocab_size() {
            bail!(
                "checkpoint vocab_size {} does not match tokenizer vocab_size {}",
                loaded.config.vocab_size,
                tokenizer.vocab_size()
            );
        }
        loaded
    } else {
        let config = model_config_for_tokenizer(&tokenizer);
        TinyModel::new_random(config, seed)?
    };

    Ok((tokenizer, model))
}

/// Build model config aligned with the loaded tokenizer.
pub fn model_config_for_tokenizer(tokenizer: &Tokenizer) -> ModelConfig {
    ModelConfig::for_vocab(tokenizer.vocab_size())
}

/// Choose the next token from the final-position logits.
fn sample_next_token(
    logits: &[f32],
    req: &GenerateRequest,
    rng: &mut StdRng,
) -> anyhow::Result<usize> {
    if logits.is_empty() {
        bail!("model produced empty logits row");
    }

    if req.temperature == 0.0 {
        return Sampler::argmax(logits);
    }

    if let Some(k) = req.top_k {
        return Sampler::sample_top_k(logits, req.temperature, k, rng);
    }

    Sampler::sample_with_temperature(logits, req.temperature, rng)
}

/// Autoregressive generation with KV cache: warm prompt → incremental forward → sample.
///
/// Before this sprint, each step called [`TinyModel::forward`] on the full sequence,
/// recomputing attention over all prior tokens. Cached decoding appends one K/V row
/// per token and reuses prior keys/values via [`TinyModel::forward_incremental`].
pub fn generate(
    req: &GenerateRequest,
    tokenizer: &Tokenizer,
    model: &TinyModel,
) -> anyhow::Result<GenerationResult> {
    validate_request(req)?;

    let mut tokens = tokenizer.encode(&req.prompt, false, false);
    let input_len = tokens.len();

    if tokens.is_empty() {
        bail!("prompt produced no tokens");
    }
    if tokens.len() > model.config.max_seq_len {
        bail!(
            "prompt length {} exceeds max_seq_len {}",
            tokens.len(),
            model.config.max_seq_len
        );
    }

    let mut sample_rng = StdRng::seed_from_u64(req.seed.unwrap_or(42));
    let mut cache = ModelKvCache::new(
        model.config.n_layers,
        model.config.n_heads,
        model.config.head_dim(),
    )?;
    let mut last_logits = None;

    // Warm the cache from the prompt (incremental forwards, same math as full forward).
    for (position, &token_id) in tokens.iter().enumerate() {
        last_logits = Some(model.forward_incremental(token_id, position, &mut cache)?);
    }

    for _ in 0..req.max_new_tokens {
        if tokens.len() >= model.config.max_seq_len {
            bail!(
                "sequence length reached max_seq_len ({}) during generation",
                model.config.max_seq_len
            );
        }

        let logits = last_logits
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing logits after prompt warm-up"))?;
        let last_logits_vec = logits.last_row()?;
        let next_token = sample_next_token(&last_logits_vec, req, &mut sample_rng)?;
        tokens.push(next_token);

        let position = tokens.len() - 1;
        last_logits = Some(model.forward_incremental(next_token, position, &mut cache)?);
    }

    let generated_tokens = tokens[input_len..].to_vec();
    let output_text = tokenizer.decode(&tokens, true)?;

    Ok(GenerationResult {
        input_tokens: tokens[..input_len].to_vec(),
        all_tokens: tokens,
        generated_tokens,
        output_text,
    })
}

/// Run `forge generate` and print results.
pub fn run_generate(args: &GenerateArgs) -> anyhow::Result<()> {
    let req = GenerateRequest::from(args);
    let seed = req.seed.unwrap_or(42);
    let (tokenizer, model) = load_tokenizer_and_model(seed, args.checkpoint.as_deref())?;
    let result = generate(&req, &tokenizer, &model)?;

    println!("Prompt: {}", req.prompt);
    println!("Input Tokens: {:?}", result.input_tokens);
    println!("Generated Tokens: {:?}", result.generated_tokens);
    println!("Output: {}", result.output_text);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> GenerateRequest {
        GenerateRequest {
            prompt: "hi".to_string(),
            max_new_tokens: 4,
            temperature: 0.0,
            seed: Some(42),
            top_k: None,
        }
    }

    fn test_setup(seed: u64) -> (Tokenizer, TinyModel) {
        load_tokenizer_and_model(seed, None).unwrap()
    }

    #[test]
    fn empty_prompt_errors() {
        let mut req = sample_request();
        req.prompt = "   ".to_string();
        let (tok, model) = test_setup(42);
        let err = generate(&req, &tok, &model).unwrap_err();
        assert!(err.to_string().contains("prompt"));
    }

    #[test]
    fn zero_max_new_tokens_errors() {
        let mut req = sample_request();
        req.max_new_tokens = 0;
        let err = validate_request(&req).unwrap_err();
        assert!(err.to_string().contains("max_new_tokens"));
    }

    #[test]
    fn negative_temperature_errors() {
        let mut req = sample_request();
        req.temperature = -1.0;
        let err = validate_request(&req).unwrap_err();
        assert!(err.to_string().contains("temperature"));
    }

    #[test]
    fn zero_top_k_errors() {
        let mut req = sample_request();
        req.top_k = Some(0);
        let err = validate_request(&req).unwrap_err();
        assert!(err.to_string().contains("top_k"));
    }

    #[test]
    fn zero_temperature_is_valid() {
        let req = sample_request();
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn generation_produces_additional_tokens() {
        let req = sample_request();
        let (tok, model) = test_setup(42);
        let result = generate(&req, &tok, &model).unwrap();

        assert_eq!(result.input_tokens.len(), 2);
        assert_eq!(result.generated_tokens.len(), 4);
        assert_eq!(result.all_tokens.len(), 6);
    }

    #[test]
    fn same_seed_produces_identical_outputs() {
        let req = sample_request();
        let (tok_a, model_a) = test_setup(99);
        let (tok_b, model_b) = test_setup(99);

        let out_a = generate(&req, &tok_a, &model_a).unwrap();
        let out_b = generate(&req, &tok_b, &model_b).unwrap();

        assert_eq!(out_a.generated_tokens, out_b.generated_tokens);
        assert_eq!(out_a.output_text, out_b.output_text);
    }

    #[test]
    fn different_seeds_can_produce_different_outputs() {
        let req = GenerateRequest {
            prompt: "hi".to_string(),
            max_new_tokens: 8,
            temperature: 1.0,
            seed: Some(1),
            top_k: None,
        };

        let (tok_a, model_a) = test_setup(1);
        let (tok_b, model_b) = test_setup(2);

        let mut req_a = req.clone();
        req_a.seed = Some(1);
        let mut req_b = req.clone();
        req_b.seed = Some(2);

        let out_a = generate(&req_a, &tok_a, &model_a).unwrap();
        let out_b = generate(&req_b, &tok_b, &model_b).unwrap();

        assert_ne!(out_a.generated_tokens, out_b.generated_tokens);
    }

    /// Baseline loop: full-sequence forward each step (pre–KV-cache behavior).
    fn generate_full_forward(
        req: &GenerateRequest,
        tokenizer: &Tokenizer,
        model: &TinyModel,
    ) -> anyhow::Result<GenerationResult> {
        validate_request(req)?;

        let mut tokens = tokenizer.encode(&req.prompt, false, false);
        let input_len = tokens.len();

        if tokens.is_empty() {
            bail!("prompt produced no tokens");
        }
        if tokens.len() > model.config.max_seq_len {
            bail!(
                "prompt length {} exceeds max_seq_len {}",
                tokens.len(),
                model.config.max_seq_len
            );
        }

        let mut sample_rng = StdRng::seed_from_u64(req.seed.unwrap_or(42));

        for _ in 0..req.max_new_tokens {
            if tokens.len() >= model.config.max_seq_len {
                bail!(
                    "sequence length reached max_seq_len ({}) during generation",
                    model.config.max_seq_len
                );
            }

            let logits = model.forward(&tokens)?;
            let last_logits = logits.last_row()?;
            let next_token = sample_next_token(&last_logits, req, &mut sample_rng)?;
            tokens.push(next_token);
        }

        let generated_tokens = tokens[input_len..].to_vec();
        let output_text = tokenizer.decode(&tokens, true)?;

        Ok(GenerationResult {
            input_tokens: tokens[..input_len].to_vec(),
            all_tokens: tokens,
            generated_tokens,
            output_text,
        })
    }

    #[test]
    fn generation_with_cache_matches_full_forward() {
        let req = sample_request();
        let (tok, model) = test_setup(42);

        let cached = generate(&req, &tok, &model).unwrap();
        let baseline = generate_full_forward(&req, &tok, &model).unwrap();

        assert_eq!(cached.generated_tokens, baseline.generated_tokens);
        assert_eq!(cached.output_text, baseline.output_text);
    }

    #[test]
    fn cache_grows_during_generation() {
        let req = sample_request();
        let (tok, model) = test_setup(42);
        let input_len = tok.encode(&req.prompt, false, false).len();

        let mut cache = ModelKvCache::new(
            model.config.n_layers,
            model.config.n_heads,
            model.config.head_dim(),
        )
        .unwrap();
        let mut tokens = tok.encode(&req.prompt, false, false);
        let mut sample_rng = StdRng::seed_from_u64(req.seed.unwrap_or(42));
        let mut last_logits = None;

        for (position, &token_id) in tokens.iter().enumerate() {
            last_logits = Some(
                model
                    .forward_incremental(token_id, position, &mut cache)
                    .unwrap(),
            );
        }
        assert_eq!(cache.len(), input_len);

        for _ in 0..req.max_new_tokens {
            let logits = last_logits.as_ref().unwrap();
            let last_logits_vec = logits.last_row().unwrap();
            let next_token = sample_next_token(&last_logits_vec, &req, &mut sample_rng).unwrap();
            tokens.push(next_token);
            let position = tokens.len() - 1;
            last_logits = Some(
                model
                    .forward_incremental(next_token, position, &mut cache)
                    .unwrap(),
            );
        }

        assert_eq!(cache.len(), input_len + req.max_new_tokens as usize);
    }

    #[test]
    fn top_k_generation_is_deterministic_with_seed() {
        let req = GenerateRequest {
            prompt: "hi".to_string(),
            max_new_tokens: 6,
            temperature: 1.0,
            seed: Some(7),
            top_k: Some(5),
        };

        let (tok_a, model_a) = test_setup(7);
        let (tok_b, model_b) = test_setup(7);

        let out_a = generate(&req, &tok_a, &model_a).unwrap();
        let out_b = generate(&req, &tok_b, &model_b).unwrap();

        assert_eq!(out_a.generated_tokens, out_b.generated_tokens);
    }
}
