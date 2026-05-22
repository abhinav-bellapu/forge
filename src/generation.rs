use crate::cli::{Command, GenerateArgs};
use crate::model::{ModelConfig, TinyModel};
use crate::sampling::Sampler;
use crate::tokenizer::{self, Tokenizer};
use anyhow::bail;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Parameters for a single generation request.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub seed: Option<u64>,
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
    Ok(())
}

/// Autoregressive generation: forward → sample → append → decode.
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

    for _ in 0..req.max_new_tokens {
        if tokens.len() >= model.config.max_seq_len {
            bail!(
                "sequence length reached max_seq_len ({}) during generation",
                model.config.max_seq_len
            );
        }

        let logits = model.forward(&tokens)?;
        let last_logits = logits.last_row()?;
        if last_logits.is_empty() {
            bail!("model produced empty logits row");
        }

        let next_token = if req.temperature == 0.0 {
            Sampler::argmax(&last_logits)?
        } else {
            Sampler::sample_with_temperature(&last_logits, req.temperature, &mut sample_rng)?
        };

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

/// Build model config aligned with the loaded tokenizer.
pub fn model_config_for_tokenizer(tokenizer: &Tokenizer) -> ModelConfig {
    ModelConfig {
        vocab_size: tokenizer.vocab_size(),
        max_seq_len: 64,
        d_model: 16,
    }
}

/// Run generation from CLI flags and print results.
pub fn run_from_cli(command: &Command) -> anyhow::Result<()> {
    match command {
        Command::Generate(args) => {
            let req = GenerateRequest::from(args);
            let tokenizer = Tokenizer::from_file(tokenizer::default_vocab_path())?;
            let seed = req.seed.unwrap_or(42);
            let config = model_config_for_tokenizer(&tokenizer);
            let model = TinyModel::new_random(config, seed)?;

            let result = generate(&req, &tokenizer, &model)?;

            println!("Prompt: {}", req.prompt);
            println!("Input Tokens: {:?}", result.input_tokens);
            println!("Generated Tokens: {:?}", result.generated_tokens);
            println!("Output: {}", result.output_text);
        }
    }
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
        }
    }

    fn test_setup(seed: u64) -> (Tokenizer, TinyModel) {
        let tokenizer = Tokenizer::from_file(tokenizer::default_vocab_path()).unwrap();
        let config = model_config_for_tokenizer(&tokenizer);
        let model = TinyModel::new_random(config, seed).unwrap();
        (tokenizer, model)
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
}
