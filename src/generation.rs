use crate::cli::{Command, GenerateArgs};

/// Parameters for a single generation request.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub seed: Option<u64>,
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
        anyhow::bail!("prompt must not be empty");
    }
    if req.max_new_tokens == 0 {
        anyhow::bail!("max_new_tokens must be greater than 0");
    }
    if req.temperature <= 0.0 {
        anyhow::bail!("temperature must be greater than 0");
    }
    Ok(())
}

/// Stub generation: returns placeholder text after validation.
pub fn generate_stub(req: &GenerateRequest) -> anyhow::Result<String> {
    validate_request(req)?;
    Ok("[stub generation from Forge]".to_string())
}

/// Run generation from CLI flags and print results.
pub fn run_from_cli(command: &Command) -> anyhow::Result<()> {
    match command {
        Command::Generate(args) => {
            let req = GenerateRequest::from(args);
            let output = generate_stub(&req)?;
            println!("Prompt: {}", req.prompt);
            println!("Generated: {}", output);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> GenerateRequest {
        GenerateRequest {
            prompt: "hello world".to_string(),
            max_new_tokens: 20,
            temperature: 1.0,
            seed: None,
        }
    }

    #[test]
    fn empty_prompt_errors() {
        let mut req = sample_request();
        req.prompt = "   ".to_string();
        let err = validate_request(&req).unwrap_err();
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
    fn non_positive_temperature_errors() {
        let mut req = sample_request();
        req.temperature = 0.0;
        let err = validate_request(&req).unwrap_err();
        assert!(err.to_string().contains("temperature"));

        req.temperature = -1.0;
        assert!(validate_request(&req).is_err());
    }

    #[test]
    fn valid_request_returns_stub() {
        let req = sample_request();
        let out = generate_stub(&req).unwrap();
        assert_eq!(out, "[stub generation from Forge]");
    }
}
