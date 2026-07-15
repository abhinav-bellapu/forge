use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Forge — a tiny Rust inference runtime for transformer language models.
#[derive(Parser, Debug)]
#[command(name = "forge", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Generate text from a prompt.
    Generate(GenerateArgs),
    /// Save a randomly initialized model checkpoint to JSON.
    SaveRandomCheckpoint(SaveRandomCheckpointArgs),
    /// Train on a local text file and save a checkpoint.
    Train(TrainArgs),
    /// Benchmark generation throughput (local timing only).
    Bench(BenchArgs),
    /// Evaluate next-token loss and perplexity on a local text file.
    Eval(EvalArgs),
    /// Inspect model architecture and parameter counts.
    Inspect(InspectArgs),
}

#[derive(clap::Args, Debug)]
pub struct GenerateArgs {
    /// Input prompt text.
    #[arg(long)]
    pub prompt: String,

    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 20)]
    pub max_new_tokens: u32,

    /// Sampling temperature (0 = greedy argmax, >0 = stochastic).
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f32,

    /// Optional RNG seed for reproducible sampling.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Top-k sampling: only consider the k highest logits (requires temperature > 0).
    #[arg(long)]
    pub top_k: Option<usize>,

    /// Nucleus sampling: keep the smallest token set reaching probability p.
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Load model weights from a JSON checkpoint instead of random init.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
pub struct SaveRandomCheckpointArgs {
    /// Output checkpoint JSON path.
    #[arg(long)]
    pub output: PathBuf,

    /// Seed for random weight initialization.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

#[derive(clap::Args, Debug)]
pub struct TrainArgs {
    /// Local UTF-8 training text file.
    #[arg(long)]
    pub input: PathBuf,

    /// Number of training epochs.
    #[arg(long, default_value_t = 5)]
    pub epochs: usize,

    /// SGD learning rate for trainable embeddings and output weights.
    #[arg(long, default_value_t = 0.01)]
    pub learning_rate: f32,

    /// Optional global L2 norm cap for averaged batch gradients.
    #[arg(long)]
    pub max_grad_norm: Option<f32>,

    /// Output checkpoint JSON path.
    #[arg(long)]
    pub output: PathBuf,

    /// Examples per batch.
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    /// RNG seed for reproducible training.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Optional starting checkpoint (otherwise random init).
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
pub struct BenchArgs {
    /// Input prompt text.
    #[arg(long)]
    pub prompt: String,

    /// Maximum number of new tokens to generate per run.
    #[arg(long, default_value_t = 20)]
    pub max_new_tokens: u32,

    /// Number of timed generation runs.
    #[arg(long, default_value_t = 5)]
    pub runs: u32,

    /// Seed for model initialization and generation.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Load model weights from a JSON checkpoint instead of random init.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
pub struct EvalArgs {
    /// Local UTF-8 evaluation text file.
    #[arg(long)]
    pub input: PathBuf,

    /// Seed for random model initialization when no checkpoint is supplied.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Load model weights from a JSON checkpoint instead of random init.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
pub struct InspectArgs {
    /// Seed for random model initialization when no checkpoint is supplied.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Load model weights from a JSON checkpoint instead of random init.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
}

pub fn parse() -> anyhow::Result<Command> {
    Ok(Cli::parse().command)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_command_parses() {
        let cli = Cli::try_parse_from([
            "forge",
            "eval",
            "--input",
            "corpus.txt",
            "--checkpoint",
            "model.json",
        ])
        .unwrap();

        let Command::Eval(args) = cli.command else {
            panic!("expected eval command");
        };
        assert_eq!(args.input, PathBuf::from("corpus.txt"));
        assert_eq!(args.checkpoint, Some(PathBuf::from("model.json")));
        assert_eq!(args.seed, 42);
    }

    #[test]
    fn train_command_parses_gradient_clip_threshold() {
        let cli = Cli::try_parse_from([
            "forge",
            "train",
            "--input",
            "corpus.txt",
            "--output",
            "model.json",
            "--max-grad-norm",
            "0.5",
        ])
        .unwrap();

        let Command::Train(args) = cli.command else {
            panic!("expected train command");
        };
        assert_eq!(args.max_grad_norm, Some(0.5));
    }

    #[test]
    fn inspect_command_parses_checkpoint() {
        let cli = Cli::try_parse_from(["forge", "inspect", "--checkpoint", "model.json"]).unwrap();

        let Command::Inspect(args) = cli.command else {
            panic!("expected inspect command");
        };
        assert_eq!(args.checkpoint, Some(PathBuf::from("model.json")));
        assert_eq!(args.seed, 42);
    }

    #[test]
    fn generate_command_parses_top_p() {
        let cli = Cli::try_parse_from(["forge", "generate", "--prompt", "hello", "--top-p", "0.9"])
            .unwrap();

        let Command::Generate(args) = cli.command else {
            panic!("expected generate command");
        };
        assert_eq!(args.top_p, Some(0.9));
    }
}
