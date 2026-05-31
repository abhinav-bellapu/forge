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

    /// SGD learning rate for output-layer updates.
    #[arg(long, default_value_t = 0.01)]
    pub learning_rate: f32,

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

pub fn parse() -> anyhow::Result<Command> {
    Ok(Cli::parse().command)
}
