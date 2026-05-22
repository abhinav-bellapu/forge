mod attention;
mod checkpoint;
mod cli;
mod generation;
mod model;
mod sampling;
mod tensor;
mod tokenizer;

use anyhow::Result;
use cli::Command;

fn main() -> Result<()> {
    match cli::parse()? {
        Command::Generate(args) => generation::run_generate(&args),
        Command::SaveRandomCheckpoint(args) => {
            checkpoint::run_save_random_checkpoint(&args.output, args.seed)
        }
    }
}
