use anyhow::Result;
use forge::cli::Command;
use forge::{benchmark, checkpoint, cli, generation, inspection, training};

fn main() -> Result<()> {
    match cli::parse()? {
        Command::Generate(args) => generation::run_generate(&args),
        Command::SaveRandomCheckpoint(args) => {
            checkpoint::run_save_random_checkpoint(&args.output, args.seed)
        }
        Command::Train(args) => training::run_train(&args),
        Command::Bench(args) => benchmark::run_bench(&args),
        Command::Eval(args) => training::run_eval(&args),
        Command::Inspect(args) => inspection::run_inspect(&args),
    }
}
