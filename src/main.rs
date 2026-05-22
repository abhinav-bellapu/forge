mod attention;
mod checkpoint;
mod cli;
mod generation;
mod model;
mod sampling;
mod tensor;
mod tokenizer;

use anyhow::Result;

fn main() -> Result<()> {
    let command = cli::parse()?;
    generation::run_from_cli(&command)
}
