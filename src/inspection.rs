//! Human-readable model architecture and parameter inspection.

use crate::cli::InspectArgs;
use crate::generation::load_tokenizer_and_model;
use crate::model::TinyModel;

/// Build a stable, line-oriented summary suitable for CLI output.
pub fn inspection_report(model: &TinyModel) -> String {
    let config = &model.config;
    let counts = model.parameter_counts();
    format!(
        "vocab_size: {}\nmax_seq_len: {}\nd_model: {}\nn_heads: {}\nn_layers: {}\ntie_embeddings: {}\nparameters.token_embeddings: {}\nparameters.positional_embeddings: {}\nparameters.transformer_layers: {}\nparameters.output_projection_stored: {}\nparameters.output_projection_active: {}\nparameters.total_stored: {}\nparameters.total_active: {}",
        config.vocab_size,
        config.max_seq_len,
        config.d_model,
        config.n_heads,
        config.n_layers,
        config.tie_embeddings,
        counts.token_embeddings,
        counts.positional_embeddings,
        counts.transformer_layers,
        counts.output_projection_stored,
        counts.output_projection_active,
        counts.total_stored,
        counts.total_active,
    )
}

/// CLI entry point for `forge inspect`.
pub fn run_inspect(args: &InspectArgs) -> anyhow::Result<()> {
    let (_tokenizer, model) = load_tokenizer_and_model(args.seed, args.checkpoint.as_deref())?;
    println!("{}", inspection_report(&model));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

    #[test]
    fn report_includes_architecture_and_parameter_totals() {
        let model = TinyModel::new_random(
            ModelConfig {
                vocab_size: 16,
                max_seq_len: 8,
                d_model: 4,
                n_heads: 2,
                n_layers: 2,
                tie_embeddings: true,
            },
            1,
        )
        .unwrap();
        let report = inspection_report(&model);

        assert!(report.contains("n_layers: 2"));
        assert!(report.contains("tie_embeddings: true"));
        assert!(report.contains("parameters.total_stored: 576"));
        assert!(report.contains("parameters.total_active: 512"));
    }
}
