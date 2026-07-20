//! GPT-2-compatible inference, Hugging Face SafeTensors loading, and BPE tokenization.

use crate::attention::{Attention, ModelKvCache, MultiHeadKvCache};
use crate::cli::{Gpt2BenchArgs, Gpt2GenerateArgs, Gpt2LogitsArgs};
use crate::model::LayerNorm;
use crate::quantization::{QuantizationAxis, QuantizedMatrix};
use crate::sampling::Sampler;
use crate::tensor::Tensor;
use anyhow::{bail, Context};
use half::{bf16, f16};
use rand::rngs::StdRng;
use rand::SeedableRng;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer as HfTokenizer;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,
}

fn default_layer_norm_epsilon() -> f32 {
    1e-5
}

impl Gpt2Config {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.vocab_size == 0
            || self.n_positions == 0
            || self.n_embd == 0
            || self.n_layer == 0
            || self.n_head == 0
        {
            bail!("GPT-2 dimensions must all be greater than zero");
        }
        if !self.n_embd.is_multiple_of(self.n_head) {
            bail!(
                "n_embd {} must be divisible by n_head {}",
                self.n_embd,
                self.n_head
            );
        }
        if !self.layer_norm_epsilon.is_finite() || self.layer_norm_epsilon <= 0.0 {
            bail!("layer_norm_epsilon must be finite and positive");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum EmbeddingWeight {
    F32(Tensor),
    Int8(QuantizedMatrix),
}

impl EmbeddingWeight {
    fn shape(&self) -> [usize; 2] {
        match self {
            Self::F32(tensor) => [tensor.shape()[0], tensor.shape()[1]],
            Self::Int8(tensor) => tensor.shape(),
        }
    }

    fn row(&self, row: usize) -> anyhow::Result<Tensor> {
        match self {
            Self::F32(tensor) => {
                if row >= tensor.shape()[0] {
                    bail!("embedding row {row} is out of bounds");
                }
                let cols = tensor.shape()[1];
                Tensor::new(
                    tensor.data[row * cols..(row + 1) * cols].to_vec(),
                    vec![1, cols],
                )
            }
            Self::Int8(tensor) => tensor.row(row),
        }
    }

    fn project_rows(&self, hidden: &Tensor) -> anyhow::Result<Tensor> {
        match self {
            Self::F32(tensor) => project_embedding_rows(hidden, tensor),
            Self::Int8(tensor) => tensor.project_rows(hidden),
        }
    }

    fn quantize(&mut self) -> anyhow::Result<()> {
        if let Self::F32(tensor) = self {
            *self = Self::Int8(QuantizedMatrix::quantize(tensor, QuantizationAxis::Row)?);
        }
        Ok(())
    }

    fn parameter_count(&self) -> usize {
        let [rows, cols] = self.shape();
        rows * cols
    }

    fn memory_bytes(&self) -> usize {
        match self {
            Self::F32(tensor) => tensor.numel() * std::mem::size_of::<f32>(),
            Self::Int8(tensor) => tensor.memory_bytes(),
        }
    }
}

/// Project against a row-major tied embedding table without materializing its
/// very large transpose for every generated token.
fn project_embedding_rows(hidden: &Tensor, embedding: &Tensor) -> anyhow::Result<Tensor> {
    if hidden.ndim() != 2 || embedding.ndim() != 2 || hidden.shape()[1] != embedding.shape()[1] {
        bail!(
            "tied projection shape mismatch: {:?} x {:?}^T",
            hidden.shape(),
            embedding.shape()
        );
    }
    use rayon::prelude::*;
    let inputs = hidden.shape()[0];
    let vocab = embedding.shape()[0];
    let width = embedding.shape()[1];
    let mut output = vec![0.0f32; inputs * vocab];
    if inputs == 1 {
        let x = &hidden.data[..width];
        const ROW_TILE: usize = 128;
        output
            .par_chunks_mut(ROW_TILE)
            .enumerate()
            .for_each(|(tile, out)| {
                let first_row = tile * ROW_TILE;
                for (offset, dst) in out.iter_mut().enumerate() {
                    let row = first_row + offset;
                    let weight = &embedding.data[row * width..(row + 1) * width];
                    *dst = x.iter().zip(weight.iter()).map(|(&a, &b)| a * b).sum();
                }
            });
    } else {
        output
            .par_chunks_mut(vocab)
            .enumerate()
            .for_each(|(input_row, out)| {
                let x = &hidden.data[input_row * width..(input_row + 1) * width];
                for (row, dst) in out.iter_mut().enumerate() {
                    let weight = &embedding.data[row * width..(row + 1) * width];
                    *dst = x.iter().zip(weight.iter()).map(|(&a, &b)| a * b).sum();
                }
            });
    }
    Tensor::new(output, vec![inputs, vocab])
}

#[derive(Debug, Clone)]
enum LinearWeight {
    F32(Tensor),
    Int8(QuantizedMatrix),
}

impl LinearWeight {
    fn shape(&self) -> [usize; 2] {
        match self {
            Self::F32(tensor) => [tensor.shape()[0], tensor.shape()[1]],
            Self::Int8(tensor) => tensor.shape(),
        }
    }

    fn matmul(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        match self {
            Self::F32(tensor) => input.matmul(tensor),
            Self::Int8(tensor) => tensor.matmul_rhs(input),
        }
    }

    fn quantize(&mut self) -> anyhow::Result<()> {
        if let Self::F32(tensor) = self {
            *self = Self::Int8(QuantizedMatrix::quantize(tensor, QuantizationAxis::Column)?);
        }
        Ok(())
    }

    fn parameter_count(&self) -> usize {
        let [rows, cols] = self.shape();
        rows * cols
    }

    fn memory_bytes(&self) -> usize {
        match self {
            Self::F32(tensor) => tensor.numel() * std::mem::size_of::<f32>(),
            Self::Int8(tensor) => tensor.memory_bytes(),
        }
    }
}

#[derive(Debug, Clone)]
struct Linear {
    weight: LinearWeight,
    bias: Tensor,
}

impl Linear {
    fn new(weight: Tensor, bias: Tensor) -> anyhow::Result<Self> {
        if weight.ndim() != 2 {
            bail!("linear weight must be 2D");
        }
        let bias = as_row(bias)?;
        if bias.shape() != [1, weight.shape()[1]] {
            bail!(
                "linear bias shape {:?} does not match output width {}",
                bias.shape(),
                weight.shape()[1]
            );
        }
        Ok(Self {
            weight: LinearWeight::F32(weight),
            bias,
        })
    }

    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        self.weight.matmul(input)?.add_broadcast_row(&self.bias)
    }

    fn quantize(&mut self) -> anyhow::Result<()> {
        self.weight.quantize()
    }

    fn parameter_count(&self) -> usize {
        self.weight.parameter_count() + self.bias.numel()
    }

    fn memory_bytes(&self) -> usize {
        self.weight.memory_bytes() + self.bias.numel() * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Clone)]
struct Gpt2Attention {
    c_attn: Linear,
    c_proj: Linear,
}

impl Gpt2Attention {
    fn forward(&self, input: &Tensor, n_head: usize) -> anyhow::Result<Tensor> {
        let qkv = self.c_attn.forward(input)?;
        let d_model = input.shape()[1];
        let q = qkv.slice_cols(0, d_model)?;
        let k = qkv.slice_cols(d_model, 2 * d_model)?;
        let v = qkv.slice_cols(2 * d_model, 3 * d_model)?;
        let attention = Attention::multi_head_causal(&q, &k, &v, n_head)?;
        self.c_proj.forward(&attention)
    }

    fn forward_incremental(
        &self,
        input: &Tensor,
        cache: &mut MultiHeadKvCache,
    ) -> anyhow::Result<Tensor> {
        let qkv = self.c_attn.forward(input)?;
        let d_model = input.shape()[1];
        let q = qkv.slice_cols(0, d_model)?;
        let k = qkv.slice_cols(d_model, 2 * d_model)?;
        let v = qkv.slice_cols(2 * d_model, 3 * d_model)?;
        let attention = Attention::multi_head_cached(&q, &k, &v, cache)?;
        self.c_proj.forward(&attention)
    }

    fn quantize(&mut self) -> anyhow::Result<()> {
        self.c_attn.quantize()?;
        self.c_proj.quantize()
    }

    fn parameter_count(&self) -> usize {
        self.c_attn.parameter_count() + self.c_proj.parameter_count()
    }

    fn memory_bytes(&self) -> usize {
        self.c_attn.memory_bytes() + self.c_proj.memory_bytes()
    }
}

#[derive(Debug, Clone)]
struct Gpt2Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Gpt2Mlp {
    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        self.c_proj.forward(&self.c_fc.forward(input)?.gelu()?)
    }

    fn quantize(&mut self) -> anyhow::Result<()> {
        self.c_fc.quantize()?;
        self.c_proj.quantize()
    }

    fn parameter_count(&self) -> usize {
        self.c_fc.parameter_count() + self.c_proj.parameter_count()
    }

    fn memory_bytes(&self) -> usize {
        self.c_fc.memory_bytes() + self.c_proj.memory_bytes()
    }
}

#[derive(Debug, Clone)]
struct Gpt2Block {
    ln_1: LayerNorm,
    attention: Gpt2Attention,
    ln_2: LayerNorm,
    mlp: Gpt2Mlp,
}

impl Gpt2Block {
    fn forward(&self, input: &Tensor, n_head: usize) -> anyhow::Result<Tensor> {
        let attention = self.attention.forward(&self.ln_1.forward(input)?, n_head)?;
        let residual = input.add(&attention)?;
        let mlp = self.mlp.forward(&self.ln_2.forward(&residual)?)?;
        residual.add(&mlp)
    }

    fn forward_incremental(
        &self,
        input: &Tensor,
        cache: &mut MultiHeadKvCache,
    ) -> anyhow::Result<Tensor> {
        let attention = self
            .attention
            .forward_incremental(&self.ln_1.forward(input)?, cache)?;
        let residual = input.add(&attention)?;
        let mlp = self.mlp.forward(&self.ln_2.forward(&residual)?)?;
        residual.add(&mlp)
    }

    fn quantize(&mut self) -> anyhow::Result<()> {
        self.attention.quantize()?;
        self.mlp.quantize()
    }

    fn parameter_count(&self) -> usize {
        self.ln_1.gamma.numel()
            + self.ln_1.beta.numel()
            + self.attention.parameter_count()
            + self.ln_2.gamma.numel()
            + self.ln_2.beta.numel()
            + self.mlp.parameter_count()
    }

    fn memory_bytes(&self) -> usize {
        (self.ln_1.gamma.numel()
            + self.ln_1.beta.numel()
            + self.ln_2.gamma.numel()
            + self.ln_2.beta.numel())
            * std::mem::size_of::<f32>()
            + self.attention.memory_bytes()
            + self.mlp.memory_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gpt2MemoryReport {
    pub parameters: usize,
    pub fp32_equivalent_bytes: usize,
    pub stored_bytes: usize,
    pub reduction_percent: f64,
}

#[derive(Debug, Clone)]
pub struct Gpt2Model {
    pub config: Gpt2Config,
    token_embeddings: EmbeddingWeight,
    position_embeddings: EmbeddingWeight,
    blocks: Vec<Gpt2Block>,
    ln_f: LayerNorm,
}

impl Gpt2Model {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");
        let config: Gpt2Config = serde_json::from_slice(
            &fs::read(&config_path)
                .with_context(|| format!("failed to read {}", config_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
        config.validate()?;

        let bytes = fs::read(&weights_path)
            .with_context(|| format!("failed to read {}", weights_path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("failed to parse {}", weights_path.display()))?;
        Self::from_safetensors(config, &tensors)
    }

    fn from_safetensors(config: Gpt2Config, tensors: &SafeTensors<'_>) -> anyhow::Result<Self> {
        let mut named = HashMap::new();
        for name in tensors.names() {
            named.insert(name.to_string(), tensor_from_view(tensors.tensor(name)?)?);
        }
        Self::from_named_tensors(config, &named)
    }

    fn from_named_tensors(
        config: Gpt2Config,
        tensors: &HashMap<String, Tensor>,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        let required = |name: &str| -> anyhow::Result<Tensor> {
            let unprefixed = name.strip_prefix("transformer.").unwrap_or(name);
            tensors
                .get(name)
                .or_else(|| tensors.get(unprefixed))
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("SafeTensors checkpoint is missing {name}"))
        };

        let token_embeddings = required("transformer.wte.weight")?;
        let position_embeddings = required("transformer.wpe.weight")?;
        expect_shape(
            &token_embeddings,
            &[config.vocab_size, config.n_embd],
            "transformer.wte.weight",
        )?;
        expect_shape(
            &position_embeddings,
            &[config.n_positions, config.n_embd],
            "transformer.wpe.weight",
        )?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer in 0..config.n_layer {
            let prefix = format!("transformer.h.{layer}");
            let ln_1 = layer_norm_from_tensors(
                required(&format!("{prefix}.ln_1.weight"))?,
                required(&format!("{prefix}.ln_1.bias"))?,
                config.layer_norm_epsilon,
                config.n_embd,
            )?;
            let attention = Gpt2Attention {
                c_attn: Linear::new(
                    required(&format!("{prefix}.attn.c_attn.weight"))?,
                    required(&format!("{prefix}.attn.c_attn.bias"))?,
                )?,
                c_proj: Linear::new(
                    required(&format!("{prefix}.attn.c_proj.weight"))?,
                    required(&format!("{prefix}.attn.c_proj.bias"))?,
                )?,
            };
            let ln_2 = layer_norm_from_tensors(
                required(&format!("{prefix}.ln_2.weight"))?,
                required(&format!("{prefix}.ln_2.bias"))?,
                config.layer_norm_epsilon,
                config.n_embd,
            )?;
            let mlp = Gpt2Mlp {
                c_fc: Linear::new(
                    required(&format!("{prefix}.mlp.c_fc.weight"))?,
                    required(&format!("{prefix}.mlp.c_fc.bias"))?,
                )?,
                c_proj: Linear::new(
                    required(&format!("{prefix}.mlp.c_proj.weight"))?,
                    required(&format!("{prefix}.mlp.c_proj.bias"))?,
                )?,
            };
            validate_block_shapes(&attention, &mlp, config.n_embd, layer)?;
            blocks.push(Gpt2Block {
                ln_1,
                attention,
                ln_2,
                mlp,
            });
        }

        let ln_f = layer_norm_from_tensors(
            required("transformer.ln_f.weight")?,
            required("transformer.ln_f.bias")?,
            config.layer_norm_epsilon,
            config.n_embd,
        )?;

        Ok(Self {
            config,
            token_embeddings: EmbeddingWeight::F32(token_embeddings),
            position_embeddings: EmbeddingWeight::F32(position_embeddings),
            blocks,
            ln_f,
        })
    }

    pub fn new_cache(&self) -> anyhow::Result<ModelKvCache> {
        ModelKvCache::new(
            self.config.n_layer,
            self.config.n_head,
            self.config.head_dim(),
        )
    }

    pub fn forward(&self, token_ids: &[usize]) -> anyhow::Result<Tensor> {
        let hidden = self.forward_hidden(token_ids)?;
        self.token_embeddings.project_rows(&hidden)
    }

    pub fn forward_hidden(&self, token_ids: &[usize]) -> anyhow::Result<Tensor> {
        let mut hidden = self.embed_tokens(token_ids)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, self.config.n_head)?;
        }
        self.ln_f.forward(&hidden)
    }

    pub fn forward_incremental(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut ModelKvCache,
    ) -> anyhow::Result<Tensor> {
        if position != cache.len() {
            bail!(
                "position {position} does not match cache length {}",
                cache.len()
            );
        }
        if cache.layers.len() != self.blocks.len() {
            bail!("cache layer count does not match GPT-2 model");
        }
        let mut hidden = self.embed_token(token_id, position)?;
        for (block, layer_cache) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            hidden = block.forward_incremental(&hidden, layer_cache)?;
        }
        hidden = self.ln_f.forward(&hidden)?;
        self.token_embeddings.project_rows(&hidden)
    }

    pub fn quantize_int8(&mut self) -> anyhow::Result<Gpt2MemoryReport> {
        self.token_embeddings.quantize()?;
        self.position_embeddings.quantize()?;
        for block in &mut self.blocks {
            block.quantize()?;
        }
        Ok(self.memory_report())
    }

    pub fn memory_report(&self) -> Gpt2MemoryReport {
        let parameters = self.parameter_count();
        let fp32_equivalent_bytes = parameters * std::mem::size_of::<f32>();
        let stored_bytes = self.token_embeddings.memory_bytes()
            + self.position_embeddings.memory_bytes()
            + self
                .blocks
                .iter()
                .map(Gpt2Block::memory_bytes)
                .sum::<usize>()
            + (self.ln_f.gamma.numel() + self.ln_f.beta.numel()) * std::mem::size_of::<f32>();
        let reduction_percent = 100.0 * (1.0 - stored_bytes as f64 / fp32_equivalent_bytes as f64);
        Gpt2MemoryReport {
            parameters,
            fp32_equivalent_bytes,
            stored_bytes,
            reduction_percent,
        }
    }

    fn parameter_count(&self) -> usize {
        self.token_embeddings.parameter_count()
            + self.position_embeddings.parameter_count()
            + self
                .blocks
                .iter()
                .map(Gpt2Block::parameter_count)
                .sum::<usize>()
            + self.ln_f.gamma.numel()
            + self.ln_f.beta.numel()
    }

    fn embed_tokens(&self, token_ids: &[usize]) -> anyhow::Result<Tensor> {
        self.validate_tokens(token_ids)?;
        let mut data = Vec::with_capacity(token_ids.len() * self.config.n_embd);
        for (position, &token_id) in token_ids.iter().enumerate() {
            let token = self.token_embeddings.row(token_id)?;
            let position = self.position_embeddings.row(position)?;
            for (&token_value, &position_value) in token.data.iter().zip(position.data.iter()) {
                data.push(token_value + position_value);
            }
        }
        Tensor::new(data, vec![token_ids.len(), self.config.n_embd])
    }

    fn embed_token(&self, token_id: usize, position: usize) -> anyhow::Result<Tensor> {
        if token_id >= self.config.vocab_size || position >= self.config.n_positions {
            bail!("token or position is outside the GPT-2 configuration");
        }
        self.token_embeddings
            .row(token_id)?
            .add(&self.position_embeddings.row(position)?)
    }

    fn validate_tokens(&self, token_ids: &[usize]) -> anyhow::Result<()> {
        if token_ids.is_empty() {
            bail!("GPT-2 input cannot be empty");
        }
        if token_ids.len() > self.config.n_positions {
            bail!("GPT-2 input exceeds n_positions");
        }
        if token_ids
            .iter()
            .any(|&token| token >= self.config.vocab_size)
        {
            bail!("GPT-2 input contains an out-of-range token");
        }
        Ok(())
    }
}

pub struct Gpt2Tokenizer {
    inner: HfTokenizer,
}

impl Gpt2Tokenizer {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = model_dir.as_ref().join("tokenizer.json");
        let inner = HfTokenizer::from_file(&path)
            .map_err(|error| anyhow::anyhow!("failed to load {}: {error}", path.display()))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<usize>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!("GPT-2 BPE encoding failed: {error}"))?;
        Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
    }

    pub fn decode(&self, token_ids: &[usize]) -> anyhow::Result<String> {
        let ids: Vec<u32> = token_ids
            .iter()
            .map(|&id| u32::try_from(id))
            .collect::<Result<_, _>>()?;
        self.inner
            .decode(&ids, true)
            .map_err(|error| anyhow::anyhow!("GPT-2 BPE decoding failed: {error}"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gpt2GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub seed: u64,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gpt2GenerationResult {
    pub input_tokens: Vec<usize>,
    pub generated_tokens: Vec<usize>,
    pub all_tokens: Vec<usize>,
    pub output_text: String,
}

pub fn generate(
    request: &Gpt2GenerateRequest,
    tokenizer: &Gpt2Tokenizer,
    model: &Gpt2Model,
) -> anyhow::Result<Gpt2GenerationResult> {
    validate_generate_request(request)?;
    let mut tokens = tokenizer.encode(&request.prompt)?;
    if tokens.is_empty() {
        bail!("prompt produced no GPT-2 tokens");
    }
    if tokens.len() + request.max_new_tokens as usize > model.config.n_positions {
        bail!("prompt and requested output exceed GPT-2 context length");
    }
    let input_tokens = tokens.clone();
    let mut cache = model.new_cache()?;
    let mut logits = None;
    for (position, &token) in tokens.iter().enumerate() {
        logits = Some(model.forward_incremental(token, position, &mut cache)?);
    }
    let mut rng = StdRng::seed_from_u64(request.seed);
    for _ in 0..request.max_new_tokens {
        let row = logits
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing GPT-2 prompt logits"))?
            .last_row()?;
        let next = sample(&row, request, &mut rng)?;
        tokens.push(next);
        logits = Some(model.forward_incremental(next, tokens.len() - 1, &mut cache)?);
    }
    let generated_tokens = tokens[input_tokens.len()..].to_vec();
    let output_text = tokenizer.decode(&tokens)?;
    Ok(Gpt2GenerationResult {
        input_tokens,
        generated_tokens,
        all_tokens: tokens,
        output_text,
    })
}

pub fn run_generate(args: &Gpt2GenerateArgs) -> anyhow::Result<()> {
    let tokenizer = Gpt2Tokenizer::from_dir(&args.model_dir)?;
    let mut model = Gpt2Model::from_dir(&args.model_dir)?;
    let before = model.memory_report();
    if args.int8 {
        let after = model.quantize_int8()?;
        println!(
            "INT8 storage: {:.1} MiB ({:.2}% reduction from FP32)",
            after.stored_bytes as f64 / 1_048_576.0,
            after.reduction_percent
        );
    } else {
        println!(
            "FP32 storage: {:.1} MiB",
            before.stored_bytes as f64 / 1_048_576.0
        );
    }
    let request = Gpt2GenerateRequest {
        prompt: args.prompt.clone(),
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        seed: args.seed,
        top_k: args.top_k,
        top_p: args.top_p,
    };
    let result = generate(&request, &tokenizer, &model)?;
    println!("{}", result.output_text);
    Ok(())
}

#[derive(Debug, Serialize)]
struct LogitExport<'a> {
    implementation: &'a str,
    prompt: &'a str,
    token_ids: &'a [usize],
    last_logits: Vec<f32>,
}

pub fn run_logits(args: &Gpt2LogitsArgs) -> anyhow::Result<()> {
    let tokenizer = Gpt2Tokenizer::from_dir(&args.model_dir)?;
    let mut model = Gpt2Model::from_dir(&args.model_dir)?;
    if args.int8 {
        model.quantize_int8()?;
    }
    let token_ids = tokenizer.encode(&args.prompt)?;
    let last_logits = model.forward(&token_ids)?.last_row()?;
    let export = LogitExport {
        implementation: if args.int8 {
            "forge-int8"
        } else {
            "forge-fp32"
        },
        prompt: &args.prompt,
        token_ids: &token_ids,
        last_logits,
    };
    let file = std::fs::File::create(&args.output)
        .with_context(|| format!("failed to create {}", args.output.display()))?;
    serde_json::to_writer(file, &export)
        .with_context(|| format!("failed to write {}", args.output.display()))?;
    println!(
        "Wrote {} logits to {}",
        export.last_logits.len(),
        args.output.display()
    );
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gpt2BenchmarkResult {
    pub threads: usize,
    pub runs: u32,
    pub generated_tokens: usize,
    pub average_duration: Duration,
    pub tokens_per_second: f64,
}

fn benchmark_with_threads(
    threads: usize,
    runs: u32,
    request: &Gpt2GenerateRequest,
    tokenizer: &Gpt2Tokenizer,
    model: &Gpt2Model,
) -> anyhow::Result<Gpt2BenchmarkResult> {
    if threads == 0 || runs == 0 {
        bail!("benchmark threads and runs must be positive");
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("failed to construct benchmark thread pool")?;
    let _ = pool.install(|| generate(request, tokenizer, model))?;
    let mut total = Duration::ZERO;
    let mut generated_tokens = 0usize;
    for _ in 0..runs {
        let start = Instant::now();
        let result = pool.install(|| generate(request, tokenizer, model))?;
        total += start.elapsed();
        generated_tokens = result.generated_tokens.len();
    }
    let average_duration = total / runs;
    let tokens_per_second = generated_tokens as f64 / average_duration.as_secs_f64();
    Ok(Gpt2BenchmarkResult {
        threads,
        runs,
        generated_tokens,
        average_duration,
        tokens_per_second,
    })
}

pub fn run_benchmark(args: &Gpt2BenchArgs) -> anyhow::Result<()> {
    let tokenizer = Gpt2Tokenizer::from_dir(&args.model_dir)?;
    let mut model = Gpt2Model::from_dir(&args.model_dir)?;
    let fp32 = model.memory_report();
    if args.int8 {
        model.quantize_int8()?;
    }
    let stored = model.memory_report();
    let request = Gpt2GenerateRequest {
        prompt: args.prompt.clone(),
        max_new_tokens: args.max_new_tokens,
        temperature: 0.0,
        seed: 42,
        top_k: None,
        top_p: None,
    };
    let available_threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let single = benchmark_with_threads(1, args.runs, &request, &tokenizer, &model)?;
    let parallel =
        benchmark_with_threads(available_threads, args.runs, &request, &tokenizer, &model)?;
    let speedup = parallel.tokens_per_second / single.tokens_per_second;
    println!(
        "GPT-2 benchmark ({}, {} runs):",
        if args.int8 { "INT8" } else { "FP32" },
        args.runs
    );
    println!("  Parameters: {}", stored.parameters);
    println!(
        "  Storage: {:.1} MiB",
        stored.stored_bytes as f64 / 1_048_576.0
    );
    println!(
        "  Memory reduction vs FP32: {:.2}%",
        100.0 * (1.0 - stored.stored_bytes as f64 / fp32.stored_bytes as f64)
    );
    println!("  1 thread: {:.3} tokens/s", single.tokens_per_second);
    println!(
        "  {available_threads} threads: {:.3} tokens/s",
        parallel.tokens_per_second
    );
    println!("  Parallel speedup: {speedup:.3}x");
    if let Some(output) = &args.json_output {
        let report = serde_json::json!({
            "model": "gpt2",
            "precision": if args.int8 { "int8-weight-only" } else { "fp32" },
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "runs": args.runs,
            "parameters": stored.parameters,
            "fp32_bytes": fp32.stored_bytes,
            "stored_bytes": stored.stored_bytes,
            "memory_reduction_percent": 100.0 * (1.0 - stored.stored_bytes as f64 / fp32.stored_bytes as f64),
            "single_thread_tokens_per_second": single.tokens_per_second,
            "parallel_threads": available_threads,
            "parallel_tokens_per_second": parallel.tokens_per_second,
            "parallel_speedup": speedup,
        });
        let file = std::fs::File::create(output)
            .with_context(|| format!("failed to create {}", output.display()))?;
        serde_json::to_writer_pretty(file, &report)
            .with_context(|| format!("failed to write {}", output.display()))?;
        println!("  JSON report: {}", output.display());
    }
    Ok(())
}

fn sample(
    logits: &[f32],
    request: &Gpt2GenerateRequest,
    rng: &mut StdRng,
) -> anyhow::Result<usize> {
    if request.temperature == 0.0 {
        Sampler::argmax(logits)
    } else if let Some(top_k) = request.top_k {
        Sampler::sample_top_k(logits, request.temperature, top_k, rng)
    } else if let Some(top_p) = request.top_p {
        Sampler::sample_top_p(logits, request.temperature, top_p, rng)
    } else {
        Sampler::sample_with_temperature(logits, request.temperature, rng)
    }
}

fn validate_generate_request(request: &Gpt2GenerateRequest) -> anyhow::Result<()> {
    if request.prompt.trim().is_empty() || request.max_new_tokens == 0 {
        bail!("GPT-2 prompt must be non-empty and max_new_tokens must be positive");
    }
    if !request.temperature.is_finite() || request.temperature < 0.0 {
        bail!("temperature must be finite and non-negative");
    }
    if request.top_k == Some(0) {
        bail!("top_k must be positive");
    }
    if let Some(top_p) = request.top_p {
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            bail!("top_p must be in (0, 1]");
        }
    }
    if request.top_k.is_some() && request.top_p.is_some() {
        bail!("top_k and top_p are mutually exclusive");
    }
    Ok(())
}

fn layer_norm_from_tensors(
    gamma: Tensor,
    beta: Tensor,
    epsilon: f32,
    d_model: usize,
) -> anyhow::Result<LayerNorm> {
    let gamma = as_row(gamma)?;
    let beta = as_row(beta)?;
    expect_shape(&gamma, &[1, d_model], "layer norm gamma")?;
    expect_shape(&beta, &[1, d_model], "layer norm beta")?;
    Ok(LayerNorm {
        gamma,
        beta,
        epsilon,
    })
}

fn as_row(tensor: Tensor) -> anyhow::Result<Tensor> {
    if tensor.ndim() == 1 {
        let cols = tensor.shape()[0];
        Tensor::new(tensor.data, vec![1, cols])
    } else if tensor.ndim() == 2 && tensor.shape()[0] == 1 {
        Ok(tensor)
    } else {
        bail!(
            "expected a vector or one-row tensor, got {:?}",
            tensor.shape()
        )
    }
}

fn validate_block_shapes(
    attention: &Gpt2Attention,
    mlp: &Gpt2Mlp,
    d_model: usize,
    layer: usize,
) -> anyhow::Result<()> {
    let expected = [
        (
            attention.c_attn.weight.shape(),
            [d_model, 3 * d_model],
            "c_attn",
        ),
        (
            attention.c_proj.weight.shape(),
            [d_model, d_model],
            "attn.c_proj",
        ),
        (mlp.c_fc.weight.shape(), [d_model, 4 * d_model], "mlp.c_fc"),
        (
            mlp.c_proj.weight.shape(),
            [4 * d_model, d_model],
            "mlp.c_proj",
        ),
    ];
    for (actual, expected, name) in expected {
        if actual != expected {
            bail!("GPT-2 layer {layer} {name} shape {actual:?}, expected {expected:?}");
        }
    }
    Ok(())
}

fn expect_shape(tensor: &Tensor, expected: &[usize], name: &str) -> anyhow::Result<()> {
    if tensor.shape() != expected {
        bail!("{name} shape {:?}, expected {expected:?}", tensor.shape());
    }
    Ok(())
}

fn tensor_from_view(view: TensorView<'_>) -> anyhow::Result<Tensor> {
    let data = match view.dtype() {
        Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
        Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| f16::from_le_bytes(bytes.try_into().unwrap()).to_f32())
            .collect(),
        Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| bf16::from_le_bytes(bytes.try_into().unwrap()).to_f32())
            .collect(),
        dtype => bail!("unsupported SafeTensors dtype {dtype:?}; expected F32/F16/BF16"),
    };
    Tensor::new(data, view.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Gpt2Config {
        Gpt2Config {
            vocab_size: 8,
            n_positions: 8,
            n_embd: 4,
            n_layer: 1,
            n_head: 2,
            layer_norm_epsilon: 1e-5,
        }
    }

    fn values(rows: usize, cols: usize, offset: usize) -> Tensor {
        let data = (0..rows * cols)
            .map(|index| (((index + offset) % 17) as f32 - 8.0) * 0.01)
            .collect();
        Tensor::new(data, vec![rows, cols]).unwrap()
    }

    fn vector(cols: usize, value: f32) -> Tensor {
        Tensor::new(vec![value; cols], vec![cols]).unwrap()
    }

    fn tiny_tensors() -> HashMap<String, Tensor> {
        let mut tensors = HashMap::new();
        tensors.insert("transformer.wte.weight".into(), values(8, 4, 0));
        tensors.insert("transformer.wpe.weight".into(), values(8, 4, 3));
        tensors.insert("transformer.h.0.ln_1.weight".into(), vector(4, 1.0));
        tensors.insert("transformer.h.0.ln_1.bias".into(), vector(4, 0.0));
        tensors.insert(
            "transformer.h.0.attn.c_attn.weight".into(),
            values(4, 12, 5),
        );
        tensors.insert("transformer.h.0.attn.c_attn.bias".into(), vector(12, 0.0));
        tensors.insert("transformer.h.0.attn.c_proj.weight".into(), values(4, 4, 7));
        tensors.insert("transformer.h.0.attn.c_proj.bias".into(), vector(4, 0.0));
        tensors.insert("transformer.h.0.ln_2.weight".into(), vector(4, 1.0));
        tensors.insert("transformer.h.0.ln_2.bias".into(), vector(4, 0.0));
        tensors.insert("transformer.h.0.mlp.c_fc.weight".into(), values(4, 16, 11));
        tensors.insert("transformer.h.0.mlp.c_fc.bias".into(), vector(16, 0.0));
        tensors.insert(
            "transformer.h.0.mlp.c_proj.weight".into(),
            values(16, 4, 13),
        );
        tensors.insert("transformer.h.0.mlp.c_proj.bias".into(), vector(4, 0.0));
        tensors.insert("transformer.ln_f.weight".into(), vector(4, 1.0));
        tensors.insert("transformer.ln_f.bias".into(), vector(4, 0.0));
        tensors
    }

    fn tiny_model() -> Gpt2Model {
        Gpt2Model::from_named_tensors(tiny_config(), &tiny_tensors()).unwrap()
    }

    #[test]
    fn full_and_incremental_logits_match() {
        let model = tiny_model();
        let tokens = [1, 3, 2];
        let expected = model.forward(&tokens).unwrap().last_row().unwrap();
        let mut cache = model.new_cache().unwrap();
        let mut actual = vec![];
        for (position, &token) in tokens.iter().enumerate() {
            actual = model
                .forward_incremental(token, position, &mut cache)
                .unwrap()
                .last_row()
                .unwrap();
        }
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
        }
    }

    #[test]
    fn int8_quantization_reduces_memory_and_preserves_logits() {
        let model = tiny_model();
        let expected = model.forward(&[1, 3, 2]).unwrap();
        let fp32 = model.memory_report();
        let mut quantized = model.clone();
        let int8 = quantized.quantize_int8().unwrap();
        let actual = quantized.forward(&[1, 3, 2]).unwrap();
        assert!(int8.stored_bytes < fp32.stored_bytes);
        assert!(int8.reduction_percent > 40.0);
        let max_error = actual
            .data
            .iter()
            .zip(expected.data.iter())
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 0.03, "max error {max_error}");
    }

    #[test]
    fn malformed_config_is_rejected() {
        let mut config = tiny_config();
        config.n_head = 3;
        assert!(config.validate().is_err());
    }

    #[test]
    fn loader_accepts_hugging_face_unprefixed_tensor_names() {
        let tensors = tiny_tensors()
            .into_iter()
            .map(|(name, tensor)| {
                (
                    name.strip_prefix("transformer.")
                        .unwrap_or(&name)
                        .to_string(),
                    tensor,
                )
            })
            .collect();
        let model = Gpt2Model::from_named_tensors(tiny_config(), &tensors).unwrap();
        assert_eq!(model.forward(&[1, 2]).unwrap().shape(), &[2, 8]);
    }
}
