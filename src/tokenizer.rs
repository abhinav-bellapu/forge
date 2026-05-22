//! Character-level tokenizer with JSON vocabulary.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

const UNK_TOKEN: &str = "<unk>";
const BOS_TOKEN: &str = "<bos>";
const EOS_TOKEN: &str = "<eos>";

/// Character-level tokenizer backed by a string-to-id vocabulary file.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    token_to_id: HashMap<String, usize>,
    id_to_token: HashMap<usize, String>,
    unk_id: usize,
    bos_id: usize,
    eos_id: usize,
}

impl Tokenizer {
    /// Load vocabulary from a JSON file mapping token strings to integer IDs.
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| anyhow::anyhow!("failed to open vocab file {}: {e}", path.display()))?;
        let reader = BufReader::new(file);
        let token_to_id: HashMap<String, usize> = serde_json::from_reader(reader)
            .map_err(|e| anyhow::anyhow!("failed to parse vocab JSON: {e}"))?;

        if token_to_id.is_empty() {
            anyhow::bail!("vocabulary must not be empty");
        }

        let id_to_token: HashMap<usize, String> = token_to_id
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        let unk_id = *token_to_id
            .get(UNK_TOKEN)
            .ok_or_else(|| anyhow::anyhow!("vocabulary missing {UNK_TOKEN}"))?;
        let bos_id = *token_to_id
            .get(BOS_TOKEN)
            .ok_or_else(|| anyhow::anyhow!("vocabulary missing {BOS_TOKEN}"))?;
        let eos_id = *token_to_id
            .get(EOS_TOKEN)
            .ok_or_else(|| anyhow::anyhow!("vocabulary missing {EOS_TOKEN}"))?;

        Ok(Self {
            token_to_id,
            id_to_token,
            unk_id,
            bos_id,
            eos_id,
        })
    }

    /// Encode text into token IDs, one ID per Unicode scalar (character).
    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<usize> {
        let mut ids = Vec::new();

        if add_bos {
            ids.push(self.bos_id);
        }

        for ch in text.chars() {
            let token = ch.to_string();
            let id = self
                .token_to_id
                .get(&token)
                .copied()
                .unwrap_or(self.unk_id);
            ids.push(id);
        }

        if add_eos {
            ids.push(self.eos_id);
        }

        ids
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[usize], skip_special_tokens: bool) -> anyhow::Result<String> {
        let mut out = String::new();

        for &id in ids {
            let token = self
                .id_to_token
                .get(&id)
                .ok_or_else(|| anyhow::anyhow!("unknown token id: {id}"))?;

            if skip_special_tokens && self.is_special_id(id) {
                continue;
            }

            out.push_str(token);
        }

        Ok(out)
    }

    /// Number of entries in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    #[allow(dead_code)]
    pub fn unk_id(&self) -> usize {
        self.unk_id
    }

    #[allow(dead_code)]
    pub fn bos_id(&self) -> usize {
        self.bos_id
    }

    #[allow(dead_code)]
    pub fn eos_id(&self) -> usize {
        self.eos_id
    }

    fn is_special_id(&self, id: usize) -> bool {
        id == self.unk_id || id == self.bos_id || id == self.eos_id
    }
}

/// Default path to `vocab.json` at the project root (works with `cargo run` / tests).
pub fn default_vocab_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vocab.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer() -> Tokenizer {
        Tokenizer::from_file(default_vocab_path()).expect("load vocab.json")
    }

    #[test]
    fn loads_vocab_json() {
        let tok = test_tokenizer();
        assert_eq!(tok.vocab_size(), 78);
        assert_eq!(tok.unk_id(), 0);
        assert_eq!(tok.bos_id(), 1);
        assert_eq!(tok.eos_id(), 2);
    }

    #[test]
    fn encode_abc_returns_expected_ids() {
        let tok = test_tokenizer();
        assert_eq!(tok.encode("abc", false, false), vec![3, 4, 5]);
    }

    #[test]
    fn decode_ids_returns_abc() {
        let tok = test_tokenizer();
        let decoded = tok.decode(&[3, 4, 5], true).unwrap();
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn unknown_character_maps_to_unk() {
        let tok = test_tokenizer();
        assert_eq!(tok.encode("a@b", false, false), vec![3, 0, 4]);
        let decoded = tok.decode(&[3, 0, 4], true).unwrap();
        assert_eq!(decoded, "ab");
    }

    #[test]
    fn bos_and_eos_added_when_requested() {
        let tok = test_tokenizer();
        assert_eq!(tok.encode("hi", true, true), vec![1, 10, 11, 2]);
    }

    #[test]
    fn skip_special_tokens_on_decode() {
        let tok = test_tokenizer();
        let ids = tok.encode("hi", true, true);
        let decoded = tok.decode(&ids, true).unwrap();
        assert_eq!(decoded, "hi");

        let with_specials = tok.decode(&ids, false).unwrap();
        assert_eq!(with_specials, "<bos>hi<eos>");
    }

    #[test]
    fn invalid_decode_id_errors() {
        let tok = test_tokenizer();
        let err = tok.decode(&[9999], true).unwrap_err();
        assert!(err.to_string().contains("unknown token id"));
    }
}
