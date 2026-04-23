//! Text tokenization for ferrotorch models.
//!
//! This crate is a thin wrapper around the HuggingFace
//! [`tokenizers`] crate — the same library powering Python's
//! `transformers.AutoTokenizer` — with an API shaped for ferrotorch
//! idioms (`Vec<u32>` token ids, `FerrotorchResult` errors).
//!
//! # Quick start
//!
//! ```no_run
//! use ferrotorch_tokenize::{load_tokenizer, encode, decode};
//!
//! // Llama 3 ships a `tokenizer.json` alongside its weights.
//! let tok = load_tokenizer("/path/to/tokenizer.json").unwrap();
//! let ids = encode(&tok, "Hello, world!", /* add_special_tokens = */ true).unwrap();
//! let text = decode(&tok, &ids, /* skip_special_tokens = */ false).unwrap();
//! ```
//!
//! # Scope
//!
//! The wrapper covers the path the Llama 3 8B PoC needs:
//! - Load a `tokenizer.json` file into a [`Tokenizer`].
//! - Encode / decode single strings and batches.
//! - Query vocab size and special-token ids.
//!
//! More advanced features (chat templates, truncation strategies,
//! added-token manipulation) are available by calling the re-exported
//! [`tokenizers`] API directly on the returned [`Tokenizer`].

use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

pub use tokenizers::Tokenizer;

/// Load a tokenizer from a HuggingFace `tokenizer.json` file.
///
/// This accepts any format that `tokenizers::Tokenizer::from_file`
/// supports — which is the full HF tokenizer format including BPE,
/// WordPiece, Unigram, pre/post processors, and added tokens.
pub fn load_tokenizer(path: impl AsRef<Path>) -> FerrotorchResult<Tokenizer> {
    let path = path.as_ref();
    Tokenizer::from_file(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to load tokenizer {}: {e}", path.display()),
    })
}

/// Encode a single text into its token ids.
///
/// `add_special_tokens` controls whether BOS / EOS and other
/// template-defined special tokens are inserted (Llama 3 prepends
/// `<|begin_of_text|>` / `128000` when true).
pub fn encode(
    tokenizer: &Tokenizer,
    text: &str,
    add_special_tokens: bool,
) -> FerrotorchResult<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, add_special_tokens)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("tokenizer encode failed: {e}"),
        })?;
    Ok(encoding.get_ids().to_vec())
}

/// Encode a batch of texts in parallel.
pub fn encode_batch(
    tokenizer: &Tokenizer,
    texts: &[&str],
    add_special_tokens: bool,
) -> FerrotorchResult<Vec<Vec<u32>>> {
    let owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
    let encodings = tokenizer
        .encode_batch(owned, add_special_tokens)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("tokenizer encode_batch failed: {e}"),
        })?;
    Ok(encodings.into_iter().map(|e| e.get_ids().to_vec()).collect())
}

/// Decode a sequence of token ids back to text.
///
/// `skip_special_tokens` drops BOS / EOS / pad tokens from the output.
pub fn decode(
    tokenizer: &Tokenizer,
    ids: &[u32],
    skip_special_tokens: bool,
) -> FerrotorchResult<String> {
    tokenizer
        .decode(ids, skip_special_tokens)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("tokenizer decode failed: {e}"),
        })
}

/// Vocabulary size the tokenizer was trained with, including any
/// special / added tokens (Llama 3: 128_256).
pub fn vocab_size(tokenizer: &Tokenizer, with_added_tokens: bool) -> usize {
    tokenizer.get_vocab_size(with_added_tokens)
}

/// Resolve a token string to its numeric id, if present in the vocab
/// (including added/special tokens).
pub fn token_to_id(tokenizer: &Tokenizer, token: &str) -> Option<u32> {
    tokenizer.token_to_id(token)
}

/// Resolve a token id to its string form.
pub fn id_to_token(tokenizer: &Tokenizer, id: u32) -> Option<String> {
    tokenizer.id_to_token(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolve the HF cache directory for a gated model.
    fn hf_cache_snapshot(repo_slug: &str) -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from)?;
        let base = home
            .join(".cache/huggingface/hub")
            .join(format!("models--{}", repo_slug.replace('/', "--")))
            .join("snapshots");
        std::fs::read_dir(&base).ok()?.next()?.ok().map(|e| e.path())
    }

    #[test]
    fn loader_rejects_missing_file() {
        let r = load_tokenizer("/nonexistent/tokenizer.json");
        assert!(r.is_err());
    }

    #[test]
    fn loader_rejects_malformed_json() {
        let tmp = std::env::temp_dir().join("ferrotorch_tok_malformed.json");
        std::fs::write(&tmp, "{ not valid").unwrap();
        let r = load_tokenizer(&tmp);
        assert!(r.is_err());
        let _ = std::fs::remove_file(&tmp);
    }

    /// End-to-end: load the real Llama 3 tokenizer.json from the HF
    /// cache and verify the basic surface works.
    /// Ignored by default so CI without the gated model skips it.
    #[test]
    #[ignore = "requires Meta-Llama-3-8B tokenizer.json in the HF cache"]
    fn llama3_tokenizer_loads_and_round_trips() {
        let snapshot = hf_cache_snapshot("meta-llama/Meta-Llama-3-8B")
            .expect("Meta-Llama-3-8B snapshot missing from HF cache");
        let tok_path = snapshot.join("tokenizer.json");
        let tok = load_tokenizer(&tok_path).unwrap();

        // Llama 3 vocab.
        assert_eq!(vocab_size(&tok, true), 128_256);

        // Special tokens the Llama 3 chat template uses.
        assert_eq!(token_to_id(&tok, "<|begin_of_text|>"), Some(128_000));
        assert_eq!(token_to_id(&tok, "<|end_of_text|>"), Some(128_001));

        // Encode with special tokens — BOS should prepend 128000.
        let ids = encode(&tok, "Hello, world!", true).unwrap();
        assert!(!ids.is_empty());
        assert_eq!(ids[0], 128_000, "BOS not prepended: {ids:?}");

        // Without add_special_tokens, BOS is not prepended.
        let ids_bare = encode(&tok, "Hello, world!", false).unwrap();
        assert_ne!(ids_bare[0], 128_000);

        // Round-trip via decode.
        let text = decode(&tok, &ids_bare, false).unwrap();
        assert!(
            text.contains("Hello") && text.contains("world"),
            "decoded text unexpected: {text:?}"
        );

        // encode_batch returns one vec per input.
        let batch = encode_batch(&tok, &["hi", "bye"], false).unwrap();
        assert_eq!(batch.len(), 2);
        assert!(!batch[0].is_empty());
        assert!(!batch[1].is_empty());
    }
}
