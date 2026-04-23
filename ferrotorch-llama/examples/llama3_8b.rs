//! Meta-Llama-3-8B end-to-end smoke example.
//!
//! Run from the repository root, with `meta-llama/Meta-Llama-3-8B`
//! already in the HuggingFace cache (either via `hf download` or via
//! the HF Hub auto-download on first use):
//!
//! ```sh
//! cargo run --release -p ferrotorch-llama --example llama3_8b \
//!     -- "The meaning of life is"
//! ```
//!
//! The example:
//!
//! 1. Resolves the cached Meta-Llama-3-8B snapshot directory.
//! 2. Parses `config.json` into an [`HfTransformerConfig`] and
//!    converts it into a typed [`LlamaConfig`].
//! 3. Loads the `tokenizer.json` via [`ferrotorch_tokenize::load_tokenizer`].
//! 4. Loads all four `model.safetensors` shards into a
//!    `StateDict<bf16>` via
//!    [`ferrotorch_serialize::safetensors_io::load_safetensors_sharded`].
//! 5. Constructs a [`LlamaForCausalLM<bf16>`] and loads the HF state
//!    dict into it.
//! 6. Tokenizes the prompt (default `"The meaning of life is"`),
//!    runs a single prefill pass, picks the argmax token, decodes it
//!    to a string, and prints it.
//!
//! Scope note: the current model implementation is a CPU-only
//! reference. On a 32 GB RAM box the full bf16 load fits, but the
//! forward pass through 32 layers of 4096 hidden × 14336 intermediate
//! is not SIMD-accelerated for bf16 — a single prefill step takes
//! minutes, and autoregressive decode is impractical without the
//! planned `ferrotorch-paged` speculative paging / mixed-precision
//! kernels. This example proves the load / tokenize / prefill path
//! is correct; full generation is the bar for a future pass.

use std::path::PathBuf;
use std::time::Instant;

use ferrotorch_hub::HfTransformerConfig;
use ferrotorch_llama::{LlamaConfig, LlamaForCausalLM};
use ferrotorch_serialize::safetensors_io::load_safetensors_sharded;
use ferrotorch_tokenize::{decode, encode, load_tokenizer};
use half::bf16;

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("$HOME not set")
}

/// Resolve the first snapshot directory for a gated HF model.
fn hf_snapshot(repo_slug: &str) -> PathBuf {
    let base = home()
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", repo_slug.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("HF cache {} missing: {e}", base.display()))
        .next()
        .unwrap_or_else(|| panic!("no snapshot in {}", base.display()))
        .unwrap()
        .path()
}

fn main() {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The meaning of life is".to_string());

    let snapshot = hf_snapshot("meta-llama/Meta-Llama-3-8B");
    println!("[llama3_8b] snapshot = {}", snapshot.display());

    // -- Step 1: parse config --------------------------------------------
    let cfg_path = snapshot.join("config.json");
    let hf_cfg = HfTransformerConfig::from_file(&cfg_path)
        .expect("failed to parse config.json");
    let cfg = LlamaConfig::from_hf(&hf_cfg).expect("invalid LlamaConfig");
    println!(
        "[llama3_8b] config: hidden={} layers={} heads={} kv={} head_dim={} vocab={} rope_theta={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        cfg.vocab_size,
        cfg.rope_theta,
    );

    // -- Step 2: load tokenizer ------------------------------------------
    let tok_path = snapshot.join("tokenizer.json");
    let tok = load_tokenizer(&tok_path).expect("failed to load tokenizer.json");
    let prompt_ids = encode(&tok, &prompt, /* add_special_tokens = */ true)
        .expect("failed to encode prompt");
    println!(
        "[llama3_8b] prompt tokens ({}): {:?}",
        prompt_ids.len(),
        prompt_ids
    );

    // -- Step 3: load sharded weights as bf16 ----------------------------
    let idx_path = snapshot.join("model.safetensors.index.json");
    println!("[llama3_8b] loading weights (this takes a while)…");
    let t_load = Instant::now();
    let state = load_safetensors_sharded::<bf16>(&idx_path)
        .expect("failed to load sharded safetensors");
    println!(
        "[llama3_8b] loaded {} tensors in {:.1}s",
        state.len(),
        t_load.elapsed().as_secs_f64()
    );

    // -- Step 4: construct the model and load state ----------------------
    let mut model = LlamaForCausalLM::<bf16>::new(cfg).expect("failed to build LlamaForCausalLM");
    model
        .load_hf_state_dict(&state, /* strict = */ true)
        .expect("failed to load HF state dict");
    drop(state); // free the staging StateDict — model owns its copies now.

    // -- Step 5: prefill and greedy-sample one token ---------------------
    println!("[llama3_8b] running prefill on {} tokens…", prompt_ids.len());
    let t_prefill = Instant::now();
    let logits = model
        .forward_from_ids(&prompt_ids)
        .expect("forward_from_ids failed");
    let t_prefill_s = t_prefill.elapsed().as_secs_f64();

    // logits shape: [1, seq_len, vocab_size]. Argmax over the last
    // position's vocab axis to pick the next token.
    let shape = logits.shape();
    assert_eq!(shape[0], 1, "batch should be 1");
    let seq_len = shape[1];
    let vocab = shape[2];
    let data = logits.data().expect("logits.data()");
    let offset = (seq_len - 1) * vocab;
    let mut best_id = 0u32;
    let mut best_val = bf16::from_f32(f32::NEG_INFINITY);
    for (i, &v) in data[offset..offset + vocab].iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_id = i as u32;
        }
    }
    let decoded = decode(&tok, &[best_id], /* skip_special_tokens = */ false)
        .expect("failed to decode next token");

    println!(
        "[llama3_8b] prefill done in {:.2}s ({:.3} tokens/sec)",
        t_prefill_s,
        prompt_ids.len() as f64 / t_prefill_s
    );
    println!(
        "[llama3_8b] next token: id={best_id} text={decoded:?}"
    );
    println!("[llama3_8b] full continuation: {}{}", prompt, decoded);
}
