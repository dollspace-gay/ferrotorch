//! GPU dispatch for [`super::JsonSchemaProcessor::compute_mask`].
//!
//! Bridges the CPU-side `JsonGrammar` state to the
//! [`ferrotorch_cubecl::compute_token_mask_dfa_to_gpu`] kernel:
//!
//! 1. Inspect the current grammar via [`super::state::JsonGrammar::boolean_emission_stage`].
//! 2. If the state is DFA-compilable (stage 2 supports `Schema::Boolean`
//!    only), build the per-token DFA tables on the host.
//! 3. Pack the processor's vocab as `(offsets, chars)` u32 buffers.
//! 4. Dispatch the CubeCL kernel; read back the allow mask.
//! 5. If the state isn't compilable, return `None` so callers fall through
//!    to the existing CPU loop in `JsonSchemaProcessor::compute_mask`.
//!
//! Compiled only when the `cuda` feature is enabled.

use cubecl::prelude::{ComputeClient, Runtime};
use ferrotorch_cubecl::{DfaMaskInputs, compute_token_mask_dfa_to_gpu};

use super::state::BooleanEmissionStage;
use super::json_schema::{JsonSchemaProcessor, TokenMask};

/// One DFA built from a grammar state. All buffers are owned `Vec<u32>`s
/// because the kernel launcher takes them by reference, and they need to
/// outlive the launcher call.
struct CompiledDfa {
    transitions: Vec<u32>,
    char_classes: Vec<u32>,
    num_classes: u32,
    start_state: u32,
    reject_state: u32,
}

/// Compile a [`BooleanEmissionStage`] into a finite DFA.
///
/// State numbering convention:
///
/// - `0` is the start state.
/// - `1..=N` are intermediate states corresponding to characters
///   already accepted along the literal.
/// - `REJECT = num_states - 1`. Defined explicitly so the kernel's
///   `state == reject_state` short-circuit fires.
///
/// For [`BooleanEmissionStage::Start`] the DFA branches: from `0`,
/// `'t'` → state 1 (head of "rue"), `'f'` → state 5 (head of "alse"),
/// any other class → REJECT. Then both branches walk linearly to their
/// respective accept positions, after which any further char rejects
/// (the grammar would be `done`, but the kernel still needs to handle
/// tokens that try to emit past the literal's end).
///
/// For [`BooleanEmissionStage::PartialTrue { remaining }`] the DFA is
/// just the linear walk over `remaining`'s chars, plus REJECT. Same for
/// `PartialFalse`.
fn compile_dfa_for_boolean(stage: &BooleanEmissionStage) -> CompiledDfa {
    match stage {
        BooleanEmissionStage::Start => compile_boolean_full(),
        BooleanEmissionStage::PartialTrue { remaining } => compile_linear_literal(remaining),
        BooleanEmissionStage::PartialFalse { remaining } => compile_linear_literal(remaining),
    }
}

/// DFA for the full `Schema::Boolean` at `Phase::Start`: accept any
/// prefix of `"true"` or `"false"`, reject everything else.
fn compile_boolean_full() -> CompiledDfa {
    // Char classes: t r u e f a l s OTHER  →  9 classes.
    let class_t = 0u32;
    let class_r = 1u32;
    let class_u = 2u32;
    let class_e = 3u32;
    let class_f = 4u32;
    let class_a = 5u32;
    let class_l = 6u32;
    let class_s = 7u32;
    let class_other = 8u32;
    let num_classes = 9u32;

    let mut char_classes = vec![class_other; 128];
    char_classes[b't' as usize] = class_t;
    char_classes[b'r' as usize] = class_r;
    char_classes[b'u' as usize] = class_u;
    char_classes[b'e' as usize] = class_e;
    char_classes[b'f' as usize] = class_f;
    char_classes[b'a' as usize] = class_a;
    char_classes[b'l' as usize] = class_l;
    char_classes[b's' as usize] = class_s;

    // States:
    //  0 = start (need 't' or 'f')
    //  1 = saw "t" (need 'r')
    //  2 = saw "tr" (need 'u')
    //  3 = saw "tru" (need 'e')
    //  4 = saw "true" (any further char rejects)
    //  5 = saw "f" (need 'a')
    //  6 = saw "fa" (need 'l')
    //  7 = saw "fal" (need 's')
    //  8 = saw "fals" (need 'e')
    //  9 = saw "false" (any further char rejects)
    // 10 = REJECT
    let num_states = 11usize;
    let reject = 10u32;
    let mut transitions = vec![reject; num_states * num_classes as usize];

    let nc = num_classes as usize;
    // Index helper makes the row * nc pattern explicit and avoids the
    // clippy::erasing_op false-positive on `0 * nc`.
    let row = |state: usize, class: u32| state * nc + class as usize;
    transitions[row(0, class_t)] = 1;
    transitions[row(0, class_f)] = 5;
    transitions[row(1, class_r)] = 2;
    transitions[row(2, class_u)] = 3;
    transitions[row(3, class_e)] = 4;
    // state 4 (= "true" complete): every class falls through to REJECT (already set).
    transitions[row(5, class_a)] = 6;
    transitions[row(6, class_l)] = 7;
    transitions[row(7, class_s)] = 8;
    transitions[row(8, class_e)] = 9;
    // state 9 (= "false" complete): every class → REJECT (already set).
    // state 10 (REJECT): every class → REJECT (already set).

    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state: 0,
        reject_state: reject,
    }
}

/// DFA accepting any prefix of `literal`. Used for the
/// `PartialTrue` / `PartialFalse` stages: we've already emitted the head
/// of "true" or "false", and `literal` is the remaining suffix to match.
fn compile_linear_literal(literal: &str) -> CompiledDfa {
    // Build a per-char class table over literal's distinct chars.
    // 8 distinct ASCII letters at most for boolean ("true" / "false"); this
    // generalises cleanly to any short literal.
    let mut classes_for_char = std::collections::BTreeMap::<char, u32>::new();
    let mut next_class: u32 = 0;
    for c in literal.chars() {
        classes_for_char.entry(c).or_insert_with(|| {
            let id = next_class;
            next_class += 1;
            id
        });
    }
    let class_other = next_class;
    let num_classes = next_class + 1;

    let mut char_classes = vec![class_other; 128];
    for (&c, &id) in &classes_for_char {
        if (c as u32) < 128 {
            char_classes[c as usize] = id;
        }
    }

    // States: 0 .. literal.len() are intermediate (state `n` is the accept
    // state — we land on it when the literal completes), literal.len() + 1
    // is REJECT. Every char emitted *past* the accept state lands on
    // REJECT, matching the CPU grammar's "already complete" rejection.
    let n = literal.chars().count();
    let reject = (n + 1) as u32;
    let num_states = n + 2;
    let nc = num_classes as usize;
    let mut transitions = vec![reject; num_states * nc];

    for (i, c) in literal.chars().enumerate() {
        let class = *classes_for_char.get(&c).expect("class table built above");
        transitions[i * nc + class as usize] = (i + 1) as u32;
    }
    // state `n` (accept): every class → REJECT (already set).
    // state `reject`: every class → REJECT (already set).

    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state: 0,
        reject_state: reject,
    }
}

/// Pre-packed vocab buffers ready for upload. Computed once per
/// (processor, vocab) and cached on the call site since vocabularies are
/// large (Llama-3 = 128k entries).
pub struct PackedVocab {
    pub offsets: Vec<u32>,
    pub chars: Vec<u32>,
    pub max_token_len: u32,
}

impl PackedVocab {
    /// Pack a string vocabulary into `(offsets, chars, max_token_len)`.
    ///
    /// `offsets[i] .. offsets[i+1]` is the slice of `chars` holding
    /// token `i`'s codepoints (one `u32` per Unicode scalar).
    /// `max_token_len` is the longest token's char count, used as the
    /// kernel's bounded-loop cap.
    pub fn pack(vocab: &[String]) -> Self {
        let mut offsets = Vec::with_capacity(vocab.len() + 1);
        let mut chars = Vec::new();
        let mut max_token_len: usize = 0;
        offsets.push(0u32);
        for tok in vocab {
            let mut tok_len = 0usize;
            for c in tok.chars() {
                chars.push(c as u32);
                tok_len += 1;
            }
            offsets.push(chars.len() as u32);
            if tok_len > max_token_len {
                max_token_len = tok_len;
            }
        }
        Self {
            offsets,
            chars,
            max_token_len: max_token_len as u32,
        }
    }
}

/// Try to compute the token-allow mask on GPU. Returns `None` if the
/// current grammar state isn't DFA-compilable (caller should fall
/// through to the CPU `compute_mask` path).
///
/// Stage 2 supports `Schema::Boolean` only; future stages extend the
/// match in [`compile_dfa_for_boolean`]'s caller to cover Null, Number,
/// StringEnum, etc.
pub fn compute_mask_gpu<R: Runtime>(
    processor: &JsonSchemaProcessor,
    client: &ComputeClient<R>,
    packed: &PackedVocab,
) -> Option<TokenMask> {
    let stage = processor.grammar().boolean_emission_stage()?;
    let dfa = compile_dfa_for_boolean(&stage);

    let inputs = DfaMaskInputs {
        transitions: &dfa.transitions,
        char_classes: &dfa.char_classes,
        vocab_offsets: &packed.offsets,
        vocab_chars: &packed.chars,
        num_classes: dfa.num_classes,
        start_state: dfa.start_state,
        reject_state: dfa.reject_state,
        max_token_len: packed.max_token_len,
    };
    let (handle, n) = compute_token_mask_dfa_to_gpu::<R>(client, &inputs);
    let bytes = client.read_one(handle).ok()?;
    if bytes.len() != n * std::mem::size_of::<u32>() {
        return None;
    }
    let mut allow = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        allow.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(TokenMask { allow })
}

// ---------------------------------------------------------------------------
// CUDA runtime tests — real GPU dispatch with byte-equality vs CPU.
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::grammar::schema::Schema;
    use cubecl_cuda::{CudaDevice, CudaRuntime};
    use serde_json::json;

    fn cuda_client() -> ComputeClient<CudaRuntime> {
        let device = CudaDevice { index: 0 };
        CudaRuntime::client(&device)
    }

    fn ascii_char_vocab() -> Vec<String> {
        (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect()
    }

    /// Parity: a fresh Boolean processor at Phase::Start. GPU mask must
    /// be byte-equal to the existing CPU `compute_mask` over the same
    /// vocab.
    #[test]
    fn boolean_gpu_mask_matches_cpu_at_start() {
        let vocab = ascii_char_vocab();
        let processor =
            JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
        let cpu_mask = processor.compute_mask();

        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask =
            compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed).expect(
                "Schema::Boolean at Phase::Start must be DFA-compilable",
            );

        assert_eq!(
            cpu_mask.allow, gpu_mask.allow,
            "GPU mask must equal CPU mask byte-for-byte for Boolean@Start",
        );
        // Sanity: 't' and 'f' should be the only ASCII single-char tokens
        // accepted at the start of a boolean.
        let allowed_chars: Vec<char> = (0..gpu_mask.allow.len())
            .filter(|&i| gpu_mask.allow[i] != 0)
            .map(|i| vocab[i].chars().next().unwrap())
            .collect();
        assert!(allowed_chars.contains(&'t'));
        assert!(allowed_chars.contains(&'f'));
        assert!(!allowed_chars.contains(&'a'));
    }

    /// Parity after the first character has already been emitted.
    /// Stepping 't' moves the grammar into PartialTrue { remaining: "rue" }.
    #[test]
    fn boolean_gpu_mask_matches_cpu_after_t() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
        let t_id = vocab.iter().position(|s| s == "t").unwrap() as u32;
        processor.step_token(t_id).unwrap();

        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("PartialTrue must be DFA-compilable");

        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // After 't', only 'r' should be allowed among single-char tokens.
        let allowed_chars: Vec<char> = (0..gpu_mask.allow.len())
            .filter(|&i| gpu_mask.allow[i] != 0)
            .map(|i| vocab[i].chars().next().unwrap())
            .collect();
        assert!(allowed_chars.contains(&'r'));
        assert!(!allowed_chars.contains(&'t'));
        assert!(!allowed_chars.contains(&'u'));
    }

    /// Parity after stepping 'f': PartialFalse { remaining: "alse" }.
    #[test]
    fn boolean_gpu_mask_matches_cpu_after_f() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
        let f_id = vocab.iter().position(|s| s == "f").unwrap() as u32;
        processor.step_token(f_id).unwrap();

        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("PartialFalse must be DFA-compilable");

        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        let allowed_chars: Vec<char> = (0..gpu_mask.allow.len())
            .filter(|&i| gpu_mask.allow[i] != 0)
            .map(|i| vocab[i].chars().next().unwrap())
            .collect();
        assert!(allowed_chars.contains(&'a'));
        assert!(!allowed_chars.contains(&'f'));
    }

    /// Stage-2 unsupported schemas return `None` so the caller can fall
    /// through to CPU compute_mask. Verifies the gate is in place — no
    /// silent mis-dispatch to a half-built DFA.
    #[test]
    fn unsupported_schema_returns_none() {
        let vocab = ascii_char_vocab();
        let processor = JsonSchemaProcessor::new(
            &json!({
                "type": "object",
                "properties": {"v": {"type": "boolean"}},
                "required": ["v"]
            }),
            vocab.clone(),
        )
        .unwrap();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let res = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed);
        assert!(
            res.is_none(),
            "stage-2 must return None for non-Boolean schemas; got Some",
        );

        // Sanity: directly verify the underlying API too.
        assert!(matches!(
            crate::grammar::schema::Schema::from_json_schema(&json!({
                "type": "object",
                "properties": {"v": {"type": "boolean"}},
                "required": ["v"]
            })),
            Ok(Schema::Object { .. })
        ));
    }
}
