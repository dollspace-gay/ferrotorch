//! Constrained-decoding grammar processors.
//!
//! Submodules:
//!
//! - [`schema`] — internal `Schema` enum and JSON-Schema parser (subset).
//! - [`state`]  — `JsonGrammar` state machine that tracks where we are in
//!   the partially-emitted JSON value.
//! - [`json_schema`] — public `JsonSchemaProcessor` that wraps a tokenizer
//!   vocabulary and produces per-step token-allow masks for use with
//!   `ferrotorch_cubecl::apply_token_mask_to_gpu`.

pub mod json_schema;
pub mod schema;
pub mod state;

#[cfg(feature = "cuda")]
pub mod gpu_dispatch;

pub use json_schema::{GrammarError, JsonSchemaProcessor, TokenMask};
pub use schema::Schema;
pub use state::{BooleanEmissionStage, JsonGrammar};

#[cfg(feature = "cuda")]
pub use gpu_dispatch::{PackedVocab, compute_mask_gpu};
