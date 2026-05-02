# Audit: `ferrotorch-hub` + `ferrotorch-tokenize` vs `torch.hub` + HF tokenizers

## ferrotorch-hub

Pretrained model registry, download, caching.

### Modules

| Module | Role |
|---|---|
| `registry` | `ModelInfo`, `WeightsFormat`, `list_models`, `get_model_info` — local registry of known models |
| `cache` | `HubCache`, `default_cache_dir` — on-disk cache directory layout |
| `download` | `download_weights`, `load_pretrained` — fetch weights, hydrate state dict |
| `discovery` (feature `http`) | `HfModelInfo`, `HfModelSummary`, `HfRepoFile`, `SearchQuery`, `get_model`, `search_models` — HuggingFace Hub API client |
| `hf_config` | `HfTransformerConfig` — parse `config.json` from HF model |

### Coverage vs torch.hub

torch.hub is a thin model registry — it knows about a handful of github
repos and downloads/loads `.pt` files. The real model hub for the
ecosystem is **HuggingFace Hub**, accessed via `huggingface_hub` Python
package.

| Feature | ferrotorch | torch.hub | huggingface_hub |
|---|---|---|---|
| List known models | `list_models()` | `torch.hub.list(github_repo)` | n/a |
| Get model info | `get_model_info(name)` | n/a (per-repo) | `HfApi().model_info(repo_id)` |
| Download weights | `download_weights(name)` | `torch.hub.load(github, model)` | `hf_hub_download` |
| Load pretrained | `load_pretrained::<f32>(name)` | `torch.hub.load(...)` | `from_pretrained()` (transformers) |
| Cache dir | `HubCache` / `default_cache_dir()` | `~/.cache/torch/hub` | `~/.cache/huggingface/hub` |
| HF Hub search | `search_models(SearchQuery)` | n/a | `HfApi().list_models()` |
| HF Hub model info | `get_model(repo_id) -> HfModelInfo` | n/a | `HfApi().model_info()` |
| HF Hub repo file listing | `HfRepoFile` | n/a | `HfApi().list_repo_files()` |
| HF auth token | tracked in #509 | n/a | env var / config |
| Sharded download | tracked in #509 | n/a | ✅ |
| Resume on failure | unclear | n/a | ✅ |
| Etag / hash verification | unclear | partial | ✅ |
| Local model layout (`refs/`, `snapshots/`, `blobs/`) | partial via HubCache | flat | ✅ structured |
| `from_pretrained(repo_id)` user-side helper | ❌ (manual: discover→download→state-dict→load) | n/a | ✅ |

**Gaps:**
- **`from_pretrained`-style one-liner** that takes `repo_id`, downloads
  config + weights + tokenizer, instantiates model, loads weights. This is
  what users coming from HF transformers expect.
- **Sharded downloads** — `pytorch_model-00001-of-00006.bin` indexed by
  `pytorch_model.bin.index.json` (#509 tracks).
- **Auth token** for gated models (#509).
- **Resume on failure / partial download recovery**.
- **Etag/sha-verification** of cached files.

### Strengths
- Has both a **curated registry** (for fast pretrained loading via known
  names) and a **HF Hub client** (for arbitrary repos). torch.hub only has
  the curated path.
- HF model info parsing (`HfTransformerConfig`) is first-class.

## ferrotorch-tokenize

Thin wrapper around the HuggingFace `tokenizers` Rust crate.

### Surface

```
load_tokenizer(path) -> Tokenizer
encode(tokenizer, text, add_special_tokens) -> Vec<u32>
encode_batch(tokenizer, texts, add_special_tokens) -> Vec<Vec<u32>>
decode(tokenizer, ids, skip_special_tokens) -> String
vocab_size(tokenizer, with_added_tokens) -> usize
token_to_id(tokenizer, token) -> Option<u32>
id_to_token(tokenizer, id) -> Option<String>
```

The crate re-exports the underlying `tokenizers::Tokenizer` so users can
call advanced APIs directly.

### Coverage vs torch / HF

torch has **no built-in tokenizer**. `transformers` (HF Python) provides
`AutoTokenizer.from_pretrained(repo_id)` which:
1. Downloads `tokenizer.json` from HF Hub
2. Loads it via the HF tokenizers Rust crate
3. Wraps in a Python class with chat templating, truncation, padding
   strategies, special-token bookkeeping.

ferrotorch-tokenize covers steps 2 and **partially** step 3.

**Gaps:**
- **Chat templating** — applying `chat_template.jinja` to `[{"role":
  "user", "content": "..."}]` lists. This is how Llama 3 / Mistral /
  Qwen-Chat tokenizers expect input. Big gap for chat-model deployment.
- **Truncation / padding strategies** — `truncation=True/False/'longest'`,
  `padding='max_length'/'longest'/False`, `max_length`, `pad_to_multiple_of`.
- **Added-token manipulation** — `add_tokens`, `add_special_tokens` (the
  programmatic API, not just the encode flag).
- **Auto-downloading** — `load_tokenizer_from_hub(repo_id)` would unify
  with `ferrotorch-hub`.

### Strengths
- Uses the same Rust crate as HF transformers. Wire-compatible.
- API is idiomatic Rust (`FerrotorchResult`, `Vec<u32>`).

## Recommendations

### ferrotorch-hub

1. **Add `from_pretrained(repo_id)` umbrella** — config + weights +
   tokenizer in one call. Returns a `(StateDict, HfTransformerConfig,
   Tokenizer)` triple or similar.
2. **Add sharded download** (#509) — index file → individual shard
   download → unified state-dict load.
3. **Add auth token** (#509) — env var `HF_TOKEN`, fall back to
   `~/.cache/huggingface/token`.
4. **Add resume / verification** — partial-file resume, etag check,
   sha256 of downloaded blobs.
5. **Match HF Hub local layout** (`refs/main`, `snapshots/<sha>/`,
   `blobs/`) so cache is interoperable with HF tools.

### ferrotorch-tokenize

1. **Add chat-template rendering** — Jinja2 template engine applied to
   message lists. Critical for chat models.
2. **Add truncation / padding strategies** — `EncodeOptions` builder.
3. **Add `load_tokenizer_from_hub(repo_id)`** — wire to `ferrotorch-hub`.

## Status

- **ferrotorch-hub**: ~60% complete. Has registry + HF Hub client + cache,
  but `from_pretrained`-style ergonomics and sharded download (the things
  Python `transformers` users expect) are gaps. #509 covers the biggest
  ones.
- **ferrotorch-tokenize**: thin and correct. Chat templating is the
  missing feature that blocks chat-model demos.

**Do not split.** Both crates have right-sized scope.

## Related issues
- #509 — HF Hub auth token + sharded download
