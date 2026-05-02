# Audit: `ferrotorch-serialize` vs `torch.save/load` + safetensors + ONNX + GGUF

Covers: native checkpoints, safetensors, PyTorch `.pt`/`.pth` import (pickle
parsing), PyTorch export, ONNX export, GGUF (llama.cpp interop).

## Modules

| Module | Capability |
|---|---|
| `state_dict` | Native ferrotorch state-dict save/load (JSON header + binary body) |
| `checkpoint` | Training checkpoint with optimizer + scheduler state, `AsyncCheckpointer` for nonblocking saves, `TrainingCheckpoint` |
| `safetensors_io` | safetensors format read/write |
| `gguf` | GGUF format (llama.cpp): parse, load, dequantize quantized tensors, load state dict |
| `pytorch_import` | Parse PyTorch `.pt`/`.pth` pickle files (`PickleValue`, `load_pytorch_state_dict`, `parse_pickle`) |
| `pytorch_export` | Save in PyTorch-compatible format (`save_pytorch`, `validate_checkpoint`) |
| `onnx_export` | Export to ONNX (`OnnxExportConfig`, `export_onnx`, `export_from_program`, `export_ir_graph_to_onnx`, `ir_graph_to_onnx`) |

## Coverage vs torch / ecosystem

| Format | ferrotorch | torch / ecosystem |
|---|---|---|
| Native ferrotorch state_dict | ✅ | (n/a) |
| Training checkpoint | ✅ + async | torch saves dict manually |
| safetensors | ✅ | safetensors crate (HF) |
| PyTorch `.pt` load | ✅ via `pytorch_import` | `torch.load` |
| PyTorch `.pt` save | ✅ via `pytorch_export` | `torch.save` |
| ONNX export | ✅ | `torch.onnx.export` |
| GGUF (llama.cpp) | ✅ + dequantize | (no torch equivalent) |
| pickle (torch.load with `weights_only=False`) | partial — only state-dict-shaped pickle | full pickle deserialization |
| TorchScript `.pt` (scripted) | ❌ — only state-dict pickle | `torch.jit.save/load` |
| `.pt2` (torch.export) | ❌ | `torch.export.save/load` |
| HF transformers `pytorch_model.bin` | ✅ via pickle | ✅ |
| HF transformers `model.safetensors` (sharded) | depends on `ferrotorch-hub` for sharded download | ✅ |
| Quantized formats (BitsAndBytes, GPTQ, AWQ) | partial via GGUF | partial in HF |

## Strengths
- **More format breadth than torch core** — ferrotorch handles GGUF (no torch
  equivalent), ONNX export (torch needs `torch.onnx`), and PyTorch interop
  in one crate.
- **Async checkpointing** (`AsyncCheckpointer`) — nice for large models.
- **Pickle parsing** — pure Rust, no Python dependency.
- **GGUF + dequantize** — important for llama.cpp interop on the model
  loading side.

## Gaps

1. **TorchScript `.pt` files** — only state-dict shape is loaded. Loading a
   serialized `ScriptModule` is much harder (requires the Torch IR
   spec). Probably not worth it; document as "weights-only" interop.
2. **`.pt2` files** (torch.export output) — would let ferrotorch run
   torch-exported graphs. Requires alignment with `ferrotorch-jit::export`
   format. Worth a follow-up issue if interop with `torch.export.save` is
   desired.
3. **Sharded loading** — multi-file safetensors (HF transformers
   convention: `model-00001-of-00006.safetensors` + `model.safetensors.index.json`)
   — needs to be confirmed in `ferrotorch-hub` (#509 covers download;
   loader needs to walk the index file). Currently `safetensors_io` likely
   single-file only.
4. **Memory-mapped loading** — torch supports `mmap=True` on `torch.load`
   to avoid copying weights into RAM. Worth checking if `safetensors_io`
   does this; if not, big-model loading is slow / RAM-bloated.
5. **Sparse-tensor serialization** — torch supports sparse in
   `state_dict`; ferrotorch's `state_dict` likely doesn't (haven't looked).
6. **Quantized tensor serialization** — `QuantizedTensor` should
   round-trip via state_dict.

## Recommendations

1. **Document supported torch formats** clearly: weights-only `.pt`/`.pth`,
   safetensors, GGUF, ONNX export, ferrotorch-native — what works and what
   doesn't.
2. **Add sharded safetensors loading** if not already present (depends on
   `ferrotorch-hub`).
3. **Add memory-mapped loading** for safetensors / GGUF / pytorch — large
   LLM weights shouldn't double-buffer through RAM.
4. **Confirm sparse + quantized state-dict round-trip** — write tests if
   missing.
5. **Decide on `.pt2` interop** as a separate work item (depends on
   alignment with `ferrotorch-jit::export` IR / ATen core opset).

## Status

**Coverage is broad: 5+ formats supported, including ones torch core
splits across multiple modules (`torch.save`, `torch.onnx`).** The crate
description ("State dict and checkpoint serialization") **understates what
it actually does** — should be updated to mention safetensors + GGUF +
ONNX + pytorch interop.

**Do not split.** This crate is a natural one-stop-shop for serialization.
