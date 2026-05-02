# Audit: `ferrotorch-profiler` vs `torch.profiler`

## ferrotorch-profiler modules

| Module | Role |
|---|---|
| `event` | `ProfileEvent`, `MemoryCategory` |
| `profiler` | `Profiler`, `ProfileConfig`, `with_profiler` |
| `report` | `ProfileReport`, `OpSummary`, table + Chrome trace export |
| `schedule` | `ProfileSchedule`, `SchedulePhase` (wait/warmup/active phases) |
| `flops` | FLOPs estimation |
| `cuda_timing` (feature `cuda`) | `CudaKernelScope`, `PendingCudaScope` (CUDA event-based timing) |

## Coverage vs `torch.profiler`

torch.profiler features:
- `torch.profiler.profile(activities, schedule, on_trace_ready, ...)` —
  context manager
- Activities: `CPU`, `CUDA`, `XPU`, `Privateuse1`, `MTIA`, `IPU`
- Schedule: `wait`, `warmup`, `active`, `repeat` phases
- Records: kernel timings, memory events, shapes, stack traces, FLOPs
- Exports:
  - `tensorboard_trace_handler(dir)` — TB profiler plugin
  - Chrome trace (`profile.export_chrome_trace(path)`)
  - Stack trace export
  - `key_averages()` — aggregated table
- ITT (Intel Trace Tools) integration for VTune (`torch.profiler.itt`)
- Memory profiling (`_memory_profiler.py`)
- Pattern matching for inefficiency detection (`_pattern_matcher.py`)
  — flags slow patterns like CPU↔GPU bouncing
- Python tracer (sys.setprofile-based, captures Python frames)

| Capability | ferrotorch | torch | Status |
|---|---|---|---|
| Op timing | ✅ | ✅ | |
| Memory events | ✅ via `MemoryCategory` | ✅ | |
| Input shapes | ✅ | ✅ | |
| Schedule (wait/warmup/active) | ✅ via `ProfileSchedule`/`SchedulePhase` | ✅ | |
| Chrome trace export | ✅ | ✅ | |
| Table report | ✅ via `ProfileReport.table(n)` | ✅ via `key_averages` | |
| FLOPs estimation | ✅ via `flops` module | ✅ | |
| CUDA kernel timing | ✅ via `cuda_timing` (feature flag) | ✅ via CUPTI | |
| TensorBoard trace handler | ❌ | ✅ | gap |
| Stack trace recording | unclear (likely none) | ✅ (per-event Python stack) | likely gap |
| Pattern-matched warnings | ❌ | ✅ (`_pattern_matcher`) | gap |
| ITT / VTune integration | ❌ | ✅ | gap |
| NVTX integration | ❌ | partial via `torch.cuda.nvtx` | gap |
| XPU / MPS profiling | ❌ | ✅ | gap |
| Memory pattern allocator viz | ❌ | ✅ (`_memory_viz`) | gap |

## Strengths
- Clean public API: `with_profiler(config, |p| { ... })` — idiomatic Rust
  scope-bound profiler.
- Schedule phases match torch's wait/warmup/active idiom exactly.
- FLOPs estimator is a separate module — easy to extend with op-specific
  formulas.
- CUDA timing is feature-gated and uses CUDA events (correct).

## Gaps

1. **TensorBoard trace handler** — would let users open profile in TB
   profiler plugin, the standard torch viz tool.
2. **Stack-trace capture** — without rust backtrace per event, can't
   answer "which Rust function called this op?"
3. **Pattern-matched warnings** — flag CPU↔GPU shuffling, sync GPU stalls,
   etc.
4. **NVTX integration** — emit ranges so Nsight can collapse high-level
   ops alongside the low-level CUPTI traces.
5. **Multi-device coverage** — only CUDA timing today; XPU / MPS / ROCm
   need their own timing scopes.

## Recommendations

1. **Add a TB trace handler** — either via Chrome-trace JSON in TB's
   format, or a real TB protobuf writer.
2. **Capture stack traces** per event — Rust `backtrace` crate.
3. **Add per-backend timing scopes** — `XpuKernelScope`,
   `RocmKernelScope`, etc., parallel to `CudaKernelScope`.
4. **Add NVTX wrapper** in `cuda_timing` for Nsight integration.
5. **Add pattern matcher** — heuristic warnings for inefficient patterns
   (high host↔device transfer time, kernel launch overhead, sync points).

## Status

ferrotorch-profiler covers the **core profiling loop** well: timing,
memory events, shapes, schedule, FLOPs, Chrome trace export, CUDA timing.
Gaps are integration points (TB, NVTX, ITT, stack traces) and multi-
backend timing.

**Do not split.** Crate matches `torch.profiler` 1:1.
