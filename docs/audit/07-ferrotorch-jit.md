# Audit: `ferrotorch-jit` vs `torch.jit` + `torch.fx` + `torch.compiler` (Dynamo + Inductor)

The pytorch side spans **four** distinct subsystems that have evolved over time:

| pytorch subsystem | Role | Stability |
|---|---|---|
| `torch.jit` (TorchScript) | Static graph capture via tracing or scripting → IR → C++ runtime | Frozen, mostly maintenance |
| `torch.fx` | Symbolic graph IR with Python proxies; programmable graph transforms | Stable |
| `torch._dynamo` | Bytecode-level frame interception → graph capture (incl. graph breaks) | Active |
| `torch._inductor` | Backend codegen (Triton GPU, C++/OpenMP CPU) | Active |
| `torch.compiler` | Public umbrella API (`torch.compile(model)`) tying it all together | Stable |
| `torch.export` | Full-graph export (no graph breaks) → ATen-level `ExportedProgram` | Active |
| AOTAutograd | Joint forward+backward graph tracing | Inside `torch._functorch` |

ferrotorch-jit covers **a subset of all four**, in a single crate.

## ferrotorch-jit modules

| Module | Role | Maps to |
|---|---|---|
| `trace` | Tensor-level eager → IR tracing | `torch.jit.trace`, `torch.fx.symbolic_trace` |
| `graph` | IR (IrGraph, IrNode, IrValue, IrOpKind) | `torch.fx.Graph` + `torch.fx.Node` |
| `interpreter` | Run an IR graph | `torch.fx.Interpreter` |
| `module` | `TracedModule`, `AotCompiledModule`, `compile`, `compile_with_config` | `torch.jit.ScriptModule`, `torch.compile()` |
| `aot_autograd` | Joint forward+backward compilation, `AotGraphPair`, `decompose_forward_backward` | `torch._functorch.aot_autograd` |
| `optimize` | Graph optimization passes | `torch.fx.passes` |
| `fusion` | Fused op patterns (FusedChain, FusedOp, ReductionKind, generate_reduction_c, generate_reduction_ptx) | Inductor fusion |
| `dag_fusion` | DAG-based fusion grouping (FusionGroup, FusionGroupKind) | Inductor scheduler |
| `codegen` | Backend abstraction (Codegen, CompiledGraph, InductorBackend, InductorTarget, NativeBackend, InterpreterBackend) | `torch._inductor` |
| `codegen_cpu` | CpuCodegen (C++/OpenMP-style) | Inductor C++ backend |
| `codegen_gpu` | GpuCodegen (CUDA/PTX) | Inductor Triton backend |
| `codegen_jit` | JIT C kernel compile (`compile_c_kernel`, `JitCompiledKernel`) | Inductor jit caching |
| `codegen_ir` | Lower-level loop IR (Expr, LoopIR, BinOpKind, UnaryOpKind) | Inductor IR / Halide |
| `autotune` | AutotuneCandidate, AutotuneKey, AutotuneResult, Autotuner | `torch._inductor.autotune_process` |
| `memory_plan` | MemoryPlan, plan_memory | Inductor memory planning / `torch._functorch` activation reuse |
| `symbolic` | SymbolicDim, ShapeSignature, Guard, SymbolicTracedModule, compile_symbolic | `torch.fx.experimental.symbolic_shapes`, dynamo guards |
| `graph_break` | GraphSegment, SegmentedModule, TraceResult, trace_with_breaks | `torch._dynamo` graph breaks |
| `export` | ExportedProgram, ExportedProgramMetadata, InputSpec, DimSpec, export, export_with_dynamic_shapes | `torch.export.export`, `torch.export.ExportedProgram` |
| `serialize` | Serialize compiled graphs | `torch.export.save/load`, `.pt2` files |

That's a remarkably broad scope for one crate.

## Coverage by area

### Tracing / graph capture

| ferrotorch | torch | Status |
|---|---|---|
| `trace(f, example_inputs) -> IrGraph` | `torch.jit.trace`, `torch.fx.symbolic_trace` | ✅ |
| `compile(module, config) -> AotCompiledModule` | `torch.compile(model)` | ✅ |
| `trace_with_breaks` → `SegmentedModule` | dynamo graph breaks | ✅ partial |
| `compile_symbolic` (with `SymbolicDim`, `ShapeSignature`, `Guard`) | dynamo guards on dynamic shapes | ✅ |

**Gap:** no scripting (Python source → AST → IR) — but scripting is
torch-specific and not a real Rust analog. ferrotorch traces eager Rust
code; users write Rust and trace it. Reasonable divergence.

### Graph IR

| ferrotorch | torch.fx | Status |
|---|---|---|
| `IrGraph`, `IrNode`, `IrValue`, `IrOpKind`, `IrValueId`, `IrNodeId` | `Graph`, `Node`, `Argument` | ✅ structurally aligned |
| `interpret(graph, inputs)`, `interpret_multi` | `Interpreter`, `Transformer` | ✅ |
| `optimize(graph, config)` (with `OptimizationConfig`) | `torch.fx.passes.*` | ✅ partial |

**Gap:** no equivalent to `torch.fx.Transformer` (rewrite IR by walking;
bodies of nodes can be edited)? Actually `optimize` covers passes. Missing:
- `replace_pattern` / `subgraph_rewriter` — pattern-based graph rewriting
- pass-manager API (compose multiple passes)
- `GraphModule` — the IR-as-Python-module wrapper. ferrotorch has
  `TracedModule` / `AotCompiledModule` which serve a similar role.

### AOTAutograd

| ferrotorch | torch | Status |
|---|---|---|
| `compile_aot(module)` → `AotGraphPair` | `aot_function` / `aot_module` | ✅ |
| `decompose_forward_backward` | dual-graph decomposition | ✅ |

This is a **first-class capability** in ferrotorch and well-aligned with
torch's AOTAutograd story.

### Codegen / backends

| ferrotorch | torch._inductor | Status |
|---|---|---|
| `Codegen` trait | scheduler hooks | ✅ |
| `CompiledGraph` | `CompiledFxGraph` | ✅ |
| `InductorBackend`, `InductorTarget` (named match) | (named after pytorch's backend) | ✅ |
| `NativeBackend`, `InterpreterBackend` | fallback / eager mode | ✅ |
| `CpuCodegen` (C++ kernels) | `codegen/cpp_*` | ✅ |
| `GpuCodegen` (PTX) | `codegen/triton.py` | partial — Triton vs raw PTX |
| `JitCompiledKernel`, `compile_c_kernel` | `cpp_wrapper.py`, kernel cache | ✅ |
| `LoopIR`, `Expr`, `BinOpKind`, `UnaryOpKind` | inductor IR (sympy-based) | ✅ structurally |
| `MemoryPlan`, `plan_memory` | inductor memory planning | ✅ |
| `Autotuner` (AutotuneCandidate, AutotuneKey, AutotuneResult) | `select_algorithm.py`, autotune flow | ✅ |

**Gap:** no Triton-equivalent (Triton is a Python kernel DSL; the analog
in Rust would be CubeCL kernels which `ferrotorch-cubecl` already provides
— jit could lower to CubeCL kernels for a closer Inductor analog).

### Fusion

| ferrotorch | torch._inductor | Status |
|---|---|---|
| `FusedChain`, `FusedOp`, `apply_fused`, `with_fusion`, `is_fusion_enabled` | scheduler fusion | ✅ |
| `FusionGroup`, `FusionGroupKind` (in `dag_fusion.rs`) | DAG-based scheduling | ✅ |
| `ReductionKind`, `generate_reduction_c`, `generate_reduction_ptx`, `estimate_matmul_dims`, `estimate_numel_for_inputs` | reduction codegen | ✅ |

### Export

| ferrotorch | torch.export | Status |
|---|---|---|
| `ExportedProgram`, `ExportedProgramMetadata`, `InputSpec`, `DimSpec` | same names | ✅ aligned |
| `export(module, args)` | `torch.export.export` | ✅ |
| `export_with_dynamic_shapes` | dynamic shapes support | ✅ |
| `serialize` module | `torch.export.save/load` (.pt2) | ✅ |

**Gap:** no equivalent to ATen-level decomposition (torch.export decomposes
into a small ATen "core" op set). ferrotorch IR is already pretty
small-op-set, but the formal "core ATen" definition / verification would
need explicit attention if interop with torch's `.pt2` format is a goal.

### Graph breaks (dynamo)

| ferrotorch | torch._dynamo | Status |
|---|---|---|
| `GraphSegment`, `SegmentedModule`, `TraceResult`, `trace_with_breaks` | dynamo's bytecode-level interception emits multiple graphs around graph breaks | ✅ structurally analogous; mechanism differs |

ferrotorch's "graph breaks" are tracer-level (the tracer sees data-dependent
control flow and segments the IR); dynamo's are bytecode-level. The
*output* — multiple graph segments stitched with eager boundaries — is the
same.

### Public umbrella API

| ferrotorch | torch.compiler | Status |
|---|---|---|
| `compile(model)`, `compile_with_config(model, config)` | `torch.compile(model, backend=..., mode=..., dynamic=...)` | ✅ |
| (config struct: `CompileConfig`) | (kwargs) | ✅ |
| `OptimizationConfig` | `torch.compiler.set_default_backend` etc. | partial |

**Gap:** no `disable(fn)`, `allow_in_graph(fn)`, `assume_constant_result(fn)`,
`is_compiling()`, `cudagraph_mark_step_begin()`. These are dynamo escape
hatches users sprinkle through their code.

## Big-picture coverage

ferrotorch-jit looks like it covers **all four torch subsystems** at a
recognizable structural level:

| Subsystem | Coverage |
|---|---|
| torch.jit (TorchScript) | trace path covered; scripting not applicable |
| torch.fx (graph IR + transforms) | IR + interpret + optimize covered; subgraph_rewriter / pattern_matcher missing |
| torch._dynamo (frame-level capture) | `trace_with_breaks` + `SymbolicDim` cover the *output*; mechanism is tracer not bytecode |
| torch._inductor (backend codegen) | Cpu + Gpu codegen, autotune, memory plan, fusion all present |
| torch.export (full-graph export) | `export` / `ExportedProgram` / `serialize` all present |
| AOTAutograd | `compile_aot` / `AotGraphPair` first-class |

That's a lot for one crate. Module list is 21 files; this is already the
biggest crate by module count after core.

## Recommendations

1. **Don't split this crate yet.** Even though the pytorch side has 4-5
   distinct subsystems, ferrotorch's implementation has clear internal
   boundaries (separate modules) and the cross-dependencies are tight (IR
   is shared, codegen depends on IR, autotune depends on codegen). Splitting
   into `ferrotorch-jit` (trace/IR) + `ferrotorch-compile` (codegen/fusion)
   + `ferrotorch-export` would create circular imports unless done very
   carefully.
2. **Add subgraph rewriting**: `replace_pattern` style API for users to
   write graph transforms by example. Big productivity win.
3. **Add a pass-manager**: compose `OptimizationConfig`-style passes,
   register custom passes. Currently `optimize` is opaque.
4. **Add escape hatches**: `disable(fn)`, `allow_in_graph(fn)`,
   `assume_constant_result(fn)` analogs. Without these, users hit
   tracer-incompatible code with no workaround.
5. **Lower to CubeCL kernels** as an alternative to raw PTX — let
   `GpuCodegen` emit CubeCL `#[cube]` for cross-vendor portability.
   Currently codegen_gpu is CUDA-PTX only; CubeCL emit would unlock ROCm/
   Metal/Vulkan via the existing `ferrotorch-cubecl` infrastructure.
6. **Document the divergence from TorchScript** — ferrotorch traces Rust
   code; there's no scripting front-end and likely never will be.
7. **`.pt2` interop** — if loading torch-exported programs is a goal,
   need ATen-core op-set adapter. Document as deferred / non-goal.

## Status

ferrotorch-jit is **structurally one of the most complete crates** vs its
pytorch counterparts. The four torch subsystems map to internal modules
inside this single crate, and most of the named entities have direct
analogs.

Real coverage gaps are subgraph rewriting, pass manager, escape-hatch
decorators (allow_in_graph etc.), and `.pt2` interop. Codegen breadth
(CPU + GPU + JIT C compile) is impressive.

**No new crate needed.** Document scope clearly: "tracing JIT + graph IR
+ AOT compile + export, in the spirit of `torch.compile` + `torch.export`."
