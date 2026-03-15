# Feature: Phase 8 — JIT / Graph Optimization (ferrotorch-jit)

## Summary
A tracing JIT compiler that captures ferrotorch-core's dynamic computation graphs into a static intermediate representation, applies graph-level optimizations (constant folding, operator fusion, dead code elimination, memory planning, kernel selection), and emits optimized executables for production inference and accelerated training. The crate intercepts the existing `GradFn<T>` autograd graph by replaying a forward function with proxy tensors, producing a `TracedModule` that can be serialized, optimized, and executed without rebuilding the graph each iteration. Codegen targets Cranelift for CPU and PTX emission via `cudarc` for CUDA GPUs. Production features include a `compile()` API (torch.compile equivalent) with shape guards and automatic recompilation, graph break handling for graceful fallback to eager mode, and an `export()` API for self-contained serialized model artifacts.

## Requirements
- REQ-1: A `trace` function must accept any closure `Fn(&[Tensor<T>]) -> Result<Tensor<T>, FerrotorchError>` and a set of example inputs, execute the closure once while recording every operation into a static IR graph, and return a `TracedModule<T>` that reproduces the same computation without dynamic graph construction overhead. Tracing must capture the complete forward pass — every `GradFn<T>` node created during execution must have a corresponding IR node. Control flow that depends on tensor values (data-dependent branching) must be detected and rejected with `FerrotorchError::TracingError` rather than silently producing an incorrect graph.
- REQ-2: The IR graph must be a directed acyclic graph of typed operation nodes (`IrNode`) connected by value edges (`IrValue`). Each `IrNode` must record: the operation kind (matching ferrotorch-core's op vocabulary — arithmetic, reduction, linalg, activation, shape, indexing), input values, output values, shape/dtype metadata, and an optional device annotation. The IR must support subgraphs for representing fused regions. The graph must be serializable to and deserializable from a binary format for ahead-of-time compilation workflows.
- REQ-3: The optimizer must implement at minimum five graph transformation passes executed in a fixed pipeline order: (1) constant folding — evaluate subgraphs whose inputs are all compile-time constants and replace them with literal tensors, (2) dead code elimination — remove nodes whose outputs are not consumed by any live node or graph output, (3) operator fusion — identify chains of elementwise operations and collapse them into a single fused kernel node, (4) memory planning — analyze the liveness of every IR value and assign buffer slots that maximize memory reuse across non-overlapping lifetimes, (5) kernel selection — annotate each node with the preferred execution strategy (e.g., SIMD width, tiling parameters, or GPU kernel choice) based on the target device and tensor shapes. Each pass must be independently toggleable via an `OptimizationConfig` struct.
- REQ-4: `TracedModule<T>` must implement ferrotorch-nn's `Module<T>` trait so that traced models are drop-in replacements for eager models in inference pipelines. Calling `forward` on a `TracedModule` must execute the optimized IR graph rather than re-tracing. `TracedModule` must also expose `backward` support: when any input `requires_grad`, the traced graph must produce correct gradients by recording the backward pass as a second IR graph (the adjoint graph) during tracing.
- REQ-5: A code generation backend must translate the optimized IR graph into executable form. The initial implementation must target a Rust-native backend (Cranelift) that emits machine code for CPU execution. The `Codegen` trait must be object-safe and extensible so that additional backends (LLVM, CUDA PTX via ferrotorch-gpu) can be added without modifying existing code. Generated code must match eager-mode numerical output within the same floating-point tolerances used by ferrotorch-core's test suite (rtol=1e-4 for f32, rtol=1e-7 for f64).
- REQ-6: All public functions must return `Result<T, FerrotorchError>`. Tracing failures (data-dependent control flow, unsupported operations, shape mismatches in the IR) must produce descriptive error variants. The crate must never panic on invalid input.
- REQ-7: Tracing must be deterministic: given the same function and the same example input shapes, repeated calls to `trace` must produce identical IR graphs. The IR must be independent of the concrete values of the example inputs — only shapes, dtypes, and devices are captured.
- REQ-8: The crate must provide a `GraphProfiler` that instruments a `TracedModule` to collect per-node execution times, memory allocation sizes, and fusion region boundaries. Profiling output must be a structured `ProfileReport` that can be printed as a human-readable summary or serialized for external tooling.
- REQ-9: A `compile` function must accept any `Module<T>` and example inputs, trace the model, apply the full optimization pipeline, and return a `CompiledModule<T>` that implements `Module<T>` as a drop-in replacement. The compiled module must handle dynamic shapes via shape guards and automatic recompilation, caching compiled variants in an LRU cache keyed by shape signature. Named parameters must be updatable via `set_parameter` without re-tracing, enabling use in training loops.
- REQ-10: When tracing encounters an unsupported operation, the tracer must insert a graph break and split the computation into segments — compiled IR subgraphs interleaved with eager fallback closures. The `SegmentedModule` must execute segments in order, passing tensor values between them via recorded dataflow connections. When `CompileConfig::fullgraph` is `true`, graph breaks must be rejected with `JitError::GraphBreak` instead of falling back.
- REQ-11: A `jit::export` function must produce a fully self-contained `ExportedModel<T>` that includes the optimized IR graph, all parameter values, input/output shape specifications, and metadata. Export mode must enforce `fullgraph=true` (no graph breaks). The exported model must be serializable to and deserializable from a `.ftm` file (MessagePack format) and must be executable without access to the original model's Rust source code. An `ExportedModel` must support validation to verify IR well-formedness.
- REQ-12: The `PtxBackend` codegen backend (gated behind the `cuda` feature) must generate PTX assembly for fused elementwise kernel chains and dispatch complex operations (matmul, convolution, batch norm) to cuBLAS/cuDNN via `ferrotorch-gpu`. PTX modules must be loaded at runtime through `cudarc`'s CUDA driver API. The PTX backend must produce numerically identical output to the Cranelift CPU backend within floating-point tolerance.

## Acceptance Criteria
- [ ] AC-1: `jit::trace(|inputs| model.forward(&inputs[0]), &[example])` on a 4-layer MLP (Linear-ReLU-Linear-ReLU-Linear-ReLU-Linear) produces a `TracedModule` whose IR contains the expected number of matmul, add (bias), and relu nodes. Re-executing the `TracedModule` with different-valued inputs of the same shape produces numerically identical output to eager-mode execution (within f32 tolerance).
- [ ] AC-2: Tracing a function that contains `if tensor.sum().item() > 0.0` (data-dependent branch) returns `Err(FerrotorchError::TracingError { .. })` with a message identifying the offending operation as data-dependent control flow.
- [ ] AC-3: Constant folding eliminates compile-time constant subgraphs: tracing `|x| x * Tensor::ones([3, 3]) + Tensor::zeros([3, 3])` produces an IR where the `ones` and `zeros` tensors are folded into a single constant node and the dead addition of zero is eliminated.
- [ ] AC-4: Operator fusion merges a chain of 5 sequential elementwise operations (add, mul, relu, sigmoid, neg) into a single fused node in the IR. Executing the fused graph produces the same output as the unfused version within floating-point tolerance.
- [ ] AC-5: Memory planning reduces peak buffer allocation by at least 30% on a 20-layer residual network compared to naive per-node allocation, measured by summing the sizes of all simultaneously live buffers.
- [ ] AC-6: `TracedModule` implements `Module<T>` and can be used as a drop-in replacement in an inference loop: `let traced = jit::trace(|x| model.forward(&x[0]), &[example])?; let out = traced.forward(&input)?;` compiles and produces correct output.
- [ ] AC-7: The Cranelift backend compiles the IR for a simple computation graph (matmul + relu + matmul) into native machine code and executes it, producing output matching eager mode. Compilation time for a 50-node graph is under 100ms on a modern x86-64 CPU.
- [ ] AC-8: `TracedModule::backward` computes correct gradients for all leaf inputs through the traced graph, verified against eager-mode backward for at least 10 representative graphs including matmul chains, activation functions, reductions, and reshape operations.
- [ ] AC-9: IR graphs survive a round-trip through serialization: `let bytes = graph.serialize()?; let restored = IrGraph::deserialize(&bytes)?;` produces a graph that executes identically to the original.
- [ ] AC-10: `jit::compile(&model, &[example], None)` on an MLP returns a `CompiledModule` that produces output identical to eager mode. Calling `forward` with a different batch size triggers automatic recompilation, and the second call with the original batch size is a cache hit (verified via `cache_info()`).
- [ ] AC-11: `compiled.set_parameter("layer1.weight", new_weight)?` updates the parameter, and subsequent `forward` calls use the new weight values without re-tracing. Backward through the compiled module produces correct gradients for the updated parameters.
- [ ] AC-12: Tracing a model that contains a `println!` (unsupported side effect) with `fullgraph: false` produces a `SegmentedModule` with 3 segments (compiled, eager, compiled). Executing the segmented module produces the same tensor output as eager mode. The same trace with `fullgraph: true` returns `Err(JitError::GraphBreak { .. })`.
- [ ] AC-13: `jit::export(&model, &[example], None)` produces an `ExportedModel` that round-trips through `save`/`load` on disk. The loaded model executes via `into_module()` and produces output identical to the original model. `exported.validate()` returns `Ok(())` for valid models and `Err` for malformed IR.
- [ ] AC-14: The PTX backend (under `cuda` feature) compiles a fused elementwise chain (add + relu + mul) into a single PTX kernel, loads it via `cudarc`, and produces output matching the Cranelift CPU backend within f32 tolerance.
- [ ] AC-15: `cargo test -p ferrotorch-jit` passes with 0 failures. Minimum 200 tests covering tracing, all five optimization passes, codegen (Cranelift + PTX under feature), backward through traced graphs, serialization round-trips, compile API, graph breaks, export/import, shape guard recompilation, parameter updates, error paths, and profiling.

## Architecture

### Crate Layout

```
ferrotorch-jit/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public API: trace, compile, export; TracedModule, CompiledModule, ExportedModel, IrGraph, OptimizationConfig
│   ├── trace.rs                  # Tracing engine — proxy tensors, op recording, control flow detection
│   ├── graph.rs                  # IrGraph, IrNode, IrValue, IrOpKind — the intermediate representation
│   ├── optimize.rs               # Optimization pipeline — constant fold, DCE, fusion, memory plan, kernel select
│   ├── codegen.rs                # Codegen trait + CraneliftBackend (CPU) + PtxBackend (CUDA, feature-gated)
│   ├── module.rs                 # TracedModule — Module<T> impl, forward/backward, shape cache, parameter indirection
│   ├── compile.rs                # compile() convenience API, CompileConfig, CompiledModule, BackendChoice
│   ├── graph_break.rs            # Graph break detection, GraphSegment, SegmentedModule, eager fallback
│   ├── export.rs                 # export() API, ExportedModel, ExportConfig, TensorSpec, .ftm format
│   ├── profile.rs                # GraphProfiler, ProfileReport — per-node timing and memory stats
│   ├── serialize.rs              # Binary serialization/deserialization of IrGraph
│   └── error.rs                  # JIT-specific error variants (TracingError, CodegenError, GraphBreak, ExportError, etc.)
└── tests/
    ├── test_trace.rs             # Tracing correctness, data-dependent branch rejection, determinism
    ├── test_graph.rs             # IR construction, node connectivity, shape inference
    ├── test_optimize.rs          # Each optimization pass in isolation + full pipeline
    ├── test_codegen.rs           # Cranelift compilation and execution correctness
    ├── test_codegen_ptx.rs       # PTX backend compilation (requires `cuda` feature + GPU)
    ├── test_backward.rs          # Gradient correctness through traced graphs
    ├── test_compile.rs           # compile() API, CompileConfig, shape guard recompilation, training loop
    ├── test_graph_break.rs       # Graph break detection, segmented execution, fullgraph mode
    ├── test_export.rs            # export() API, .ftm round-trip, standalone execution, validation
    ├── test_serialize.rs         # Round-trip serialization
    └── test_profile.rs           # Profiling instrumentation output
```

### Core Types

**IrGraph** (`graph.rs`):
```rust
/// A static computation graph in SSA form. Every value is defined exactly once.
pub struct IrGraph<T: Element> {
    nodes: Vec<IrNode<T>>,
    inputs: Vec<IrValue>,           // Graph-level input placeholders
    outputs: Vec<IrValue>,          // Graph-level outputs
    constants: HashMap<IrValue, Tensor<T>>,  // Folded constant tensors
    metadata: GraphMetadata,        // Source function name, trace timestamp
}

/// A single operation in the IR.
pub struct IrNode<T: Element> {
    id: NodeId,
    op: IrOpKind,
    inputs: Vec<IrValue>,
    outputs: Vec<IrValue>,
    shape: Vec<usize>,              // Output shape (inferred during tracing)
    dtype: DType,
    device: Device,
    fusion_group: Option<FusionGroupId>,
    _marker: PhantomData<T>,
}

/// Every differentiable operation in ferrotorch-core has a corresponding variant.
pub enum IrOpKind {
    // Arithmetic
    Add, Sub, Mul, Div, Neg, Pow, Sqrt, Abs,
    // Reduction
    Sum { dims: Vec<usize>, keep_dim: bool },
    Mean { dims: Vec<usize>, keep_dim: bool },
    Prod { dims: Vec<usize>, keep_dim: bool },
    // Linalg
    Matmul, Bmm, Mm, Mv, Dot,
    // Activation
    Relu, Sigmoid, Tanh, Gelu, Silu, Softmax { dim: usize }, LogSoftmax { dim: usize },
    // Shape
    Reshape { shape: Vec<usize> }, Transpose { dim0: usize, dim1: usize },
    Permute { dims: Vec<usize> }, Expand { shape: Vec<usize> },
    Cat { dim: usize }, Stack { dim: usize }, Split { dim: usize, sizes: Vec<usize> },
    Squeeze { dim: Option<usize> }, Unsqueeze { dim: usize }, Flatten { start: usize, end: usize },
    // Indexing
    Gather { dim: usize }, ScatterAdd { dim: usize }, IndexSelect { dim: usize }, MaskedFill,
    // Comparison
    Where,
    // Special
    Constant,                       // Literal tensor embedded in the graph
    FusedElementwise { ops: Vec<IrOpKind> },  // Result of operator fusion
}

/// A typed handle to a value produced by a node.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct IrValue(u32);

pub struct NodeId(u32);
pub struct FusionGroupId(u32);
```

**Tracing Engine** (`trace.rs`):
```rust
/// Trace a forward function into a frozen IR graph.
pub fn trace<T, F>(
    f: F,
    example_inputs: &[Tensor<T>],
) -> Result<TracedModule<T>, FerrotorchError>
where
    T: Element,
    F: Fn(&[Tensor<T>]) -> Result<Tensor<T>, FerrotorchError>,
{
    // 1. Create proxy tensors that mirror example_inputs' shapes/dtypes/devices
    // 2. Enable tracing mode (thread-local flag, similar to no_grad)
    // 3. Execute f(&proxies) — every op records an IrNode instead of computing
    // 4. Disable tracing mode
    // 5. Collect recorded nodes into an IrGraph
    // 6. Build adjoint graph for backward support
    // 7. Run optimization pipeline
    // 8. Return TracedModule wrapping the optimized graph
}
```

Tracing intercepts operation dispatch at the `GradFn<T>` boundary. When tracing mode is active (controlled by a `TRACING_ACTIVE` thread-local, mirroring the `GRAD_ENABLED` pattern from ferrotorch-core's `no_grad`), each op that would normally construct a `GradFn<T>` node instead records an `IrNode` into a thread-local `TraceRecorder`. The proxy tensors carry shape metadata but no real data — operations infer output shapes from input shapes using the same broadcasting and shape-propagation rules as eager mode.

Data-dependent control flow detection works by tagging proxy tensors with a `is_proxy: bool` flag. Any attempt to extract a scalar value from a proxy (via `.item()`, comparison to a concrete value, or boolean coercion) raises `FerrotorchError::TracingError` immediately, identifying the operation that would introduce data-dependent branching.

**Adjoint Graph Construction**: During tracing, every `IrNode` has a known `GradFn<T>` counterpart from ferrotorch-core. The adjoint graph is built by symbolically applying each `GradFn`'s backward rule in reverse topological order over the IR, producing a second `IrGraph` that maps output gradients to input gradients. This adjoint graph receives the same optimization passes as the forward graph.

**Optimization Pipeline** (`optimize.rs`):
```rust
pub struct OptimizationConfig {
    pub constant_folding: bool,         // Default: true
    pub dead_code_elimination: bool,    // Default: true
    pub operator_fusion: bool,          // Default: true
    pub memory_planning: bool,          // Default: true
    pub kernel_selection: bool,         // Default: true
}

pub fn optimize<T: Element>(
    graph: &mut IrGraph<T>,
    config: &OptimizationConfig,
) -> Result<(), FerrotorchError> {
    if config.constant_folding { constant_fold(graph)?; }
    if config.dead_code_elimination { dead_code_eliminate(graph)?; }
    if config.operator_fusion { fuse_operators(graph)?; }
    if config.memory_planning { plan_memory(graph)?; }
    if config.kernel_selection { select_kernels(graph)?; }
    Ok(())
}
```

The passes execute in a fixed order because each pass's output feeds the next:
1. **Constant folding** runs first because it materializes constant subgraphs into literal `Constant` nodes, exposing new dead code and fusion opportunities.
2. **Dead code elimination** removes nodes orphaned by folding (and any that were unused in the original graph). Uses a reverse reachability pass from graph outputs — any node not reachable is dead.
3. **Operator fusion** identifies maximal chains of elementwise `IrOpKind` variants (Add, Mul, Relu, Sigmoid, etc.) connected by single-use edges. Each chain is collapsed into a `FusedElementwise` node whose `ops` field records the original sequence. Fusion boundaries are: non-elementwise ops, nodes with multiple consumers (fan-out), and device transitions.
4. **Memory planning** performs liveness analysis on all `IrValue` handles, computes overlapping lifetimes, and assigns buffer slots using a greedy first-fit algorithm. The result is a `MemoryPlan` stored as metadata on the graph — a mapping from `IrValue` to `(slot_id, offset, size)`. The executor allocates one contiguous arena per slot and sub-allocates from it.
5. **Kernel selection** annotates each node with execution hints based on the target `Device` and the node's shapes. For CPU: SIMD width (AVX2/AVX-512 detection at trace time), tiling parameters for matmul, parallelism degree for reductions. For CUDA (when ferrotorch-gpu is present): grid/block dimensions, shared memory usage, and whether to use cuBLAS or a custom kernel.

**Code Generation** (`codegen.rs`):
```rust
/// Backend-agnostic code generation trait.
pub trait Codegen<T: Element>: Send + Sync {
    /// Compile an optimized IR graph into an executable artifact.
    fn compile(&self, graph: &IrGraph<T>) -> Result<CompiledGraph<T>, FerrotorchError>;

    /// Return the target device this backend compiles for.
    fn target_device(&self) -> Device;
}

/// An executable compiled graph.
pub struct CompiledGraph<T: Element> {
    execute_fn: Box<dyn Fn(&[Tensor<T>], &HashMap<String, Tensor<T>>) -> Result<Tensor<T>, FerrotorchError> + Send + Sync>,
    memory_plan: MemoryPlan,
    _marker: PhantomData<T>,
}

/// Cranelift-based CPU code generator (default backend).
pub struct CraneliftBackend {
    opt_level: cranelift_codegen::settings::OptLevel,
}

impl<T: Element> Codegen<T> for CraneliftBackend {
    fn compile(&self, graph: &IrGraph<T>) -> Result<CompiledGraph<T>, FerrotorchError> {
        // 1. Create Cranelift IR function
        // 2. For each IrNode, emit Cranelift instructions:
        //    - Elementwise/fused: inline SIMD loop
        //    - Matmul: call into ferray-linalg (extern function reference)
        //    - Reduction: emit reduction loop with accumulator
        // 3. Finalize and compile to native machine code
        // 4. Wrap in CompiledGraph with the memory plan from optimization
    }

    fn target_device(&self) -> Device { Device::Cpu }
}

/// PTX code generator for CUDA GPUs. Emits PTX assembly for fused elementwise
/// kernels and dispatches complex ops (matmul, conv) to cuBLAS/cuDNN via
/// ferrotorch-gpu. Requires the `cuda` feature flag and a CUDA-capable GPU.
pub struct PtxBackend {
    device_ordinal: usize,
    compute_capability: (u32, u32),  // e.g. (8, 0) for SM 80
}

impl<T: Element> Codegen<T> for PtxBackend {
    fn compile(&self, graph: &IrGraph<T>) -> Result<CompiledGraph<T>, FerrotorchError> {
        // 1. Walk the IR graph, partitioning nodes into:
        //    a. FusedElementwise groups → generate PTX kernel source
        //    b. Complex ops (Matmul, Bmm, etc.) → emit cuBLAS/cuDNN dispatch calls
        // 2. For each fused kernel:
        //    a. Generate PTX source string with grid-stride loop over elements
        //    b. Load the PTX module via cudarc's CUDA driver API
        //    c. Extract the kernel function handle
        // 3. Build an execution plan that sequences kernel launches and library calls
        // 4. Wrap in CompiledGraph with GPU memory plan (arena allocation on device)
    }

    fn target_device(&self) -> Device { Device::Cuda(self.device_ordinal) }
}
```

The Cranelift backend handles elementwise and fused ops by emitting tight SIMD loops directly. For complex operations (matmul, convolutions, FFT), it emits calls to the same ferray functions that eager mode uses — the JIT benefit comes from eliminating graph overhead, fusing surrounding elementwise ops, and pre-planning memory, not from replacing BLAS kernels.

The PTX backend generates CUDA PTX assembly for fused elementwise chains, where the primary benefit is eliminating intermediate global memory round-trips between ops. A chain of 5 elementwise ops that would normally require 5 kernel launches and 4 intermediate buffers becomes a single kernel that reads input once and writes output once. Complex ops (matmul, convolutions, batch norm) dispatch to cuBLAS/cuDNN through `ferrotorch-gpu` because hand-written PTX cannot compete with vendor-tuned libraries for these operations.

**TracedModule** (`module.rs`):
```rust
pub struct TracedModule<T: Element> {
    /// The original traced forward function (retained for re-tracing on shape changes).
    trace_fn: Option<Box<dyn Fn(&[Tensor<T>]) -> Result<Tensor<T>, FerrotorchError> + Send + Sync>>,
    forward_graph: IrGraph<T>,
    backward_graph: Option<IrGraph<T>>,  // Present when any traced input requires_grad
    /// LRU cache of compiled graphs keyed by input shape signature.
    compiled_cache: LruCache<ShapeSignature, CompiledGraph<T>>,
    /// Maximum number of shape variants to cache before evicting the least-recently-used entry.
    cache_capacity: usize,              // Default: 8
    /// Named parameters that can be updated between training steps without re-tracing.
    parameters: HashMap<String, Tensor<T>>,
    config: OptimizationConfig,
    /// Cumulative cache hit/miss statistics.
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

/// A tuple of input shapes used as a cache key.
type ShapeSignature = Vec<Vec<usize>>;

/// Statistics about the shape-specialized compilation cache.
pub struct CacheInfo {
    pub cached_variants: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
}

impl<T: Element> TracedModule<T> {
    /// Update a named parameter captured during tracing. The new tensor must match
    /// the original's shape and dtype. Takes effect on the next `forward` call
    /// without re-tracing.
    pub fn set_parameter(&mut self, name: &str, value: Tensor<T>) -> Result<(), FerrotorchError> {
        // Validate shape/dtype match, update the parameter in the indirection table
    }

    /// Return statistics about the shape-specialized compilation cache.
    pub fn cache_info(&self) -> CacheInfo { ... }

    /// Clear all cached compilations, forcing recompilation on the next `forward` call.
    pub fn clear_cache(&mut self) { ... }
}

impl<T: Element> Module<T> for TracedModule<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, FerrotorchError> {
        // 1. Compute shape signature from input
        // 2. Check compiled_cache for a matching compiled graph
        // 3. On cache miss: re-trace with the new shape, optimize, compile, insert into cache
        // 4. Execute the compiled graph, passing current parameter values via indirection table
        // 5. If input.requires_grad, attach backward_graph as the grad_fn
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        // Return references to named parameters captured during tracing
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        // Return mutable references to named parameters for optimizer updates
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn named_parameters(&self) -> Vec<(&str, &Parameter<T>)> {
        // Return (name, parameter) pairs from the parameters HashMap
    }
    fn load_state_dict(&mut self, state: &StateDict<T>) -> Result<(), FerrotorchError> {
        // Update each named parameter from the state dict via set_parameter
    }
    fn state_dict(&self) -> StateDict<T> {
        // Export current parameter values as a StateDict
    }
}
```

`TracedModule` compiles lazily: the first call to `forward` invokes `Codegen::compile` and caches the result keyed by the input shape signature. Subsequent calls with the same shapes reuse the cached artifact. When called with a new input shape, the module re-traces (if a `trace_fn` is available), re-optimizes, and compiles a new variant, inserting it into the LRU cache. This avoids paying compilation cost if the module is serialized before execution, and gracefully handles variable input shapes (e.g., different batch sizes) at the cost of a one-time recompilation per new shape.

**Profiling** (`profile.rs`):
```rust
pub struct GraphProfiler;

impl GraphProfiler {
    /// Execute a TracedModule with instrumentation enabled.
    pub fn profile<T: Element>(
        module: &TracedModule<T>,
        inputs: &[Tensor<T>],
    ) -> Result<ProfileReport, FerrotorchError> { ... }
}

pub struct ProfileReport {
    pub node_timings: Vec<NodeTiming>,       // Per-node wall time
    pub memory_usage: Vec<NodeMemory>,       // Per-node allocation size
    pub fusion_regions: Vec<FusionRegion>,   // Fused op boundaries
    pub total_time: std::time::Duration,
    pub peak_memory_bytes: usize,
}
```

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | `Tensor<T>`, `GradFn<T>`, `Device`, `FerrotorchError`, autograd graph traversal |
| `ferrotorch-nn` | workspace | `Module<T>` trait (for `TracedModule` and `CompiledModule` impls) |
| `ferrotorch-gpu` | workspace, optional | GPU kernel dispatch for `PtxBackend` (gated behind `cuda` feature) |
| `cranelift-codegen` | 0.115 | CPU native code generation (`CraneliftBackend`) |
| `cranelift-module` | 0.115 | JIT module management for Cranelift |
| `cranelift-jit` | 0.115 | In-process JIT compilation |
| `cudarc` | 0.16, optional | CUDA driver API for PTX loading and kernel launch (gated behind `cuda` feature) |
| `lru` | 0.12 | LRU cache for shape-specialized compiled graph variants |
| `thiserror` | 2.0 | Error derive macros |
| `serde` | 1.0 | IR graph and export model serialization |
| `rmp-serde` | 1.3 | MessagePack binary format for IR serialization and `.ftm` export format |

### Error Variants

`error.rs` extends `FerrotorchError` with JIT-specific variants:

```rust
#[derive(Debug, thiserror::Error)]
pub enum JitError {
    #[error("tracing error: {message}")]
    TracingError { message: String },
    #[error("data-dependent control flow detected at op '{op}': tracing requires static control flow")]
    DataDependentControlFlow { op: String },
    #[error("unsupported operation during tracing: {op}")]
    UnsupportedOp { op: String },
    #[error("shape mismatch: traced with {traced:?}, called with {actual:?}")]
    ShapeMismatch { traced: Vec<usize>, actual: Vec<usize> },
    #[error("codegen error: {message}")]
    CodegenError { message: String },
    #[error("serialization error: {message}")]
    SerializationError { message: String },
    #[error("graph break at op '{op}': {reason}")]
    GraphBreak { op: String, reason: String },
    #[error("export error at op '{op}': {reason} (export mode requires fullgraph — no graph breaks allowed)")]
    ExportError { op: String, reason: String },
    #[error("parameter error: {message}")]
    ParameterError { message: String },
    #[error("recompilation failed for shape {shape:?}: {message}")]
    RecompilationError { shape: Vec<Vec<usize>>, message: String },
}

impl From<JitError> for FerrotorchError {
    fn from(e: JitError) -> Self {
        FerrotorchError::InvalidArgument { message: e.to_string() }
    }
}
```

### Test Strategy

1. **Tracing correctness**: Trace 15+ model architectures (MLP, CNN patterns, attention, residual connections) and verify the IR node count and connectivity match expectations. Re-execute with varied inputs and compare against eager mode.
2. **Optimization passes**: Unit test each pass in isolation — construct a small IR graph, run one pass, assert the expected transformation occurred (node count, fusion groups, eliminated nodes, memory plan slot count).
3. **Codegen numerical parity (Cranelift)**: For every `IrOpKind`, compile a single-node graph through Cranelift and compare output against eager execution using the same tolerances as ferrotorch-core (rtol=1e-4 for f32, rtol=1e-7 for f64).
4. **Codegen numerical parity (PTX)**: Under the `cuda` feature, repeat codegen parity tests using the `PtxBackend`. Verify fused elementwise PTX kernels produce identical output to unfused eager execution. Verify cuBLAS/cuDNN dispatch for complex ops (matmul, conv) matches eager-mode GPU execution.
5. **Backward correctness**: Trace forward+backward for 10+ graphs, compute gradients through the traced backward graph, and verify against eager-mode `backward()` output.
6. **Serialization round-trip**: Serialize and deserialize IR graphs, verify the restored graph produces identical output.
7. **Error paths**: Verify that data-dependent control flow, unsupported ops, shape mismatches at execution time, codegen failures, graph breaks in fullgraph mode, export failures on unsupported ops, and parameter shape mismatches all produce the correct `JitError` variant.
8. **Determinism**: Trace the same function 10 times and assert all IR graphs are byte-identical after serialization.
9. **Compile API**: Test `jit::compile` end-to-end on MLP, CNN, and transformer architectures. Verify `CompiledModule` is a drop-in replacement for the original model. Test `CompileConfig` options (backend selection, cache capacity, fullgraph mode).
10. **Shape guard recompilation**: Compile a model with batch size 16, call forward with batch size 32 (triggers recompilation), call again with batch size 16 (cache hit). Verify `cache_info()` reports correct hit/miss counts. Test LRU eviction with more shape variants than cache capacity.
11. **Parameter updates for training**: Compile a model, run a training step (forward + backward + optimizer step), call `set_parameter` with updated weights, run another forward pass. Verify gradients flow correctly through the parameter indirection and output changes reflect the updated weights.
12. **Graph break handling**: Trace models with unsupported ops (I/O, custom ops). Verify `SegmentedModule` splits correctly into compiled and eager segments. Verify dataflow connections pass tensors between segments. Test `fullgraph: true` rejects graph breaks with `JitError::GraphBreak`.
13. **Export round-trip**: Export a model, save to disk as `.ftm`, load in a fresh context, execute via `into_module()`. Verify output matches original model. Test `validate()` on valid and intentionally corrupted models. Verify export rejects models with graph breaks.
14. **Profiling**: Run the profiler on a traced graph and verify the report contains timing entries for every node and correct fusion group boundaries.

## Resolved Questions

### Q1: Primary codegen backend — Cranelift for CPU, PTX emission for CUDA
**Decision: Cranelift for CPU, PTX emission for CUDA.** Cranelift is pure Rust, has fast compile times (~100ms for small graphs), and is the pragmatic choice for a Rust project. LLVM would produce ~10-20% better optimized kernels but brings a 200MB+ build dependency (`libLLVM`), second-scale compilation times, and complicates cross-compilation. PyTorch uses its own Inductor backend which generates Triton (GPU) and C++ (CPU). We use Cranelift (CPU) and direct PTX emission via `cudarc` (GPU). The `Codegen` trait remains extensible — an LLVM backend can be added behind an optional feature flag in the future if Cranelift's code quality proves insufficient for specific workloads.

### Q2: GPU codegen path — PTX emission via CUDA driver API
**Decision: PTX emission via CUDA driver API.** `cudarc` can load PTX strings at runtime through the CUDA driver API. We generate PTX from the IR for fused GPU kernels — specifically fused elementwise chains identified by the operator fusion pass. Individual complex ops (matmul, convolution, batch norm) still dispatch to cuBLAS/cuDNN through `ferrotorch-gpu`'s existing kernel library — fusion only applies to elementwise chains where the JIT can emit a single PTX kernel that avoids intermediate memory traffic. This hybrid approach gets the highest-value optimization (elementwise fusion) without reinventing BLAS/DNN libraries.

### Q3: Dynamic shapes — Shape guards with recompilation
**Decision: Shape guards with recompilation.** Tracing captures concrete shapes and inserts shape guards into the compiled artifact. When `TracedModule::forward` is called with inputs whose shapes do not match the traced shapes, the module automatically re-traces and recompiles for the new shape, caching the compiled variant keyed by the shape signature (a tuple of input shapes). Previously compiled variants are retained in an LRU cache (default capacity: 8 entries) so that alternating between a small number of shapes (e.g., variable batch sizes of 1, 16, 32) does not trigger repeated recompilation. This is simpler than full symbolic shape tracing (which would complicate every optimization pass) and handles the vast majority of real-world cases. The `TracedModule` exposes a `cache_info()` method returning the number of cached compilations and hit/miss statistics.

### Q4: Training integration — Mutable parameters via indirection
**Decision: Support mutable parameters via indirection.** `TracedModule` captures parameter values as constants during tracing, but supports a `set_parameter(name: &str, new_tensor: Tensor<T>)` API for updating weights between training steps without re-tracing. Internally, named parameters are stored in a `parameters: HashMap<String, Tensor<T>>` on `TracedModule`, and the compiled graph references these by name through an indirection table rather than embedding the tensor data directly. This allows the optimizer to update weights in-place. Backward through traced graphs works by recording the adjoint graph during tracing — the backward graph maps output gradients to gradients for both inputs and named parameters. This matches PyTorch's approach where `torch.compile` works seamlessly with training loops.

## Production Features

### `compile` — torch.compile equivalent

A convenience API that traces a model with example inputs, optimizes the graph, and returns a `CompiledModule<T>` ready for immediate execution. This is the primary user-facing entry point for JIT compilation, analogous to `torch.compile`.

```rust
/// Compile a model for optimized execution. Traces the forward function with the
/// given example inputs, applies the full optimization pipeline, and returns a
/// CompiledModule that handles shape guards and recompilation transparently.
pub fn compile<T, M>(
    model: &M,
    example_inputs: &[Tensor<T>],
    config: Option<CompileConfig>,
) -> Result<CompiledModule<T>, FerrotorchError>
where
    T: Element,
    M: Module<T>,
{
    // 1. Trace the model's forward method with example inputs
    // 2. Apply optimization pipeline with the given config (or defaults)
    // 3. Compile for the target device (auto-detected from example inputs)
    // 4. Return CompiledModule with shape guard cache and parameter indirection
}

/// Configuration for `compile`.
pub struct CompileConfig {
    pub optimization: OptimizationConfig,   // Which passes to run (default: all enabled)
    pub backend: BackendChoice,             // Cpu, Cuda, or Auto (default: Auto)
    pub cache_capacity: usize,              // Shape cache LRU capacity (default: 8)
    pub fullgraph: bool,                    // If true, error on graph breaks instead of falling back (default: false)
}

pub enum BackendChoice {
    Auto,                                   // Detect from input devices
    Cpu,                                    // Force Cranelift backend
    Cuda(usize),                            // Force PTX backend on given device ordinal
}

/// A compiled model that wraps TracedModule with an ergonomic API.
/// Implements Module<T> for drop-in replacement.
pub struct CompiledModule<T: Element> {
    inner: TracedModule<T>,
}

impl<T: Element> Module<T> for CompiledModule<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, FerrotorchError> {
        self.inner.forward(input)
    }
    // Delegates all Module methods to inner TracedModule
}
```

Usage:
```rust
let model = MyModel::new();
let example = Tensor::<f32>::randn(&[16, 784])?;
let compiled = jit::compile(&model, &[example], None)?;
// compiled implements Module<T>, use it exactly like the original model
let output = compiled.forward(&input)?;
```

### Graph Break Handling

When tracing encounters an operation that cannot be represented in the IR (e.g., a custom op without a registered `IrOpKind`, or I/O side effects), the tracer inserts a **graph break** rather than failing outright. The traced computation is split into subgraphs, with the unsupported operation executed in eager mode between compiled subgraphs. This matches PyTorch's `torch.compile` graph break behavior.

```rust
/// A segment of a traced computation — either a compiled subgraph or an eager fallback.
pub enum GraphSegment<T: Element> {
    /// A compiled IR subgraph that runs through the codegen backend.
    Compiled(CompiledGraph<T>),
    /// An eager fallback for a single unsupported operation.
    /// The closure captures the original op and executes it in eager mode.
    Eager(Box<dyn Fn(&[Tensor<T>]) -> Result<Vec<Tensor<T>>, FerrotorchError> + Send + Sync>),
}

/// A traced module that may contain graph breaks.
pub struct SegmentedModule<T: Element> {
    segments: Vec<GraphSegment<T>>,
    /// Dataflow connections between segments: which outputs of segment N
    /// feed into which inputs of segment N+1.
    connections: Vec<SegmentConnection>,
    parameters: HashMap<String, Tensor<T>>,
}
```

When `CompileConfig::fullgraph` is `true`, graph breaks are treated as errors — the tracer returns `Err(JitError::GraphBreak { op, reason })` instead of inserting a fallback segment. This is useful for deployment scenarios where partial eager execution is unacceptable (e.g., `export` mode).

Graph break detection works by extending the `TraceRecorder` with an `is_supported(op: &IrOpKind) -> bool` check. When an unsupported op is encountered:
1. The current subgraph is finalized and added to the segment list as `GraphSegment::Compiled`.
2. The unsupported op is wrapped in a `GraphSegment::Eager` closure.
3. A new subgraph recording begins for subsequent operations.
4. Dataflow connections are recorded so that tensor values flow correctly between segments.

### Export Mode

`jit::export` produces a fully self-contained serialized artifact that can be loaded and executed without the original Rust model code, similar to `torch.export`. The exported artifact includes the optimized IR graph, all parameter values, shape metadata, and serialized constants — everything needed to reconstruct and execute the model.

```rust
/// Export a model into a self-contained binary artifact.
pub fn export<T, M>(
    model: &M,
    example_inputs: &[Tensor<T>],
    config: Option<ExportConfig>,
) -> Result<ExportedModel<T>, FerrotorchError>
where
    T: Element,
    M: Module<T>,
{
    // 1. Trace with fullgraph=true (no graph breaks allowed in export mode)
    // 2. Apply full optimization pipeline
    // 3. Serialize the IR graph, parameters, and metadata into ExportedModel
    // 4. Validate the exported model reproduces the same output as eager mode
}

pub struct ExportConfig {
    pub optimization: OptimizationConfig,
    pub target_device: Device,              // Target device for kernel selection
    pub validate: bool,                     // Run validation pass after export (default: true)
}

/// A self-contained exported model. Can be serialized to disk and loaded
/// in any process that has ferrotorch-jit, without the original model source code.
pub struct ExportedModel<T: Element> {
    graph: IrGraph<T>,
    parameters: HashMap<String, Tensor<T>>,
    input_specs: Vec<TensorSpec>,           // Expected input shapes, dtypes, devices
    output_specs: Vec<TensorSpec>,          // Output shape/dtype metadata
    metadata: ExportMetadata,
    _marker: PhantomData<T>,
}

pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
}

pub struct ExportMetadata {
    pub ferrotorch_version: String,
    pub export_timestamp: u64,
    pub source_model_name: Option<String>,
}

impl<T: Element> ExportedModel<T> {
    /// Serialize the exported model to bytes (MessagePack format).
    pub fn save(&self, path: &std::path::Path) -> Result<(), FerrotorchError> { ... }

    /// Deserialize an exported model from bytes.
    pub fn load(path: &std::path::Path) -> Result<Self, FerrotorchError> { ... }

    /// Load and compile the exported model for execution.
    pub fn into_module(self) -> Result<TracedModule<T>, FerrotorchError> {
        // Compile the stored IR graph and wrap in a TracedModule
    }

    /// Execute the exported model directly (compiles on first call).
    pub fn forward(&self, inputs: &[Tensor<T>]) -> Result<Tensor<T>, FerrotorchError> { ... }

    /// Validate that the exported model's IR is well-formed: all node inputs
    /// are defined, shapes are consistent, and no unsupported ops are present.
    pub fn validate(&self) -> Result<(), FerrotorchError> { ... }
}
```

Usage:
```rust
// Export
let exported = jit::export(&model, &[example], None)?;
exported.save(Path::new("model.ftm"))?;

// Load and run (no original model code needed)
let loaded = ExportedModel::<f32>::load(Path::new("model.ftm"))?;
let module = loaded.into_module()?;
let output = module.forward(&input)?;
```

The `.ftm` (ferrotorch model) file format is a MessagePack-encoded blob containing the IR graph, parameter tensors, and metadata. Export mode enforces `fullgraph=true` — graph breaks are not permitted because eager fallback segments cannot be serialized (they contain Rust closures). If tracing encounters an unsupported op during export, it returns `Err(JitError::ExportError { op, reason })`.

## Out of Scope
- Scripting (source-level analysis or AST transformation) — ferrotorch-jit is tracing-only, not a Rust-to-IR compiler
- Custom operator registration for the JIT — third-party ops must go through the standard `GradFn<T>` trait and are automatically captured during tracing
- Quantization-aware tracing (int8/int4 inference) — this belongs in a future ferrotorch-quantize crate
- Multi-device graph partitioning (splitting a single traced graph across CPU and GPU) — ferrotorch-distributed handles device placement
- ONNX export from IR — ferrotorch-serialize already handles ONNX; the JIT IR is an internal representation
- Metal and Vulkan codegen backends — these can be added later via the `Codegen` trait but are not in Phase 8's scope
- Automatic differentiation of the compiled code itself — backward support works by tracing the backward pass into a separate IR graph during the original trace, not by differentiating generated machine code
