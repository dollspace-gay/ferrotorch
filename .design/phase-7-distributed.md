# Feature: Phase 7 — Distributed Training (ferrotorch-distributed)

## Summary
A multi-GPU and multi-node distributed training crate for ferrotorch, providing communication primitives, process group management, and high-level parallelism strategies. Wraps ferrotorch-nn modules with automatic gradient synchronization (DDP), parameter sharding (FSDP), pipeline parallelism (GPipe-style microbatching), and tensor parallelism (Megatron-style column/row splitting). Communication is abstracted behind a `Backend` trait with NCCL (GPU), Gloo (CPU), and MPI implementations, allowing users to scale training from a single machine with multiple GPUs to clusters of hundreds of nodes without changing model code.

## Requirements
- REQ-1: A `Backend` trait must abstract inter-process communication, with concrete implementations for NCCL (GPU-to-GPU via NVLink/PCIe/InfiniBand), Gloo (CPU via FFI to the C++ Gloo library, with a pure-Rust TCP fallback for testing), and MPI (via system MPI library). Each backend must support the full set of collective operations defined in REQ-3. Backend selection must auto-detect based on device availability (NCCL for CUDA tensors, Gloo for CPU tensors) when no backend is specified, but allow explicit override via `init_process_group(backend: Option<Backend>)`.
- REQ-2: A `ProcessGroup` must manage a set of ranks participating in collective communication. The default group spans all ranks. Users must be able to create sub-groups (e.g., for tensor-parallel communication within a node vs. data-parallel communication across nodes). Each rank has a unique integer ID within its group, and the group tracks the total world size. Process groups must be `Send + Sync` and reference-counted (`Arc`).
- REQ-3: The following collective operations must be implemented for all backends: `allreduce` (sum, mean, min, max), `broadcast` (root to all), `allgather` (concatenate from all ranks), `reduce_scatter` (reduce then scatter), `barrier` (synchronization fence), and point-to-point `send`/`recv`. All collectives must operate on `Tensor<T>` directly and return `Result<(), FerrotorchError>`. Async variants must return a `Work` handle that can be `wait()`-ed.
- REQ-4: `DistributedDataParallel<M>` must wrap any `Module<T>` and synchronize gradients across ranks after each backward pass via allreduce on the default process group. It must bucket small parameter gradients into larger buffers before allreduce to amortize communication latency (matching PyTorch's gradient bucketing strategy). Forward calls must broadcast parameters from rank 0 on the first iteration to ensure all ranks start with identical weights.
- REQ-5: `FullyShardedDataParallel<M>` must shard model parameters, gradients, and optimizer states across ranks in the process group, materializing full parameters only during forward and backward via allgather, then re-sharding via reduce_scatter after backward. It must support configurable sharding strategies: full shard (ZeRO-3 equivalent, parameters + gradients + optimizer state), grad-only shard (ZeRO-2 equivalent, gradients + optimizer state), and no shard (DDP fallback). Mixed-precision with f32 parameter copies for the optimizer and bf16/f16 for computation must be supported. FSDP must be compatible with gradient checkpointing: when a checkpointed layer recomputes its forward during backward, FSDP must detect this and re-allgather parameters for the recomputed forward pass, prefetching the next layer's shards to overlap communication with recomputation.
- REQ-6: Pipeline parallelism must partition a `Sequential` module across devices (one stage per device group) and execute forward/backward using GPipe-style microbatching to fill the pipeline. The number of microbatches must be configurable. The implementation must handle the 1F1B (one-forward-one-backward) schedule to reduce peak memory compared to naive GPipe fill-drain.
- REQ-7: Tensor parallelism must provide `ColumnParallelLinear`, `RowParallelLinear`, `ParallelEmbedding`, and `ParallelAttention` modules. `ColumnParallelLinear` and `RowParallelLinear` split weight matrices across ranks along the output and input dimensions respectively. Column-parallel gathers outputs via allgather after forward; row-parallel reduces inputs via reduce_scatter. `ParallelEmbedding` partitions the embedding table rows across ranks, with each rank holding `num_embeddings / world_size` rows. `ParallelAttention` distributes attention heads across ranks (each rank computes `num_heads / world_size` heads), handling Q/K/V column-parallel projections and row-parallel output projection internally. These modules must be drop-in replacements for their non-parallel counterparts in transformer architectures.
- REQ-8: All public functions must return `Result<T, FerrotorchError>`. Failures in communication (rank timeout, NCCL error, connection refused) must produce descriptive errors including the failing rank, operation, and backend. No panics on communication failure.
- REQ-9: An `init_process_group` entry point must initialize the distributed runtime from environment variables (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) or explicit configuration, matching PyTorch's `torch.distributed.init_process_group` interface. The `backend` parameter must be `Option<BackendKind>`: `None` auto-detects (NCCL if CUDA devices are available, Gloo otherwise), `Some(kind)` forces a specific backend. Cleanup must happen via `Drop` on the process group, releasing backend resources.
- REQ-10: A `ferrotorch-run` CLI binary must be provided as the canonical process launcher, equivalent to PyTorch's `torchrun`. It must spawn `--nproc-per-node` worker processes on the local node, setting `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` environment variables for each. For multi-node training, it must accept `--nnodes`, `--node-rank`, and `--rdzv-endpoint` arguments. Single-node usage must work with zero configuration beyond specifying the number of processes and the training script.
- REQ-11: Elastic training must support dynamic worker join and leave during training. A rendezvous mechanism (backed by `etcd` or a built-in TCP store) must allow workers to register, detect membership changes, and trigger re-rendezvous. When workers join or leave, the runtime must re-form process groups, re-shard data, and resume training from the last checkpoint. `DistributedDataParallel` and `FullyShardedDataParallel` must handle world size changes by re-bucketing gradients and re-sharding parameters respectively. Minimum and maximum worker counts must be configurable.
- REQ-12: `DistributedDataParallel` must overlap gradient allreduce with backward computation by default (communication overlap). As each gradient bucket is filled during backward, its async allreduce must be issued immediately while backward continues computing gradients for earlier layers. This overlap must be enabled by default and configurable via a `overlap_comm: bool` parameter on the DDP builder.
- REQ-13: A monitoring subsystem must expose per-rank metrics: training throughput (samples/sec, tokens/sec), communication time per collective (allreduce, allgather, reduce_scatter), communication-to-computation ratio, peak and current memory usage per rank, and gradient norm statistics. Metrics must be accessible programmatically via a `DistributedMonitor` struct and optionally exported to a log file or socket for external monitoring tools. Metrics collection must have negligible overhead when disabled and less than 2% overhead when enabled.

## Acceptance Criteria
- [ ] AC-1: `init_process_group(backend, rank, world_size, master_addr, master_port)` initializes a process group. All ranks can execute `barrier()` and return successfully. Destroying the group (via `Drop`) releases all backend resources without leaks.
- [ ] AC-2: `allreduce` on a `Tensor<f32>` across 4 ranks produces the correct sum (verified numerically). `broadcast` from rank 0 results in identical tensors on all ranks. `allgather` concatenates rank-local tensors into the correct global tensor. `reduce_scatter` produces correct sharded results. All collectives tested for both NCCL and Gloo backends.
- [ ] AC-3: `DistributedDataParallel` wrapping a 3-layer MLP produces identical gradients on all ranks after backward (within f32 tolerance), and parameter updates are identical after optimizer step. Training loss curves match single-GPU training within statistical noise over 100 iterations.
- [ ] AC-4: `DistributedDataParallel` gradient bucketing reduces the number of allreduce calls — a model with 50 small parameters (each under 1KB) issues fewer than 10 allreduce calls per backward pass, not 50.
- [ ] AC-5: `FullyShardedDataParallel` with full sharding reduces per-rank memory usage by at least 60% compared to DDP for a 100M-parameter model across 4 ranks, while producing numerically identical training results.
- [ ] AC-6: Pipeline parallelism with 4 stages and 8 microbatches trains a 4-segment Sequential model across 4 devices. The 1F1B schedule achieves at least 70% pipeline utilization (measured as compute time / wall time per rank) and produces correct gradients verified against single-device training on the same model.
- [ ] AC-7: `ColumnParallelLinear` and `RowParallelLinear` produce numerically identical outputs to a standard `Linear` layer when gathered across ranks. `ParallelEmbedding` produces identical outputs to a standard `Embedding` layer. `ParallelAttention` distributing 8 heads across 4 ranks (2 heads per rank) produces the same output as single-rank 8-head attention (within f32 tolerance). A transformer block using tensor-parallel attention and MLP produces the same logits as a single-rank transformer block (within f32 tolerance).
- [ ] AC-8: All public functions return `Result`. Attempting `allreduce` on a destroyed process group returns `Err(FerrotorchError::DistributedError { .. })`. Timeout on a hung rank returns an error with the rank ID and operation name within the configured timeout window.
- [ ] AC-9: `FullyShardedDataParallel` with `FullShard` strategy combined with gradient checkpointing correctly recomputes forward passes during backward by re-allgathering parameters for checkpointed layers. Training a 4-layer checkpointed model with FSDP across 4 ranks produces numerically identical results to FSDP without checkpointing, while reducing peak activation memory.
- [ ] AC-10: `ferrotorch-run --nproc-per-node 4 train.rs` spawns 4 worker processes with correct `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` environment variables. All workers initialize successfully and complete a barrier. For multi-node, `ferrotorch-run --nnodes 2 --node-rank 0 --rdzv-endpoint host:port --nproc-per-node 4 train.rs` spawns 4 local workers that rendezvous with workers on the other node.
- [ ] AC-11: Elastic training: starting with 4 workers, removing 1 worker triggers re-rendezvous and training continues with 3 workers without crashing. Adding a 5th worker triggers re-rendezvous and training continues with 5 workers. DDP re-buckets gradients and FSDP re-shards parameters after each membership change. A minimum of 2 workers and maximum of 8 workers is enforced by configuration.
- [ ] AC-12: `DistributedDataParallel` with `overlap_comm: true` (the default) overlaps allreduce communication with backward computation. Measured wall time for backward + allreduce is at least 15% less than sequential backward followed by allreduce on a model with 4+ gradient buckets across 4 ranks.
- [ ] AC-13: `DistributedMonitor` reports per-rank metrics: throughput (samples/sec), total communication time, total computation time, communication/computation ratio, and peak memory. Metrics are accurate within 5% of manually instrumented measurements. Enabling monitoring adds less than 2% overhead to total training time. Metrics can be retrieved programmatically and written to a JSON log file.
- [ ] AC-14: `cargo test -p ferrotorch-distributed` passes with 0 failures. Multi-rank tests use `std::process::Command` to spawn child processes simulating ranks (no external launcher required for testing). Minimum 120 tests covering all collectives, DDP, FSDP, pipeline, tensor parallelism, elastic training, communication overlap, launcher, and monitoring.

## Architecture

### Crate Layout

```
ferrotorch-distributed/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports, init_process_group
│   ├── backend/
│   │   ├── mod.rs                # Backend trait definition, auto-detection logic
│   │   ├── nccl.rs               # NCCL backend (GPU, via nccl-rs bindings)
│   │   ├── gloo.rs               # Gloo backend (CPU, FFI to C++ Gloo library)
│   │   ├── tcp_fallback.rs       # Pure-Rust TCP fallback for testing without C++ Gloo
│   │   └── mpi.rs                # MPI backend (via rsmpi bindings)
│   ├── process_group.rs          # ProcessGroup, sub-group creation, rank/world_size
│   ├── collective.rs             # allreduce, broadcast, allgather, reduce_scatter, barrier, send, recv
│   ├── ddp.rs                    # DistributedDataParallel<M> wrapper with comm overlap
│   ├── fsdp.rs                   # FullyShardedDataParallel<M> wrapper, checkpoint-aware
│   ├── pipeline.rs               # PipelineParallel, GPipe schedule, 1F1B schedule
│   ├── tensor_parallel.rs        # ColumnParallelLinear, RowParallelLinear, ParallelEmbedding, ParallelAttention
│   ├── elastic.rs                # Elastic training: rendezvous, membership changes, re-sharding
│   └── monitor.rs                # DistributedMonitor: throughput, comm time, memory stats
├── ferrotorch-run/
│   ├── Cargo.toml                # Binary crate for the process launcher
│   └── src/
│       └── main.rs               # CLI: spawn workers, set env vars, elastic rendezvous
└── tests/
    ├── test_process_group.rs     # Init, sub-groups, cleanup, auto-detection
    ├── test_collectives.rs       # allreduce, broadcast, allgather, reduce_scatter, barrier
    ├── test_ddp.rs               # Gradient sync, bucketing, convergence, comm overlap
    ├── test_fsdp.rs              # Sharding, memory, numerical equivalence, checkpoint compat
    ├── test_pipeline.rs          # Microbatch scheduling, utilization, correctness
    ├── test_tensor_parallel.rs   # Column/row parallel, parallel embedding/attention, transformer integration
    ├── test_elastic.rs           # Worker join/leave, re-rendezvous, re-sharding
    ├── test_launcher.rs          # ferrotorch-run spawning, env var correctness
    └── test_monitor.rs           # Metrics accuracy, overhead measurement
```

### Backend Trait (`backend/mod.rs`)

```rust
/// Reduction operations for collective communication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Min,
    Max,
}

/// Handle for an in-flight async collective operation.
pub trait Work: Send + Sync {
    fn wait(&self) -> Result<(), FerrotorchError>;
    fn is_completed(&self) -> bool;
}

/// Communication backend abstraction.
pub trait Backend: Send + Sync {
    fn allreduce(&self, tensor: &mut Tensor<f32>, op: ReduceOp, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn allreduce_async(&self, tensor: &mut Tensor<f32>, op: ReduceOp, group: &ProcessGroup)
        -> Result<Box<dyn Work>, FerrotorchError>;

    fn broadcast(&self, tensor: &mut Tensor<f32>, src: usize, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn allgather(&self, output: &mut Tensor<f32>, input: &Tensor<f32>, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn reduce_scatter(&self, output: &mut Tensor<f32>, input: &Tensor<f32>, op: ReduceOp, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn barrier(&self, group: &ProcessGroup) -> Result<(), FerrotorchError>;

    fn send(&self, tensor: &Tensor<f32>, dst: usize, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn recv(&self, tensor: &mut Tensor<f32>, src: usize, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn backend_name(&self) -> &'static str;
}
```

The trait is defined over `Tensor<f32>` in the signatures above for clarity, but the actual implementation is generic over `T: Element` using a sealed helper trait `DistElement` that restricts to types the backend can transmit (f32, f64, bf16, f16, i32, i64).

### NCCL Backend (`backend/nccl.rs`)

Wraps the NCCL library via Rust bindings (either `cudarc`'s NCCL support or a dedicated `nccl-rs` crate). Each `NcclBackend` instance holds a `ncclComm_t` communicator per process group. Tensors must reside on `Device::Cuda` — passing a CPU tensor returns `Err(FerrotorchError::DeviceMismatch)`. All operations are launched on the CUDA stream associated with the tensor's device, enabling overlap with compute kernels.

### Gloo Backend (`backend/gloo.rs`)

Wraps Facebook's C++ Gloo library via FFI bindings, matching PyTorch's approach. The C++ Gloo library provides highly optimized collectives with TCP, shared-memory, and RDMA transports. Rust FFI bindings are generated via `cc` + `bindgen` linking against the system-installed or vendored `libgloo`. Each `GlooBackend` instance holds a `gloo::Context` per process group. Tensors must reside on `Device::Cpu` — passing a GPU tensor returns `Err(FerrotorchError::DeviceMismatch)`.

For testing without C++ Gloo installed (and for lightweight CI environments), a pure-Rust TCP fallback backend (`TcpBackend`) is provided. This fallback uses `std::net::TcpStream` connections between ranks and implements allreduce via ring-reduce. It is not intended for production use but ensures distributed tests can run anywhere. The fallback is selected automatically when the `gloo-ffi` feature is disabled.

Shared-memory transport (via C++ Gloo) is used when ranks detect they are on the same host (comparing `MASTER_ADDR` against local interfaces), avoiding TCP overhead for intra-node communication.

### MPI Backend (`backend/mpi.rs`)

Wraps the system MPI library via `rsmpi`. Delegates collectives to `MPI_Allreduce`, `MPI_Bcast`, `MPI_Allgather`, etc. Intended for HPC clusters where MPI is the standard communication fabric. The MPI environment must be initialized before `init_process_group` (via `mpi::initialize()` from rsmpi). Feature-gated behind `features = ["mpi"]` since rsmpi requires a system MPI installation.

### Process Group (`process_group.rs`)

```rust
pub struct ProcessGroup {
    rank: usize,
    world_size: usize,
    ranks: Vec<usize>,              // Global rank IDs in this group
    backend: Arc<dyn Backend>,
    sub_groups: Vec<Arc<ProcessGroup>>,
}

impl ProcessGroup {
    /// Create a sub-group from a subset of ranks.
    pub fn new_group(&self, ranks: &[usize]) -> Result<Arc<ProcessGroup>, FerrotorchError>;

    /// This rank's ID within the group.
    pub fn rank(&self) -> usize;

    /// Total number of ranks in the group.
    pub fn world_size(&self) -> usize;
}
```

Sub-groups are created by calling `new_group` with a subset of global ranks. The backend creates a new communicator scoped to those ranks (e.g., `ncclCommSplit` for NCCL, `MPI_Comm_create_group` for MPI). Sub-groups enable hybrid parallelism: a tensor-parallel group within each node (e.g., ranks [0,1,2,3]) and a data-parallel group across nodes (e.g., ranks [0,4,8,12]).

### Collective Operations (`collective.rs`)

Module-level functions that dispatch to the backend attached to the given process group:

```rust
pub fn allreduce(tensor: &mut Tensor<f32>, op: ReduceOp, group: &ProcessGroup)
    -> Result<(), FerrotorchError>
{
    group.backend.allreduce(tensor, op, group)
}
```

Async variants return a `Box<dyn Work>` handle. The caller must `wait()` before reading the tensor. This enables overlapping communication with computation — DDP issues async allreduce on one gradient bucket while backward is still computing gradients for the next bucket.

### DistributedDataParallel (`ddp.rs`)

```rust
pub struct DistributedDataParallel<M: Module<f32>> {
    module: M,
    process_group: Arc<ProcessGroup>,
    buckets: Vec<GradientBucket>,
    bucket_size_mb: f64,
}

struct GradientBucket {
    params: Vec<usize>,             // Indices into module.parameters()
    buffer: Tensor<f32>,            // Flat buffer for allreduce
    work: Option<Box<dyn Work>>,    // In-flight async allreduce handle
}
```

**Initialization**: Parameters are assigned to buckets in reverse declaration order (matching PyTorch's strategy — parameters used later in forward are communicated first in backward). Bucket boundaries are placed when cumulative parameter size exceeds `bucket_size_mb` (default: 25 MB). On the first forward call, rank 0's parameters are broadcast to all ranks to ensure identical starting weights.

**Backward hook**: After each parameter's gradient is computed during `backward()`, the DDP wrapper checks whether the gradient's bucket is full. If so, it flattens all gradients in the bucket into the contiguous buffer and issues an async allreduce. This overlaps communication of early buckets with gradient computation for later layers.

**Synchronization**: Before `optimizer.step()`, DDP calls `wait()` on all in-flight allreduce handles, then copies the averaged gradients from bucket buffers back to each parameter's `.grad`. The optimizer then steps on the synchronized gradients.

### FullyShardedDataParallel (`fsdp.rs`)

```rust
pub enum ShardingStrategy {
    FullShard,      // ZeRO-3: shard params + grads + optimizer state
    GradOnlyShard,  // ZeRO-2: shard grads + optimizer state, replicate params
    NoShard,        // DDP-equivalent: replicate everything
}

pub struct FullyShardedDataParallel<M: Module<f32>> {
    module: M,
    process_group: Arc<ProcessGroup>,
    strategy: ShardingStrategy,
    sharded_params: Vec<ShardedParameter>,
    mixed_precision: Option<MixedPrecisionConfig>,
}

struct ShardedParameter {
    local_shard: Tensor<f32>,       // This rank's slice of the parameter
    full_param: Option<Tensor<f32>>,// Materialized during forward/backward, then dropped
    shard_offsets: (usize, usize),  // Start/end indices in the flat parameter
}

pub struct MixedPrecisionConfig {
    param_dtype: DType,             // Storage dtype (f32)
    compute_dtype: DType,           // Forward/backward dtype (bf16 or f16)
    reduce_dtype: DType,            // Communication dtype (f32 for accuracy)
}
```

**Forward**: Before the wrapped module's forward, FSDP calls `allgather` to materialize full parameters from shards across ranks. The full parameter tensors are attached to the module for the duration of forward.

**Backward**: Full parameters are re-materialized via `allgather` (if they were freed after forward). After gradient computation, `reduce_scatter` distributes gradient shards — each rank keeps only its shard of the gradient. Full parameter tensors are freed immediately after use.

**Memory lifecycle**: In `FullShard` mode, each rank holds only `1/world_size` of the parameters at rest. Peak memory during forward/backward includes one full parameter set at a time (the layer currently executing). FSDP processes layers one at a time, prefetching the next layer's allgather while the current layer executes.

### Pipeline Parallelism (`pipeline.rs`)

```rust
pub struct PipelineParallel<T: Element = f32> {
    stages: Vec<PipelineStage<T>>,
    num_microbatches: usize,
    schedule: PipelineSchedule,
    process_group: Arc<ProcessGroup>,
}

pub enum PipelineSchedule {
    /// Fill all microbatches forward, then drain all backward. Simple but high memory.
    FillDrain,
    /// Interleave forward and backward passes. Lower memory, better utilization.
    OneFOneBSchedule,
}

struct PipelineStage<T: Element> {
    module: Box<dyn Module<T>>,
    device: Device,
    rank: usize,
}
```

**FillDrain (GPipe)**: All microbatches execute forward through all stages, then all microbatches execute backward. Peak memory is proportional to `num_microbatches` since all intermediate activations are held simultaneously.

**1F1B schedule**: After the pipeline fills (first `num_stages` microbatches in forward), each rank alternates one forward and one backward pass. This limits peak activation memory to `num_stages` microbatches rather than `num_microbatches`, at the cost of slightly more complex scheduling logic.

**Inter-stage communication**: Stage `i` sends its output tensor to stage `i+1` via point-to-point `send`/`recv`. During backward, stage `i+1` sends the gradient of its input back to stage `i`. Each stage runs on its assigned device, so send/recv crosses device boundaries (GPU-to-GPU via NCCL or CPU-to-CPU via Gloo).

### Tensor Parallelism (`tensor_parallel.rs`)

```rust
pub struct ColumnParallelLinear<T: Element = f32> {
    weight_shard: Parameter<T>,     // Shape: [out_features / world_size, in_features]
    bias_shard: Option<Parameter<T>>,
    process_group: Arc<ProcessGroup>,
    gather_output: bool,
}

pub struct RowParallelLinear<T: Element = f32> {
    weight_shard: Parameter<T>,     // Shape: [out_features, in_features / world_size]
    bias: Option<Parameter<T>>,     // Only rank 0 holds bias (added after reduce)
    process_group: Arc<ProcessGroup>,
    input_is_parallel: bool,
}

pub struct ParallelEmbedding<T: Element = f32> {
    embedding_shard: Parameter<T>,  // Shape: [num_embeddings / world_size, embedding_dim]
    num_embeddings: usize,          // Total vocabulary size
    embedding_dim: usize,
    shard_start: usize,             // First row index owned by this rank
    shard_end: usize,               // Last row index (exclusive) owned by this rank
    process_group: Arc<ProcessGroup>,
}

pub struct ParallelAttention<T: Element = f32> {
    q_proj: ColumnParallelLinear<T>,    // Q projection, split across heads
    k_proj: ColumnParallelLinear<T>,    // K projection, split across heads
    v_proj: ColumnParallelLinear<T>,    // V projection, split across heads
    out_proj: RowParallelLinear<T>,     // Output projection, reduce across ranks
    num_heads: usize,                   // Total number of attention heads
    num_local_heads: usize,             // Heads on this rank (num_heads / world_size)
    head_dim: usize,
    process_group: Arc<ProcessGroup>,
}
```

**ColumnParallelLinear**: Splits the weight along the output dimension. Each rank computes `Y_local = X @ W_local^T`. If `gather_output` is true, an allgather concatenates `Y_local` across ranks to produce the full output. In transformer architectures, `gather_output` is false for the first linear in a paired column+row sequence — the partial outputs feed directly into `RowParallelLinear`.

**RowParallelLinear**: Splits the weight along the input dimension. Each rank computes `Y_local = X_local @ W_local^T` where `X_local` is the rank-local slice of the input. A reduce_scatter (or allreduce) combines partial sums across ranks. When `input_is_parallel` is true (typical in a transformer), the input is already split across ranks from a preceding `ColumnParallelLinear`.

**ParallelEmbedding**: Partitions the embedding table across ranks along the vocabulary dimension. Each rank holds rows `[shard_start, shard_end)`. For input token IDs that fall within the local shard, the rank performs a local lookup; for IDs outside the shard, the rank produces zeros. An allreduce (sum) across ranks yields the correct embedding for every token. This avoids replicating large embedding tables (which can be gigabytes for large vocabularies).

**ParallelAttention**: Encapsulates the full Megatron-style parallel attention pattern as a single module. Q/K/V projections use `ColumnParallelLinear` (splitting `num_heads` across ranks, so each rank computes `num_heads / world_size` heads). Attention computation is local to each rank's head subset. The output projection uses `RowParallelLinear` to reduce across ranks. This requires exactly 1 allreduce per attention block.

**Transformer pattern**: A typical Megatron-style transformer block uses:
1. `ParallelAttention` (or manually: `ColumnParallelLinear` for Q/K/V + local attention + `RowParallelLinear` for output projection)
2. `ColumnParallelLinear` for MLP up-projection
3. `RowParallelLinear` for MLP down-projection

This requires only 2 allreduce operations per transformer block (one after attention output projection, one after MLP down-projection) regardless of the number of ranks.

### Initialization (`lib.rs`)

```rust
pub fn init_process_group(
    backend: Option<BackendKind>,
    rank: usize,
    world_size: usize,
    master_addr: &str,
    master_port: u16,
) -> Result<Arc<ProcessGroup>, FerrotorchError>;

pub fn init_process_group_from_env(
    backend: Option<BackendKind>,
) -> Result<Arc<ProcessGroup>, FerrotorchError>;

#[derive(Clone, Copy, Debug)]
pub enum BackendKind {
    Nccl,
    Gloo,
    Mpi,
}
```

When `backend` is `None`, the runtime auto-detects the best backend: NCCL if CUDA devices are available (checked via `cudarc::driver::CudaDevice::count()`), Gloo (C++ FFI) if the `gloo-ffi` feature is enabled, or the pure-Rust TCP fallback otherwise. When `backend` is `Some(kind)`, the specified backend is used directly, returning `Err(BackendUnavailable)` if prerequisites are not met (e.g., `Some(Nccl)` without CUDA). This matches PyTorch's `init_process_group(backend=None)` behavior.

`init_process_group_from_env` reads `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` from environment variables, matching PyTorch's `torchrun` / `ferrotorch-run` conventions. This allows ferrotorch distributed programs to be launched with `ferrotorch-run`, `torchrun`, `mpirun`, or SLURM `srun`.

### Error Handling

The `FerrotorchError` enum (defined in ferrotorch-core) is extended with distributed variants:

```rust
#[error("distributed error on rank {rank}, operation {operation}: {message}")]
DistributedError { rank: usize, operation: String, message: String },

#[error("rank {rank} timed out after {timeout_secs}s during {operation}")]
DistributedTimeout { rank: usize, timeout_secs: u64, operation: String },

#[error("backend {backend} not available: {reason}")]
BackendUnavailable { backend: String, reason: String },
```

### Communication Overlap (`ddp.rs`)

DDP overlaps gradient allreduce with backward computation by default, matching PyTorch's behavior. The mechanism works as follows:

1. During backward, gradient computation proceeds layer by layer from output to input.
2. As each parameter's gradient is computed, DDP checks whether the gradient's bucket is full.
3. When a bucket is full, DDP immediately issues an **async** allreduce on that bucket while backward continues computing gradients for earlier layers.
4. By the time backward completes, most or all allreduce operations have already finished or are nearly finished.

```rust
impl<M: Module<f32>> DistributedDataParallel<M> {
    pub fn builder(module: M, process_group: Arc<ProcessGroup>) -> DdpBuilder<M>;
}

pub struct DdpBuilder<M: Module<f32>> {
    module: M,
    process_group: Arc<ProcessGroup>,
    bucket_size_mb: f64,            // Default: 25.0
    overlap_comm: bool,             // Default: true
    broadcast_buffers: bool,        // Default: true
    find_unused_parameters: bool,   // Default: false
}
```

When `overlap_comm` is `false`, DDP waits until the entire backward pass completes, then issues allreduce calls sequentially. This is useful for debugging but slower in production. When `find_unused_parameters` is `true`, DDP marks parameters that did not receive gradients in the current iteration and excludes them from allreduce (necessary for models with conditional execution paths).

### Elastic Training (`elastic.rs`)

Elastic training allows workers to join or leave during training without stopping the job. This is critical for cloud environments where spot instances can be preempted and for fault tolerance.

```rust
pub struct ElasticConfig {
    min_workers: usize,             // Minimum workers to continue training
    max_workers: usize,             // Maximum workers allowed
    rdzv_backend: RendezvousBackend,// etcd, tcp-store, or file-based
    rdzv_endpoint: String,          // e.g., "etcd-host:2379" or "master:29400"
    rdzv_timeout: Duration,         // How long to wait for re-rendezvous
    max_restarts: usize,            // Max consecutive restart attempts per worker
}

pub enum RendezvousBackend {
    /// etcd-based rendezvous (production, supports multi-node)
    Etcd,
    /// Built-in TCP store (single-node or simple multi-node)
    TcpStore,
    /// File-based rendezvous (shared filesystem required)
    FileStore { path: PathBuf },
}

pub struct ElasticManager {
    config: ElasticConfig,
    current_group: Arc<ProcessGroup>,
    membership_version: AtomicU64,
}

impl ElasticManager {
    /// Block until enough workers have joined (>= min_workers).
    pub fn initial_rendezvous(&mut self) -> Result<Arc<ProcessGroup>, FerrotorchError>;

    /// Called when a membership change is detected. Re-forms the process group.
    pub fn re_rendezvous(&mut self) -> Result<Arc<ProcessGroup>, FerrotorchError>;

    /// Register a callback invoked when membership changes.
    pub fn on_membership_change(&self, callback: Box<dyn Fn(usize, usize) + Send + Sync>);
}
```

**Membership change protocol**: When a worker detects a peer failure (allreduce timeout, connection reset), it initiates a re-rendezvous. All surviving workers synchronize at the rendezvous point, form a new process group with the updated world size, and resume training. DDP re-buckets gradients for the new world size. FSDP re-shards parameters and optimizer states. The data loader re-partitions the dataset across the new worker count. Training resumes from the last saved checkpoint (elastic training requires periodic checkpointing).

### Monitoring (`monitor.rs`)

```rust
pub struct DistributedMonitor {
    rank: usize,
    world_size: usize,
    enabled: bool,
    metrics: Arc<Mutex<RankMetrics>>,
}

pub struct RankMetrics {
    // Throughput
    pub samples_processed: u64,
    pub tokens_processed: u64,
    pub elapsed: Duration,

    // Communication
    pub comm_time_allreduce: Duration,
    pub comm_time_allgather: Duration,
    pub comm_time_reduce_scatter: Duration,
    pub comm_time_broadcast: Duration,
    pub comm_time_barrier: Duration,
    pub comm_bytes_sent: u64,
    pub comm_bytes_recv: u64,

    // Computation
    pub compute_time_forward: Duration,
    pub compute_time_backward: Duration,

    // Memory
    pub peak_memory_bytes: u64,
    pub current_memory_bytes: u64,
    pub gradient_memory_bytes: u64,

    // Gradient statistics
    pub gradient_norm_l2: f64,
    pub gradient_norm_inf: f64,
}

impl DistributedMonitor {
    pub fn new(rank: usize, world_size: usize) -> Self;
    pub fn enable(&mut self);
    pub fn disable(&mut self);
    pub fn snapshot(&self) -> RankMetrics;
    pub fn reset(&mut self);
    pub fn comm_computation_ratio(&self) -> f64;
    pub fn throughput_samples_per_sec(&self) -> f64;
    pub fn throughput_tokens_per_sec(&self) -> f64;
    pub fn export_json(&self, path: &Path) -> Result<(), FerrotorchError>;
    pub fn export_to_socket(&self, addr: &str) -> Result<(), FerrotorchError>;
}
```

The monitor wraps backend calls with timing instrumentation. When enabled, each collective operation records its start/end time via `std::time::Instant`. Memory stats are queried from the allocator (system allocator stats on CPU, `cudaMemGetInfo` on GPU). Metrics are per-rank and can be aggregated across ranks via an allreduce of the metrics themselves for global summaries. The overhead is minimal: one `Instant::now()` call before and after each collective, stored in lock-free counters.

### Process Launcher (`ferrotorch-run/`)

The `ferrotorch-run` binary is a standalone CLI tool that spawns distributed training workers, equivalent to PyTorch's `torchrun`.

```
USAGE:
    ferrotorch-run [OPTIONS] <SCRIPT> [-- <SCRIPT_ARGS>...]

OPTIONS:
    --nproc-per-node <N>     Number of worker processes per node [default: 1]
    --nnodes <N>             Number of nodes [default: 1]
    --node-rank <N>          Rank of this node [default: 0]
    --master-addr <ADDR>     Master address [default: 127.0.0.1]
    --master-port <PORT>     Master port [default: 29500]
    --rdzv-backend <BACKEND> Rendezvous backend: static, etcd, tcp [default: static]
    --rdzv-endpoint <EP>     Rendezvous endpoint (for etcd/tcp backends)
    --max-restarts <N>       Max worker restarts for elastic training [default: 0]
    --monitor                Enable distributed monitoring, write metrics to ./dist_metrics/
```

**Environment variables set per worker**: `RANK` (global rank), `LOCAL_RANK` (rank within this node), `WORLD_SIZE` (total workers across all nodes), `LOCAL_WORLD_SIZE` (workers on this node), `MASTER_ADDR`, `MASTER_PORT`, `GROUP_RANK` (node rank), `ROLE_RANK` (same as `LOCAL_RANK` for homogeneous setups).

**Single-node example**: `ferrotorch-run --nproc-per-node 4 train` spawns 4 processes with `RANK=0..3`, `LOCAL_RANK=0..3`, `WORLD_SIZE=4`.

**Multi-node example**: On node 0: `ferrotorch-run --nnodes 2 --node-rank 0 --nproc-per-node 4 --master-addr 10.0.0.1 train`. On node 1: `ferrotorch-run --nnodes 2 --node-rank 1 --nproc-per-node 4 --master-addr 10.0.0.1 train`. Total `WORLD_SIZE=8`, node 0 gets `RANK=0..3`, node 1 gets `RANK=4..7`.

**Elastic mode**: With `--rdzv-backend etcd --rdzv-endpoint etcd-host:2379 --max-restarts 3`, workers register with etcd for rendezvous. If a worker dies, it is restarted up to 3 times. If new workers join, the group re-rendezvouses.

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | `Tensor<T>`, `Device`, `FerrotorchError` |
| `ferrotorch-nn` | workspace | `Module<T>`, `Parameter<T>`, `Sequential` |
| `ferrotorch-gpu` | workspace | CUDA device, NCCL communicator access |
| `cudarc` | latest | NCCL bindings for GPU communication |
| `rsmpi` | latest | MPI bindings (feature-gated behind `mpi`) |
| `cc` | latest | Build C++ Gloo library (feature-gated behind `gloo-ffi`) |
| `bindgen` | latest | Generate FFI bindings for C++ Gloo (feature-gated behind `gloo-ffi`) |
| `crossbeam-channel` | 0.5 | Lock-free channels for async work notification |
| `clap` | 4 | CLI argument parsing for `ferrotorch-run` |
| `etcd-client` | latest | etcd rendezvous for elastic training (feature-gated behind `elastic-etcd`) |
| `serde` | 1 | Serialization for monitoring metrics export |
| `serde_json` | 1 | JSON export for monitoring metrics |

### Test Strategy

Multi-rank tests spawn child processes via `std::process::Command`, each running with different `RANK` environment variables. The test binary is re-invoked with a special flag (`--distributed-worker`) to enter worker mode. This avoids requiring `mpirun` or `torchrun` in the test harness.

```rust
#[test]
fn test_allreduce_4_ranks() {
    let world_size = 4;
    let handles: Vec<_> = (0..world_size).map(|rank| {
        Command::new(env::current_exe().unwrap())
            .env("RANK", rank.to_string())
            .env("WORLD_SIZE", world_size.to_string())
            .env("MASTER_ADDR", "127.0.0.1")
            .env("MASTER_PORT", "29500")
            .arg("--distributed-worker")
            .arg("allreduce")
            .spawn()
            .unwrap()
    }).collect();
    for mut h in handles { assert!(h.wait().unwrap().success()); }
}
```

NCCL tests require multiple GPUs and are gated behind `#[cfg(feature = "nccl-tests")]`. CI runs Gloo-backend tests by default.

## Resolved Questions

### Q1: Communication backend selection strategy
**Decision: Auto-detect with explicit override.** PyTorch auto-selects NCCL for CUDA tensors and Gloo for CPU tensors, but allows explicit specification. We follow the same approach: `init_process_group(backend: Option<Backend>)` where `None` auto-detects based on device availability (NCCL if CUDA devices are present, Gloo otherwise) and `Some(kind)` forces a specific backend. This matches PyTorch's `init_process_group(backend=None)` default while preserving full user control. The `backend` parameter in both `init_process_group` and `init_process_group_from_env` is `Option<BackendKind>`.

### Q2: Gloo implementation language
**Decision: FFI to C++ Gloo library with a pure-Rust TCP fallback.** PyTorch uses the C++ Gloo library directly, and rewriting its optimized shared-memory, TCP, and RDMA transports in pure Rust is massive scope with no clear benefit. We use FFI bindings (via `cc` + `bindgen`) to the C++ Gloo library, feature-gated behind `gloo-ffi`. For testing without C++ Gloo installed (lightweight CI, quick development), a pure-Rust TCP fallback backend (`TcpBackend`) implements ring-reduce over `std::net::TcpStream`. The fallback is not intended for production use but ensures distributed tests run anywhere.

### Q3: FSDP + gradient checkpointing compatibility
**Decision: Must work together, handled internally by FSDP.** PyTorch supports FSDP + gradient checkpointing natively. When a checkpointed layer recomputes its forward during backward, FSDP must detect this and re-allgather parameters for the recomputed forward pass. FSDP hooks into the checkpoint recomputation to trigger allgather before recomputed forward and prefetches the next layer's shards to overlap communication with recomputation. This is a requirement (REQ-5) and acceptance criterion (AC-9), not left to the user.

### Q4: Tensor parallelism scope
**Decision: Column/row parallel Linear + parallel Embedding + parallel attention.** PyTorch (via Megatron-LM) provides these three building blocks, which are sufficient for all transformer architectures. We provide `ColumnParallelLinear`, `RowParallelLinear` (primitives), `ParallelEmbedding` (vocabulary sharding), and `ParallelAttention` (composed module that handles Q/K/V column-parallel projections, local attention on a head subset, and row-parallel output projection). This gives users both low-level primitives and a high-level composed module.

### Q5: Launcher integration
**Decision: Provide `ferrotorch-run` binary.** PyTorch has `torchrun` (elastic launch). We provide `ferrotorch-run` as a standalone binary crate in the workspace that spawns worker processes with correct environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, etc.). It supports single-node (`--nproc-per-node`), multi-node (`--nnodes`, `--node-rank`), and elastic training (`--rdzv-backend`, `--max-restarts`). Compatibility with external launchers (`torchrun`, `mpirun`, SLURM `srun`) is maintained since we read the same environment variables.

## Out of Scope
- Model parallelism strategies beyond pipeline and tensor parallelism (e.g., expert parallelism / MoE routing) — future work
- Gradient compression (quantized allreduce, sparsification) — optimization pass after correctness is established
- Heterogeneous device training (mixing GPU and CPU ranks in the same group) — each process group is tied to one backend
- Automatic parallelism planning (deciding how to split a model across devices) — users specify the strategy explicitly
- RDMA/InfiniBand transport for the pure-Rust TCP fallback — C++ Gloo via FFI provides RDMA when available
- Cross-framework distributed communication (e.g., interop with PyTorch distributed processes) — not a goal
- Custom rendezvous backends beyond etcd, TCP store, and file store — extensible but not provided
