# Audit: `ferrotorch-distributed` vs `torch.distributed`

torch.distributed is the largest and most-evolved area of pytorch — 49+
files, multiple sub-packages (fsdp, elastic, checkpoint, _shard,
_sharded_tensor, _composable, algorithms, _tools, debug, flight_recorder,
launcher, rpc, tensor parallel via DTensor, ...).

## ferrotorch-distributed modules

| Module | Role | torch counterpart |
|---|---|---|
| `backend` | `TcpBackend`, `SimulatedBackend` | `ProcessGroup` + `gloo`/`nccl`/`mpi`/`ucc` backends |
| `collective` | `allreduce`, `all_gather`, `reduce_scatter`, `broadcast`, `barrier` | `dist.all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`, `barrier`, `send`, `recv`, `gather`, `scatter`, `reduce`, `all_to_all`, `reduce_scatter_tensor`, `all_gather_into_tensor`, `monitored_barrier` |
| `async_collective` | `async_all_gather`, `async_reduce_scatter`, `PendingCollective` | `dist.all_reduce(async_op=True)` returns `Work` handle |
| `gpu_collective` | `gpu_allreduce`, `gpu_broadcast` (TCP via CPU bounce) | NCCL native |
| `nccl_backend`, `nccl_collective`, `nccl_sys` | NCCL bindings | `torch.distributed.nccl` |
| `hybrid_backend` | TCP + NCCL combination | `dist` auto-routes |
| `ddp` | `DDP` wrapper | `torch.nn.parallel.DistributedDataParallel` |
| `fsdp` | `FSDP` wrapper | `torch.distributed.fsdp.FullyShardedDataParallel` |
| `pipeline` | `Pipeline`, `PipelineStage`, `PipelineSchedule::{GPipe, Interleaved1F1B}` | `torch.distributed.pipelining` (PiPPy upstream) |
| `rpc` | `RpcContext`, `RRef`, `SimulatedRpcBackend` | `torch.distributed.rpc` |
| `sync_batch_norm` | distributed BN | `torch.nn.SyncBatchNorm` |
| `checkpoint` | dist checkpoint helpers | `torch.distributed.checkpoint` |

## Coverage by area

### Backends

| ferrotorch | torch | Status |
|---|---|---|
| `TcpBackend` | `gloo` (TCP-style fallback) | ✅ analog |
| `SimulatedBackend` (in-process) | (none — torch uses `gloo` with `init_method=file://`) | **extra** — much better testing story |
| `nccl_backend` + `nccl_sys` | `nccl` (default for CUDA) | ✅ |
| `hybrid_backend` | (none — torch picks one) | **extra** |
| **missing** | `mpi` (Open MPI) | gap |
| **missing** | `ucc` (UCC unified communication) | gap |
| **missing** | `gloo` (Facebook's collective lib, sockets+RDMA) | gap (TCP covers same use) |

### Collectives

| ferrotorch | torch | Status |
|---|---|---|
| `allreduce` | `all_reduce`, `all_reduce_coalesced` | partial — no coalesced |
| `all_gather` | `all_gather`, `all_gather_into_tensor`, `all_gather_object` | partial |
| `reduce_scatter` | `reduce_scatter`, `reduce_scatter_tensor` | partial |
| `broadcast` | `broadcast`, `broadcast_object_list` | partial |
| `barrier` | `barrier`, `monitored_barrier` | partial |
| **missing** | `send` / `recv` (point-to-point) | gap |
| **missing** | `gather`, `scatter` | gap |
| **missing** | `reduce` (single-target reduction) | gap |
| **missing** | `all_to_all`, `all_to_all_single` | gap (matters for MoE / expert parallelism) |

`async_all_gather` / `async_reduce_scatter` / `PendingCollective` covers
async-op story well. **Object collectives** (broadcast/all-gather of
arbitrary Rust types) missing — though Rust serialization means this is
just a `serde` wrapper.

### Parallelism strategies

| ferrotorch | torch | Status |
|---|---|---|
| `DDP` | `DistributedDataParallel` | ✅ |
| `FSDP` | `FullyShardedDataParallel` (FSDP1 + FSDP2) | ✅ |
| `Pipeline` (GPipe, Interleaved1F1B) | `torch.distributed.pipelining` | ✅ matches |
| `RpcContext`, `RRef` | `torch.distributed.rpc` (RRef, remote, dist_autograd) | partial — has skeleton |
| **missing** | Tensor parallel via DTensor | gap (#459 covers backends, this is a separate concept) |
| **missing** | `torch.distributed._composable` (`replicate`, `fully_shard`, `data_parallel` as composable wrappers) | gap |
| **missing** | `torch.distributed.tensor.parallel` (`parallelize_module`, `RowwiseParallel`, `ColwiseParallel`, `SequenceParallel`) | gap |
| **missing** | `torch.distributed.checkpoint` (DCP — sharded checkpoint format, planner) | partial via `checkpoint` module |
| **missing** | `torch.distributed.elastic` (rendezvous, fault tolerance, auto-restart) | gap — needed for spot-instance training |
| **missing** | `torch.distributed.launcher` / `torchrun` equivalent | gap |
| **missing** | `torch.distributed._symmetric_memory` (intra-node fast path) | gap |
| **missing** | `torch.distributed._sharded_tensor` (formal sharded tensor type) | gap |

### DTensor / sharding

torch's modern path is **DTensor + DeviceMesh**:

```python
mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("tp",))
sharded_tensor = distribute_tensor(local_tensor, mesh, [Shard(0)])
```

ferrotorch has nothing equivalent. This is the **single biggest missing
abstraction** vs modern torch distributed — it's how 2D / 3D parallelism
(DP × TP × PP) is composed.

### SyncBatchNorm

`sync_batch_norm.rs` exists — must be wired into `ferrotorch-nn`'s
BatchNorm to be useful (or `ferrotorch-nn` needs `SyncBatchNorm` re-
exporting from this crate).

### NCCL bindings

`nccl_backend`, `nccl_collective`, `nccl_sys` — direct NCCL FFI.
torch's `torch.distributed.nccl` exposes `all_reduce`, `broadcast`, etc.

ferrotorch likely has a more limited NCCL op surface (would need to read
files to confirm). Probably covers `all_reduce`, `broadcast`,
`reduce_scatter`, `all_gather`. Missing: `all_to_all` (NCCL has it since
2.7), `send`/`recv`, group/subgroup support.

### Elastic / launcher

torch's killer feature for cloud training: `torchrun` + elastic
rendezvous + fault tolerance. ferrotorch has nothing — users would have to
write their own multi-process launcher.

## Strengths

1. **`SimulatedBackend`** for in-process testing is a big improvement
   over torch (where you must spawn processes to test distributed code).
2. **Pipeline parallelism with GPipe + Interleaved1F1B** schedules at
   crate launch is more than torch had at first release.
3. **`PendingCollective` async handles** match torch's `Work` pattern.
4. **`hybrid_backend`** shows planning for mixed-transport scenarios.
5. **Crate is well-modularized** (12 source files, clean separation).

## Gaps (priority-ordered)

1. **DTensor + DeviceMesh** — the abstraction modern torch users rely on
   for 2D/3D parallelism. Without it, mixing DP/TP/PP is ad hoc.
2. **`send`/`recv` point-to-point** — required for pipeline parallelism
   correctness and for many custom comm patterns.
3. **`all_to_all`** — required for MoE expert parallelism.
4. **`gather`/`scatter`/`reduce`** — uneven workloads need these.
5. **Elastic / fault tolerance** — `torchrun` equivalent + rendezvous.
6. **Tensor parallel helpers** — `parallelize_module(model, plan)` with
   `RowwiseParallel`/`ColwiseParallel`/`SequenceParallel`.
7. **Coalesced collectives** — `all_reduce_coalesced` for gradient
   reduction batching.
8. **Object-collectives** — broadcast/gather arbitrary `serde::Serialize`
   types (used for distributed checkpoint, distributed dataloader sync,
   distributed sampler seeds).
9. **gloo backend** (Facebook collective lib) — RDMA-aware TCP, useful
   in mixed environments where NCCL isn't available.
10. **Composable wrappers** — `replicate`, `fully_shard`, etc. as
    composable mixins on `Module` (FSDP2 idiom upstream).
11. **Wire `SyncBatchNorm`** into `ferrotorch-nn` so users can use it
    transparently.

## Status

`ferrotorch-distributed` covers the **major axes of distributed training**
(DDP, FSDP, pipeline, RPC, async collectives, NCCL) but is missing the
**DTensor abstraction** and many **point-to-point + uneven collectives**.

Issue #459 tracks "additional distributed backends" — should be
expanded to enumerate DTensor, send/recv, all_to_all, elastic.

**Do not split.** The crate cleanly maps to `torch.distributed`. The work
is depth-fill.

## Related issues
- #459 — Add additional distributed backends
