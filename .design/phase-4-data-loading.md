# Feature: Phase 4 — Data Loading (ferrotorch-data)

## Summary
A data loading pipeline for ferrotorch: a `Dataset` trait for random-access sample retrieval, an `IterableDataset` trait for streaming data that does not fit in memory, a `DataLoader` that batches, shuffles, and prefetches samples using rayon-based parallelism with persistent workers and pin-memory support, pluggable samplers for index generation, a collation system that stacks heterogeneous samples into batched tensors (with automatic batching for both map-style and iterable datasets), memory-mapped datasets for zero-copy disk access, and a composable `Transform` pipeline for preprocessing. This crate bridges raw data sources to the `Tensor<T>` and `Module` types from ferrotorch-core and ferrotorch-nn, completing the training loop.

## Requirements
- REQ-1: A `Dataset` trait must provide thread-safe random access to samples. It must be `Send + Sync`, declare an associated `Sample` type, and expose `len()` for the dataset size and `get(index)` for sample retrieval. `get()` must return `FerrotorchResult<Self::Sample>` to handle I/O errors, missing files, and out-of-bounds indices without panicking. An `IterableDataset` trait must provide a streaming interface for datasets too large to fit in memory or generated on-the-fly (web datasets, log streams, database cursors). It must be `Send + Sync`, declare an associated `Sample` type, and expose `iter()` returning an iterator that yields `FerrotorchResult<Self::Sample>`. `IterableDataset` does not require `len()` or random access. When used with multi-worker data loading, each worker must receive a disjoint partition of the stream (via a `worker_id` and `num_workers` context passed to the dataset through a `WorkerInfo` struct) to avoid duplicate samples.
- REQ-2: `DataLoader<D: Dataset>` must wrap an `Arc<D>` and support configurable `batch_size`, `shuffle`, `num_workers` (rayon thread count for parallel prefetch), `drop_last` (discard the final incomplete batch), `pin_memory` (allocate batches in host-pinned / page-locked memory for faster CPU-to-GPU transfers), `persistent_workers` (keep worker threads alive between epochs to avoid thread creation overhead), and a user-replaceable `collate_fn`. `DataLoader::iter()` must return a `DataLoaderIter` that yields `FerrotorchResult<B>` where `B` is the collated batch type. Iteration must be lazy — batches are assembled on demand, not materialized up front. `DataLoader` must also accept an `IterableDataset` in place of a `Dataset`; when backed by an `IterableDataset`, samplers are ignored and iteration order is determined by the dataset's own `iter()` method.
- REQ-3: Samplers must control the order of index generation independently from the dataset and dataloader. Three samplers are required: `SequentialSampler` (indices 0..len in order), `RandomSampler` (shuffled indices, seeded for reproducibility), and `DistributedSampler` (partitions indices across `num_replicas` ranks for data-parallel training, with configurable `rank`, `shuffle`, and `seed`). The `DataLoader` must accept any sampler implementing the `Sampler` trait.
- REQ-4: A `CollateFn` trait must define how a `Vec<Sample>` is combined into a single batch value. A default `StackCollate` implementation must stack numeric tensors along a new batch dimension (dim 0). Users must be able to supply custom collate functions for non-uniform sample types (variable-length sequences, mixed modalities).
- REQ-5: A `Transform<Input, Output>` trait must define a single `apply(&self, input: Input) -> FerrotorchResult<Output>` method for data preprocessing. Implementations must be `Send + Sync` for use across rayon worker threads. Built-in transforms: `Compose` (chain transforms sequentially), `Normalize` (subtract mean, divide by std per channel), `ToTensor` (convert raw data to `Tensor<T>`), `RandomCrop` (extract a random spatial sub-region), `RandomHorizontalFlip` (flip with configurable probability). The `Compose` type must be generic over its input/output chain.
- REQ-6: All public functions must return `FerrotorchResult<T>`. No panics on invalid input: zero batch size, empty dataset, out-of-bounds index, worker thread failure, transform pipeline error, or shape mismatch during collation. Errors that occur in worker threads (panics, `FerrotorchError` returns from `dataset.get()` or `transform.apply()`, and I/O failures) must be propagated to the main thread as `FerrotorchError` variants, never silently dropped. If multiple workers fail in the same batch, the first error must be returned and the remaining workers' results discarded.
- REQ-7: `DataLoader` must support fully reproducible iteration. When `shuffle = true`, the user must be able to set a seed on the sampler. Two iterations with the same seed must produce identical batch orderings. `set_epoch()` on `DistributedSampler` must change the shuffle permutation per epoch (standard practice for distributed training). Setting a `seed` on the `DataLoader` must make iteration deterministic across workers: each worker must receive a deterministic per-worker seed derived from the base seed and the worker ID (e.g., `base_seed ^ worker_id`), ensuring that stochastic transforms (`RandomCrop`, `RandomHorizontalFlip`) produce identical results across runs. A `worker_init_fn` callback must be accepted by the builder, allowing users to customize per-worker RNG seeding or other initialization. Reproducibility must hold regardless of `num_workers` — the same seed must produce the same output whether `num_workers` is 0 or 8.
- REQ-8: Parallel prefetching via `num_workers > 0` must use a rayon thread pool scoped to the dataloader, not the global rayon pool. Worker threads call `dataset.get()` and `transform.apply()` in parallel, feeding completed samples into a bounded channel for the main thread to collate. Worker panics must be caught and converted to `FerrotorchError::WorkerPanic`, not propagated to the caller. When `persistent_workers = true`, the thread pool and worker state must be kept alive between calls to `DataLoader::iter()` across epochs, avoiding thread creation/destruction overhead. When `persistent_workers = false` (the default), the pool is still created once at `build()` but worker-local state (caches, file handles) may be reset between epochs.
- REQ-9: `DataLoader` must support automatic batching in two modes. In the default mode (map-style `Dataset`), the dataloader fetches individual samples via `dataset.get()` and groups them into batches using `collate_fn`. In pre-batched mode (used with `IterableDataset` or when `batch_size = None`), the dataset itself yields complete batches and the collate function is bypassed. The builder must expose a `.automatic_batching(bool)` option; when disabled, samples pass through to the caller without collation, and `batch_size` controls only the prefetch buffer size.
- REQ-10: A `MmapDataset` implementation must provide memory-mapped access to contiguous on-disk data files. `MmapDataset` must map the file into the process address space using `mmap` (via the `memmap2` crate) and serve samples by computing byte offsets without reading the entire file into RAM. It must implement the `Dataset` trait. The constructor must accept a file path and a record layout (fixed-size records or an index file mapping sample indices to byte offsets and lengths). `MmapDataset` must be `Send + Sync` (the `memmap2::Mmap` type is both). It must handle files larger than available RAM, and must return `FerrotorchError::DataLoading` on I/O errors or corrupted index files.

## Acceptance Criteria
- [ ] AC-1: A struct implementing `Dataset` with `type Sample = (Tensor<f32>, usize)` can be constructed, and `DataLoader::new(&dataset, batch_size=4)` produces an iterator that yields batches of 4 samples. The total number of batches equals `ceil(len / batch_size)` with `drop_last = false` and `floor(len / batch_size)` with `drop_last = true`.
- [ ] AC-2: `SequentialSampler` yields indices `[0, 1, 2, ..., len-1]`. `RandomSampler` with a fixed seed yields a deterministic permutation. Two `RandomSampler` instances with the same seed and length produce identical index sequences.
- [ ] AC-3: `DistributedSampler` with `num_replicas=3, rank=0` yields approximately `ceil(len / 3)` indices, and the union of indices across all 3 ranks covers the full dataset with no gaps and minimal overlap (at most `num_replicas - 1` duplicated samples for padding).
- [ ] AC-4: `StackCollate` applied to a `Vec<Tensor<f32>>` where each tensor has shape `[3, 32, 32]` produces a single `Tensor<f32>` with shape `[batch_size, 3, 32, 32]`. If tensors have mismatched shapes, `StackCollate` returns `Err(FerrotorchError::ShapeMismatch { .. })`.
- [ ] AC-5: `Compose::new(vec![Box::new(ToTensor::<f32>::new()), Box::new(Normalize::new(mean, std))])` applied to raw data produces a normalized `Tensor<f32>`. `Normalize` with `mean=[0.5, 0.5, 0.5]` and `std=[0.5, 0.5, 0.5]` maps the range `[0.0, 1.0]` to `[-1.0, 1.0]`.
- [ ] AC-6: `RandomHorizontalFlip::new(1.0)` always flips (verified by comparing pixel columns), and `RandomHorizontalFlip::new(0.0)` never flips. `RandomCrop::new(16, 16)` on a `[3, 32, 32]` tensor produces a `[3, 16, 16]` tensor with valid spatial content.
- [ ] AC-7: `DataLoader` with `num_workers=4` produces identical batch contents (ignoring order within a batch) as `num_workers=0` for the same seed and dataset. No data is lost or duplicated.
- [ ] AC-8: Calling `DataLoader::iter()` on an empty dataset (`len() == 0`) yields zero batches and does not panic. Constructing a `DataLoader` with `batch_size=0` returns `Err(FerrotorchError::InvalidArgument { .. })`.
- [ ] AC-9: A struct implementing `IterableDataset` that yields samples from a `Vec<T>` can be wrapped in a `DataLoader`. The resulting iterator yields all samples in order, batched according to `batch_size`. With `num_workers=2`, each worker receives a disjoint partition of the stream and the union of all yielded samples equals the full dataset with no duplicates.
- [ ] AC-10: `DataLoader::builder(dataset).pin_memory(true).build()` succeeds. When `pin_memory = true`, the returned batch tensors must have their `is_pinned()` flag set to `true`. When `pin_memory = false` (default), `is_pinned()` returns `false`.
- [ ] AC-11: A worker that returns `Err(FerrotorchError::DataLoading { .. })` from `dataset.get()` causes the corresponding `DataLoaderIter::next()` call to return `Err(FerrotorchError::DataLoading { .. })` on the main thread. A worker that panics causes `next()` to return `Err(FerrotorchError::WorkerPanic { .. })`. In both cases, the error is not silently swallowed.
- [ ] AC-12: With `persistent_workers = true`, calling `DataLoader::iter()` twice in succession reuses the same thread pool (verified by checking that thread IDs across epochs overlap). With `persistent_workers = false`, the thread pool may be recycled but worker-local state is not guaranteed to persist.
- [ ] AC-13: With `automatic_batching = false`, `DataLoader` passes individual samples from the dataset directly to the caller without collation, and `batch_size` controls only the prefetch buffer depth. With an `IterableDataset` that yields pre-batched `Tensor<f32>` of shape `[B, C, H, W]`, the loader yields those tensors unchanged.
- [ ] AC-14: Two `DataLoader` instances with identical seeds and `num_workers=4` produce byte-identical batch sequences across two separate runs, including the output of stochastic transforms (`RandomCrop`, `RandomHorizontalFlip`). Changing the seed produces a different sequence.
- [ ] AC-15: `MmapDataset::open("data.bin", FixedRecordLayout { record_size: 3080 })` memory-maps the file and serves `dataset.get(i)` by reading bytes `[i * 3080 .. (i+1) * 3080]` without loading the entire file into RAM. `dataset.len()` equals `file_size / record_size`. Accessing an index beyond `len()` returns `Err(FerrotorchError::DataLoading { .. })`.
- [ ] AC-16: `cargo test -p ferrotorch-data` passes with 0 failures. Minimum 120 tests covering: both dataset traits (`Dataset` and `IterableDataset`), all three samplers, collation (success and error paths), all built-in transforms, dataloader iteration (sequential and shuffled, map-style and iterable), multi-worker prefetching, persistent workers, pin memory, worker error propagation, automatic vs manual batching, mmap datasets, reproducibility across workers, edge cases (empty dataset, single sample, batch_size > len), and distributed sampler epoch rotation.

## Architecture

### Crate Layout

```
ferrotorch-data/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports
│   ├── dataset.rs                # Dataset trait, IterableDataset trait, WorkerInfo
│   ├── mmap_dataset.rs           # MmapDataset, FixedRecordLayout, IndexedRecordLayout
│   ├── dataloader.rs             # DataLoader struct, DataLoaderIter, builder
│   ├── sampler.rs                # Sampler trait, SequentialSampler, RandomSampler, DistributedSampler
│   ├── collate.rs                # CollateFn trait, StackCollate
│   └── transforms.rs            # Transform trait, Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
└── tests/
    ├── test_dataset.rs           # Dataset + IterableDataset trait contracts, mock datasets
    ├── test_mmap_dataset.rs      # MmapDataset: fixed-record, indexed, large files, error paths
    ├── test_dataloader.rs        # Batching, shuffling, drop_last, multi-worker, pin_memory, persistent_workers, error propagation, automatic batching, reproducibility
    ├── test_sampler.rs           # Sequential, random (reproducibility), distributed (partitioning)
    ├── test_collate.rs           # StackCollate success + shape mismatch errors
    └── test_transforms.rs        # Each transform individually + Compose chains
```

### Core Traits

**Dataset** (`dataset.rs`):
```rust
/// A dataset of indexable samples, safe to share across worker threads.
pub trait Dataset: Send + Sync {
    /// The type of a single sample returned by `get()`.
    type Sample: Send;

    /// Returns the total number of samples in the dataset.
    fn len(&self) -> usize;

    /// Returns true if the dataset contains no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieves the sample at the given index.
    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample>;
}
```

**IterableDataset** (`dataset.rs`):
```rust
/// Context passed to each worker so the dataset can partition its stream.
#[derive(Debug, Clone, Copy)]
pub struct WorkerInfo {
    /// This worker's ID (0-based).
    pub worker_id: usize,
    /// Total number of workers.
    pub num_workers: usize,
    /// Per-worker seed derived from the DataLoader's base seed.
    pub seed: u64,
}

/// A streaming dataset that yields samples without random access.
///
/// Use this for data sources too large to fit in memory, or for streams
/// that are generated on-the-fly (web datasets, log tails, database cursors).
/// When used with multi-worker loading, the dataset receives a `WorkerInfo`
/// and must partition its stream so that each worker yields a disjoint subset.
pub trait IterableDataset: Send + Sync {
    /// The type of a single sample yielded by the iterator.
    type Sample: Send;

    /// Returns an iterator over samples.
    ///
    /// `worker_info` is `None` when `num_workers == 0` (main-thread loading).
    /// When `Some`, the implementation must use `worker_info.worker_id` and
    /// `worker_info.num_workers` to yield only its partition of the data.
    fn iter(
        &self,
        worker_info: Option<WorkerInfo>,
    ) -> Box<dyn Iterator<Item = FerrotorchResult<Self::Sample>> + Send + '_>;
}
```

**MmapDataset** (`mmap_dataset.rs`):
```rust
/// Record layout for memory-mapped datasets with fixed-size records.
pub struct FixedRecordLayout {
    /// Size of each record in bytes.
    pub record_size: usize,
}

/// Record layout for memory-mapped datasets with variable-size records.
/// An index file maps each sample index to a (byte_offset, byte_length) pair.
pub struct IndexedRecordLayout {
    /// Path to the index file containing (offset, length) pairs.
    pub index_path: PathBuf,
}

/// A dataset backed by a memory-mapped file, providing zero-copy access
/// to on-disk data without loading the entire file into RAM.
pub struct MmapDataset {
    mmap: memmap2::Mmap,
    // ... layout, record count, index table (if indexed)
}

impl MmapDataset {
    /// Memory-map the file at `path` with a fixed-record layout.
    pub fn open(path: impl AsRef<Path>, layout: FixedRecordLayout) -> FerrotorchResult<Self> { .. }

    /// Memory-map the file at `path` with a variable-record layout and index file.
    pub fn open_indexed(path: impl AsRef<Path>, layout: IndexedRecordLayout) -> FerrotorchResult<Self> { .. }
}

impl Dataset for MmapDataset {
    type Sample = Vec<u8>; // Raw bytes; callers wrap in a Transform to decode

    fn len(&self) -> usize { .. }
    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> { .. }
}
```

**Sampler** (`sampler.rs`):
```rust
/// Generates a sequence of dataset indices for a DataLoader to consume.
pub trait Sampler: Send + Sync {
    /// Returns an iterator over dataset indices for one epoch.
    fn indices(&self, epoch: u64) -> Vec<usize>;

    /// Returns the total number of indices produced.
    fn len(&self) -> usize;
}
```

`SequentialSampler` stores `len` and returns `(0..len).collect()`. `RandomSampler` stores `len` and `seed`; `indices()` creates an `StdRng::seed_from_u64(seed ^ epoch)`, generates `(0..len)`, and shuffles with `SliceRandom::shuffle`. `DistributedSampler` stores `len`, `num_replicas`, `rank`, `shuffle`, and `seed`; `indices()` computes a global permutation (if `shuffle`), pads to a multiple of `num_replicas`, and takes every `num_replicas`-th element starting at `rank`.

**CollateFn** (`collate.rs`):
```rust
/// Combines a vector of individual samples into a single batched value.
pub trait CollateFn<S>: Send + Sync {
    /// The batched output type.
    type Batch;

    /// Collate a vector of samples into one batch.
    fn collate(&self, samples: Vec<S>) -> FerrotorchResult<Self::Batch>;
}

/// Default collation: stack tensors along a new dim 0.
pub struct StackCollate;

impl<T: Float> CollateFn<Tensor<T>> for StackCollate {
    type Batch = Tensor<T>;

    fn collate(&self, samples: Vec<Tensor<T>>) -> FerrotorchResult<Tensor<T>> {
        // Verify all samples have identical shape, then stack along dim 0.
        // Returns ShapeMismatch if shapes differ.
    }
}
```

For tuple samples like `(Tensor<T>, Tensor<T>)` (input + label), users implement `CollateFn` for their specific tuple type, or use a provided `TupleCollate` that collates each position independently.

**Transform** (`transforms.rs`):
```rust
/// A data preprocessing step, safe to apply across worker threads.
pub trait Transform<Input, Output = Input>: Send + Sync {
    /// Apply this transform to a single input.
    fn apply(&self, input: Input) -> FerrotorchResult<Output>;
}
```

**Compose** chains transforms sequentially. Because Rust's type system cannot erase a heterogeneous chain of `Transform<A, B>`, `Transform<B, C>`, etc. into a single dynamic list, `Compose` is limited to transforms where `Input == Output`:

```rust
/// Chain multiple transforms of the same type signature.
pub struct Compose<T> {
    transforms: Vec<Box<dyn Transform<T, T>>>,
}

impl<T: Send> Transform<T, T> for Compose<T> {
    fn apply(&self, mut input: T) -> FerrotorchResult<T> {
        for t in &self.transforms {
            input = t.apply(input)?;
        }
        Ok(input)
    }
}
```

**Built-in transforms:**

| Transform | Input | Output | Behavior |
|-----------|-------|--------|----------|
| `ToTensor<T>` | `Vec<T>` + shape | `Tensor<T>` | Constructs a tensor from raw numeric data |
| `Normalize` | `Tensor<T>` | `Tensor<T>` | Per-channel: `(x - mean) / std` |
| `RandomCrop` | `Tensor<T>` | `Tensor<T>` | Extracts `[C, H', W']` from `[C, H, W]` at a random offset |
| `RandomHorizontalFlip` | `Tensor<T>` | `Tensor<T>` | Reverses the last spatial dimension with probability `p` |

`Normalize`, `RandomCrop`, and `RandomHorizontalFlip` operate on `Tensor<T>` and implement `Transform<Tensor<T>, Tensor<T>>`, making them composable via `Compose<Tensor<T>>`.

### DataLoader

**DataLoader** (`dataloader.rs`):
```rust
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    sampler: Box<dyn Sampler>,
    collate_fn: Arc<dyn CollateFn<D::Sample, Batch = /* caller-specified */>>,
    num_workers: usize,
    drop_last: bool,
    pin_memory: bool,
    persistent_workers: bool,
    automatic_batching: bool,
    seed: Option<u64>,
    worker_init_fn: Option<Arc<dyn Fn(WorkerInfo) + Send + Sync>>,
    // Private: rayon ThreadPool if num_workers > 0, created once at construction.
    // When persistent_workers = true, kept alive across iter() calls.
    pool: Option<rayon::ThreadPool>,
}
```

An analogous `IterableDataLoader<D: IterableDataset>` wraps an `Arc<D>` with the same builder options except `sampler` (which is inapplicable to streaming datasets). Both types share a common builder interface and may be unified behind an enum internally.

Construction uses the builder pattern:

```rust
// Map-style dataset
let loader = DataLoader::builder(dataset)
    .batch_size(32)
    .shuffle(true)               // Convenience: sets RandomSampler
    .num_workers(4)
    .drop_last(true)
    .pin_memory(true)            // Page-locked memory for GPU transfers
    .persistent_workers(true)    // Keep threads alive between epochs
    .seed(42)                    // Deterministic across workers
    .worker_init_fn(|info| { /* per-worker setup */ })
    .sampler(DistributedSampler::new(len, num_replicas, rank))
    .collate_fn(StackCollate)
    .build()?;                   // Validates: batch_size > 0, etc.

// Iterable-style dataset (streaming)
let loader = IterableDataLoader::builder(stream_dataset)
    .batch_size(32)
    .num_workers(4)
    .pin_memory(true)
    .persistent_workers(true)
    .seed(42)
    .build()?;
```

`.shuffle(true)` is syntactic sugar for `.sampler(RandomSampler::new(len, seed))`. If both `.shuffle()` and `.sampler()` are called, the explicit sampler wins.

**Pin memory**: When `pin_memory = true`, after collation the dataloader copies each batch tensor into a pinned (page-locked) memory allocation. This is a no-op on CPU-only builds and requires the ferrotorch-gpu backend to provide a `pin_memory()` function on `Tensor<T>`. The pinned tensor is returned to the caller, enabling asynchronous CPU-to-GPU copies via `tensor.to(device)`. If ferrotorch-gpu is not available, `pin_memory(true)` returns `Err(FerrotorchError::InvalidArgument)`.

**Persistent workers**: When `persistent_workers = true`, the rayon `ThreadPool` and any worker-local state (open file handles, decoder caches) are preserved across calls to `DataLoader::iter()`. This avoids the overhead of thread creation/destruction between epochs. When `false`, the pool itself persists (it is created once at `build()`) but the dataloader does not guarantee preservation of worker-local state.

**DataLoaderIter** yields batches lazily:

```rust
pub struct DataLoaderIter<'a, D: Dataset> {
    loader: &'a DataLoader<D>,
    indices: Vec<usize>,         // Pre-generated by sampler for this epoch
    pos: usize,                  // Current position in indices
    epoch: u64,
}

impl<'a, D: Dataset> Iterator for DataLoaderIter<'a, D> {
    type Item = FerrotorchResult</* Batch type */>;

    fn next(&mut self) -> Option<Self::Item> {
        // 1. Slice indices[pos..pos+batch_size] (or skip if drop_last and remainder)
        // 2. If num_workers == 0: sequentially call dataset.get() for each index
        //    If num_workers > 0: dispatch get() calls to rayon pool, collect via channel
        // 3. Collate the Vec<Sample> into a batch via collate_fn
        // 4. Advance pos
    }
}
```

**Parallel prefetch** (when `num_workers > 0`):

The rayon `ThreadPool` is created at `DataLoader::build()` with `num_workers` threads. During iteration, each batch's indices are dispatched to the pool via `pool.install(|| indices.par_iter().map(|&i| dataset.get(i)).collect())`. This parallelizes I/O-bound `get()` calls (file reads, decoding) across workers. `std::panic::catch_unwind` wraps each `get()` call to convert worker panics into `FerrotorchError::WorkerPanic`.

**Worker error propagation**: Every worker result is a `Result<Sample, FerrotorchError>`. After parallel collection, the dataloader inspects all results. If any worker returned an `Err`, the first error is returned from `DataLoaderIter::next()` and all other results (including successes) for that batch are dropped. Panics caught by `catch_unwind` are wrapped in `FerrotorchError::WorkerPanic { message }` where `message` includes the worker ID and the panic payload (via `std::any::Any` downcast to `&str` or `String`). Errors are never silently discarded.

**Worker seeding for reproducibility**: When a `seed` is set on the dataloader, each worker receives `WorkerInfo { worker_id, num_workers, seed: base_seed.wrapping_add(worker_id as u64) }`. The `worker_init_fn` (if provided) is called with this `WorkerInfo` before the worker processes any samples. Stochastic transforms must use thread-local RNGs seeded from `WorkerInfo::seed` to ensure deterministic output across runs.

**Automatic batching**: In the default mode (`automatic_batching = true`), the dataloader fetches individual samples and groups them into batches of `batch_size` using `collate_fn`. When `automatic_batching = false`, the dataloader passes raw samples through without collation — this is intended for `IterableDataset` implementations that yield pre-batched tensors, or for use cases where the caller wants full control over batching. In this mode, `batch_size` controls only the prefetch buffer depth (how many samples are fetched ahead), not the grouping of samples.

**Iterable dataset iteration**: When backed by an `IterableDataset`, each worker calls `dataset.iter(Some(WorkerInfo { .. }))` and pulls samples from the returned iterator. The dataset is responsible for partitioning its stream across workers. Samples from all workers are fed into a shared bounded channel; the main thread drains the channel in arrival order and groups samples into batches. Samplers are not used — iteration order is determined by the dataset.

### Error Variants

New variants added to `FerrotorchError` in ferrotorch-core (or re-exported from ferrotorch-data if core cannot be modified):

```rust
#[error("data loading error: {message}")]
DataLoading { message: String },

#[error("worker thread {worker_id} panicked: {message}")]
WorkerPanic { worker_id: usize, message: String },

#[error("worker thread {worker_id} failed: {source}")]
WorkerError { worker_id: usize, source: Box<FerrotorchError> },

#[error("transform failed: {message}")]
TransformError { message: String },

#[error("memory map error on {path}: {message}")]
MmapError { path: String, message: String },
```

`WorkerPanic` is produced when `catch_unwind` catches a panic in a worker thread. `WorkerError` is produced when a worker's `dataset.get()` or `transform.apply()` returns an `Err`. Both include the `worker_id` for diagnostics. The main thread always surfaces the first error encountered; concurrent errors from other workers are logged at `tracing::warn` level but not returned.

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | `Tensor<T>`, `FerrotorchError`, `FerrotorchResult` |
| `rayon` | 1.11 | Parallel sample prefetching (scoped thread pool) |
| `rand` | 0.9 | Shuffle permutations, random crop offsets, flip coin |
| `crossbeam-channel` | 0.5 | Bounded channel between worker pool and iterator (if prefetch buffering is needed beyond rayon collect) |
| `memmap2` | 0.9 | Memory-mapped file I/O for `MmapDataset` |

### Test Strategy

1. **Mock datasets**: `VecDataset<S>` wrapping a `Vec<S>` for deterministic testing without I/O. `VecIterableDataset<S>` wrapping a `Vec<S>` behind the `IterableDataset` trait, partitioning by worker ID.
2. **Sampler verification**: Assert exact index sequences for `SequentialSampler` and seeded `RandomSampler`. For `DistributedSampler`, verify full coverage across all ranks and correct padding.
3. **Collation**: Test `StackCollate` with uniform shapes (success) and mixed shapes (error). Test tuple collation. Test automatic batching enabled vs disabled.
4. **Transform correctness**: Numerical checks for `Normalize` (known input/output pairs). Spatial checks for `RandomCrop` (output shape, bounds). Deterministic checks for `RandomHorizontalFlip` at p=0.0 and p=1.0.
5. **DataLoader integration**: End-to-end iteration over a mock dataset, verifying batch count, sample coverage (every index visited exactly once per epoch), and reproducibility across two iterations with the same seed. Separate integration tests for `IterableDataLoader` with streaming datasets.
6. **Multi-worker**: Same-output test: `num_workers=0` vs `num_workers=4` produce the same set of samples (order within a batch may differ due to parallel collection, so compare as sorted sets). Verify that stochastic transforms produce identical results across runs when seeded.
7. **Worker error propagation**: Mock dataset that returns `Err` for specific indices — verify the error surfaces on the main thread. Mock dataset that panics — verify `WorkerPanic` error is returned. Verify no silent error swallowing by checking that all error indices are reported.
8. **Persistent workers**: Run two epochs with `persistent_workers = true` and verify thread pool reuse (thread ID overlap). Run two epochs with `persistent_workers = false` and verify correct behavior without assuming state persistence.
9. **Pin memory**: With `pin_memory = true`, verify that returned tensors report `is_pinned() == true`. With `pin_memory = false`, verify `is_pinned() == false`.
10. **Reproducibility**: Run two `DataLoader` instances with the same seed, `num_workers=4`, and stochastic transforms. Verify byte-identical output. Change the seed and verify different output. Test `worker_init_fn` is called with correct `WorkerInfo`.
11. **MmapDataset**: Create a temporary binary file with known fixed-size records, open it via `MmapDataset::open`, and verify that `get(i)` returns the correct bytes. Test `len()` correctness. Test out-of-bounds access returns `DataLoading` error. Test with an indexed record layout and variable-size records. Test that files larger than available RAM can be opened (by checking that RSS does not spike).
12. **Edge cases**: Empty dataset, single-sample dataset, `batch_size > len`, `batch_size == len`, `drop_last` with exact divisor, `drop_last` with remainder of 1. Empty `IterableDataset` yields zero batches. `IterableDataset` with fewer samples than `batch_size`.

## Resolved Questions

### Q1: Collate function generics — trait object or generic parameter?
**Decision**: Trait object via `Arc<dyn CollateFn<S, Batch = B>>`.

A generic parameter `C: CollateFn<S>` on `DataLoader` would infect every type that holds a `DataLoader` with an extra generic. Since collate functions are called once per batch (not per element), the dynamic dispatch overhead is negligible. `Arc` wrapping allows the collate function to be shared with the iterator without lifetime entanglement.

### Q2: Compose type erasure — heterogeneous chain or same-type?
**Decision**: Same-type (`Compose<T>` where all transforms are `Transform<T, T>`).

A heterogeneous chain (`Transform<A, B>` then `Transform<B, C>`) would require either a macro-generated tuple type or runtime type erasure with `Box<dyn Any>`. Both approaches add complexity with minimal practical benefit — in real pipelines, transforms operate on `Tensor<T>` after an initial `ToTensor` conversion. Users who need heterogeneous chains can compose manually: `let sample = to_tensor.apply(raw)?; let sample = normalize.apply(sample)?;`.

### Q3: Rayon global pool vs scoped pool?
**Decision**: Scoped `rayon::ThreadPool` per `DataLoader`, not the global pool.

The global rayon pool is shared across the entire application. If a training loop creates multiple `DataLoader` instances (train + validation), or if user code uses rayon for other purposes, the global pool becomes a contention point. A dedicated `ThreadPool` with `num_workers` threads per `DataLoader` provides isolation and predictable performance. The pool is created once at `DataLoader::build()` and reused across epochs.

### Q4: IterableDataset — separate trait or enum variant?
**Decision**: Separate trait (`IterableDataset`) with a separate `IterableDataLoader` type.

Combining map-style and iterable datasets into a single enum would force runtime dispatch on every `get()` call and complicate the type signatures (iterable datasets have no `len()`, map-style datasets require samplers). Separate traits allow the compiler to enforce correct usage at build time — you cannot pass a sampler to an `IterableDataLoader`, and you cannot use an `IterableDataset` with `DistributedSampler`. The two loader types share builder methods via a common builder trait or macro, minimizing code duplication.

### Q5: Pin memory — runtime or compile-time GPU gate?
**Decision**: Runtime check via feature flag.

`pin_memory` is gated behind a `gpu` Cargo feature. When the feature is disabled, calling `.pin_memory(true)` returns `Err(FerrotorchError::InvalidArgument)` at build time (the builder's `build()` method). When enabled, it delegates to `ferrotorch-gpu::pin_memory()`. This avoids unconditional GPU dependencies while keeping the API surface uniform.

### Q6: Persistent workers — what state is preserved?
**Decision**: The rayon `ThreadPool` is always preserved (created once at `build()`). `persistent_workers` controls whether `thread_local!` state inside workers (file handles, decoder caches, RNG state) is preserved across `iter()` calls. When `true`, workers are not reset between epochs. When `false`, each `iter()` call re-initializes worker-local state by calling `worker_init_fn` (if provided) at the start of each epoch.

### Q7: MmapDataset — raw bytes or typed samples?
**Decision**: `MmapDataset` yields `Vec<u8>` (raw bytes). Type-specific decoding is handled by a `Transform` in the pipeline.

This keeps `MmapDataset` format-agnostic — it handles the memory mapping and byte slicing, while transforms handle deserialization (e.g., `BytesToTensor<f32>`, `BytesToImage`). Users compose: `MmapDataset` -> `Transform<Vec<u8>, Tensor<f32>>` -> `Normalize` -> etc. This separation of concerns matches PyTorch's pattern where datasets return raw data and transforms handle preprocessing.

## Out of Scope
- Image decoding (JPEG, PNG) — that is ferrotorch-vision's responsibility; ferrotorch-data provides the pipeline, not format-specific readers
- Built-in dataset implementations (MNIST, CIFAR, ImageNet) — those belong in ferrotorch-vision (Phase 5)
- GPU-direct storage (GPUDirect Storage / GDS for NVMe-to-GPU bypass) — that requires kernel-level integration in ferrotorch-gpu (Phase 6); `pin_memory` (host-pinned memory for faster CPU-to-GPU copies) is in scope
- Data augmentation beyond the five listed transforms — domain-specific transforms (color jitter, mixup, cutout) belong in ferrotorch-vision
- Automatic batching of variable-length sequences (padding/packing) — users handle this via custom `CollateFn` implementations; a `PadCollate` utility may be added in a future iteration
- Multi-node data loading (sharding across machines) — `DistributedSampler` handles single-node multi-GPU; cross-machine coordination belongs in a distributed training crate
