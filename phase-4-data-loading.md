---
title: "Phase 4 — Data Loading (ferrotorch-data)"
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-15
updated: 2026-03-15
---


## Design Specification

### Summary

A data loading pipeline for ferrotorch: a `Dataset` trait for random-access sample retrieval, a `DataLoader` that batches, shuffles, and prefetches samples using rayon-based parallelism, pluggable samplers for index generation, a collation system that stacks heterogeneous samples into batched tensors, and a composable `Transform` pipeline for preprocessing. This crate bridges raw data sources to the `Tensor<T>` and `Module` types from ferrotorch-core and ferrotorch-nn, completing the training loop.

### Requirements

- REQ-1: A `Dataset` trait must provide thread-safe random access to samples. It must be `Send + Sync`, declare an associated `Sample` type, and expose `len()` for the dataset size and `get(index)` for sample retrieval. `get()` must return `FerrotorchResult<Self::Sample>` to handle I/O errors, missing files, and out-of-bounds indices without panicking.
- REQ-2: `DataLoader<D: Dataset>` must wrap an `Arc<D>` and support configurable `batch_size`, `shuffle`, `num_workers` (rayon thread count for parallel prefetch), `drop_last` (discard the final incomplete batch), and a user-replaceable `collate_fn`. `DataLoader::iter()` must return a `DataLoaderIter` that yields `FerrotorchResult<B>` where `B` is the collated batch type. Iteration must be lazy — batches are assembled on demand, not materialized up front.
- REQ-3: Samplers must control the order of index generation independently from the dataset and dataloader. Three samplers are required: `SequentialSampler` (indices 0..len in order), `RandomSampler` (shuffled indices, seeded for reproducibility), and `DistributedSampler` (partitions indices across `num_replicas` ranks for data-parallel training, with configurable `rank`, `shuffle`, and `seed`). The `DataLoader` must accept any sampler implementing the `Sampler` trait.
- REQ-4: A `CollateFn` trait must define how a `Vec<Sample>` is combined into a single batch value. A default `StackCollate` implementation must stack numeric tensors along a new batch dimension (dim 0). Users must be able to supply custom collate functions for non-uniform sample types (variable-length sequences, mixed modalities).
- REQ-5: A `Transform<Input, Output>` trait must define a single `apply(&self, input: Input) -> FerrotorchResult<Output>` method for data preprocessing. Implementations must be `Send + Sync` for use across rayon worker threads. Built-in transforms: `Compose` (chain transforms sequentially), `Normalize` (subtract mean, divide by std per channel), `ToTensor` (convert raw data to `Tensor<T>`), `RandomCrop` (extract a random spatial sub-region), `RandomHorizontalFlip` (flip with configurable probability). The `Compose` type must be generic over its input/output chain.
- REQ-6: All public functions must return `FerrotorchResult<T>`. No panics on invalid input: zero batch size, empty dataset, out-of-bounds index, worker thread failure, transform pipeline error, or shape mismatch during collation.
- REQ-7: `DataLoader` must support reproducible iteration. When `shuffle = true`, the user must be able to set a seed on the sampler. Two iterations with the same seed must produce identical batch orderings. `set_epoch()` on `DistributedSampler` must change the shuffle permutation per epoch (standard practice for distributed training).
- REQ-8: Parallel prefetching via `num_workers > 0` must use a rayon thread pool scoped to the dataloader, not the global rayon pool. Worker threads call `dataset.get()` and `transform.apply()` in parallel, feeding completed samples into a bounded channel for the main thread to collate. Worker panics must be caught and converted to `FerrotorchError`, not propagated to the caller.

### Acceptance Criteria

- [ ] AC-1: A struct implementing `Dataset` with `type Sample = (Tensor<f32>, usize)` can be constructed, and `DataLoader::new(&dataset, batch_size=4)` produces an iterator that yields batches of 4 samples. The total number of batches equals `ceil(len / batch_size)` with `drop_last = false` and `floor(len / batch_size)` with `drop_last = true`.
- [ ] AC-2: `SequentialSampler` yields indices `[0, 1, 2, ..., len-1]`. `RandomSampler` with a fixed seed yields a deterministic permutation. Two `RandomSampler` instances with the same seed and length produce identical index sequences.
- [ ] AC-3: `DistributedSampler` with `num_replicas=3, rank=0` yields approximately `ceil(len / 3)` indices, and the union of indices across all 3 ranks covers the full dataset with no gaps and minimal overlap (at most `num_replicas - 1` duplicated samples for padding).
- [ ] AC-4: `StackCollate` applied to a `Vec<Tensor<f32>>` where each tensor has shape `[3, 32, 32]` produces a single `Tensor<f32>` with shape `[batch_size, 3, 32, 32]`. If tensors have mismatched shapes, `StackCollate` returns `Err(FerrotorchError::ShapeMismatch { .. })`.
- [ ] AC-5: `Compose::new(vec![Box::new(ToTensor::<f32>::new()), Box::new(Normalize::new(mean, std))])` applied to raw data produces a normalized `Tensor<f32>`. `Normalize` with `mean=[0.5, 0.5, 0.5]` and `std=[0.5, 0.5, 0.5]` maps the range `[0.0, 1.0]` to `[-1.0, 1.0]`.
- [ ] AC-6: `RandomHorizontalFlip::new(1.0)` always flips (verified by comparing pixel columns), and `RandomHorizontalFlip::new(0.0)` never flips. `RandomCrop::new(16, 16)` on a `[3, 32, 32]` tensor produces a `[3, 16, 16]` tensor with valid spatial content.
- [ ] AC-7: `DataLoader` with `num_workers=4` produces identical batch contents (ignoring order within a batch) as `num_workers=0` for the same seed and dataset. No data is lost or duplicated.
- [ ] AC-8: Calling `DataLoader::iter()` on an empty dataset (`len() == 0`) yields zero batches and does not panic. Constructing a `DataLoader` with `batch_size=0` returns `Err(FerrotorchError::InvalidArgument { .. })`.
- [ ] AC-9: `cargo test -p ferrotorch-data` passes with 0 failures. Minimum 80 tests covering: dataset trait implementations, all three samplers, collation (success and error paths), all built-in transforms, dataloader iteration (sequential and shuffled), multi-worker prefetching, reproducibility, edge cases (empty dataset, single sample, batch_size > len), and distributed sampler epoch rotation.

### Architecture

### Crate Layout

```
ferrotorch-data/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports
│   ├── dataset.rs                # Dataset trait
│   ├── dataloader.rs             # DataLoader struct, DataLoaderIter, builder
│   ├── sampler.rs                # Sampler trait, SequentialSampler, RandomSampler, DistributedSampler
│   ├── collate.rs                # CollateFn trait, StackCollate
│   └── transforms.rs            # Transform trait, Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
└── tests/
    ├── test_dataset.rs           # Dataset trait contract, mock datasets
    ├── test_dataloader.rs        # Batching, shuffling, drop_last, multi-worker, edge cases
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
    // Private: rayon ThreadPool if num_workers > 0, created once at construction
    pool: Option<rayon::ThreadPool>,
}
```

Construction uses the builder pattern:

```rust
let loader = DataLoader::builder(dataset)
    .batch_size(32)
    .shuffle(true)               // Convenience: sets RandomSampler
    .num_workers(4)
    .drop_last(true)
    .sampler(DistributedSampler::new(len, num_replicas, rank))
    .collate_fn(StackCollate)
    .build()?;                   // Validates: batch_size > 0, etc.
```

`.shuffle(true)` is syntactic sugar for `.sampler(RandomSampler::new(len, seed))`. If both `.shuffle()` and `.sampler()` are called, the explicit sampler wins.

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

### Error Variants

New variants added to `FerrotorchError` in ferrotorch-core (or re-exported from ferrotorch-data if core cannot be modified):

```rust
#[error("data loading error: {message}")]
DataLoading { message: String },

#[error("worker thread panicked: {message}")]
WorkerPanic { message: String },

#[error("transform failed: {message}")]
TransformError { message: String },
```

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | `Tensor<T>`, `FerrotorchError`, `FerrotorchResult` |
| `rayon` | 1.11 | Parallel sample prefetching (scoped thread pool) |
| `rand` | 0.9 | Shuffle permutations, random crop offsets, flip coin |
| `crossbeam-channel` | 0.5 | Bounded channel between worker pool and iterator (if prefetch buffering is needed beyond rayon collect) |

### Test Strategy

1. **Mock datasets**: `VecDataset<S>` wrapping a `Vec<S>` for deterministic testing without I/O.
2. **Sampler verification**: Assert exact index sequences for `SequentialSampler` and seeded `RandomSampler`. For `DistributedSampler`, verify full coverage across all ranks and correct padding.
3. **Collation**: Test `StackCollate` with uniform shapes (success) and mixed shapes (error). Test tuple collation.
4. **Transform correctness**: Numerical checks for `Normalize` (known input/output pairs). Spatial checks for `RandomCrop` (output shape, bounds). Deterministic checks for `RandomHorizontalFlip` at p=0.0 and p=1.0.
5. **DataLoader integration**: End-to-end iteration over a mock dataset, verifying batch count, sample coverage (every index visited exactly once per epoch), and reproducibility across two iterations with the same seed.
6. **Multi-worker**: Same-output test: `num_workers=0` vs `num_workers=4` produce the same set of samples (order within a batch may differ due to parallel collection, so compare as sorted sets).
7. **Edge cases**: Empty dataset, single-sample dataset, `batch_size > len`, `batch_size == len`, `drop_last` with exact divisor, `drop_last` with remainder of 1.

### Out of Scope

- Image decoding (JPEG, PNG) — that is ferrotorch-vision's responsibility; ferrotorch-data provides the pipeline, not format-specific readers
- Built-in dataset implementations (MNIST, CIFAR, ImageNet) — those belong in ferrotorch-vision (Phase 5)
- GPU-direct data loading (CUDA pinned memory, GPUDirect Storage) — that requires ferrotorch-gpu (Phase 6)
- Iterable-style datasets (streaming / infinite) — only map-style (indexed) datasets are in scope for Phase 4; streaming datasets can be added later as an `IterableDataset` trait
- Data augmentation beyond the five listed transforms — domain-specific transforms (color jitter, mixup, cutout) belong in ferrotorch-vision
- Automatic batching of variable-length sequences (padding/packing) — users handle this via custom `CollateFn` implementations; a `PadCollate` utility may be added in a future iteration

### resolved questions

### Q1: Collate function generics — trait object or generic parameter?
**Decision**: Trait object via `Arc<dyn CollateFn<S, Batch = B>>`.

A generic parameter `C: CollateFn<S>` on `DataLoader` would infect every type that holds a `DataLoader` with an extra generic. Since collate functions are called once per batch (not per element), the dynamic dispatch overhead is negligible. `Arc` wrapping allows the collate function to be shared with the iterator without lifetime entanglement.

### Q2: Compose type erasure — heterogeneous chain or same-type?
**Decision**: Same-type (`Compose<T>` where all transforms are `Transform<T, T>`).

A heterogeneous chain (`Transform<A, B>` then `Transform<B, C>`) would require either a macro-generated tuple type or runtime type erasure with `Box<dyn Any>`. Both approaches add complexity with minimal practical benefit — in real pipelines, transforms operate on `Tensor<T>` after an initial `ToTensor` conversion. Users who need heterogeneous chains can compose manually: `let sample = to_tensor.apply(raw)?; let sample = normalize.apply(sample)?;`.

### Q3: Rayon global pool vs scoped pool?
**Decision**: Scoped `rayon::ThreadPool` per `DataLoader`, not the global pool.

The global rayon pool is shared across the entire application. If a training loop creates multiple `DataLoader` instances (train + validation), or if user code uses rayon for other purposes, the global pool becomes a contention point. A dedicated `ThreadPool` with `num_workers` threads per `DataLoader` provides isolation and predictable performance. The pool is created once at `DataLoader::build()` and reused across epochs.

