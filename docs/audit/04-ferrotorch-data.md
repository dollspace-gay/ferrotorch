# Audit: `ferrotorch-data` vs `torch.utils.data`

Covers: `torch.utils.data.{Dataset, IterableDataset, DataLoader, Sampler}`,
`torch.utils.data.distributed`, `torch.utils.data.datapipes` (experimental).

## Datasets

| ferrotorch | torch | Notes |
|---|---|---|
| `Dataset` (trait: `len`, `get`, `iter`, `is_empty`) | `Dataset` (`__len__`, `__getitem__`) | ✅ |
| `IterableDataset` | `IterableDataset` | ✅ |
| `TensorDataset` | `TensorDataset` | ✅ |
| `ConcatDataset` | `ConcatDataset` | ✅ |
| `ChainDataset` | `ChainDataset` | ✅ |
| `MappedDataset` (`.map(fn)`) | (manual via subclassing) | **extra** |
| `VecDataset` (in-memory wrapper) | (use TensorDataset) | **extra** |
| `WorkerInfo` | `get_worker_info()` returns analogous struct | ✅ |
| **missing** | `StackDataset` | gap |
| **missing** | `Subset` (slice via index list) | gap |

## Samplers

| ferrotorch | torch | Notes |
|---|---|---|
| `Sampler` (trait) | `Sampler` (ABC) | ✅ |
| `SequentialSampler` | `SequentialSampler` | ✅ |
| `RandomSampler` | `RandomSampler` | ✅ |
| `WeightedRandomSampler` | `WeightedRandomSampler` | ✅ |
| `BatchSampler` | `BatchSampler` | ✅ |
| `DistributedSampler` | `DistributedSampler` (in `data.distributed`) | ✅ |
| `shuffle_with_seed` helper | (use `torch.Generator`) | **extra** |
| **missing** | `SubsetRandomSampler` | gap |

## DataLoader

ferrotorch builder API:
```rust
DataLoader::new(dataset, batch_size)
    .worker_mode(WorkerMode::...)        // single | threaded | multiprocess
    .shuffle(bool)
    .drop_last(bool)
    .seed(u64)
    .num_workers(n)
    .prefetch_factor(n)
    .device(Device)                       // auto-move batches to device
    .pin_memory(bool)
    .with_sampler(Box<dyn Sampler>)
    .with_collate(closure)
```

Iterators: `BatchIter`, `CollatedIter`, `MultiWorkerIter`, `PrefetchIter`,
`ToDevice`.

torch DataLoader supports: `dataset, batch_size, shuffle, sampler,
batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout,
worker_init_fn, multiprocessing_context, generator, prefetch_factor,
persistent_workers, pin_memory_device, in_order`.

**Missing in ferrotorch:**
- `timeout` — worker fetch timeout
- `worker_init_fn` — per-worker initialization callback
- `persistent_workers` — keep workers alive across epochs
- `pin_memory_device` (which device to pin to; with multi-GPU)
- `in_order` (preserve fetch order across workers; new in torch ≥2.6)
- explicit `batch_sampler` (separate from `sampler` and `batch_size`)
- `Generator` parameter for deterministic shuffling (ferrotorch has `seed()`
  which covers most of the use case)

**Extra in ferrotorch:**
- `device(Device)` chaining — auto-move every batch to device. Idiomatic Rust.
- `WorkerMode::{Single, Threaded, Multiprocess}` exposed explicitly. torch
  uses `num_workers=0` vs `>0` and never threaded mode.
- `ToDevice` iterator adapter as a public type.

## Collate

| ferrotorch | torch |
|---|---|
| `default_collate` | `default_collate` (in `_utils/collate.py`) |
| `default_collate_pair` | (no analog) |
| custom via `with_collate(fn)` | custom via `collate_fn` arg |

✅ matches; ferrotorch has a paired-tuple convenience.

## Transforms

ferrotorch (`transforms.rs`): `Transform` trait, `Compose`, `Normalize`,
`RandomCrop`, `RandomHorizontalFlip`, `ToTensor`, `manual_seed`.

torch puts transforms in `torchvision.transforms` (not
`torch.utils.data`). ferrotorch keeps a small core in `ferrotorch-data` and
the bigger image-specific suite in `ferrotorch-vision`. **Reasonable
divergence.**

## Datapipes (experimental in torch)

torch has a fairly large `torch.utils.data.datapipes` directory implementing
a DAG-based pipeline (`IterDataPipe`, `MapDataPipe`, `traverse_dps`,
graph manipulation, standard pipes like `shuffle`, `batch`, `collate`).

ferrotorch has nothing equivalent. Datapipes are deprioritized upstream
(stable but quietly maintained); not a high priority unless dataset graph
manipulation is desired. **Skip unless needed.**

## Recommendations

1. **Add `Subset` and `SubsetRandomSampler`** — common pattern for
   train/val/test splits.
2. **Add `StackDataset`** — combines multiple datasets sample-wise (paired
   image/label/mask, e.g. for semantic segmentation).
3. **Add `worker_init_fn`** to DataLoader builder. Required for setting
   per-worker random seeds and opening per-worker file handles in
   multi-process mode.
4. **Add `persistent_workers`** option (useful at high `num_workers`).
5. **Add `timeout`** option for worker fetch (defensive for stuck I/O).
6. **Document the divergence**: `worker_mode(WorkerMode::Threaded)` is a
   ferrotorch concept (Rust threads share memory cheaply, unlike Python
   processes), and exists alongside the torch-equivalent `Multiprocess` and
   `Single` modes.

## Status

**Coverage ~85%.** Core abstractions (Dataset, Sampler, DataLoader,
collate, transforms) are present. Gaps are in supporting helpers (`Subset`,
`StackDataset`, `SubsetRandomSampler`) and DataLoader knobs
(`worker_init_fn`, `persistent_workers`, `timeout`).

**Do not split.** Maps cleanly to `torch.utils.data`.
