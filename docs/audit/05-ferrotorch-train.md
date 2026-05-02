# Audit: `ferrotorch-train` vs PyTorch training conventions

PyTorch has **no built-in training loop**. The closest analogs are:
- **Lightning / fastai / Ignite / Hugging Face Trainer** (third-party)
- The implicit "training loop pattern" in PyTorch tutorials
- `torch.utils.tensorboard` for logging (in `torch/utils`)

This crate is therefore *additive* relative to torch: it provides what
torch deliberately doesn't.

## Components

| ferrotorch | Equivalent in PyTorch ecosystem |
|---|---|
| `Learner::new(model, optimizer, loss_fn).fit(...)` | Lightning `Trainer().fit(LightningModule)` / fastai `Learner` |
| `Callback` trait (on_epoch_start/end, on_batch_start/end, on_train_end, should_stop) | Lightning `Callback` |
| `EarlyStopping` | Lightning `EarlyStopping` |
| `ProgressLogger` | Lightning `ProgressBar` / fastai `ProgressCallback` |
| `EmaCallback` | Lightning `StochasticWeightAveraging`-adjacent / EMA callbacks |
| `TensorBoardCallback`, `TensorBoardWriter` | `torch.utils.tensorboard.SummaryWriter` |
| `Metric` trait, `LossMetric`, `AccuracyMetric`, `TopKAccuracy`, `RunningAverage` | `torchmetrics` (third-party) / Lightning `MetricCollection` |
| `EpochResult`, `EvalResult`, `TrainingHistory` | Lightning `Trainer.callback_metrics` / per-step dict |
| `checkpoint`, `checkpoint_sequential` | `torch.utils.checkpoint.checkpoint` (alias for ferrotorch-core's checkpoint) |
| `clip_grad_norm_`, `clip_grad_value_` | `torch.nn.utils.clip_grad_norm_` (also in `ferrotorch-nn::utils` — duplication?) |
| `amp` module | `torch.amp` (autocast + GradScaler integration in optim) |

## Architectural notes

- **`Learner` is the single entry point**, mirroring fastai. Builder
  pattern with `.with_scheduler()`, `.with_train_metric()`,
  `.with_val_metric()`, `.with_callback()`, `.with_checkpointing()`.
- **`fit(train_loader, val_loader, epochs)`** vs Lightning's
  `Trainer().fit(model, train_loader, val_loader)` — semantically equivalent.
- **`evaluate(val_data)`** for inference-mode eval pass.
- **`load_checkpoint(path)`** built-in (Lightning has it too).

## Coverage assessment

Comparing to **Lightning** (the most-used PyTorch trainer):

| Lightning feature | ferrotorch-train | Status |
|---|---|---|
| Trainer.fit / .test / .predict / .validate | `Learner::fit` / `evaluate` | partial — no `predict`/`test` distinct from `evaluate` |
| LightningModule lifecycle (training_step, validation_step, configure_optimizers) | model trait + loss_fn + optimizer constructor args | ✅ direct equivalent (Rust trait-based instead of method overrides) |
| Callbacks (EarlyStopping, ModelCheckpoint, LearningRateMonitor, GradientAccumulationScheduler, BackboneFinetuning, ...) | `EarlyStopping`, `EmaCallback`, `ProgressLogger`, `TensorBoardCallback`, `with_checkpointing` | partial |
| Loggers (TensorBoard, MLflow, Weights & Biases, Comet, CSV, Neptune) | `TensorBoardCallback` only | partial |
| Multi-GPU / distributed | (depends on `ferrotorch-distributed`) | not yet integrated |
| Mixed precision (16/bf16, native autocast) | `amp.rs` (autocast + GradScaler) | ✅ |
| Gradient clipping | `clip_grad_norm_`, `clip_grad_value_` | ✅ |
| Gradient accumulation | not on `Learner` directly (use `GradientAccumulator` from optim) | partial |
| `accelerator='cpu'/'gpu'/'mps'/'tpu'` | implicit via Device on tensors | partial — no explicit accelerator config |
| Checkpoint resume | `load_checkpoint(path)` | ✅ |
| Profiler integration | (use `ferrotorch-profiler` directly) | partial — no callback wrapper |
| Model summary (params + FLOPs estimate) | not exposed | gap |
| `LearningRateMonitor` callback (logs LR per step) | not exposed | gap |
| `ModelCheckpoint(save_top_k, monitor, mode)` | basic save-every-epoch via `with_checkpointing` | gap on top-K, monitor metric, save_last |
| LR finder | not exposed | gap |

## Gaps to fill

1. **`predict` / `test` modes** distinct from `evaluate`, with their own
   callback hooks (`on_predict_*`, `on_test_*`).
2. **`ModelCheckpoint` callback** with `save_top_k`, `monitor`, `mode`,
   `save_last`, `every_n_epochs`. Currently `with_checkpointing(dir)` is
   indiscriminate.
3. **`LearningRateMonitor`** callback — log LR per step alongside loss.
4. **`GradientAccumulationScheduler`** integrated as a callback (currently
   manual via `GradientAccumulator` in optim).
5. **More logger integrations** — at minimum a CSV logger and a
   pluggable `Logger` trait so MLflow/W&B/Comet hooks are externable.
6. **Model summary** — print parameter counts per layer, total params,
   trainable params, estimated FLOPs.
7. **LR finder** — sweep LR exponentially, plot loss, recommend.
8. **Hooks for distributed** — once `ferrotorch-distributed` integration
   lands, `Learner` should auto-wrap model in DDP and stride samplers.
9. **De-duplicate `clip_grad_*`** — currently in both
   `ferrotorch-train::grad_utils` and `ferrotorch-nn::utils`. Pick one home
   (probably `ferrotorch-nn::utils`, re-export from train if convenient).

## Status

ferrotorch-train fills a niche torch leaves to third parties. It's
broadly a **subset of Lightning** today: solid base loop, callback trait,
metric trait, history, AMP, checkpointing, tensorboard. The 9 gaps above
are all incremental additions inside the existing crate boundary.

**Do not split.** `Learner`/`Callback`/`Metric` belong together.

## Note on philosophy

If the goal is "PyTorch in Rust" (pure parity), this crate could be
*omitted* — torch users write their own loops, and `ferrotorch-train` is
extra. If the goal is "ergonomic deep learning in Rust", this crate is the
single biggest UX improvement over a literal pytorch port. Recommend
keeping and expanding it.
