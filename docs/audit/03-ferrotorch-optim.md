# Audit: `ferrotorch-optim` vs `torch.optim`

Covers: `torch.optim.*` optimizers, `torch.optim.lr_scheduler.*`,
`torch.optim.swa_utils`, `torch.amp.GradScaler`.

## Optimizers

| ferrotorch | torch.optim | Notes |
|---|---|---|
| `Sgd` (+ momentum, dampening, nesterov) | `SGD` | ✅ |
| `Adam` | `Adam` | ✅ |
| `AdamW` | `AdamW` | ✅ |
| `Adamax` | `Adamax` | ✅ |
| `NAdam` | `NAdam` | ✅ |
| `RAdam` | `RAdam` | ✅ |
| `Adagrad` | `Adagrad` | ✅ |
| `Adadelta` | `Adadelta` | ✅ |
| `Adafactor` | `Adafactor` | ✅ |
| `Asgd` | `ASGD` | ✅ |
| `Rmsprop` | `RMSprop` | ✅ |
| `Rprop` | `Rprop` | ✅ |
| `Lbfgs` (+ `LineSearchFn`) | `LBFGS` | ✅ |
| `SparseAdam` | `SparseAdam` | ✅ |
| `Muon` | `_muon` (private) | ✅ exposed by ferrotorch |
| `Kfac` (KFAC natural gradient) | none | **extra** — natural-gradient family |
| `ExponentialMovingAverage` | none (Lightning-style external) | **extra** |
| `GradientAccumulator` | done via manual `loss.backward()` looping | **extra** |
| `differentiable::diff_sgd_step`, `diff_sgd_momentum_step` | `torch.optim._functional` (partial) | ✅ ferrotorch surfaces them publicly |
| `foreach_utils` | `torch.optim._multi_tensor` | ✅ structurally aligned |

**Coverage: 14 of 14 torch optimizers, plus 3 extras (Muon promoted, KFAC,
EMA, GradientAccumulator).**

## LR schedulers

| ferrotorch | torch.optim.lr_scheduler |
|---|---|
| `LrScheduler` (trait) | `LRScheduler` (base class) |
| `StepLR` | `StepLR` |
| `MultiStepLR` | `MultiStepLR` |
| `LambdaLR` | `LambdaLR` |
| `MultiplicativeLR` | `MultiplicativeLR` |
| `ConstantLR` | `ConstantLR` |
| `LinearLR` | `LinearLR` |
| `ExponentialLR` | `ExponentialLR` |
| `PolynomialLR` | `PolynomialLR` |
| `CosineAnnealingLR` | `CosineAnnealingLR` |
| `CosineAnnealingWarmRestarts` | `CosineAnnealingWarmRestarts` |
| `CyclicLR` (+ `CyclicMode`, `AnnealStrategy`) | `CyclicLR` |
| `OneCycleLR` | `OneCycleLR` |
| `SequentialLr` | `SequentialLR` |
| `ChainedScheduler` | `ChainedScheduler` |
| `ReduceLROnPlateau` (+ `PlateauMode`, `MetricScheduler`) | `ReduceLROnPlateau` |
| `LinearWarmup`, `cosine_warmup_scheduler` | none in torch core | **extra** |

**Coverage: 16 of 16 torch schedulers, plus warmup helpers.**

## Mixed precision

| ferrotorch | torch |
|---|---|
| `GradScaler` (`grad_scaler.rs`) — `GradScalerConfig`, `GradScalerState` | `torch.amp.GradScaler` (was `torch.cuda.amp.GradScaler`) |

**Coverage: ✅** — `GradScaler` lives in `ferrotorch-optim`. (Earlier
`ferrotorch-core` audit flagged "no GradScaler in core" — it's correctly in
optim, matching torch where `torch.amp.GradScaler` is reused by optimizers.)

## SWA

| ferrotorch | torch |
|---|---|
| `swa::AveragedModel` | `swa_utils.AveragedModel` |
| `swa::AveragingStrategy` | (functions: `update_parameters`, `_get_ema_avg_fn`) |
| `swa::Swalr` | `swa_utils.SWALR` |

**Coverage: ✅** — full SWA path.

## Optimizer trait

| Method | ferrotorch | torch.optim.Optimizer |
|---|---|---|
| Step | `step()` | `step(closure=None)` |
| Zero gradients | `zero_grad()` | `zero_grad(set_to_none=True)` |
| Read LR | `lr()` | property on `param_groups[0]['lr']` |
| Write LR | `set_lr()` | mutate `param_groups[i]['lr']` |
| Param groups | `param_groups()` / `param_groups_mut()` | `param_groups` (list-of-dict) |
| Add param group | `add_param_group(group)` | `add_param_group(group)` |
| State dict | `state_dict()` / `load_state_dict()` | same |
| Closure | (not on trait) | passed to `step(closure)` for LBFGS-like |

**Gaps on Optimizer trait:**
- No closure-form `step(closure)` — needed by LBFGS to allow re-evaluation.
  (`Lbfgs` likely takes its closure differently — should verify.)
- No `register_step_pre_hook` / `register_step_post_hook` /
  `register_state_dict_pre_hook` / `register_state_dict_post_hook`.
- No `set_to_none` flag on `zero_grad()` — torch defaults to setting grads
  to `None` rather than zero (faster, makes `.grad is None` checks work).

## Recommendations

1. **Add closure-style `step(closure)`** on the `Optimizer` trait (or a
   sibling trait) so LBFGS can be polymorphic with everything else.
2. **Add `zero_grad(set_to_none: bool)`** option, or document the chosen
   semantic and stick to it.
3. **Add hook-registration methods** to `Optimizer` trait (parallel to the
   `Module` trait recommendation in `02-ferrotorch-nn.md`).
4. **Confirm `Kfac`/Muon/EMA naming** matches what users expect coming from
   torch (Muon is `torch.optim._muon` — currently private upstream — keep
   ferrotorch's public exposure).

## Status

**ferrotorch-optim is the most-complete crate vs its torch counterpart.**
Every torch optimizer and every torch scheduler is present, plus extras
(KFAC, EMA, Muon, gradient accumulator, warmup helpers). Trait gaps are the
only remaining work: closure-style step, zero_grad mode, hooks.

**Do not split.** The crate boundary matches torch.optim cleanly.
