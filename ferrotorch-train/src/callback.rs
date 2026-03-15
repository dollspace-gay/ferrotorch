//! Training callbacks.
//!
//! Callbacks hook into the training loop at well-defined points (epoch
//! start/end, batch start/end, training end) and can observe or modify
//! training behavior.
//!
//! # Provided callbacks
//!
//! | Callback | Description |
//! |----------|-------------|
//! | [`EarlyStopping`] | Stop training when validation loss stops improving |
//! | [`ProgressLogger`] | Print epoch/batch progress to stdout |

use ferrotorch_core::Float;

use crate::history::{EpochResult, TrainingHistory};

// ---------------------------------------------------------------------------
// Callback trait
// ---------------------------------------------------------------------------

/// A callback that hooks into the [`Learner`](crate::Learner) training loop.
///
/// All methods have default no-op implementations so callbacks only need to
/// override the hooks they care about.
pub trait Callback<T: Float>: Send + Sync {
    /// Called at the start of each epoch.
    fn on_epoch_start(&mut self, _epoch: usize) {}

    /// Called at the end of each epoch with the epoch result.
    fn on_epoch_end(&mut self, _epoch: usize, _result: &EpochResult) {}

    /// Called at the start of each batch.
    fn on_batch_start(&mut self, _batch: usize) {}

    /// Called at the end of each batch with the batch loss.
    fn on_batch_end(&mut self, _batch: usize, _loss: f64) {}

    /// Called when the entire training run finishes.
    fn on_train_end(&mut self, _history: &TrainingHistory) {}

    /// Whether this callback requests early stopping.
    ///
    /// The [`Learner`](crate::Learner) checks this after each epoch. If any
    /// callback returns `true`, training stops.
    fn should_stop(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// EarlyStopping
// ---------------------------------------------------------------------------

/// Stop training when validation loss fails to improve for `patience` epochs.
///
/// "Improve" means decreasing by at least `min_delta`. The callback tracks the
/// best validation loss seen so far and increments a wait counter when no
/// improvement occurs. Training stops when `wait >= patience`.
///
/// # Examples
///
/// ```
/// use ferrotorch_train::EarlyStopping;
/// use ferrotorch_train::Callback;
///
/// let es = EarlyStopping::new(3, 0.001);
/// assert!(!Callback::<f32>::should_stop(&es));
/// ```
pub struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    best: f64,
    wait: usize,
    stopped: bool,
}

impl EarlyStopping {
    /// Create a new `EarlyStopping` callback.
    ///
    /// # Arguments
    ///
    /// * `patience` - Number of epochs with no improvement before stopping.
    /// * `min_delta` - Minimum decrease in validation loss to count as improvement.
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best: f64::INFINITY,
            wait: 0,
            stopped: false,
        }
    }

    /// Return the current best validation loss.
    pub fn best_loss(&self) -> f64 {
        self.best
    }

    /// Return the current wait counter.
    pub fn wait(&self) -> usize {
        self.wait
    }

    /// Return the patience value.
    pub fn patience(&self) -> usize {
        self.patience
    }
}

impl<T: Float> Callback<T> for EarlyStopping {
    fn on_epoch_end(&mut self, _epoch: usize, result: &EpochResult) {
        let val_loss = match result.val_loss {
            Some(vl) => vl,
            // No validation loss: nothing to monitor.
            None => return,
        };

        if val_loss < self.best - self.min_delta {
            // Improvement: reset wait counter.
            self.best = val_loss;
            self.wait = 0;
        } else {
            // No improvement.
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped = true;
            }
        }
    }

    fn should_stop(&self) -> bool {
        self.stopped
    }
}

// ---------------------------------------------------------------------------
// ProgressLogger
// ---------------------------------------------------------------------------

/// Prints training progress to stdout.
///
/// Logs epoch start/end and batch-level loss for visibility during training.
pub struct ProgressLogger {
    log_every_n_batches: usize,
}

impl ProgressLogger {
    /// Create a new `ProgressLogger`.
    ///
    /// # Arguments
    ///
    /// * `log_every_n_batches` - Print batch-level loss every N batches.
    ///   Set to 0 to disable batch-level logging.
    pub fn new(log_every_n_batches: usize) -> Self {
        Self {
            log_every_n_batches,
        }
    }
}

impl Default for ProgressLogger {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T: Float> Callback<T> for ProgressLogger {
    fn on_epoch_start(&mut self, epoch: usize) {
        println!("--- Epoch {epoch} ---");
    }

    fn on_epoch_end(&mut self, _epoch: usize, result: &EpochResult) {
        println!("{result}");
    }

    fn on_batch_end(&mut self, batch: usize, loss: f64) {
        if self.log_every_n_batches > 0 && batch % self.log_every_n_batches == 0 {
            println!("  batch {batch}: loss={loss:.6}");
        }
    }

    fn on_train_end(&mut self, history: &TrainingHistory) {
        println!("Training complete. {} epochs.", history.len());
        if let Some((epoch, loss)) = history.best_train_loss() {
            println!("Best train loss: {loss:.6} (epoch {epoch})");
        }
        if let Some((epoch, loss)) = history.best_val_loss() {
            println!("Best val loss: {loss:.6} (epoch {epoch})");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn make_epoch_result(epoch: usize, val_loss: Option<f64>) -> EpochResult {
        EpochResult {
            epoch,
            train_loss: 1.0,
            val_loss,
            metrics: HashMap::new(),
            lr: 0.001,
            duration_secs: 1.0,
        }
    }

    /// Helper: call `on_epoch_end` with `f32` as the `Float` type parameter.
    fn epoch_end(es: &mut EarlyStopping, epoch: usize, result: &EpochResult) {
        Callback::<f32>::on_epoch_end(es, epoch, result);
    }

    /// Helper: call `should_stop` with `f32` as the `Float` type parameter.
    fn stopped(es: &EarlyStopping) -> bool {
        Callback::<f32>::should_stop(es)
    }

    // -- EarlyStopping -------------------------------------------------------

    #[test]
    fn test_early_stopping_no_trigger_on_improvement() {
        let mut es = EarlyStopping::new(3, 0.001);
        epoch_end(&mut es, 0, &make_epoch_result(0, Some(1.0)));
        assert!(!stopped(&es));
        epoch_end(&mut es, 1, &make_epoch_result(1, Some(0.9)));
        assert!(!stopped(&es));
        epoch_end(&mut es, 2, &make_epoch_result(2, Some(0.8)));
        assert!(!stopped(&es));
    }

    #[test]
    fn test_early_stopping_triggers_after_patience() {
        let mut es = EarlyStopping::new(2, 0.0);
        // Set a baseline.
        epoch_end(&mut es, 0, &make_epoch_result(0, Some(1.0)));
        assert!(!stopped(&es));
        // No improvement.
        epoch_end(&mut es, 1, &make_epoch_result(1, Some(1.0)));
        assert!(!stopped(&es));
        assert_eq!(es.wait(), 1);
        // Still no improvement: patience exhausted.
        epoch_end(&mut es, 2, &make_epoch_result(2, Some(1.1)));
        assert!(stopped(&es));
        assert_eq!(es.wait(), 2);
    }

    #[test]
    fn test_early_stopping_resets_on_improvement() {
        let mut es = EarlyStopping::new(3, 0.0);
        epoch_end(&mut es, 0, &make_epoch_result(0, Some(1.0)));
        epoch_end(&mut es, 1, &make_epoch_result(1, Some(1.1))); // wait=1
        epoch_end(&mut es, 2, &make_epoch_result(2, Some(1.2))); // wait=2
        assert_eq!(es.wait(), 2);
        epoch_end(&mut es, 3, &make_epoch_result(3, Some(0.5))); // improvement, reset
        assert_eq!(es.wait(), 0);
        assert!(!stopped(&es));
    }

    #[test]
    fn test_early_stopping_min_delta() {
        let mut es = EarlyStopping::new(2, 0.1);
        epoch_end(&mut es, 0, &make_epoch_result(0, Some(1.0)));
        // Improvement is only 0.05 < min_delta (0.1).
        epoch_end(&mut es, 1, &make_epoch_result(1, Some(0.95)));
        assert_eq!(es.wait(), 1);
        // Real improvement: 1.0 - 0.8 = 0.2 > 0.1.
        epoch_end(&mut es, 2, &make_epoch_result(2, Some(0.8)));
        assert_eq!(es.wait(), 0);
    }

    #[test]
    fn test_early_stopping_ignores_no_val_loss() {
        let mut es = EarlyStopping::new(2, 0.0);
        epoch_end(&mut es, 0, &make_epoch_result(0, None));
        epoch_end(&mut es, 1, &make_epoch_result(1, None));
        epoch_end(&mut es, 2, &make_epoch_result(2, None));
        // No val_loss means nothing to monitor: should never stop.
        assert!(!stopped(&es));
    }

    #[test]
    fn test_early_stopping_best_loss() {
        let mut es = EarlyStopping::new(5, 0.0);
        assert!(es.best_loss().is_infinite());
        epoch_end(&mut es, 0, &make_epoch_result(0, Some(2.0)));
        assert!((es.best_loss() - 2.0).abs() < 1e-12);
        epoch_end(&mut es, 1, &make_epoch_result(1, Some(1.5)));
        assert!((es.best_loss() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_early_stopping_patience_accessor() {
        let es = EarlyStopping::new(7, 0.01);
        assert_eq!(es.patience(), 7);
    }

    // -- ProgressLogger ------------------------------------------------------

    #[test]
    fn test_progress_logger_construction() {
        let pl = ProgressLogger::new(10);
        assert_eq!(pl.log_every_n_batches, 10);
    }

    #[test]
    fn test_progress_logger_default() {
        let pl = ProgressLogger::default();
        assert_eq!(pl.log_every_n_batches, 0);
    }

    #[test]
    fn test_progress_logger_should_stop_always_false() {
        let pl = ProgressLogger::new(10);
        assert!(!Callback::<f32>::should_stop(&pl));
    }

    // -- Send + Sync ---------------------------------------------------------

    #[test]
    fn test_callbacks_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EarlyStopping>();
        assert_send_sync::<ProgressLogger>();
    }
}
