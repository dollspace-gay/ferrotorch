use std::cell::Cell;

thread_local! {
    static AUTOCAST_ENABLED: Cell<bool> = const { Cell::new(false) };
    static AUTOCAST_DTYPE: Cell<AutocastDtype> = const { Cell::new(AutocastDtype::F16) };
    static AUTOCAST_DEBUG: Cell<bool> = const { Cell::new(false) };
}

/// Enable or disable autocast event recording on this thread.
///
/// When enabled, [`autocast_guard`](super::autocast_ops::autocast_guard)
/// appends entries to the per-thread event log, which is useful for tests
/// and diagnostics. When disabled (the default), only the category is
/// returned and no events are recorded.
///
/// This is a thread-local setting (like autocast itself) so enabling debug
/// on one thread does not affect others.
pub fn set_autocast_debug(enabled: bool) {
    AUTOCAST_DEBUG.with(|d| d.set(enabled));
}

/// Returns `true` if autocast debug event recording is active on this thread.
pub fn is_autocast_debug() -> bool {
    AUTOCAST_DEBUG.with(|d| d.get())
}

/// The reduced-precision dtype used during autocast regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutocastDtype {
    /// IEEE 754 half-precision (1-5-10).
    F16,
    /// Brain floating point (1-8-7). Wider dynamic range than f16.
    BF16,
}

/// Returns `true` if mixed-precision autocast is currently enabled on this thread.
pub fn is_autocast_enabled() -> bool {
    AUTOCAST_ENABLED.with(|e| e.get())
}

/// Returns the target dtype for autocast regions on this thread.
///
/// Only meaningful when [`is_autocast_enabled`] returns `true`.
pub fn autocast_dtype() -> AutocastDtype {
    AUTOCAST_DTYPE.with(|d| d.get())
}

/// A snapshot of the per-thread autocast state at a point in time.
///
/// Used by gradient checkpointing (and any other op that needs to recreate
/// the same autocast context later) to restore the exact (enabled, dtype)
/// pair that was active during the forward pass when the backward
/// recomputation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutocastSnapshot {
    /// Whether autocast was enabled when the snapshot was captured.
    pub enabled: bool,
    /// The autocast dtype at snapshot time. Only meaningful when `enabled`
    /// is true, but we always carry it so the snapshot can round-trip
    /// faithfully (e.g. if a future API also restores the dtype cell when
    /// disabled).
    pub dtype: AutocastDtype,
}

/// Capture the current per-thread autocast state.
///
/// Pair this with [`with_autocast_state`] to recreate the same autocast
/// context later, e.g. for gradient checkpointing where the forward and
/// recomputed-during-backward passes must produce numerically identical
/// activations.
pub fn current_autocast_snapshot() -> AutocastSnapshot {
    AutocastSnapshot {
        enabled: is_autocast_enabled(),
        dtype: autocast_dtype(),
    }
}

/// Execute a closure with the autocast state set to a captured snapshot.
///
/// On entry, the previous (enabled, dtype) is saved; on exit (including
/// panic unwind), the previous state is restored via an RAII guard.
///
/// This is more general than [`autocast`]: it can also restore the
/// "disabled" state if the snapshot was captured outside any autocast
/// region. Used by gradient checkpointing to ensure the backward
/// recomputation runs with the exact autocast configuration that was
/// active when the original forward pass ran.
pub fn with_autocast_state<F, R>(snapshot: AutocastSnapshot, f: F) -> R
where
    F: FnOnce() -> R,
{
    struct StateGuard {
        prev_enabled: bool,
        prev_dtype: AutocastDtype,
    }

    impl Drop for StateGuard {
        fn drop(&mut self) {
            AUTOCAST_ENABLED.with(|e| e.set(self.prev_enabled));
            AUTOCAST_DTYPE.with(|d| d.set(self.prev_dtype));
        }
    }

    let _guard = StateGuard {
        prev_enabled: is_autocast_enabled(),
        prev_dtype: autocast_dtype(),
    };
    AUTOCAST_ENABLED.with(|e| e.set(snapshot.enabled));
    AUTOCAST_DTYPE.with(|d| d.set(snapshot.dtype));
    f()
}

/// Execute a closure with mixed-precision autocast enabled.
///
/// Operations that benefit from lower precision (matmul, conv) will use
/// `dtype` (f16 or bf16), while operations needing full precision
/// (reductions, norms) stay in f32. The actual casting is handled by
/// individual ops that check [`is_autocast_enabled`].
///
/// Uses an RAII guard so that the previous autocast state is restored even
/// if `f` panics (unwinding safety).
///
/// Calls can be nested safely — the outermost `autocast` restores the
/// previous state.
///
/// # Example
///
/// ```
/// use ferrotorch_core::autograd::autocast::{autocast, is_autocast_enabled, AutocastDtype};
///
/// autocast(AutocastDtype::F16, || {
///     assert!(is_autocast_enabled());
///     // matmul, conv, etc. would run in f16 here.
/// });
/// assert!(!is_autocast_enabled());
/// ```
pub fn autocast<F, R>(dtype: AutocastDtype, f: F) -> R
where
    F: FnOnce() -> R,
{
    // RAII guard restores previous state on drop (including panic unwind).
    struct AutocastGuard {
        prev_enabled: bool,
        prev_dtype: AutocastDtype,
    }

    impl Drop for AutocastGuard {
        fn drop(&mut self) {
            AUTOCAST_ENABLED.with(|e| e.set(self.prev_enabled));
            AUTOCAST_DTYPE.with(|d| d.set(self.prev_dtype));
        }
    }

    let _guard = AutocastGuard {
        prev_enabled: is_autocast_enabled(),
        prev_dtype: autocast_dtype(),
    };
    AUTOCAST_ENABLED.with(|e| e.set(true));
    AUTOCAST_DTYPE.with(|d| d.set(dtype));
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocast_default_disabled() {
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_autocast_enables() {
        assert!(!is_autocast_enabled());
        autocast(AutocastDtype::F16, || {
            assert!(is_autocast_enabled());
        });
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_autocast_nested() {
        assert!(!is_autocast_enabled());
        autocast(AutocastDtype::F16, || {
            assert!(is_autocast_enabled());
            assert_eq!(autocast_dtype(), AutocastDtype::F16);

            autocast(AutocastDtype::BF16, || {
                assert!(is_autocast_enabled());
                assert_eq!(autocast_dtype(), AutocastDtype::BF16);
            });

            // Inner autocast restores the outer dtype.
            assert!(is_autocast_enabled());
            assert_eq!(autocast_dtype(), AutocastDtype::F16);
        });
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_autocast_dtype_selection() {
        autocast(AutocastDtype::BF16, || {
            assert_eq!(autocast_dtype(), AutocastDtype::BF16);
        });

        autocast(AutocastDtype::F16, || {
            assert_eq!(autocast_dtype(), AutocastDtype::F16);
        });
    }

    #[test]
    fn test_default_dtype_is_f16() {
        // Before any autocast call, the default dtype cell value is F16.
        assert_eq!(autocast_dtype(), AutocastDtype::F16);
    }

    #[test]
    fn test_autocast_panic_safety() {
        // Verify the RAII guard restores state after a panic.
        let result = std::panic::catch_unwind(|| {
            autocast(AutocastDtype::BF16, || {
                assert!(is_autocast_enabled());
                panic!("intentional panic inside autocast");
            });
        });
        assert!(result.is_err());
        // State must be restored even after the panic.
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_autocast_debug_flag() {
        // Default is off.
        assert!(!is_autocast_debug());
        set_autocast_debug(true);
        assert!(is_autocast_debug());
        set_autocast_debug(false);
        assert!(!is_autocast_debug());
    }
}
