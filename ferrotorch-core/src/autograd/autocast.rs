use std::cell::Cell;

thread_local! {
    static AUTOCAST_ENABLED: Cell<bool> = const { Cell::new(false) };
    static AUTOCAST_DTYPE: Cell<AutocastDtype> = const { Cell::new(AutocastDtype::F16) };
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

/// Execute a closure with mixed-precision autocast enabled.
///
/// Operations that benefit from lower precision (matmul, conv) will use
/// `dtype` (f16 or bf16), while operations needing full precision
/// (reductions, norms) stay in f32. The actual casting is handled by
/// individual ops that check [`is_autocast_enabled`].
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
    let prev_enabled = AUTOCAST_ENABLED.with(|e| {
        let prev = e.get();
        e.set(true);
        prev
    });
    let prev_dtype = AUTOCAST_DTYPE.with(|d| {
        let prev = d.get();
        d.set(dtype);
        prev
    });
    let result = f();
    AUTOCAST_DTYPE.with(|d| d.set(prev_dtype));
    AUTOCAST_ENABLED.with(|e| e.set(prev_enabled));
    result
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
}
