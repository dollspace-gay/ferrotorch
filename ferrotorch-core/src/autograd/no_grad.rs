use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns `true` if gradient tracking is currently enabled on this thread.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// Execute a closure with gradient tracking disabled.
///
/// Tensors created inside the closure will have `requires_grad = false`
/// regardless of their inputs. This is used for inference and for
/// manually updating parameters (e.g., SGD step) without recording
/// the update in the computation graph.
///
/// Calls can be nested safely — the outermost `no_grad` restores the
/// previous state.
///
/// # Example
///
/// ```
/// use ferrotorch_core::autograd::no_grad;
///
/// no_grad(|| {
///     // All tensor operations here are untracked.
/// });
/// ```
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev = GRAD_ENABLED.with(|g| {
        let prev = g.get();
        g.set(false);
        prev
    });
    let result = f();
    GRAD_ENABLED.with(|g| g.set(prev));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_enabled_default() {
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_no_grad_disables() {
        assert!(is_grad_enabled());
        no_grad(|| {
            assert!(!is_grad_enabled());
        });
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_no_grad_nested() {
        assert!(is_grad_enabled());
        no_grad(|| {
            assert!(!is_grad_enabled());
            no_grad(|| {
                assert!(!is_grad_enabled());
            });
            // Inner no_grad restores to false (the outer no_grad's state).
            assert!(!is_grad_enabled());
        });
        assert!(is_grad_enabled());
    }
}
