//! Integration tests: end-to-end computation graphs, thread safety,
//! edge cases, and numerical gradient verification.

use ferrotorch_core::*;

// ============================================================
// End-to-end training-like graphs
// ============================================================

#[test]
fn test_linear_regression_graph() {
    // y = w * x + b, loss = (y - target)^2
    let x = from_slice(&[2.0f32], &[1]).unwrap().requires_grad_(false);
    let w = from_slice(&[0.5f32], &[1]).unwrap().requires_grad_(true);
    let b = from_slice(&[0.1f32], &[1]).unwrap().requires_grad_(true);
    let target = from_slice(&[1.0f32], &[1]).unwrap();

    let wx = grad_fns::arithmetic::mul(&w, &x).unwrap();
    let y = grad_fns::arithmetic::add(&wx, &b).unwrap();
    let diff = grad_fns::arithmetic::sub(&y, &target).unwrap();
    let loss = grad_fns::arithmetic::mul(&diff, &diff).unwrap();
    let loss_scalar = grad_fns::reduction::sum(&loss).unwrap();
    loss_scalar.backward().unwrap();

    // y = 1.1, diff = 0.1, loss = 0.01
    // dloss/dw = 2 * 0.1 * 2.0 = 0.4
    // dloss/db = 2 * 0.1 = 0.2
    let w_grad = w.grad().unwrap().unwrap();
    let b_grad = b.grad().unwrap().unwrap();
    assert!((w_grad.data().unwrap()[0] - 0.4).abs() < 1e-5);
    assert!((b_grad.data().unwrap()[0] - 0.2).abs() < 1e-5);
}

#[test]
fn test_multi_layer_graph() {
    // z = relu(w1 * x), out = w2 * z
    let x = scalar(3.0f32).unwrap().requires_grad_(true);
    let w1 = scalar(0.5f32).unwrap().requires_grad_(true);
    let w2 = scalar(-2.0f32).unwrap().requires_grad_(true);

    let h = grad_fns::arithmetic::mul(&w1, &x).unwrap();
    let z = grad_fns::activation::relu(&h).unwrap();
    let out = grad_fns::arithmetic::mul(&w2, &z).unwrap();
    out.backward().unwrap();

    let w2_grad = w2.grad().unwrap().unwrap();
    assert!((w2_grad.item().unwrap() - 1.5).abs() < 1e-5);

    let w1_grad = w1.grad().unwrap().unwrap();
    assert!((w1_grad.item().unwrap() - (-6.0)).abs() < 1e-5);
}

// ============================================================
// Thread safety (AC-12)
// ============================================================

#[test]
fn test_send_sync_across_threads() {
    use std::thread;

    let a = scalar(2.0f32).unwrap().requires_grad_(true);
    let b = scalar(3.0f32).unwrap().requires_grad_(true);
    let c = grad_fns::arithmetic::mul(&a, &b).unwrap();

    let a_clone = a.clone();
    let b_clone = b.clone();
    let handle = thread::spawn(move || {
        c.backward().unwrap();
        let a_grad = a_clone.grad().unwrap().unwrap();
        let b_grad = b_clone.grad().unwrap().unwrap();
        (a_grad.item().unwrap(), b_grad.item().unwrap())
    });

    let (a_g, b_g) = handle.join().unwrap();
    assert!((a_g - 3.0).abs() < 1e-6);
    assert!((b_g - 2.0).abs() < 1e-6);
}

#[test]
fn test_tensor_clone_across_threads() {
    use std::thread;

    let t = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
    let t_clone = t.clone();
    let handle = thread::spawn(move || t_clone.data().unwrap().to_vec());
    assert_eq!(handle.join().unwrap(), vec![1.0, 2.0, 3.0]);
}

// ============================================================
// no_grad context
// ============================================================

#[test]
fn test_no_grad_inside_grad_context() {
    let a = scalar(2.0f32).unwrap().requires_grad_(true);
    let result = no_grad(|| {
        let b = grad_fns::arithmetic::mul(&a, &a).unwrap();
        assert!(b.grad_fn().is_none());
        b
    });
    assert!(result.grad_fn().is_none());
}

// ============================================================
// Edge cases
// ============================================================

#[test]
fn test_scalar_tensor_operations() {
    let a = scalar(5.0f32).unwrap().requires_grad_(true);
    let b = scalar(3.0f32).unwrap().requires_grad_(true);
    let c = grad_fns::arithmetic::add(&a, &b).unwrap();
    assert!(c.is_scalar());
    assert_eq!(c.item().unwrap(), 8.0);
}

#[test]
fn test_zero_element_tensor() {
    let t: Tensor<f32> = zeros(&[0, 3]).unwrap();
    assert_eq!(t.numel(), 0);
}

#[test]
fn test_large_rank_tensor() {
    let t: Tensor<f32> = ones(&[2, 2, 2, 2, 2, 2, 2, 2]).unwrap();
    assert_eq!(t.ndim(), 8);
    assert_eq!(t.numel(), 256);
}

#[test]
fn test_single_element_tensor_item() {
    let t = from_slice(&[42.0f32], &[1, 1, 1]).unwrap();
    assert_eq!(t.item().unwrap(), 42.0);
}

#[test]
fn test_backward_on_non_scalar_errors() {
    let t = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
    assert!(t.backward().is_err());
}

#[test]
fn test_detach_stops_gradient() {
    let a = scalar(3.0f32).unwrap().requires_grad_(true);
    let b = grad_fns::arithmetic::mul(&a, &a).unwrap();
    let c = b.detach();
    assert!(c.grad_fn().is_none());
    assert!(!c.requires_grad());
}

#[test]
fn test_requires_grad_toggle() {
    let t = scalar(1.0f32).unwrap();
    assert!(!t.requires_grad());
    let t2 = t.requires_grad_(true);
    assert!(t2.requires_grad());
}

#[test]
fn test_is_contiguous() {
    let t = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert!(t.is_contiguous());
}

// ============================================================
// Numerical gradient checks (finite differences)
// ============================================================

fn numerical_grad<F>(f: F, x: f32, h: f32) -> f32
where
    F: Fn(f32) -> f32,
{
    (f(x + h) - f(x - h)) / (2.0 * h)
}

#[test]
fn test_numerical_grad_mul() {
    // f(x) = x * 3.0 at x = 2.0 => df/dx = 3.0
    // Both inputs need requires_grad for mul to attach a grad_fn,
    // but we only need the gradient w.r.t. x.
    let analytic = {
        let x = scalar(2.0f32).unwrap().requires_grad_(true);
        let three = scalar(3.0f32).unwrap().requires_grad_(true);
        let y = grad_fns::arithmetic::mul(&x, &three).unwrap();
        y.backward().unwrap();
        x.grad().unwrap().unwrap().item().unwrap()
    };
    let numeric = numerical_grad(|x| x * 3.0, 2.0, 1e-4);
    assert!(
        (analytic - numeric).abs() < 1e-2,
        "analytic={analytic}, numeric={numeric}"
    );
}

#[test]
fn test_numerical_grad_pow() {
    let analytic = {
        let x = scalar(2.0f32).unwrap().requires_grad_(true);
        let y = grad_fns::arithmetic::pow(&x, 3.0).unwrap();
        y.backward().unwrap();
        x.grad().unwrap().unwrap().item().unwrap()
    };
    let numeric = numerical_grad(|x| x.powi(3), 2.0, 1e-4);
    assert!((analytic - numeric).abs() < 1e-2);
}

#[test]
fn test_numerical_grad_sqrt() {
    let analytic = {
        let x = scalar(4.0f32).unwrap().requires_grad_(true);
        let y = grad_fns::arithmetic::sqrt(&x).unwrap();
        y.backward().unwrap();
        x.grad().unwrap().unwrap().item().unwrap()
    };
    let numeric = numerical_grad(|x| x.sqrt(), 4.0, 1e-4);
    assert!((analytic - numeric).abs() < 1e-3);
}

#[test]
fn test_numerical_grad_sigmoid() {
    let analytic = {
        let x = scalar(1.0f32).unwrap().requires_grad_(true);
        let y = grad_fns::activation::sigmoid(&x).unwrap();
        y.backward().unwrap();
        x.grad().unwrap().unwrap().item().unwrap()
    };
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let numeric = numerical_grad(sigmoid, 1.0, 1e-4);
    assert!((analytic - numeric).abs() < 1e-3);
}

#[test]
fn test_numerical_grad_tanh() {
    let analytic = {
        let x = scalar(0.5f32).unwrap().requires_grad_(true);
        let y = grad_fns::activation::tanh(&x).unwrap();
        y.backward().unwrap();
        x.grad().unwrap().unwrap().item().unwrap()
    };
    let numeric = numerical_grad(|x| x.tanh(), 0.5, 1e-4);
    assert!((analytic - numeric).abs() < 1e-3);
}

#[test]
fn test_numerical_grad_relu_positive() {
    let analytic = {
        let x = scalar(2.0f32).unwrap().requires_grad_(true);
        let y = grad_fns::activation::relu(&x).unwrap();
        y.backward().unwrap();
        x.grad().unwrap().unwrap().item().unwrap()
    };
    assert!((analytic - 1.0).abs() < 1e-6);
}

#[test]
fn test_numerical_grad_relu_negative() {
    let analytic = {
        let x = scalar(-1.0f32).unwrap().requires_grad_(true);
        let y = grad_fns::activation::relu(&x).unwrap();
        y.backward().unwrap();
        x.grad().unwrap().unwrap().item().unwrap()
    };
    assert!((analytic - 0.0).abs() < 1e-6);
}

#[test]
fn test_numerical_grad_div() {
    let (da, db) = {
        let a = scalar(6.0f32).unwrap().requires_grad_(true);
        let b = scalar(3.0f32).unwrap().requires_grad_(true);
        let y = grad_fns::arithmetic::div(&a, &b).unwrap();
        y.backward().unwrap();
        (
            a.grad().unwrap().unwrap().item().unwrap(),
            b.grad().unwrap().unwrap().item().unwrap(),
        )
    };
    assert!((da - 1.0 / 3.0).abs() < 1e-4);
    assert!((db - (-6.0 / 9.0)).abs() < 1e-4);
}

#[test]
fn test_numerical_grad_chain_rule() {
    // f(x) = sigmoid(x^2) at x = 1.0
    let analytic = {
        let x = scalar(1.0f32).unwrap().requires_grad_(true);
        let x_sq = grad_fns::arithmetic::pow(&x, 2.0).unwrap();
        let y = grad_fns::activation::sigmoid(&x_sq).unwrap();
        y.backward().unwrap();
        x.grad().unwrap().unwrap().item().unwrap()
    };
    let s1 = 1.0f32 / (1.0 + (-1.0f32).exp());
    let expected = s1 * (1.0 - s1) * 2.0;
    assert!((analytic - expected).abs() < 1e-3);
}

// ============================================================
// Creation function edge cases
// ============================================================

#[test]
fn test_eye_1x1() {
    let t: Tensor<f32> = eye(1).unwrap();
    assert_eq!(t.item().unwrap(), 1.0);
}

#[test]
fn test_arange_negative_step() {
    let t: Tensor<f32> = arange(5.0, 0.0, -1.0).unwrap();
    assert_eq!(t.data().unwrap(), &[5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_rand_different_values() {
    let a: Tensor<f32> = rand(&[100]).unwrap();
    let b: Tensor<f32> = rand(&[100]).unwrap();
    assert_ne!(a.data().unwrap(), b.data().unwrap());
}

#[test]
fn test_f64_operations() {
    let a = scalar(2.0f64).unwrap().requires_grad_(true);
    let b = scalar(3.0f64).unwrap().requires_grad_(true);
    let c = grad_fns::arithmetic::mul(&a, &b).unwrap();
    c.backward().unwrap();
    let a_grad = a.grad().unwrap().unwrap();
    assert!((a_grad.item().unwrap() - 3.0f64).abs() < 1e-10);
}
