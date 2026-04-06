#!/usr/bin/env python3
"""NumPy performance benchmarks for comparison with ferrotorch/ferray."""

import time
import json
import numpy as np

def bench(name, fn, warmup=5, iters=100):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) / iters * 1e6
    print(f"  {name}: {elapsed:.1f} us")
    return elapsed

def main():
    print(f"NumPy {np.__version__}")
    np_config = np.show_config(mode="dicts") if hasattr(np, "show_config") else {}
    print()

    results = {}

    print("=" * 60)
    print("NUMPY CPU BENCHMARKS")
    print("=" * 60)

    # Tensor creation
    print("\n--- Array Creation ---")
    results["zeros_1000x1000"] = bench("zeros [1000,1000]", lambda: np.zeros((1000, 1000), dtype=np.float32))
    results["rand_1000x1000"] = bench("rand [1000,1000]", lambda: np.random.rand(1000, 1000).astype(np.float32))

    # Elementwise ops
    print("\n--- Elementwise Ops ---")
    a = np.random.rand(1000, 1000).astype(np.float32)
    b = np.random.rand(1000, 1000).astype(np.float32)
    results["add_1000x1000"] = bench("add [1000,1000]", lambda: np.add(a, b))
    results["sub_1000x1000"] = bench("sub [1000,1000]", lambda: np.subtract(a, b))
    results["mul_1000x1000"] = bench("mul [1000,1000]", lambda: np.multiply(a, b))
    results["div_1000x1000"] = bench("div [1000,1000]", lambda: np.divide(a, b))
    results["neg_1000x1000"] = bench("neg [1000,1000]", lambda: np.negative(a))
    results["abs_1000x1000"] = bench("abs [1000,1000]", lambda: np.abs(a))

    # Activation-like
    print("\n--- Activations ---")
    results["relu_1000x1000"] = bench("relu [1000,1000]", lambda: np.maximum(a, 0))
    results["sigmoid_1000x1000"] = bench("sigmoid [1000,1000]", lambda: 1.0 / (1.0 + np.exp(-a)))
    results["tanh_1000x1000"] = bench("tanh [1000,1000]", lambda: np.tanh(a))

    # Matrix multiply
    print("\n--- Matrix Multiply ---")
    for size in [64, 256, 1024]:
        a_m = np.random.rand(size, size).astype(np.float32)
        b_m = np.random.rand(size, size).astype(np.float32)
        iters = 100 if size <= 256 else 20
        results[f"matmul_{size}x{size}"] = bench(f"matmul [{size},{size}]", lambda: a_m @ b_m, iters=iters)

    # Transcendentals
    print("\n--- Transcendental Ops ---")
    t = np.random.rand(1000, 1000).astype(np.float32)
    results["exp_1000x1000"] = bench("exp [1000,1000]", lambda: np.exp(t))
    results["log_1000x1000"] = bench("log [1000,1000]", lambda: np.log(t + 1e-6))
    results["sqrt_1000x1000"] = bench("sqrt [1000,1000]", lambda: np.sqrt(t))
    results["sin_1000x1000"] = bench("sin [1000,1000]", lambda: np.sin(t))
    results["cos_1000x1000"] = bench("cos [1000,1000]", lambda: np.cos(t))
    results["pow_1000x1000"] = bench("pow x^2 [1000,1000]", lambda: np.power(t, 2.0))

    # Reductions
    print("\n--- Reduction Ops ---")
    r = np.random.rand(1000, 1000).astype(np.float32)
    results["sum_all_1000x1000"] = bench("sum_all [1000,1000]", lambda: np.sum(r))
    results["sum_dim0_1000x1000"] = bench("sum dim=0 [1000,1000]", lambda: np.sum(r, axis=0))
    results["sum_dim1_1000x1000"] = bench("sum dim=1 [1000,1000]", lambda: np.sum(r, axis=1))
    results["mean_dim0_1000x1000"] = bench("mean dim=0 [1000,1000]", lambda: np.mean(r, axis=0))
    results["mean_dim1_1000x1000"] = bench("mean dim=1 [1000,1000]", lambda: np.mean(r, axis=1))
    results["max_dim0_1000x1000"] = bench("max dim=0 [1000,1000]", lambda: np.max(r, axis=0))
    results["prod_all_1000x1000"] = bench("prod_all [1000,1000]", lambda: np.prod(r))

    # Cumulative
    print("\n--- Cumulative Ops ---")
    results["cumsum_dim0_1000x1000"] = bench("cumsum dim=0 [1000,1000]", lambda: np.cumsum(r, axis=0))
    results["cumprod_dim0_1000x1000"] = bench("cumprod dim=0 [1000,1000]", lambda: np.cumprod(r, axis=0))

    # Shape ops
    print("\n--- Shape Ops ---")
    m = np.random.rand(1000, 1000).astype(np.float32)
    results["transpose_1000x1000"] = bench("transpose [1000,1000]", lambda: np.ascontiguousarray(m.T))
    results["reshape_1000x1000"] = bench("reshape [1000,1000]->[1M]", lambda: m.reshape(-1))
    cs = np.array_split(m, 4, axis=0)
    results["concatenate_4x250x1000"] = bench("concatenate [4x250,1000]", lambda: np.concatenate(cs, axis=0))

    # Broadcast
    print("\n--- Broadcast Ops ---")
    a_bc = np.random.rand(1000, 1).astype(np.float32)
    b_bc = np.random.rand(1, 1000).astype(np.float32)
    results["broadcast_add_1000"] = bench("broadcast add [1000,1]+[1,1000]", lambda: a_bc + b_bc)
    a3 = np.random.rand(64, 1, 256).astype(np.float32)
    b3 = np.random.rand(1, 128, 1).astype(np.float32)
    results["broadcast_mul_3d"] = bench("broadcast mul [64,1,256]*[1,128,1]", lambda: a3 * b3)

    # Sorting/indexing
    print("\n--- Indexing Ops ---")
    idx = np.random.randint(0, 1000, size=10000)
    src = np.random.rand(1000, 128).astype(np.float32)
    results["gather_10k_from_1kx128"] = bench("gather 10k from [1000,128]", lambda: src[idx])

    # f64
    print("\n--- f64 Ops ---")
    a64 = np.random.rand(1000, 1000)
    b64 = np.random.rand(1000, 1000)
    results["f64_add_1000x1000"] = bench("f64 add [1000,1000]", lambda: a64 + b64)
    results["f64_matmul_512x512"] = bench("f64 matmul [512,512]", lambda: a64[:512,:512] @ b64[:512,:512], iters=20)
    results["f64_exp_1000x1000"] = bench("f64 exp [1000,1000]", lambda: np.exp(a64))

    print("\n" + "=" * 60)
    print("SUMMARY (all times in microseconds)")
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.1f} us")

    with open("numpy_reference.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to numpy_reference.json")

if __name__ == "__main__":
    main()
