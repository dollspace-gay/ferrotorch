#!/usr/bin/env python3
"""PyTorch performance benchmarks for comparison with ferrotorch."""

import time
import torch
import torch.nn as nn

def bench(name, fn, warmup=5, iters=100):
    """Run a benchmark and return average time in microseconds."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1e6  # microseconds
    print(f"  {name}: {elapsed:.1f} us")
    return elapsed

def main():
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda") if torch.cuda.is_available() else None

    print(f"PyTorch {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if device_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    results = {}

    # ========== CPU Benchmarks ==========
    print("=" * 60)
    print("CPU BENCHMARKS")
    print("=" * 60)

    # 1. Tensor creation
    print("\n--- Tensor Creation ---")
    results["cpu_zeros_1000x1000"] = bench("zeros [1000,1000]", lambda: torch.zeros(1000, 1000))
    results["cpu_rand_1000x1000"] = bench("rand [1000,1000]", lambda: torch.rand(1000, 1000))

    # 2. Elementwise ops
    print("\n--- Elementwise Ops ---")
    a = torch.rand(1000, 1000)
    b = torch.rand(1000, 1000)
    results["cpu_add_1000x1000"] = bench("add [1000,1000]", lambda: a + b)
    results["cpu_mul_1000x1000"] = bench("mul [1000,1000]", lambda: a * b)
    results["cpu_relu_1000x1000"] = bench("relu [1000,1000]", lambda: torch.relu(a))
    results["cpu_sigmoid_1000x1000"] = bench("sigmoid [1000,1000]", lambda: torch.sigmoid(a))

    # 3. Matrix multiply
    print("\n--- Matrix Multiply ---")
    for size in [64, 256, 1024]:
        a = torch.rand(size, size)
        b = torch.rand(size, size)
        iters = 100 if size <= 256 else 20
        results[f"cpu_matmul_{size}x{size}"] = bench(f"matmul [{size},{size}]", lambda: a @ b, iters=iters)

    # 4. Forward pass (MLP)
    print("\n--- Forward Pass (MLP 784->256->10) ---")
    mlp = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    x_mlp = torch.rand(32, 784)
    results["cpu_mlp_fwd_b32"] = bench("MLP forward B=32", lambda: mlp(x_mlp))

    # 5. Backward pass
    print("\n--- Backward Pass (MLP 784->256->10) ---")
    def mlp_backward():
        x = torch.rand(32, 784, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
    results["cpu_mlp_bwd_b32"] = bench("MLP backward B=32", mlp_backward, iters=50)

    # 6. Full training step
    print("\n--- Full Training Step (MLP + Adam) ---")
    optimizer = torch.optim.Adam(mlp.parameters())
    loss_fn = nn.CrossEntropyLoss()
    def training_step():
        x = torch.rand(32, 784)
        target = torch.randint(0, 10, (32,))
        optimizer.zero_grad()
        out = mlp(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
    results["cpu_train_step_b32"] = bench("training step B=32", training_step, iters=50)

    # ========== New Benchmarks (Waves 1-5) ==========

    # 7. Transcendental ops
    print("\n--- Transcendental Ops ---")
    t = torch.rand(1000, 1000)
    results["cpu_exp_1000x1000"] = bench("exp [1000,1000]", lambda: torch.exp(t))
    results["cpu_log_1000x1000"] = bench("log [1000,1000]", lambda: torch.log(t.abs() + 1e-6))
    results["cpu_sin_1000x1000"] = bench("sin [1000,1000]", lambda: torch.sin(t))
    results["cpu_cos_1000x1000"] = bench("cos [1000,1000]", lambda: torch.cos(t))
    results["cpu_tanh_1000x1000"] = bench("tanh [1000,1000]", lambda: torch.tanh(t))

    # 8. Reduction ops with axis
    print("\n--- Reduction Ops (with axis) ---")
    r = torch.rand(1000, 1000)
    results["cpu_sum_all_1000x1000"] = bench("sum_all [1000,1000]", lambda: r.sum())
    results["cpu_sum_dim0_1000x1000"] = bench("sum dim=0 [1000,1000]", lambda: r.sum(dim=0))
    results["cpu_mean_dim1_1000x1000"] = bench("mean dim=1 [1000,1000]", lambda: r.mean(dim=1))

    # 9. Tensor manipulation ops
    print("\n--- Tensor Manipulation ---")
    m = torch.rand(1000, 1000)
    results["cpu_permute_1000x1000"] = bench("permute [1000,1000]", lambda: m.permute(1, 0).contiguous())
    results["cpu_chunk_1000x1000"] = bench("chunk [1000,1000] into 4", lambda: m.chunk(4, dim=0))
    chunks = m.chunk(4, dim=0)
    results["cpu_cat_4x250x1000"] = bench("cat [4x 250,1000]", lambda: torch.cat(chunks, dim=0))

    # 10. GRU forward pass
    print("\n--- GRU Forward ---")
    gru = nn.GRU(128, 256, batch_first=True)
    x_gru = torch.rand(16, 32, 128)
    results["cpu_gru_fwd"] = bench("GRU forward (128->256, seq=32, B=16)", lambda: gru(x_gru), iters=50)

    # 11. Larger MLP (784->512->256->10, B=128)
    print("\n--- Larger MLP (784->512->256->10, B=128) ---")
    mlp_large = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    x_large = torch.rand(128, 784)
    results["cpu_mlp_large_fwd_b128"] = bench("MLP forward B=128 (784->512->256->10)", lambda: mlp_large(x_large), iters=50)
    def mlp_large_backward():
        x = torch.rand(128, 784, requires_grad=True)
        out = mlp_large(x)
        loss = out.sum()
        loss.backward()
    results["cpu_mlp_large_bwd_b128"] = bench("MLP backward B=128", mlp_large_backward, iters=30)
    opt_large = torch.optim.Adam(mlp_large.parameters())
    loss_fn_large = nn.CrossEntropyLoss()
    def training_step_large():
        x = torch.rand(128, 784)
        target = torch.randint(0, 10, (128,))
        opt_large.zero_grad()
        out = mlp_large(x)
        loss = loss_fn_large(out, target)
        loss.backward()
        opt_large.step()
    results["cpu_train_step_b128"] = bench("training step B=128", training_step_large, iters=30)

    # 12. Conv2d forward
    print("\n--- Conv2d Forward ---")
    conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
    x_conv = torch.rand(32, 3, 32, 32)
    results["cpu_conv2d_fwd"] = bench("Conv2d forward [32,3,32,32]->[32,16,30,30]", lambda: conv(x_conv), iters=50)

    # 13. Broadcast operations
    print("\n--- Broadcast Ops ---")
    a_bc = torch.rand(1000, 1)
    b_bc = torch.rand(1, 1000)
    results["cpu_broadcast_add"] = bench("broadcast add [1000,1]+[1,1000]", lambda: a_bc + b_bc)
    a_bc3 = torch.rand(64, 1, 256)
    b_bc3 = torch.rand(1, 128, 1)
    results["cpu_broadcast_mul"] = bench("broadcast mul [64,1,256]*[1,128,1]", lambda: a_bc3 * b_bc3)

    # 14. Creation ops (like)
    print("\n--- Creation Ops (like) ---")
    tpl = torch.rand(1000, 1000)
    results["cpu_zeros_like_1000x1000"] = bench("zeros_like [1000,1000]", lambda: torch.zeros_like(tpl))
    results["cpu_randn_like_1000x1000"] = bench("randn_like [1000,1000]", lambda: torch.randn_like(tpl))

    # ========== GPU Benchmarks ==========
    if device_gpu:
        print("\n" + "=" * 60)
        print("GPU BENCHMARKS")
        print("=" * 60)

        # 1. Tensor creation
        print("\n--- Tensor Creation ---")
        results["gpu_zeros_1000x1000"] = bench("zeros [1000,1000]", lambda: torch.zeros(1000, 1000, device="cuda"))
        results["gpu_rand_1000x1000"] = bench("rand [1000,1000]", lambda: torch.rand(1000, 1000, device="cuda"))

        # 2. Elementwise
        print("\n--- Elementwise Ops ---")
        a_g = torch.rand(1000, 1000, device="cuda")
        b_g = torch.rand(1000, 1000, device="cuda")
        results["gpu_add_1000x1000"] = bench("add [1000,1000]", lambda: a_g + b_g)
        results["gpu_mul_1000x1000"] = bench("mul [1000,1000]", lambda: a_g * b_g)
        results["gpu_relu_1000x1000"] = bench("relu [1000,1000]", lambda: torch.relu(a_g))

        # 3. Matmul
        print("\n--- Matrix Multiply ---")
        for size in [64, 256, 1024, 4096]:
            a_g = torch.rand(size, size, device="cuda")
            b_g = torch.rand(size, size, device="cuda")
            results[f"gpu_matmul_{size}x{size}"] = bench(f"matmul [{size},{size}]", lambda: a_g @ b_g)

        # 4. Transfer
        print("\n--- Host <-> Device Transfer ---")
        t_cpu = torch.rand(1000, 1000)
        t_gpu = t_cpu.cuda()
        results["gpu_h2d_1000x1000"] = bench("CPU->GPU [1000,1000]", lambda: t_cpu.cuda())
        results["gpu_d2h_1000x1000"] = bench("GPU->CPU [1000,1000]", lambda: t_gpu.cpu())

        # 5. Forward/backward
        print("\n--- Forward/Backward (MLP on GPU) ---")
        mlp_g = mlp.cuda()
        x_g = torch.rand(32, 784, device="cuda")
        results["gpu_mlp_fwd_b32"] = bench("MLP forward B=32", lambda: mlp_g(x_g))

        def mlp_backward_gpu():
            x = torch.rand(32, 784, device="cuda", requires_grad=True)
            out = mlp_g(x)
            loss = out.sum()
            loss.backward()
        results["gpu_mlp_bwd_b32"] = bench("MLP backward B=32", mlp_backward_gpu, iters=50)

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("SUMMARY (all times in microseconds)")
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.1f} us")

if __name__ == "__main__":
    main()
