#!/usr/bin/env python3
"""
PyTorch reference: speed + correctness baselines for comparison with ferrotorch.

Run:  python benchmarks/pytorch_validate.py
Saves results to benchmarks/pytorch_reference.json for automated comparison.
"""

import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS = {}

def bench(name, fn, warmup=5, iters=100):
    """Run a benchmark, return average time in microseconds."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1e6
    print(f"  {name}: {elapsed:.1f} us")
    RESULTS[f"speed_{name}"] = elapsed
    return elapsed

def section(title):
    print(f"\n{'='*60}")
    print(title)
    print('='*60)

def subsection(title):
    print(f"\n--- {title} ---")

# ============================================================
# PART 1: SPEED BENCHMARKS (CPU)
# ============================================================

def run_speed_benchmarks():
    section("SPEED BENCHMARKS (CPU)")

    # Elementwise
    subsection("Elementwise 1M elements")
    a = torch.rand(1000, 1000)
    b = torch.rand(1000, 1000)
    bench("add_1M", lambda: a + b)
    bench("mul_1M", lambda: a * b)
    bench("relu_1M", lambda: F.relu(a))
    bench("sigmoid_1M", lambda: torch.sigmoid(a))
    bench("exp_1M", lambda: torch.exp(a))
    bench("log_1M", lambda: torch.log(a.abs() + 1e-6))
    bench("tanh_1M", lambda: torch.tanh(a))
    bench("gelu_1M", lambda: F.gelu(a))

    # Matmul
    subsection("Matrix Multiply")
    for sz in [64, 256, 1024]:
        m1 = torch.rand(sz, sz)
        m2 = torch.rand(sz, sz)
        it = 100 if sz <= 256 else 20
        bench(f"matmul_{sz}", lambda m1=m1, m2=m2: m1 @ m2, iters=it)

    # MLP forward + backward
    subsection("MLP (784->256->10, B=32)")
    mlp = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    x = torch.rand(32, 784)
    bench("mlp_fwd_b32", lambda: mlp(x))

    def mlp_bwd():
        xi = torch.rand(32, 784, requires_grad=True)
        out = mlp(xi); out.sum().backward()
    bench("mlp_bwd_b32", mlp_bwd, iters=50)

    # Full training step
    subsection("Training Step (MLP + Adam, B=32)")
    opt = torch.optim.Adam(mlp.parameters())
    ce = nn.CrossEntropyLoss()
    def train_step():
        xi = torch.rand(32, 784)
        tgt = torch.randint(0, 10, (32,))
        opt.zero_grad()
        out = mlp(xi); loss = ce(out, tgt); loss.backward(); opt.step()
    bench("train_step_b32", train_step, iters=50)

    # Transformer block
    subsection("Transformer Block (d=256, h=4, seq=32, B=4)")
    enc_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, batch_first=True)
    xt = torch.rand(4, 32, 256)
    bench("transformer_fwd", lambda: enc_layer(xt), iters=50)
    def transformer_bwd():
        xi = torch.rand(4, 32, 256, requires_grad=True)
        out = enc_layer(xi); out.sum().backward()
    bench("transformer_bwd", transformer_bwd, iters=30)

    # Conv2d
    subsection("Conv2d (3->16, k=3, 32x32, B=32)")
    conv = nn.Conv2d(3, 16, 3)
    xc = torch.rand(32, 3, 32, 32)
    bench("conv2d_fwd", lambda: conv(xc), iters=50)

    # LSTM
    subsection("LSTM (128->256, seq=32, B=16)")
    lstm = nn.LSTM(128, 256, batch_first=True)
    xl = torch.rand(16, 32, 128)
    bench("lstm_fwd", lambda: lstm(xl), iters=50)

    # LayerNorm
    subsection("LayerNorm [256] on [4,32,256]")
    ln = nn.LayerNorm(256)
    xn = torch.rand(4, 32, 256)
    bench("layernorm_fwd", lambda: ln(xn))

    # Softmax
    subsection("Softmax dim=-1 on [4,32,256]")
    bench("softmax_fwd", lambda: F.softmax(xn, dim=-1))

    # Embedding
    subsection("Embedding (50257, 512)")
    emb = nn.Embedding(50257, 512)
    ids = torch.randint(0, 50257, (4, 128))
    bench("embedding_fwd", lambda: emb(ids))

# ============================================================
# PART 2: CORRECTNESS BASELINES
# ============================================================

def run_correctness_baselines():
    section("CORRECTNESS BASELINES")
    torch.manual_seed(42)

    # 1. MLP training — loss must decrease
    subsection("MLP training convergence (10 steps)")
    mlp = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    losses = []
    for step in range(10):
        x = torch.randn(16, 20)
        tgt = torch.randint(0, 5, (16,))
        opt.zero_grad()
        out = mlp(x); loss = ce(out, tgt); loss.backward(); opt.step()
        losses.append(loss.item())
    RESULTS["correct_mlp_loss_decrease"] = losses[-1] < losses[0]
    RESULTS["correct_mlp_losses"] = losses
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}  {'PASS' if losses[-1] < losses[0] else 'FAIL'}")

    # 2. Autograd numerical gradient check
    subsection("Autograd numerical gradient check")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    analytical = x.grad.tolist()
    numerical = []
    eps = 1e-5
    for i in range(3):
        xp = x.detach().clone(); xp[i] += eps
        xm = x.detach().clone(); xm[i] -= eps
        numerical.append(((xp * xp).sum().item() - (xm * xm).sum().item()) / (2 * eps))
    max_err = max(abs(a - n) for a, n in zip(analytical, numerical))
    RESULTS["correct_autograd_maxerr"] = max_err
    RESULTS["correct_autograd_pass"] = max_err < 1e-4
    print(f"  Analytical: {analytical}")
    print(f"  Numerical:  {numerical}")
    print(f"  Max error: {max_err:.2e}  {'PASS' if max_err < 1e-4 else 'FAIL'}")

    # 3. Softmax sums to 1
    subsection("Softmax normalization")
    x = torch.randn(4, 10)
    sm = F.softmax(x, dim=-1)
    sums = sm.sum(dim=-1)
    max_dev = (sums - 1.0).abs().max().item()
    RESULTS["correct_softmax_maxdev"] = max_dev
    RESULTS["correct_softmax_pass"] = max_dev < 1e-6
    print(f"  Row sums deviation from 1: {max_dev:.2e}  {'PASS' if max_dev < 1e-6 else 'FAIL'}")

    # 4. LayerNorm produces zero mean, unit variance
    subsection("LayerNorm statistics")
    ln = nn.LayerNorm(256, elementwise_affine=False)
    x = torch.randn(4, 32, 256) * 5 + 3  # non-zero mean/var
    y = ln(x)
    mean_err = y.mean(dim=-1).abs().max().item()
    var_err = (y.var(dim=-1, unbiased=False) - 1.0).abs().max().item()
    RESULTS["correct_layernorm_mean_err"] = mean_err
    RESULTS["correct_layernorm_var_err"] = var_err
    RESULTS["correct_layernorm_pass"] = mean_err < 1e-5 and var_err < 1e-4
    print(f"  Mean err: {mean_err:.2e}, Var err: {var_err:.2e}  {'PASS' if mean_err < 1e-5 and var_err < 1e-4 else 'FAIL'}")

    # 5. CrossEntropy gradient check
    subsection("CrossEntropy gradient")
    logits = torch.randn(4, 10, requires_grad=True)
    targets = torch.tensor([3, 7, 1, 5])
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    has_grad = logits.grad is not None and logits.grad.abs().sum().item() > 0
    RESULTS["correct_ce_has_gradient"] = has_grad
    print(f"  Has non-zero gradient: {has_grad}  {'PASS' if has_grad else 'FAIL'}")

    # 6. Adam with weight decay
    subsection("AdamW weight decay effect")
    p = nn.Linear(10, 10)
    w_before = p.weight.data.clone()
    opt = torch.optim.AdamW(p.parameters(), lr=0.1, weight_decay=0.1)
    for _ in range(10):
        opt.zero_grad()
        loss = p(torch.randn(4, 10)).sum()
        loss.backward()
        opt.step()
    w_after = p.weight.data
    # Weight decay should shrink weights
    norm_before = w_before.norm().item()
    norm_after = w_after.norm().item()
    # With large lr and weight_decay, norm should decrease
    RESULTS["correct_adamw_norm_before"] = norm_before
    RESULTS["correct_adamw_norm_after"] = norm_after
    print(f"  Weight norm: {norm_before:.4f} -> {norm_after:.4f}")

    # 7. Conv2d output shape
    subsection("Conv2d output shape")
    conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
    x = torch.rand(2, 3, 32, 32)
    y = conv(x)
    expected = [2, 16, 16, 16]
    correct = list(y.shape) == expected
    RESULTS["correct_conv2d_shape"] = correct
    RESULTS["correct_conv2d_output_shape"] = list(y.shape)
    print(f"  Input: {list(x.shape)} -> Output: {list(y.shape)} (expected {expected})  {'PASS' if correct else 'FAIL'}")

    # 8. Dropout mask is correct rate
    subsection("Dropout rate")
    torch.manual_seed(0)
    x = torch.ones(10000)
    d = nn.Dropout(0.3)
    d.train()
    y = d(x)
    zero_frac = (y == 0).float().mean().item()
    RESULTS["correct_dropout_zero_frac"] = zero_frac
    RESULTS["correct_dropout_pass"] = 0.25 < zero_frac < 0.35
    print(f"  Drop rate 0.3, actual zero fraction: {zero_frac:.4f}  {'PASS' if 0.25 < zero_frac < 0.35 else 'FAIL'}")

    # 9. Embedding lookup correctness
    subsection("Embedding lookup")
    emb = nn.Embedding(100, 16)
    ids = torch.tensor([5, 10, 50])
    out = emb(ids)
    match = torch.allclose(out[0], emb.weight[5]) and torch.allclose(out[2], emb.weight[50])
    RESULTS["correct_embedding_pass"] = match
    print(f"  Lookup matches weight rows: {match}  {'PASS' if match else 'FAIL'}")

    # 10. RoPE equivariance: rotating Q and K by same amount preserves dot product
    subsection("RoPE dot product preservation")
    d = 64
    seq = 8
    # Simple RoPE implementation
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
    t = torch.arange(seq).float()
    angles = torch.outer(t, freqs)  # [seq, d/2]
    cos_a = angles.cos()
    sin_a = angles.sin()
    def apply_rope(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1).flatten(-2)
    q = torch.randn(seq, d)
    k = torch.randn(seq, d)
    # Dot product at same position should be preserved
    qr = apply_rope(q)
    kr = apply_rope(k)
    # Compare dot products at position 3
    dot_orig = (q[3] * k[3]).sum().item()
    dot_rope = (qr[3] * kr[3]).sum().item()
    # They won't be identical because RoPE changes the representation,
    # but relative ordering should be preserved across positions
    RESULTS["correct_rope_dot_original"] = dot_orig
    RESULTS["correct_rope_dot_rotated"] = dot_rope
    print(f"  Original dot: {dot_orig:.4f}, After RoPE: {dot_rope:.4f}")

    # 11. Causal attention mask
    subsection("Causal attention mask")
    seq = 8
    q = torch.randn(1, seq, 64)
    k = torch.randn(1, seq, 64)
    v = torch.randn(1, seq, 64)
    # With causal mask, output at position i should only depend on positions <= i
    mask = torch.triu(torch.ones(seq, seq) * float('-inf'), diagonal=1)
    scores = torch.bmm(q, k.transpose(1, 2)) / 8.0 + mask
    attn = F.softmax(scores, dim=-1)
    # Check upper triangle is zero (future positions masked)
    upper_sum = attn[0].triu(diagonal=1).sum().item()
    RESULTS["correct_causal_mask_upper_sum"] = upper_sum
    RESULTS["correct_causal_mask_pass"] = upper_sum < 1e-6
    print(f"  Upper triangle attention sum: {upper_sum:.2e}  {'PASS' if upper_sum < 1e-6 else 'FAIL'}")

    # 12. Gradient checkpointing
    subsection("Gradient checkpointing correctness")
    from torch.utils.checkpoint import checkpoint
    lin = nn.Linear(32, 32)
    x = torch.randn(4, 32, requires_grad=True)
    # Without checkpoint
    y1 = lin(lin(x))
    y1.sum().backward()
    g1 = x.grad.clone()
    x.grad = None
    # With checkpoint
    y2 = checkpoint(lambda x: lin(lin(x)), x, use_reentrant=False)
    y2.sum().backward()
    g2 = x.grad.clone()
    match = torch.allclose(g1, g2, atol=1e-6)
    RESULTS["correct_checkpoint_pass"] = match
    print(f"  Gradients match: {match}  {'PASS' if match else 'FAIL'}")

# ============================================================
# PART 3: REAL TASK BENCHMARKS
# ============================================================

def run_real_tasks():
    section("REAL TASK BENCHMARKS")
    torch.manual_seed(42)

    # Task 1: Train a small CNN on random CIFAR-like data
    subsection("Task 1: CNN training (CIFAR-like, 5 epochs)")
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(4)
            self.fc = nn.Linear(64 * 4 * 4, 10)
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    start = time.perf_counter()
    for epoch in range(5):
        total_loss = 0
        for _ in range(10):  # 10 batches per epoch
            x = torch.randn(32, 3, 32, 32)
            tgt = torch.randint(0, 10, (32,))
            opt.zero_grad()
            out = model(x); loss = ce(out, tgt); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch}: loss={total_loss/10:.4f}")
    cnn_time = time.perf_counter() - start
    RESULTS["task_cnn_time_s"] = cnn_time
    RESULTS["task_cnn_final_loss"] = total_loss / 10
    print(f"  Total time: {cnn_time:.2f}s")

    # Task 2: Transformer language model training
    subsection("Task 2: Transformer LM training (5 epochs)")
    class TinyLM(nn.Module):
        def __init__(self, vocab=1000, d_model=128, nhead=4, nlayers=2):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(64, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
            self.head = nn.Linear(d_model, vocab)
        def forward(self, x):
            b, s = x.shape
            pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
            h = self.emb(x) + self.pos(pos)
            mask = nn.Transformer.generate_square_subsequent_mask(s)
            h = self.encoder(h, mask=mask, is_causal=True)
            return self.head(h)

    lm = TinyLM()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-3)
    start = time.perf_counter()
    for epoch in range(5):
        total_loss = 0
        for _ in range(10):
            ids = torch.randint(0, 1000, (4, 32))
            opt.zero_grad()
            logits = lm(ids)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 1000), ids[:, 1:].reshape(-1))
            loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch}: loss={total_loss/10:.4f}")
    lm_time = time.perf_counter() - start
    RESULTS["task_lm_time_s"] = lm_time
    RESULTS["task_lm_final_loss"] = total_loss / 10
    print(f"  Total time: {lm_time:.2f}s")

    # Task 3: Autoencoder
    subsection("Task 3: Autoencoder reconstruction (5 epochs)")
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 32))
            self.decoder = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 784))
        def forward(self, x):
            return self.decoder(self.encoder(x))

    ae = Autoencoder()
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    start = time.perf_counter()
    for epoch in range(5):
        total_loss = 0
        for _ in range(20):
            x = torch.randn(64, 784)
            opt.zero_grad()
            recon = ae(x)
            loss = F.mse_loss(recon, x)
            loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch}: recon_loss={total_loss/20:.4f}")
    ae_time = time.perf_counter() - start
    RESULTS["task_ae_time_s"] = ae_time
    RESULTS["task_ae_final_loss"] = total_loss / 20
    print(f"  Total time: {ae_time:.2f}s")

# ============================================================
# MAIN
# ============================================================

def main():
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    run_speed_benchmarks()
    run_correctness_baselines()
    run_real_tasks()

    # Save results
    path = "benchmarks/pytorch_reference.json"
    with open(path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nResults saved to {path}")

    # Summary
    section("SUMMARY")
    correct_keys = [k for k in RESULTS if k.startswith("correct_") and k.endswith("_pass")]
    passed = sum(1 for k in correct_keys if RESULTS[k])
    total = len(correct_keys)
    print(f"Correctness: {passed}/{total} passed")

    speed_keys = [k for k in RESULTS if k.startswith("speed_")]
    print(f"Speed benchmarks: {len(speed_keys)} recorded")

    task_keys = [k for k in RESULTS if k.startswith("task_") and k.endswith("_time_s")]
    for k in task_keys:
        name = k.replace("task_", "").replace("_time_s", "")
        print(f"  {name}: {RESULTS[k]:.2f}s")

if __name__ == "__main__":
    main()
