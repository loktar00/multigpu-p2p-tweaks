# Bit-exactness + timing of the V-split GDN decode kernel against vLLM's stock op. Needs one GPU:
#   python gdnsplit/gdn_split_test.py
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_split_ext as G
from vllm import _custom_ops as ops
torch.manual_seed(0)
dev = "cuda"
H, HV, K = 2, 6, 128
ok_all = True
for N, T, nw_dtype, act in ((1, 5, torch.float32, "silu"), (3, 5, torch.float32, "silu"), (1, 5, torch.bfloat16, "sigmoid"),
                            (8, 5, torch.float32, "silu"), (2, 3, torch.float32, "silu")):
    L = N * T
    mixed = (torch.randn(L, 2 * H * K + HV * K, device=dev) * 0.5).to(torch.bfloat16)
    a = torch.randn(L, HV, device=dev).to(torch.bfloat16); b = torch.randn(L, HV, device=dev).to(torch.bfloat16)
    A_log = torch.randn(HV, device=dev) * 0.1; dt_bias = torch.randn(HV, device=dev) * 0.1
    slots = 64
    state0 = torch.randn(slots, HV, K, K, device=dev) * 0.05
    si = torch.randperm(slots - 1, device=dev)[: N * T].view(N, T).int() + 1
    cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
    acc = torch.randint(1, T + 1, (N,), device=dev, dtype=torch.int32)
    gate = torch.randn(L, HV, K, device=dev).to(torch.bfloat16)
    nw = (torch.rand(K, device=dev) + 0.5).to(nw_dtype)
    s1, s2 = state0.clone(), state0.clone()
    o1 = ops.fused_gdn_decode_post_conv_mtp(mixed, a, b, A_log, dt_bias, si, cu, acc, s1, gate, nw, output_gate_activation=act)
    o2 = G.fused_gdn_decode_post_conv_mtp(mixed, a, b, A_log, dt_bias, si, cu, acc, s2, gate, nw, output_gate_activation=act)
    torch.cuda.synchronize()
    eo, es = torch.equal(o1, o2), torch.equal(s1, s2)
    ok_all &= eo and es
    print(f"N={N} T={T} nw={nw_dtype} act={act}: out identical {eo} (max diff {(o1.float()-o2.float()).abs().max().item():.3g}), state identical {es}")
    # timing (CUDA graph, 50 calls)
    def bench(fn):
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            for _ in range(3): fn()
        torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(50): fn()
        g.replay(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
        for _ in range(10): g.replay()
        e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / 500
    t1 = bench(lambda: ops.fused_gdn_decode_post_conv_mtp(mixed, a, b, A_log, dt_bias, si, cu, acc, s1, gate, nw, out=o1, output_gate_activation=act))
    t2 = bench(lambda: G.fused_gdn_decode_post_conv_mtp(mixed, a, b, A_log, dt_bias, si, cu, acc, s2, gate, nw, out=o2, output_gate_activation=act))
    print(f"   stock {t1:.2f} us   split {t2:.2f} us")
print("ALL IDENTICAL" if ok_all else "MISMATCH")
