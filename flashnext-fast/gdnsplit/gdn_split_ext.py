# Loader + drop-in replacement for vllm._custom_ops.fused_gdn_decode_post_conv_mtp (V-split kernel).
# Builds gdn_split.cu for sm_86 into ./build on first use (needs nvcc).
# MIT License, see LICENSE at the repository root.
import os
import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None
_BUF = {}


def ext():
    global _ext
    if _ext is None:
        os.makedirs(os.path.join(_HERE, "build"), exist_ok=True)
        _ext = load(name="fn_gdn_split", sources=[os.path.join(_HERE, "gdn_split.cu")],
                    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_86,code=sm_86", "-std=c++17"],
                    build_directory=os.path.join(_HERE, "build"), verbose=False)
    return _ext


def _bufs(device, n_out, n_cnt):
    b = _BUF.get(device)
    if b is None or b[0].numel() < n_out or b[1].numel() < n_cnt:
        b = (torch.empty(max(n_out, 512 * 8 * 128), device=device, dtype=torch.bfloat16),
             torch.zeros(max(n_cnt, 4096), device=device, dtype=torch.int32))
        _BUF[device] = b
    return b


def fused_gdn_decode_post_conv_mtp(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens,
                                   state, output_gate, norm_weight, out=None, scale=128 ** -0.5, norm_eps=1e-5,
                                   output_gate_activation="silu"):
    if out is None:
        out = torch.empty_like(output_gate)
    raw, cnt = _bufs(out.device, out.numel(), state_indices.size(0) * state.size(1))
    ext().gdn_split(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state,
                    output_gate, norm_weight, out, raw, cnt, float(scale), float(norm_eps),
                    output_gate_activation == "sigmoid")
    return out
