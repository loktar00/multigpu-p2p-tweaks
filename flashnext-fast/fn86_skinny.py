# Skinny bf16 decode GEMMs for sm_86: y[M,N] = x[M,K] @ W[N,K]^T for M <= 16.
# Two kernels, chosen per (N, K, M-bucket) by a table measured on an RTX 3090 (configs.json):
#   gemv : CUDA-core dot products, best at M == 1 (one CTA owns BN rows and streams them once)
#   dot  : tensor-core tl.dot with x padded to 16 rows, optional split-K over SPLIT CTAs whose
#          fp32 partials are summed by the last-arriving CTA (self-resetting counter, graph-safe)
#   cublas: torch F.linear (kept wherever it measured fastest)
# Weight streaming is the whole cost at these M, so the goal is bandwidth: enough CTAs, deep loads.
#
# MIT License, see LICENSE at the repository root.
import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, y_ptr, M, N, K, sxm, swn, sym,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        kmask = rk < K
        w = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=nmask[:, None] & kmask[None, :],
                    other=0.0, eviction_policy="evict_first").to(tl.float32)
        for m in tl.static_range(BM):
            xv = tl.load(x_ptr + m * sxm + rk, mask=kmask & (m < M), other=0.0).to(tl.float32)
            part = tl.sum(w * xv[None, :], axis=1)
            acc += tl.where(rm[:, None] == m, part[None, :], 0.0)
    tl.store(y_ptr + rm[:, None] * sym + rn[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & nmask[None, :])


@triton.jit
def _dot_kernel(x_ptr, w_ptr, y_ptr, ws_ptr, lock_ptr, M, N, K, sxm, swn, sym,
                KCH, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, 16)
    rn = pid_n * BN + tl.arange(0, BN)
    nmask = rn < N
    acc = tl.zeros((16, BN), dtype=tl.float32)
    kbeg = pid_k * KCH
    for k0 in range(kbeg, kbeg + KCH, BK):
        rk = k0 + tl.arange(0, BK)
        kmask = rk < K
        x = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :], mask=(rm[:, None] < M) & kmask[None, :], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=nmask[:, None] & kmask[None, :], other=0.0,
                    eviction_policy="evict_first")
        acc += tl.dot(x, tl.trans(w))
    omask = (rm[:, None] < M) & nmask[None, :]
    if SPLIT == 1:
        tl.store(y_ptr + rm[:, None] * sym + rn[None, :], acc.to(y_ptr.dtype.element_ty), mask=omask)
    else:
        wofs = rm[:, None] * N + rn[None, :]
        tl.store(ws_ptr + pid_k * 16 * N + wofs, acc, mask=omask)
        tl.debug_barrier()
        old = tl.atomic_add(lock_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if old == SPLIT - 1:
            tot = tl.zeros((16, BN), dtype=tl.float32)
            for s in range(SPLIT):
                tot += tl.load(ws_ptr + s * 16 * N + wofs, mask=omask, other=0.0, cache_modifier=".cg")
            tl.store(y_ptr + rm[:, None] * sym + rn[None, :], tot.to(y_ptr.dtype.element_ty), mask=omask)
            tl.atomic_xchg(lock_ptr + pid_n, 0, sem="relaxed", scope="gpu")


def bucket(M):
    return 1 if M == 1 else 2 if M == 2 else 4 if M <= 4 else 8 if M <= 8 else 16


CONFIGS = {}   # (N, K, bucket) -> ("cublas",) | ("gemv", BN, BK, nw) | ("dot", BN, BK, SPLIT, nw, stages)
_BUF = {}      # (device, N, K) -> (workspace fp32, lock int32): one pair per weight shape


def _bufs(dev, N, K, split, nblk):
    key = (dev, N, K)
    b = _BUF.get(key)
    if b is None or b[0].numel() < split * 16 * N or b[1].numel() < nblk:
        b = (torch.empty(max(split, 16) * 16 * N, device=dev, dtype=torch.float32),
             torch.zeros(max(nblk, 1024), device=dev, dtype=torch.int32))
        _BUF[key] = b
    return b


def prealloc(dev, N, K):
    """Allocate this shape's split-K buffers (call outside CUDA-graph capture)."""
    for bm in (1, 2, 4, 8, 16):
        c = CONFIGS.get((N, K, bm))
        if c and c[0] == "dot" and c[3] > 1:
            _bufs(dev, N, K, c[3], triton.cdiv(N, c[1]))


def run(x, w, c):
    M, K = x.shape
    N = w.shape[0]
    if c[0] == "cublas":
        return torch.nn.functional.linear(x, w)
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if c[0] == "gemv":
        _, BN, BK, nw = c
        _gemv_kernel[(triton.cdiv(N, BN),)](x, w, y, M, N, K, x.stride(0), w.stride(0), y.stride(0),
                                            BM=bucket(M), BN=BN, BK=BK, num_warps=nw)
        return y
    _, BN, BK, SPLIT, nw, st = c
    nblk = triton.cdiv(N, BN)
    kch = triton.cdiv(triton.cdiv(K, SPLIT), BK) * BK
    if SPLIT > 1:
        ws, lock = _bufs(x.device, N, K, SPLIT, nblk)
    else:
        ws, lock = y, y  # unused
    _dot_kernel[(nblk, SPLIT)](x, w, y, ws, lock, M, N, K, x.stride(0), w.stride(0), y.stride(0), kch,
                               BN=BN, BK=BK, SPLIT=SPLIT, num_warps=nw, num_stages=st)
    return y


_MISS = set()


def skinny_gemm(x, w):
    c = CONFIGS.get((w.shape[0], x.shape[1], bucket(x.shape[0])))
    if c is None:
        key = (w.shape[0], x.shape[1], x.shape[0])
        if key not in _MISS:
            _MISS.add(key)
            import sys
            print(f"[flashnext-fast] skinny MISS N={key[0]} K={key[1]} M={key[2]}", file=sys.stderr, flush=True)
        return torch.nn.functional.linear(x, w)
    return run(x, w, c)


# ---------------------------------------------------------------------------------------------------
# int8 weight-only (per-output-row scale) variants, used ONLY for the MTP drafter's weights: draft
# quality affects acceptance, never the output (the target verifies every draft token in bf16).
@triton.jit
def _gemv_i8_kernel(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, sxm, sqn, sym,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        kmask = rk < K
        w = tl.load(q_ptr + rn[:, None] * sqn + rk[None, :], mask=nmask[:, None] & kmask[None, :],
                    other=0, eviction_policy="evict_first").to(tl.float32)
        for m in tl.static_range(BM):
            xv = tl.load(x_ptr + m * sxm + rk, mask=kmask & (m < M), other=0.0).to(tl.float32)
            part = tl.sum(w * xv[None, :], axis=1)
            acc += tl.where(rm[:, None] == m, part[None, :], 0.0)
    s = tl.load(s_ptr + rn, mask=nmask, other=0.0)
    acc = acc * s[None, :]
    tl.store(y_ptr + rm[:, None] * sym + rn[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & nmask[None, :])


@triton.jit
def _dot_i8_kernel(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, sxm, sqn, sym,
                   BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    rm = tl.arange(0, 16)
    rn = pid_n * BN + tl.arange(0, BN)
    nmask = rn < N
    acc = tl.zeros((16, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        kmask = rk < K
        x = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :], mask=(rm[:, None] < M) & kmask[None, :], other=0.0)
        w = tl.load(q_ptr + rn[:, None] * sqn + rk[None, :], mask=nmask[:, None] & kmask[None, :], other=0,
                    eviction_policy="evict_first").to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w))
    s = tl.load(s_ptr + rn, mask=nmask, other=0.0)
    acc = acc * s[None, :]
    tl.store(y_ptr + rm[:, None] * sym + rn[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & nmask[None, :])


CONFIGS_I8 = {}   # (N, K, bucket) -> ("gemv", BN, BK, nw) | ("dot", BN, BK, nw, stages)


def quant_i8(w):
    """Per-output-row symmetric int8: returns (q int8 [N,K], s fp32 [N])."""
    wf = w.float()
    s = wf.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.round(wf / s[:, None]).clamp_(-127, 127).to(torch.int8).contiguous()
    return q, s.contiguous()


def run_i8(x, q, s, c):
    M, K = x.shape
    N = q.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if c[0] == "gemv":
        _, BN, BK, nw = c
        _gemv_i8_kernel[(triton.cdiv(N, BN),)](x, q, s, y, M, N, K, x.stride(0), q.stride(0), y.stride(0),
                                               BM=bucket(M), BN=BN, BK=BK, num_warps=nw)
    else:
        _, BN, BK, nw, st = c
        _dot_i8_kernel[(triton.cdiv(N, BN),)](x, q, s, y, M, N, K, x.stride(0), q.stride(0), y.stride(0),
                                              BN=BN, BK=BK, num_warps=nw, num_stages=st)
    return y


def skinny_i8(x, q, s):
    M = x.shape[0]
    c = CONFIGS_I8.get((q.shape[0], x.shape[1], bucket(M)))
    if c is None:
        c = ("gemv", 4, 256, 4) if M == 1 else ("dot", 32, 128, 4, 3)
    return run_i8(x, q, s, c)
