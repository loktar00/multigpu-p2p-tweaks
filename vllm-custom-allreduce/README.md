# Forced custom all-reduce for vLLM on PCIe GPUs

**Correction, 2026-10-05.** This README used to say the overlay made GLM-5.3-Flash decode 26%
faster and freed KV memory. Compared fairly, it didn't beat NCCL's defaults on any model we ran:

- GLM-5.3-Flash: the 68.5 t/s baseline had `NCCL_PROTO=LL128` forced, which is slow for small
  messages. The overlay just routed around it. With NCCL's default protocol and no overlay,
  decode was 91.75 t/s, against 87.7 t/s with the overlay.
- MiMo-V2.6-Flash and Qwen3.8-Flash-Next: no decode change. With `CA_MAX_BYTES=8192` their
  verify all-reduces (over 8 KB) still go through NCCL; raising the cap made them 16-22% slower.
- KV memory: the overlay needs `expandable_segments` off, and turning that off is what grew the
  KV cache on Flash-Next. See "Memory" below for what we could and couldn't separate.

It stays here as an experiment, not a recommended speedup. The changes that did pay off on this
box are the P2P driver with `NCCL_P2P_LEVEL=SYS` (`../driver/`) and `expandable_segments` off.

## What it does

vLLM turns off its custom all-reduce kernel on more than two GPUs unless every pair is linked
by NVLink. On a PCIe box you get this in the log and NCCL does all the work:

    Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs.

The `sitecustomize.py` here makes vLLM's NVLink check return true, so the custom kernel runs.
vLLM's own P2P test still runs after it and disables custom all-reduce if peer access doesn't
actually work. `CA_MAX_BYTES` caps which all-reduces use it.

## Measurements

8x RTX 3090, TP8, x16 and x4 links, 4 NVLink pairs, 210 W, P2P driver, `NCCL_P2P_LEVEL=SYS`,
single stream.

GLM-5.3-Flash AWQ W4A16, no speculative decoding, 8 prompts x 2 runs, 768 tokens, greedy,
median (2026-10-05):

| | decode |
|---|---|
| overlay, `CA_MAX_BYTES=8192`, `NCCL_PROTO=LL128` (what we ran, 2 loads) | 87.7 t/s |
| overlay, `CA_MAX_BYTES=8192`, NCCL default protocol | 87.2 t/s |
| no overlay, NCCL default protocol | 91.75 t/s |

The no-overlay run used a 131,072 window; see "Memory" for why. All-reduce time per call on the
same box:

| | 8 KB | 32 KB | 64 KB |
|---|---|---|---|
| NCCL default | 19.6 us | 25.7 us | 43.7 us |
| NCCL, `NCCL_PROTO=LL128` | 58.5 us | 54.9 us | 56.7 us |
| custom all-reduce, one-stage | 29.1 us | 97.6 us | 188.9 us |
| custom all-reduce, two-stage | 20.4 us | 33.5 us | 61.2 us |

The earlier table (2026-10-03, 68.5 -> 86.5 t/s) compared the overlay against NCCL with LL128
forced. That measured getting out of our own LL128 setting, not the kernel.

Spec-decode models, `CA_MAX_BYTES=8192` against no overlay:

- MiMo-V2.6-Flash TP8, MTP k=1: 199.0 -> 198.0 t/s. A 16 KB cap: 155.0 t/s (-22%).
- Qwen3.8-Flash-Next FP8 TP8, MTP k=4: 129.5 -> 127.1 t/s, same step time (21.3 ms). A 32 KB
  cap: -19%.
- A 27B DFlash TP4 lane with a 96 KB cap: -16%.

## Memory

The custom kernel exports its buffers over CUDA IPC, which fails on expandable segments, so the
overlay needs `expandable_segments:True` out of `PYTORCH_CUDA_ALLOC_CONF`. Turning that off
gave more KV cache. Per 24 GB card, before = expandable_segments on, no overlay; after =
expandable_segments off, overlay loaded:

| | before | after | window |
|---|---|---|---|
| MiMo-V2.6-Flash TP8, MTP k=1 | 1.26 GiB | 1.51 GiB | 196,608 -> 262,144 |
| MiMo-V2.6-Flash TP8, MTP k=3 (`../mimo-mtp3/`) | 0.99 GiB | 1.24 GiB | back to 196,608 |
| Qwen3.8-Flash-Next FP8 TP8 | 2.90 GiB | 3.68 GiB | 196,608 -> 262,144 |

Where the two were separated:

- Flash-Next: `expandable_segments` off alone gave 3.68 GiB; adding the overlay, 3.69.
- GLM-5.3-Flash at 159,744: 0.97 GiB with the overlay, 0.96 without. Without it 159,744
  refused to start at `--gpu-memory-utilization 0.97` (vLLM's estimate: 158,976 max).
- MiMo: the two were changed together. One load at k=1 with `expandable_segments` off and no
  overlay got 1.31 GiB (262,144 didn't start); it wasn't repeated, and k=3 wasn't tried without
  the overlay.

So turn off `expandable_segments` first, on its own, and measure. Decode speed didn't change
in any of these, and needle tests passed at the new maximum.

## Use

To try it on other hardware or models (one candidate: a small hidden size, where a decoding
sequence's all-reduce stays at or under 8 KB), compare against no overlay with NCCL defaults
(`NCCL_PROTO` and `NCCL_ALGO` unset), not against a tuned NCCL setting.

Copy `sitecustomize.py` into a directory of its own and put that directory first on the
server's `PYTHONPATH`. Remove `--disable-custom-all-reduce` if you had it.

    mkdir -p /opt/vllm-force-ca && cp sitecustomize.py /opt/vllm-force-ca/
    export PYTHONPATH=/opt/vllm-force-ca${PYTHONPATH:+:$PYTHONPATH}
    export CA_MAX_BYTES=8192
    export NCCL_P2P_LEVEL=SYS
    unset PYTORCH_CUDA_ALLOC_CONF      # or at least drop expandable_segments from it
    vllm serve <model> --tensor-parallel-size 8 ...

The log should show:

    [force-ca] is_fully_connected -> True (CudaPlatformBase, NvmlCudaPlatform, NonNvmlCudaPlatform)
    [force-ca] custom all-reduce capped at 8192 bytes

and no "Custom allreduce is disabled" warning. If you still see "your platform lacks GPU P2P
capability or P2P test failed", P2P isn't working between some pair; fix that first.

To undo, take the directory off `PYTHONPATH` (or put `--disable-custom-all-reduce` back).

It runs the next `sitecustomize.py` found after its own directory on `PYTHONPATH`, so overlays
stack: `PYTHONPATH=/opt/vllm-mimo-mtp3:/opt/vllm-force-ca`.

## Settings that matter

`CA_MAX_BYTES` caps which all-reduces use the custom kernel. It's a one-shot kernel, so over
x4 links it loses badly on big messages: uncapped, GLM prefill halved and 3-stream decode
dropped 10%. 8192 bytes is one decoding sequence for a model with hidden size 4096 in bf16
(4096 x 2 bytes). For another model start at `hidden_size * 2` and measure; on an all-x16 box a
higher cap may do better. Unset or 0 means no extra cap (vLLM's own size limit applies).

`PYTORCH_CUDA_ALLOC_CONF` must not contain `expandable_segments:True`, or the server crashes in
CUDA graph capture with `custom_all_reduce.cuh:... invalid argument`.

P2P must work between every pair of ranks. Check with `nvidia-smi topo -p2p r` (all `OK`) and
`../driver/p2p_check.py`.

## Risk

This skips a safety check vLLM put there on purpose. The kernel assumes fast, reliable peer
access between all ranks. Before trusting a server with it, run the same prompts at
temperature 0 with and without the overlay and compare the output, and keep an eye on anything
that looks like corruption under load.

On current vLLM `main`, `fully_connected` also enables the custom all-gather and
reduce-scatter paths, which `CA_MAX_BYTES` does not cap. Those paths were present in the
tested tree too, but whether a given model hits them depends on its parallel layout.

## vLLM versions

It patches `CudaPlatformBase` / `NvmlCudaPlatform` / `NonNvmlCudaPlatform.is_fully_connected`
in `vllm/platforms/cuda.py` and `CustomAllreduce.should_custom_ar` in
`vllm/distributed/device_communicators/custom_all_reduce.py`. If either is missing it prints a
warning and patches nothing.

- Run and measured on [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport)
  v0.13.1 (commit cde54e8): the GLM-5.3-Flash numbers.
- Served on vLLM 0.28.1rc1.dev78 (commit 696cdcb) with MiMo-V2.6-Flash and
  Qwen3.8-Flash-Next: the MiMo and Flash-Next numbers.
- The patched functions exist with the same shape in upstream v0.10.0, v0.11.0 and `main`
  as of 2026-10. Not run there.
