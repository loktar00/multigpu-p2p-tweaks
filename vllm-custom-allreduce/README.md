# Forced custom all-reduce for vLLM on PCIe GPUs

vLLM turns off its custom all-reduce kernel on more than two GPUs unless every pair is linked
by NVLink. On a PCIe box you get this in the log and NCCL does all the work:

    Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs.

With a P2P-enabled driver (see `../driver/`) the PCIe peers work fine and the custom kernel is
faster for small, latency-bound all-reduces, which is what single-stream decode is. The
`sitecustomize.py` here makes vLLM's NVLink check return true. vLLM's own P2P test still runs
after it and disables custom all-reduce if peer access doesn't actually work.

Measured on 8x RTX 3090 (TP8, x16 and x4 links, 4 NVLink pairs, P2P driver,
`NCCL_P2P_LEVEL=SYS`), GLM-5.3-Flash AWQ W4A16, single stream:

| | decode @4k | decode @50k | 3 streams | prefill 16k |
|---|---|---|---|---|
| NCCL only (stock vLLM) | 68.5 t/s | 67.5 t/s | 159.7 t/s | 2,037 t/s |
| forced, no cap | 86.6 t/s | 84.9 t/s | 144.3 t/s | 1,036 t/s |
| forced, `CA_MAX_BYTES=8192` | 86.5 t/s | 84.7 t/s | 162.0 t/s | 2,043 t/s |

Dropping `expandable_segments` (required, see below) also freed enough memory to raise the
context from 57,344 to 65,536 tokens.

It only wins at about 8 KB, i.e. one decoding sequence without speculative decoding. Every
MTP/DFlash verify step all-reduces 16-80 KB, and pushing those through the custom kernel lost
16-22% decode on the same box (MiMo-V2.6-Flash MTP: 199 -> 155 t/s at a 16 KB cap; Qwen3.8-Flash-Next
MTP k=4: -19% at 32 KB; a 27B DFlash TP4 lane: -16% at 96 KB). Keep `CA_MAX_BYTES=8192`.
With a spec-decode model the kernel then sits idle, which is fine, because of the next point.

## Side effect: more KV cache

With the overlay loaded and `expandable_segments` off, vLLM ends up with more memory for KV
cache. Measured per 24 GB card:

| | before | after | window |
|---|---|---|---|
| MiMo-V2.6-Flash TP8, MTP k=1 | 1.26 GiB | 1.51 GiB | 196,608 -> 262,144 |
| MiMo-V2.6-Flash TP8, MTP k=3 (`../mimo-mtp3/`) | 0.99 GiB | 1.24 GiB | back to 196,608 |
| Qwen3.8-Flash-Next FP8 TP8 | 2.90 GiB | 3.68 GiB | 196,608 -> 262,144 |

On Flash-Next `expandable_segments` off alone did it. On MiMo, turning it off without the
overlay only reached 1.31 GiB and 262,144 refused to start. Decode speed was unchanged in all
three, and needle tests passed at the new maximum.

## Use

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
x4 links it wins small messages and loses big ones. Uncapped, prefill halved and 3-stream
decode dropped 10%. 8192 bytes is one decoding sequence for a model with hidden size 4096 in
bf16 (4096 x 2 bytes). For another model start at `hidden_size * 2` and measure; on an all-x16
box a higher cap may pay off. Unset or 0 means no extra cap (vLLM's own size limit applies).

`PYTORCH_CUDA_ALLOC_CONF` must not contain `expandable_segments:True`. The custom kernel
exports its buffers over CUDA IPC, which fails on expandable segments; the server crashes in
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
  v0.13.1 (commit cde54e8), the tree the numbers above come from.
- Served on vLLM 0.28.1rc1.dev78 (commit 696cdcb) with MiMo-V2.6-Flash and
  Qwen3.8-Flash-Next (the numbers in the two sections above).
- The patched functions exist with the same shape in upstream v0.10.0, v0.11.0 and `main`
  as of 2026-10. Not run there.
