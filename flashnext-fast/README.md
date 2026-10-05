# Faster Qwen3.8-Flash-Next decode on 8x RTX 3090

Qwen3.8-Flash-Next FP8 on vLLM, TP8 on eight RTX 3090s, MTP with 4 draft tokens. Single-stream
decode went from 166.5 to 222.7 t/s (+34%) with the same 262,144 window and KV pool.

Most of it is one serve flag change and a `sitecustomize.py` overlay with five parts, each behind
its own env var:

- `FN_SKINNY=1`: Triton GEMMs tuned on a 3090 for the bf16 linears and LM head at decode sizes
  (M <= 16 tokens): a GEMV for M=1, tensor-core `tl.dot` with split-K for M=2..16 (the last CTA
  sums the partials, CUDA-graph safe). vLLM's own low-latency GEMM for this model is
  Blackwell-only, so on sm_86 these ran on cuBLAS small-M kernels at about half the memory
  bandwidth. Shapes not in `configs.json` stay on cuBLAS.
- `FN_DRAFT_I8=1` (needs `FN_SKINNY=1`): int8 weight-only copies (per-row scales) of the MTP
  drafter's own linears and LM head. Draft only: the target verifies every draft token in bf16,
  so the output can't change. LM head per draft 184 -> 91 us.
- `FN_TOPK_GATHER=64`: the target's logits are exchanged between ranks as each rank's top 64
  (value, id) pairs instead of an all-gather of the full vocab (537 us per step on x4 links).
  Only used when that is exact: greedy, or top_k <= 48 with temperature/top_p, and no logprobs,
  penalties, logit bias, bad words, min_p, thinking budget, grammar or more than 16 stop-token
  masks. Anything else uses the normal gather.
- `FN_GDN_SPLIT=1`: the GDN spec-decode kernel with each head's 128 value rows split over 8 CTAs.
  Stock runs one CTA per (request, head), 6 CTAs on an 82-SM card at TP8. 16.8-22 -> 8-9 us per
  call, 36 layers. Bit-identical outputs and states. Built from `gdnsplit/gdn_split.cu` on
  first use.
- `FN_ASYNC_H2D=1`: the short-conv attention metadata builder's pageable host-to-device copies
  become pinned `non_blocking` copies, so the host doesn't wait for the GPU each step.

Always on: `get_top_tokens` for the MTP drafter, so
`"use_local_argmax_reduction": true` works (drafts all-gather one value/id pair per rank instead
of full-vocab logits).

The serve flag change: the model's 47.7 GB n-gram embedding table (PLE) lives in pinned host RAM
and the GPUs read it zero-copy (`--cpu-offload-gb 6.5 --cpu-offload-params ngram_embedding`),
instead of `VLLM_PLE_CPU_OFFLOAD=1`, which runs lookups in a separate CPU process and stalled
every verify step 1.5-2 ms.

## Results

8x RTX 3090, TP8, 210 W per card, P2P driver, MTP k=4, single stream. 8 prompts x 2 runs, 768
tokens, thinking on. Before = the config we ran until now (same vLLM tree, CPU PLE worker,
custom all-reduce overlay), 2 loads; after = this, 3 loads.

| | before | after |
|---|---|---|
| decode, greedy, median / mean | 166.5 / 160.9 t/s | 222.7 / 217.8 t/s (+34%) |
| decode, temperature 1.0 / top_p 0.95 / top_k 20, median | 143.6 t/s | 194.2 t/s (+35%) |
| draft steps per second | 49.5 | 67.1 (+36%) |
| tokens per step | 3.26 | 3.24 |
| 3 concurrent streams, total | 253.7 t/s | 287.2 t/s |
| prefill 16k / 64k | 3,137 / 3,042 t/s | 3,123 / 3,005 t/s |
| time to first token, 129k / 251k prompt | 80 / 194-202 s | 78-81 / 198-205 s |
| window / KV pool | 262,144 / 269,736 tokens | same |

Acceptance swings from load to load, so steps per second is the cleaner number. Step rate as
each change was stacked:

| | steps/s | |
|---|---|---|
| before | 49.5 | |
| + skinny GEMMs, local-argmax drafts, async H2D | 56.3 | +14% |
| + n-gram table zero-copy instead of the CPU worker | 63.2 | +12% |
| + int8 drafter | 65.5 | +3.5% |
| + top-64 logits exchange | 66.5 | +1.5% |
| + GDN split (vs 66.0 on a reload of the line above) | 67.6 | +2.4% |
| final, KV pool pinned, 3 loads | 67.1 | |

What's left of the step: about 23% is the 96 all-reduces per verify step (around 20 us of ring
latency each at this size), the rest is GEMMs now at ~85% of memory bandwidth, Marlin MoE,
MoE routing and the 4 drafts.

## Correctness

This model on vLLM doesn't reproduce itself across loads: two loads of the same config split
after a median of 144 greedy tokens, always at near-ties (top two within 0.25 nats). So the gate
was an envelope: 8 prompts, 400 greedy tokens, top-2 logprobs against a baseline load; a change
passes if it stays inside the load-to-load noise of the unchanged config. Every load also had to
pass short-answer checks and needle retrieval at 4k, 50k and 256k prompt tokens.

- Top-64 exchange: run beside the full gather in-process (`FN_TOPK_CHECK=1`), 0 argmax and
  0 top-48 value mismatches in 8,500 calls.
- GDN split: bit-identical outputs and states against the stock op
  (`gdnsplit/gdn_split_test.py`, one GPU). Needs `--use_fast_math`, like the vLLM build.
- int8 only touches the drafter; tokens per step stayed the same.

## Tried, no gain

- Smaller draft vocab (drafter scores only the 16.7k or 46.7k most used tokens): LM head per
  draft 159 -> 21-61 MB, but acceptance dropped 4-18%. Flat to slower.
- k=5 and k=6 (needs a QSA ring-size fix to load): +5% tokens per step, but each extra draft
  costs ~8% of the step. 183 / 176 vs 187 t/s.
- `VLLM_MARLIN_USE_ATOMIC_ADD=1`: flat.
- 16-way GDN split: bit-identical, no faster than 8.
- Letting the KV pool take the memory the zero-copy table frees (280,526 tokens): 251k-token
  prompts took 220-232 s to first token instead of ~200 s. Hence the pin below.
- Custom all-reduce for the 25.6 KB verify all-reduces (`CA_MAX_BYTES=32768`): -19%
  (see `../vllm-custom-allreduce/`).

## vLLM base

Qwen3.8-Flash-Next (`qwen4_exp`) support came in vLLM PR
[#53896](https://github.com/vllm-project/vllm/pull/53896). We ran:

- PR #53896 at `82399a9fcb` (merged into `main` 2026-08-31 with later changes),
- plus the two PLE-offload commits of PR [#53899](https://github.com/vllm-project/vllm/pull/53899)
  (`f561eca6ca`, `95dc96d1d0`; closed, not merged), conflict resolved by hand,
- plus an Ampere guard so FP8 linears fall back from CUTLASS (SM89+ only) to Marlin, and two
  small fixes so the PLE offload process doesn't create CUDA contexts.

`vllm-base.patch` is all of that against `82399a9fcb` (vLLM files only, Apache-2.0). Build:

    git init vllm-flashnext && cd vllm-flashnext
    git fetch https://github.com/vllm-project/vllm 82399a9fcbebd892d9b2560827a3f8d3e050d2fd
    git checkout FETCH_HEAD
    git apply /path/to/flashnext-fast/vllm-base.patch
    pip install -r requirements/build/cuda.txt     # into a venv with torch 2.13.0+cu130
    TORCH_CUDA_ARCH_LIST=8.6 pip install -e . --no-build-isolation

About 40 minutes on our box. Python 3.13, torch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0.
`VLLM_USE_PRECOMPILED=1` doesn't work here: the kernels differ from any published wheel.

Upstream `main` has the model, an equivalent CUTLASS check and already-async copies in the
short-conv builder (`FN_ASYNC_H2D` then reports nothing to patch). Its sampler state is laid out
differently, so the top-64 exchange will warn and stay off there. Not run on `main`.

## Use

Needs: 8 GPUs with working P2P (`../driver/`), the custom all-reduce overlay
(`../vllm-custom-allreduce/`), about 48 GB of free host RAM for the pinned n-gram table, and
`nvcc` for the GDN kernel.

    mkdir -p /opt/vllm-flashnext-fast && cp -r sitecustomize.py fn86_skinny.py configs.json \
        configs_i8.json gdnsplit /opt/vllm-flashnext-fast/
    # build the GDN kernel once with the server's python (about a minute; needs nvcc / CUDA_HOME)
    cd /opt/vllm-flashnext-fast/gdnsplit && python -c "import gdn_split_ext; gdn_split_ext.ext()"

    export PYTHONPATH=/opt/vllm-flashnext-fast:/opt/vllm-force-ca
    export FN_SKINNY=1 FN_DRAFT_I8=1 FN_TOPK_GATHER=64 FN_GDN_SPLIT=1 FN_ASYNC_H2D=1
    export VLLM_DISABLE_COMPILE_CACHE=1
    export CA_MAX_BYTES=8192 NCCL_P2P_LEVEL=SYS
    unset PYTORCH_CUDA_ALLOC_CONF       # no expandable_segments
    vllm serve Qwen/Qwen3.8-Flash-Next-FP8 \
        --tensor-parallel-size 8 --enable-expert-parallel \
        --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 1024 \
        --gpu-memory-utilization 0.97 --kv-cache-memory-bytes 3972844748 --kv-cache-dtype auto \
        --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL"}' \
        --no-enable-flashinfer-autotune --enable-prefix-caching \
        --cpu-offload-gb 6.5 --cpu-offload-params ngram_embedding \
        --speculative-config '{"method":"mtp","num_speculative_tokens":4,"use_local_argmax_reduction":true}' \
        --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice

The log should show, among others:

    [flashnext-fast] skinny GEMM on <n> bf16 linears (M <= 16)
    [flashnext-fast] draft int8: <n> linears (...) + lm_head (...)
    [flashnext-fast] GDN MTP decode -> V-split kernel (FN_GDN_SPLIT=1)
    [flashnext-fast] vllm.v1.attention.backends.short_conv_attn: 5 host-to-device copies made async
    [force-ca] custom all-reduce capped at 8192 bytes

Settings that matter:

- `VLLM_DISABLE_COMPILE_CACHE=1`. vLLM's torch.compile cache is keyed on the config, not on
  the overlay, so with the cache on it silently reuses graphs compiled without the overlay
  (cuBLAS inlined) and the GEMM change mostly doesn't happen. Loading takes ~285 s instead of
  ~255 s.
- `--kv-cache-memory-bytes 3972844748` (3.70 GiB per card) keeps the KV pool where it was. The
  zero-copy table frees ~0.24 GiB per card; letting the pool take it made 251k-token prompts
  ~15% slower to first token (less headroom for the indexer's context-sized prefill buffers).
  Drop the flag if you never send prompts over ~200k tokens.
- `--cpu-offload-gb 6.5` is per GPU: each rank keeps its 1/8 of the table (5.96 GiB) pinned.
- `--enable-expert-parallel` is required for this checkpoint at TP8.
- `configs.json` and `configs_i8.json` hold measured 3090 configs for this model's shapes at
  TP8. Other shapes (other TP sizes, other models) stay on cuBLAS and log `skinny MISS`.
- Each `FN_*` switch can be turned off on its own. With none set, only `get_top_tokens` is added.

To undo: take `/opt/vllm-flashnext-fast` off `PYTHONPATH`, unset the `FN_*` variables, and go
back to your previous PLE setting.

## Credits

- Qwen for Qwen3.8-Flash-Next and its MTP module.
- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0): the `qwen4_exp` model (PR #53896)
  and PLE offload (PR #53899) this runs on. `gdnsplit/gdn_split.cu` is a modified copy of
  vLLM's `gdn_decode_post_conv_mtp_kernel` and `vllm-base.patch` modifies vLLM files; both are
  Apache-2.0 (`LICENSE.Apache-2.0`). The rest of this directory is MIT.
- [Triton](https://github.com/triton-lang/triton) (MIT) for the GEMM kernels.
- [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules),
  the P2P driver the box runs on (see `../driver/`).
