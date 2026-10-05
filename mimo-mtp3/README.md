# All three MTP modules for MiMo-V2.6-Flash in vLLM

MiMo-V2.6-Flash ships three MTP (multi-token prediction) modules. vLLM loads only the first
(`_MIMO_V2_FLASH_NUM_MTP_LAYERS = 1` in `mimo_v2_mtp.py`), so `num_speculative_tokens` above 1
just runs module 0 again for each extra position, and k=2 or 3 is slower than k=1.

`sitecustomize.py` builds all three and feeds each one the target model's hidden state, the way
SGLang runs MiMo. vLLM's default for multi-module MTP chains each module's output into the next;
for MiMo that drops second-position acceptance to 0.10. With the target hidden state it's 0.81.
It also adds `get_top_tokens` so `use_local_argmax_reduction` works.

## Results

8x RTX 3090, TP8, 210 W per card, P2P driver, single stream. 8 prompts x 2 runs, 768 tokens,
temperature 0, median. The k=3 runs also had the custom all-reduce overlay loaded (see Memory);
on its own it measured flat on MiMo (199.0 vs 198.0 t/s at k=1).

| | k=1 (stock) | k=3 (this) |
|---|---|---|
| decode | 211.3 t/s | 252.7 t/s |
| tokens per step | 1.86 | 3.02 |
| acceptance by position (k=3) | - | 0.85 / 0.68 / 0.54 |
| decode, one prompt at 129k context | 149.7 t/s | 237.4 t/s |
| prefill 16k / 64k | 2,868 / 2,487 t/s | 2,703 / 2,353 t/s (-6%) |

At temperature 1.0 / top_p 0.95: 189.2 -> 233.6 t/s. Prefill drops because the extra modules
also run over the prompt. Each step takes longer (8.9 -> 12.0 ms); all the gain comes from
accepting more tokens per step. Needle tests passed at 4k, 128k and 190k. Greedy output
eventually differs from k=1 on near-tie tokens, which two loads of the stock config also do.

## Use

    mkdir -p /opt/vllm-mimo-mtp3 && cp sitecustomize.py /opt/vllm-mimo-mtp3/
    export PYTHONPATH=/opt/vllm-mimo-mtp3${PYTHONPATH:+:$PYTHONPATH}
    vllm serve <MiMo-V2.6-Flash checkpoint> --tensor-parallel-size 8 \
        --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' ...

We ran the MXFP4 checkpoint text-only (`--trust-remote-code --hf-overrides '{"vision_config": null}'`)
to fit 8x 24 GB. The log should show `[mimo-mtp3] MTP modules: 3, feedback: target`.
`num_speculative_tokens` 2 builds two modules (246 t/s median here, at a 65k window). `MIMO_MTP_FEEDBACK=chain` switches
back to vLLM's chaining, for comparison.

## Memory

The extra modules cost KV cache. On 24 GB cards at k=3 the pool dropped to 0.99 GiB per card
(about 152k tokens), below the 196,608 window we ran at k=1. Removing `expandable_segments:True`
from `PYTORCH_CUDA_ALLOC_CONF` got it back to 1.24 GiB (213-219k tokens):

    unset PYTORCH_CUDA_ALLOC_CONF      # or drop expandable_segments from it
    PYTHONPATH=/opt/vllm-mimo-mtp3 NCCL_P2P_LEVEL=SYS vllm serve ...

The custom all-reduce overlay (`../vllm-custom-allreduce/`, `CA_MAX_BYTES=8192`) was also
loaded in those k=3 runs. We don't credit it: it measured no faster on MiMo, and k=3 wasn't run
without it, so whether the 196,608 window fits without it is untested. If it doesn't, the
overlay is the thing to try (`PYTHONPATH=/opt/vllm-mimo-mtp3:/opt/vllm-force-ca
CA_MAX_BYTES=8192`); at k=1 one load without it got 1.31 GiB against 1.51 GiB with it.

262,144 tokens doesn't fit at k=3 on 8x 24 GB (max about 210k).

## vLLM version

It patches `vllm/model_executor/models/mimo_v2_mtp.py` (`MiMoV2MultiTokenPredictor`,
`MiMoV2MTPLayer`, `MiMoV2MTP`) and relies on the v1 GPU runner's `MultiModuleMTPSpeculator`
accepting a `(logits_hidden, feedback_hidden)` tuple. If the MiMo classes are missing it prints
a warning and patches nothing.

Measured on vLLM 0.28.1rc1.dev78 (commit 696cdcb) with `mimo_v2.py` and `mimo_v2_mtp.py`
taken from upstream `main`, which carry two fixes MiMo-V2 needs to load at all:
[#57508](https://github.com/vllm-project/vllm/pull/57508) (fused fp8 qkv_proj sharding; without
it TP8 and TP4 both fail to load) and [#57784](https://github.com/vllm-project/vllm/pull/57784)
(MXFP4 MoE and bf16 router). Both were merged upstream on 2026-09-19 and 2026-09-20, so a vLLM
build from after that has them. The tree also had an unmerged long-context Triton attention fix
in both the k=1 and k=3 runs, which matters for the 129k numbers.

Upstream `main` still hard-codes one MTP layer as of 2026-10, and the classes and speculator
contract this patches look the same there. Not run on it.

## Credits

- [Xiaomi MiMo](https://github.com/XiaomiMiMo) for MiMo-V2.6-Flash and its MTP modules.
- [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0). The target-hidden-state
  feedback follows its multi-layer EAGLE worker,
  [multi_layer_eagle_worker_v2.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py),
  which chains MTP hidden states only for a few other architectures (`chain_mtp_hidden_states`)
  and keeps the target hidden state for MiMo. Its MiMo MTP model is
  [mimo_v2_nextn.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/mimo_v2_nextn.py).
  No code copied.
- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0), whose MiMo MTP model and
  multi-module MTP speculator this builds on at runtime.
