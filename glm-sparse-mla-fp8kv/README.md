# fp8 KV cache for GLM-5.3-Flash on Ampere (vllm-backport)

GLM-5.3-Flash runs on Ampere through [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport),
whose `TRITON_MLA_SPARSE` attention backend only accepts a bf16 KV cache. On 8x 24 GB that
caps the context window at 65,536. `fp8kv.patch` lets that backend store the cache as
`fp8_e5m2`, and adds an env var to shrink an oversized indexer reservation. Together the window
goes from 65,536 to 159,744 tokens at the same decode speed.

What the patch does (4 Python/Triton files, no rebuild):

- `TRITON_MLA_SPARSE` accepts `--kv-cache-dtype fp8_e5m2`. vLLM's existing
  `concat_and_cache_mla` writes e5m2 already; the backend re-views the cache as e5m2 and the
  kernel up-casts V to bf16 on load instead of casting P down to the cache dtype.
  e5m2 because Triton can't load e4m3 (`fp8e4nv`) on sm_80/86. No scales needed (k_scale 1.0).
- Only the MLA latent caches change. The DSA indexer cache already had its own fp8 format.
- `VLLM_INDEXER_PREFILL_BUFFER_FACTOR` (default 40, unchanged behaviour). The indexer reserves
  `max_model_len * 40 * 132` bytes at startup, sized for DeepSeek-V3.2 (692 MB at 131k). GLM's
  pooled indexer needs a fraction of that. Use 8. Don't go lower: the fused-MoE buffers share
  that workspace and need 128 MiB on GLM-5.3-Flash TP8; below that the MoE grows the workspace
  itself and you lose more KV than you saved (factor 2-6 gave 0.63 GiB of pool, 8 gave 1.0 GiB
  at 131k).

## Results

8x RTX 3090, TP8, 210 W, P2P driver, GLM-5.3-Flash AWQ W4A16, single stream, 512 tokens.
Both runs also used the custom all-reduce overlay (`CA_MAX_BYTES=8192`, no expandable_segments).

| | bf16 KV, 65,536 window | fp8_e5m2 KV, 159,744 window |
|---|---|---|
| KV pool per card | 0.80 GiB = 67,876 tokens | 0.97 GiB = 161,902 tokens |
| decode @4k / 32k / 60k | 88.1 / 86.4 / 84.5 t/s | 87.6 / 86.2 / 84.5 t/s |
| decode @100k / 131k / 149k | - | 82.3 / 84.7 / 84.3 t/s |
| prefill | ~2,000 t/s | 1-3% lower |

Needle tests passed at 4k, 50k, 100k, 130k and 148k prompt tokens. At temperature 0, 6 of 8
short answers matched bf16 exactly; the other two split at near-tie tokens (top two within
0.25 nats). A unit test on one card matched the bf16 kernel within bf16 rounding.

## Apply

Patch a copy, not the tree something else is serving from. The backport has to be built
already (with sm_86 kernels for 3090s); `cp -a` keeps the compiled extensions.

    cp -a /opt/vllm-backport /opt/vllm-backport-fp8kv
    cd /opt/vllm-backport-fp8kv
    git apply /path/to/fp8kv.patch

Then serve from the copy:

    PYTHONPATH=/opt/vllm-backport-fp8kv \
    VLLM_INDEXER_PREFILL_BUFFER_FACTOR=8 \
    vllm serve <GLM-5.3-Flash AWQ checkpoint> --tensor-parallel-size 8 \
        --kv-cache-dtype fp8_e5m2 --max-model-len 159744 --gpu-memory-utilization 0.97 ...

Tested against backport commit cde54e8; it also applies cleanly to master at 29e66dad4 (not
run there). Undo: point `PYTHONPATH` back at the unpatched tree.

## Credits

- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) (Apache-2.0), the vLLM
  backport for older GPUs this patches, including the Triton sparse-MLA backend.
- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
- Z.ai for GLM-5.3-Flash; [wtdcode/GLM-5.3-Flash-AWQ-W4A16](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16)
  for the AWQ quant we ran.

`fp8kv.patch` modifies Apache-2.0 files and is under Apache-2.0 (`LICENSE.Apache-2.0`).
