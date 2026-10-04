# AGENTS.md

Instructions for applying this repo to a Linux machine with several NVIDIA GPUs. Run commands
as root unless noted. Do one component at a time and verify it before starting the next.

## Rules

- Never replace kernel modules, restart services that use GPUs, or reboot while GPU jobs are
  running. Check first: `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`
  must print nothing. If it prints anything, stop and ask the owner.
- Back up every file you replace before replacing it, and tell the owner where the backup is.
- Do not edit BIOS settings or the kernel command line yourself. Tell the owner what is needed.
- Do not set `NVreg_*` module options, change power limits, or disable the IOMMU unless asked.
- If a verify step fails, roll back that component and report the output. Do not improvise fixes
  to the driver source.
- This repo is for boxes where all P2P cards are the same architecture. If `nvidia-smi -L`
  shows both Ampere (RTX 30xx, A-series) and Blackwell (RTX 50xx, RTX PRO Blackwell) cards,
  stop: use https://github.com/loktar00/p2p-mixed-arch-fix instead.

## Component 1: P2P driver (driver/)

What it does: replaces the five NVIDIA kernel modules with the aikitoria P2P fork so consumer
GPUs can do direct GPU-to-GPU transfers over PCIe.

Prerequisites and checks:

    cat /proc/driver/nvidia/version          # note the version, e.g. 595.58.03
    git ls-remote --heads https://github.com/aikitoria/open-gpu-kernel-modules | grep -- "-p2p"
                                              # a branch <exact version>-p2p must exist; if not, stop
    grep -o 'iommu=pt' /proc/cmdline          # must print iommu=pt
    grep -oE '(amd|intel)_iommu=on' /proc/cmdline
    nvidia-smi -q -d MEMORY | grep -A1 BAR1   # BAR1 Total should be large (e.g. 32768 MiB), not 256 MiB
    mokutil --sb-state 2>/dev/null            # SecureBoot disabled, or the owner signs modules
    ls /lib/modules/$(uname -r)/build         # kernel headers present
    gcc --version && make --version

Missing `iommu=pt` or a 256 MiB BAR1: stop and tell the owner (kernel command line /
BIOS Above 4G Decoding). Record `nvidia-smi topo -p2p r` output as the before state.

Install (replace VERSION):

    VERSION=595.58.03
    KVER=$(uname -r)
    DIR=$(dirname "$(modinfo -n nvidia)")
    BACKUP=/root/nvidia-stock-$KVER-$(date +%Y%m%d-%H%M%S)
    mkdir -p "$BACKUP"
    for m in nvidia nvidia-uvm nvidia-modeset nvidia-drm nvidia-peermem; do
        f=$(modinfo -n $m 2>/dev/null) && cp -p "$f" "$BACKUP/"
    done
    ls "$BACKUP"                               # must contain nvidia.ko* and nvidia-uvm.ko*
    cd /root && git clone -b $VERSION-p2p https://github.com/aikitoria/open-gpu-kernel-modules
    cd open-gpu-kernel-modules
    make modules -j"$(nproc)" SYSSRC=/lib/modules/$KVER/build
    ls kernel-open/nvidia*.ko                  # five .ko files
    rm -f "$DIR"/nvidia*.ko*
    cp kernel-open/nvidia*.ko "$DIR/"
    depmod -a "$KVER"

Then ask the owner to reboot (or reboot if they already approved it).

Verify after reboot:

    cat /proc/driver/nvidia/version            # same version as before
    nvidia-smi                                 # all GPUs listed
    nvidia-smi topo -p2p r                     # OK between every GPU pair
    python3 driver/p2p_check.py                # needs PyTorch with CUDA; every pair "p2p" and "ok"

Expected on 3090s: NVLink pairs about 50 GB/s, PCIe x16 pairs about 26 GB/s, any pair with an
x4 link about 6.6 GB/s. A pair showing "host" or "DATA MISMATCH" is a failure.

Rollback:

    rm -f "$DIR"/nvidia*.ko
    cp -p "$BACKUP"/* "$DIR/"
    depmod -a "$KVER"
    # reboot, then nvidia-smi topo -p2p r shows the stock state again

After any kernel update: DKMS installs the stock module for the new kernel and P2P is gone
without errors. Rebuild with `SYSSRC=/lib/modules/<new>/build`, copy into that kernel's module
directory, `depmod -a <new>`, reboot, verify.

Serving: set `NCCL_P2P_LEVEL=SYS` in the environment of vLLM / NCCL jobs, or NCCL only uses
NVLink pairs.

## Component 2: vLLM forced custom all-reduce (vllm-custom-allreduce/)

What it does: a `sitecustomize.py` that makes vLLM treat PCIe GPUs as fully connected, so its
custom all-reduce runs at tensor-parallel size > 2. `CA_MAX_BYTES` limits it to small messages.

Prerequisites and checks:

    nvidia-smi topo -p2p r                     # OK between every pair of GPUs the server will use
    python3 -c "import vllm; print(vllm.__version__, vllm.__file__)"
    grep -n "def is_fully_connected" "$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')/platforms/cuda.py"
    grep -n "def should_custom_ar" "$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')/distributed/device_communicators/custom_all_reduce.py"

Both greps must match. Run the python commands with the same interpreter the server uses.

Install:

    mkdir -p /opt/vllm-force-ca
    cp vllm-custom-allreduce/sitecustomize.py /opt/vllm-force-ca/

Add to the server's environment (systemd unit, launcher script, or proxy config), and remove
`--disable-custom-all-reduce` from its arguments:

    PYTHONPATH=/opt/vllm-force-ca            # prepend if PYTHONPATH is already set
    CA_MAX_BYTES=<hidden_size * 2>           # 8192 for hidden size 4096; keep 8192 with spec decode
    NCCL_P2P_LEVEL=SYS

Remove `expandable_segments:True` from `PYTORCH_CUDA_ALLOC_CONF` if present. With it the server
crashes during CUDA graph capture (`custom_all_reduce.cuh ... invalid argument`).

Verify:

- Server log contains `[force-ca] is_fully_connected -> True` and
  `[force-ca] custom all-reduce capped at <n> bytes`.
- Server log does not contain `Custom allreduce is disabled`.
- Send the same prompts at temperature 0 with and without the overlay; outputs must match or be
  equally coherent. Compare single-stream decode t/s and prefill t/s against the old setup;
  if prefill dropped, lower `CA_MAX_BYTES`. Never raise it to cover speculative-decoding verify
  steps (16-80 KB): that measured 16-22% slower.
- Compare the KV cache size line in the server log before and after; with `expandable_segments`
  off it is usually larger, which allows a larger `--max-model-len`.

Rollback: remove `/opt/vllm-force-ca` from `PYTHONPATH`, unset `CA_MAX_BYTES`, restart the
server. Optionally add back `--disable-custom-all-reduce`.

## Component 3: gpu-temp-guard (gpu-temp-guard/)

What it does: systemd service that lowers a card's power limit by 20 W steps when hotspot or
VRAM reaches 94 C (floor 150 W) and restores it after 5 minutes under 85 C.

Prerequisites and checks:

    which nvidia-smi python3
    gputemps --once --json                     # numeric junction and vram for each card

If `gputemps` is missing, install it:

    git clone https://github.com/ThomasBaruzier/gddr6-core-junction-vram-temps /root/gputemps
    cd /root/gputemps && make && make install  # needs gcc make libpci-dev and NVML headers

If junction/vram are null, the kernel may need `iomem=relaxed`; tell the owner, do not add it.

Install:

    gpu-temp-guard/install.sh                  # installs files, runs --selftest, does not start
    gpu-temp-guard --selftest | tail -1        # "N scenarios, 0 failed"

Edit `/etc/default/gpu-temp-guard`: set `GUARD_MATCH` to the card names to guard (empty = all)
and put any card with implausible gputemps readings in `GUARD_EXCLUDE_BUSES`. Then:

    gpu-temp-guard --dry-run --verbose --polls 3   # each card listed, sane temps, no errors
    systemctl enable --now gpu-temp-guard

Verify:

    systemctl is-active gpu-temp-guard         # active
    tail -n 5 /var/log/gpu-temp-guard.log      # START line with the expected cards and exclusions

Rollback:

    systemctl disable --now gpu-temp-guard     # restores any card it was holding
    rm /usr/local/sbin/gpu-temp-guard /etc/systemd/system/gpu-temp-guard.service \
       /etc/logrotate.d/gpu-temp-guard /etc/default/gpu-temp-guard
    rm -rf /var/lib/gpu-temp-guard
    systemctl daemon-reload

## Component 4: MiMo-V2 three-module MTP (mimo-mtp3/)

What it does: a `sitecustomize.py` that makes vLLM build all three MiMo-V2 MTP modules
(vLLM hard-codes one) and feed each the target model's hidden state.

Prerequisites and checks (same interpreter as the server):

    V=$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
    grep -n "_MIMO_V2_FLASH_NUM_MTP_LAYERS\|class MiMoV2MTPLayer\|class _MiMoV2MTPLayers" $V/model_executor/models/mimo_v2_mtp.py
    grep -n "isinstance(ret_hidden_states, tuple)" $V/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py
    grep -n "use_local_argmax_reduction" $V/config/speculative.py

All must match. MiMo-V2 must already serve on this vLLM with MTP k=1 (it needs upstream PRs
#57508 and #57784). If it does not, stop and report.

Install:

    mkdir -p /opt/vllm-mimo-mtp3
    cp mimo-mtp3/sitecustomize.py /opt/vllm-mimo-mtp3/

In the server environment put `/opt/vllm-mimo-mtp3` first on `PYTHONPATH` (before
`/opt/vllm-force-ca` if that is used) and set the speculative config to:

    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'

If the server no longer starts because the KV cache is too small, add component 2 with
`CA_MAX_BYTES=8192` and remove `expandable_segments:True`; only then lower `--max-model-len`.

Verify:

- Log contains `[mimo-mtp3] MTP modules: 3, feedback: target`.
- Spec-decode metrics (`/metrics`, `vllm:spec_decode_num_accepted_tokens_per_pos`) show
  acceptance at positions 1 and 2 well above 0.1 (about 0.68 and 0.54 measured).
- Same prompts at temperature 0: answers coherent and equivalent to the k=1 server; single-stream
  decode t/s higher than before.

Rollback: remove `/opt/vllm-mimo-mtp3` from `PYTHONPATH`, set `num_speculative_tokens` back to 1,
restart the server.

## Component 5: fp8 KV for GLM-5.3-Flash on vllm-backport (glm-sparse-mla-fp8kv/)

What it does: a patch to wtdcode/vllm-backport (4 Python/Triton files) that lets the
TRITON_MLA_SPARSE backend use `--kv-cache-dtype fp8_e5m2`, plus the env var
`VLLM_INDEXER_PREFILL_BUFFER_FACTOR`.

Prerequisites and checks:

    cd <backport tree> && git log -1 --format=%H    # tested: cde54e8ed390aef9e7d0670474365ae033b38a40
    git status --short                              # note any local changes
    git apply --check /path/to/glm-sparse-mla-fp8kv/fp8kv.patch

If `--check` fails, stop and report. The tree must already serve GLM-5.3-Flash with a bf16 KV
cache. Never patch the tree a running server uses.

Install (on a copy; no rebuild needed):

    cp -a <backport tree> <backport tree>-fp8kv
    cd <backport tree>-fp8kv && git apply /path/to/glm-sparse-mla-fp8kv/fp8kv.patch

Server changes: `PYTHONPATH` points at the copy, add `--kv-cache-dtype fp8_e5m2` and
`VLLM_INDEXER_PREFILL_BUFFER_FACTOR=8` (never below 8), then raise `--max-model-len` step by step
(159,744 fit on 8x 24 GB with component 2 loaded and expandable_segments off).

Verify:

- Server starts; log shows the fp8_e5m2 KV cache and a KV size covering `--max-model-len`.
- Needle test: a fact placed mid-way in a prompt near the maximum length is retrieved.
- Same short prompts at temperature 0 against the bf16 server: answers equivalent. Decode t/s
  within a few percent of bf16.

Rollback: point `PYTHONPATH` back at the unpatched tree, drop `--kv-cache-dtype` and the env var,
restore the old `--max-model-len`, restart.
