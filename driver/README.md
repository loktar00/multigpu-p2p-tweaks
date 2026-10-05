# P2P driver notes (same-generation cards)

Stock NVIDIA drivers don't allow peer-to-peer transfers between GeForce cards, so NCCL and
vLLM bounce every GPU-to-GPU transfer through host memory. The community P2P fork of NVIDIA's
open kernel modules turns it on: NVLink where a bridge exists, PCIe BAR1 everywhere else.

Use [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules)
and pick the branch that matches your installed driver version exactly (`<version>-p2p`).
Its README covers the build. This page is what we learned running it on 8x RTX 3090
(595.58.03-p2p, commit 6dd6ba3).

Mixing Ampere and Blackwell in one box? The fork breaks CUDA init there; don't use it.

## Before you start

- The fork only replaces the kernel modules. The user-space driver must already be installed
  and be the same version: `cat /proc/driver/nvidia/version`.
- BIOS: Above 4G Decoding on, which also enables Resizable BAR on most boards. Each 3090 then
  shows a 32 GiB BAR1 instead of 256 MiB: `nvidia-smi -q -d MEMORY | grep -A3 BAR1`.
- Kernel command line: `amd_iommu=on iommu=pt` (Intel: `intel_iommu=on iommu=pt`), then
  `update-grub` and reboot. Translated IOMMU mode breaks the transfers. Passthrough mode also
  means devices can DMA anywhere in memory; don't do this on a box running untrusted code.
- Secure Boot must be off or the modules signed, same as any out-of-tree module.
- Stop everything using the GPUs. Swapping modules under running jobs is how you lose a day.

## Install without losing your way back

On most distros the NVIDIA modules are built by DKMS into
`/lib/modules/$(uname -r)/updates/dkms/`, which wins over where the fork's `install.sh` puts
them. So build the fork and copy over the files the kernel actually loads:

    KVER=$(uname -r)
    DIR=$(dirname "$(modinfo -n nvidia)")
    BACKUP=/root/nvidia-stock-$KVER-$(date +%Y%m%d-%H%M%S)
    mkdir -p "$BACKUP"
    for m in nvidia nvidia-uvm nvidia-modeset nvidia-drm nvidia-peermem; do
        f=$(modinfo -n $m 2>/dev/null) && cp -p "$f" "$BACKUP/"
    done
    ls "$BACKUP"                        # check nvidia.ko and nvidia-uvm.ko are there
    git clone -b 595.58.03-p2p https://github.com/aikitoria/open-gpu-kernel-modules
    cd open-gpu-kernel-modules
    make modules -j"$(nproc)" SYSSRC=/lib/modules/$KVER/build
    rm -f "$DIR"/nvidia*.ko*            # removes the stock files you just backed up
    cp kernel-open/nvidia*.ko "$DIR/"
    depmod -a "$KVER"
    reboot

If the stock files are compressed (`.ko.zst`, `.ko.xz`) that's fine; the backup keeps them as
they were.

Rollback: copy the backup back and remove the fork's files.

    rm -f "$DIR"/nvidia*.ko
    cp -p "$BACKUP"/* "$DIR/"
    depmod -a "$KVER"
    reboot

You can leave `iommu=pt` in place; the stock driver works with it.

## Kernel updates

DKMS rebuilds the stock module for a new kernel and P2P silently disappears. Nothing breaks,
NCCL just falls back and everything gets slower. After every kernel update, rebuild the fork
against the new kernel and copy again (`SYSSRC=/lib/modules/<new>/build`, `depmod -a <new>`).
Check after each reboot with `nvidia-smi topo -p2p r`.

## Verify

    nvidia-smi topo -p2p r            # OK between every pair you expect (was CNS/NS before)
    python3 p2p_check.py              # peer matrix, then measured GB/s per pair with a data check

`p2p_check.py` asks `torch.cuda.can_device_access_peer` before timing anything. Skip that
and CUDA can stage the copy through host memory, and even an NVLink pair reads around 7 GB/s.

## Using it

NCCL limits P2P to NVLink unless told otherwise. Set this for vLLM and anything else on NCCL:

    export NCCL_P2P_LEVEL=SYS

`../vllm-custom-allreduce/` is an experiment with vLLM's own all-reduce on more than two PCIe
cards. It did not beat NCCL defaults here.

## Numbers (8x RTX 3090, 595.58.03-p2p)

Copy bandwidth, 1 GiB, data checked: NVLink pairs about 52 GB/s, PCIe P2P 26 GB/s between
x16 cards, 6.6 GB/s when an x4 card is involved.

Qwen3.8-Flash-Next FP8, vLLM TP8, 210 W per card, `NCCL_P2P_LEVEL=SYS`, against the stock driver:

| | stock | P2P | |
|---|---|---|---|
| decode, 1 request | 110.6 t/s | 128.9 t/s | +17% |
| decode, 3 requests | 239 t/s | 251.6 t/s | +5% |
| prefill 16k | 2,444 t/s | 3,152 t/s | +29% |
| prefill 64k | 2,382 t/s | 3,055 t/s | +28% |

Same power draw, so tokens per joule went up by the same margins.
