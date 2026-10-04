#!/usr/bin/env python3
"""Check GPU peer access and time GPU-to-GPU copies with PyTorch.

Prints the peer-access matrix the driver reports, then copies a buffer between every pair,
checks the data and prints GB/s. Pairs without peer access are copied through host memory
and marked "host".

    python3 p2p_check.py                 # all visible GPUs, 1 GiB, 5 timed copies per pair
    python3 p2p_check.py --gpus 0,1,4 --size-mib 256

MIT License, see LICENSE at the repository root.
"""
import argparse
import time

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", help="comma-separated device indices (default: all visible)")
    ap.add_argument("--size-mib", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    n = torch.cuda.device_count()
    ids = [int(x) for x in args.gpus.split(",")] if args.gpus else list(range(n))
    if len(ids) < 2:
        raise SystemExit("need at least two GPUs, found %d" % n)
    for i in ids:
        p = torch.cuda.get_device_properties(i)
        print("cuda:%d  %s  %d MiB" % (i, p.name, p.total_memory // 2**20))

    # Ask the driver first. Timing a copy before this lets CUDA stage it through host memory,
    # and even NVLink pairs then read around 7 GB/s.
    peer = {(a, b): torch.cuda.can_device_access_peer(a, b) for a in ids for b in ids if a != b}
    print("\npeer access (row = source):")
    print("      " + "".join("%5d" % b for b in ids))
    for a in ids:
        print("%5d " % a + "".join("    -" if a == b else ("  yes" if peer[(a, b)] else "   no") for b in ids))

    nbytes = args.size_mib * 2**20
    print("\ncopy %d MiB, %d timed copies per pair:" % (args.size_mib, args.iters))
    bad = 0
    for a in ids:
        for b in ids:
            if a == b:
                continue
            src = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=a)
            dst = torch.empty(nbytes, dtype=torch.uint8, device=b)
            dst.copy_(src)  # warm-up, sets up peer mappings
            torch.cuda.synchronize(a)
            torch.cuda.synchronize(b)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                dst.copy_(src)
            torch.cuda.synchronize(a)
            torch.cuda.synchronize(b)
            dt = time.perf_counter() - t0
            ok = torch.equal(src, dst.to(a))
            bad += not ok
            print("  %d -> %d  %7.1f GB/s  %s  %s" % (a, b, nbytes * args.iters / dt / 1e9,
                                                    "p2p " if peer[(a, b)] else "host", "ok" if ok else "DATA MISMATCH"))
            del src, dst
            torch.cuda.empty_cache()
    if bad:
        raise SystemExit("%d pair(s) returned wrong data" % bad)


if __name__ == "__main__":
    main()
