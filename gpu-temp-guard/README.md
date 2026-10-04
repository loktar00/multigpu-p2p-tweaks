# gpu-temp-guard

Overheat protection for NVIDIA cards that report hotspot (junction) and VRAM temperatures.
It leaves your power limits alone until a card gets too hot, then trims that one card's limit
and gives it back once the card has cooled down.

Every 5 s it reads `gputemps --once --json`. If a guarded card's hotspot or VRAM reaches 94 C,
its power limit drops 20 W. Still at 94 C or more 60 s later, another 20 W. Never below 150 W.
Once both temperatures stay under 85 C for 5 minutes, the card gets back the limit it had before
the guard touched it, and never more than that. If you change a held card's limit yourself, the
guard forgets that card and keeps your value. Missing or junk readings mean no action that poll.
Cards are tracked by PCI bus id, so index reshuffles don't matter.

The defaults were tuned on RTX 3090s. Change them in `/etc/default/gpu-temp-guard`.

## Requirements

- `nvidia-smi` and root.
- [gputemps](https://github.com/ThomasBaruzier/gddr6-core-junction-vram-temps) installed. It reads
  the memory-mapped temperature registers that NVML does not expose, and often needs
  `iomem=relaxed` on the kernel command line. Check it works before going further:

      sudo gputemps --once --json

  Every card you want guarded should show numeric `junction` and `vram` values.

## Install

    git clone https://github.com/ThomasBaruzier/gddr6-core-junction-vram-temps
    (cd gddr6-core-junction-vram-temps && make && sudo make install)
    sudo ./install.sh
    sudoedit /etc/default/gpu-temp-guard
    sudo gpu-temp-guard --dry-run --verbose --polls 3
    sudo systemctl enable --now gpu-temp-guard

`--dry-run` reads the real temperatures and logs what it would do without setting anything.
`--selftest` runs the decision logic against synthetic readings and prints a pass/fail table.

`GUARD_MATCH` picks cards by name substring (empty = all). `GUARD_EXCLUDE_BUSES` leaves cards
out by bus id; use it for any card whose gputemps readings are wrong (one Blackwell workstation
card here reported nonsense hotspot/VRAM values).

## Turning it off

    sudo touch /etc/gpu-temp-guard.disabled    # keeps running, takes no action, hands back held cards
    sudo rm /etc/gpu-temp-guard.disabled       # back on
    sudo systemctl disable --now gpu-temp-guard  # off across reboots, hands back held cards as it stops

## Files

| file | installed to |
|---|---|
| `gpu-temp-guard.py` | `/usr/local/sbin/gpu-temp-guard` |
| `gpu-temp-guard.service` | `/etc/systemd/system/gpu-temp-guard.service` |
| `gpu-temp-guard.logrotate` | `/etc/logrotate.d/gpu-temp-guard` |
| `gpu-temp-guard.conf` | `/etc/default/gpu-temp-guard` (not overwritten on reinstall) |

Logs go to `/var/log/gpu-temp-guard.log` (daily, 14 kept) and `journalctl -u gpu-temp-guard`,
with one `OK` summary line per hour. Held cards are saved in `/var/lib/gpu-temp-guard/state.json`
so a crash restart keeps them; the file is ignored after a reboot.

Uninstall:

    sudo systemctl disable --now gpu-temp-guard
    sudo rm /usr/local/sbin/gpu-temp-guard /etc/systemd/system/gpu-temp-guard.service \
        /etc/logrotate.d/gpu-temp-guard /etc/default/gpu-temp-guard
    sudo rm -rf /var/lib/gpu-temp-guard
    sudo systemctl daemon-reload
