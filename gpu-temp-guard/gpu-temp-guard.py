#!/usr/bin/env python3
"""gpu-temp-guard: lower an NVIDIA card's power limit when its hotspot or VRAM gets too hot.

Every POLL_S seconds it reads `gputemps --once --json`. When a guarded card's hotspot
(junction) or VRAM temperature reaches TRIP_C it lowers that card's power limit by STEP_W
(never below FLOOR_W), and by another STEP_W for every STEP_INTERVAL_S the card stays that hot.
Once the card has stayed below RESTORE_C (hotspot and VRAM) for RESTORE_HOLD_S it gets back the
cap it had before the guard first touched it.

It never raises a card above that remembered cap, never touches a card that is not guarded,
and forgets a card (leaving the new value alone) when its cap is changed by someone else while
held.

Config:   /etc/default/gpu-temp-guard (KEY=VALUE, see gpu-temp-guard.conf); environment
          variables of the same name override the file.

Disable:  touch /etc/gpu-temp-guard.disabled   (keeps running, takes no action, hands back
                                                any card it is holding)
          rm /etc/gpu-temp-guard.disabled      (active again)
          systemctl disable --now gpu-temp-guard   (stopped, stays off across reboots)

Modes:    --selftest             synthetic readings through the decision logic, prints a table
          --dry-run --verbose    real temps, logs what it WOULD do, never sets a power limit
"""

import os

CONFIG_FILE = "/etc/default/gpu-temp-guard"


def _read_config(path):
    conf = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    conf[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    conf.update({k: v for k, v in os.environ.items() if k.startswith("GUARD_")})
    return conf


def _csv(v):
    return [x.strip() for x in v.split(",") if x.strip()]


def configure(path=CONFIG_FILE):
    """Set the module settings from the config file and GUARD_* environment variables."""
    global TRIP_C, RESTORE_C, RESTORE_HOLD_S, STEP_W, STEP_INTERVAL_S, FLOOR_W, POLL_S
    global GPUTEMPS, MATCH, EXCLUDE_BUSES
    c = _read_config(path)
    TRIP_C = int(c.get("GUARD_TRIP_C", 94))                  # hotspot or VRAM at/above this trips the card
    RESTORE_C = int(c.get("GUARD_RESTORE_C", 85))            # hotspot and VRAM both below this ...
    RESTORE_HOLD_S = int(c.get("GUARD_RESTORE_HOLD_S", 300))  # ... this long restores the remembered cap
    STEP_W = int(c.get("GUARD_STEP_W", 20))                  # watts removed per step
    STEP_INTERVAL_S = int(c.get("GUARD_STEP_INTERVAL_S", 60))  # still hot this long after a step: step again
    FLOOR_W = int(c.get("GUARD_FLOOR_W", 150))               # never lower a cap below this
    POLL_S = int(c.get("GUARD_POLL_S", 5))
    GPUTEMPS = [c.get("GUARD_GPUTEMPS") or "gputemps", "--once", "--json"]
    MATCH = _csv(c.get("GUARD_MATCH", ""))                   # name substrings; empty = every card
    EXCLUDE_BUSES = [b.upper() for b in _csv(c.get("GUARD_EXCLUDE_BUSES", ""))]


configure(os.devnull)   # defaults; main() reads the real config file

DISABLE_FLAG = "/etc/gpu-temp-guard.disabled"
LOG_FILE = "/var/log/gpu-temp-guard.log"
STATE_FILE = "/var/lib/gpu-temp-guard/state.json"
PLAUSIBLE_C = (1, 130)                     # readings outside this range count as missing
HEARTBEAT_S = 3600                         # one "OK" summary line per hour
ERROR_LOG_EVERY_S = 300                    # repeated warnings of one kind at most this often
CMD_TIMEOUT_S = 15
# ---------------------------------------------------------------------------------------------

import argparse
import json
import logging
import logging.handlers
import signal
import subprocess
import sys
import threading
import time

log = logging.getLogger("gpu-temp-guard")


class DataError(Exception):
    pass


def _plausible(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and PLAUSIBLE_C[0] <= v <= PLAUSIBLE_C[1]


def parse_temps(raw):
    """gputemps JSON -> {index: (hotspot, vram) or None}. Raises DataError on garbage."""
    try:
        doc = json.loads(raw)
    except (ValueError, TypeError) as e:
        raise DataError("gputemps output is not JSON (%s)" % e)
    if not isinstance(doc, dict) or not isinstance(doc.get("gpus"), list) or not doc["gpus"]:
        raise DataError("gputemps JSON has no gpus list")
    out = {}
    for g in doc["gpus"]:
        if not isinstance(g, dict):
            continue
        idx = g.get("index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            continue
        hs, vr = g.get("junction"), g.get("vram")
        out[idx] = (hs, vr) if _plausible(hs) and _plausible(vr) else None
    return out


def parse_gpus(raw):
    """nvidia-smi index,pci.bus_id,name,power.limit csv -> list of dicts. Raises DataError."""
    gpus = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            raise DataError("unexpected nvidia-smi line: %r" % line)
        try:
            idx = int(parts[0])
        except ValueError:
            raise DataError("unexpected nvidia-smi index: %r" % line)
        try:
            limit = float(parts[3])
        except ValueError:
            limit = None
        gpus.append({"index": idx, "bus": parts[1].upper(), "name": parts[2], "limit": limit})
    if not gpus:
        raise DataError("nvidia-smi listed no GPUs")
    return gpus


def _w(v):
    return "n/a" if v is None else "%d" % v


def guarded(g):
    """MATCH: name substrings (empty = every card). EXCLUDE_BUSES match on the end of the bus id,
    so 21:00.0, 0000:21:00.0 and 00000000:21:00.0 all exclude the same card."""
    if MATCH and not any(m in g["name"] for m in MATCH):
        return False
    return not any(g["bus"].endswith(b) for b in EXCLUDE_BUSES)


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=CMD_TIMEOUT_S)
    if r.returncode != 0:
        raise DataError("%s exited %d: %s" % (cmd[0], r.returncode, (r.stderr or r.stdout).strip()[:200]))
    return r.stdout


class RealIO:
    def flag_exists(self):
        return os.path.exists(DISABLE_FLAG)

    def read_temps(self):
        return _run(GPUTEMPS)

    def read_gpus(self):
        return parse_gpus(_run(["nvidia-smi", "--query-gpu=index,pci.bus_id,name,power.limit",
                                "--format=csv,noheader,nounits"]))

    def set_limit(self, bus, watts):
        try:
            _run(["nvidia-smi", "-i", bus, "-pl", str(int(round(watts)))])
            now = [g for g in self.read_gpus() if g["bus"] == bus]
        except (DataError, OSError, subprocess.SubprocessError) as e:
            log.error("ERROR    bus=%s setting %dW failed: %s", bus, watts, e)
            return False
        if not now or now[0]["limit"] is None or abs(now[0]["limit"] - watts) > 0.5:
            log.error("ERROR    bus=%s set %dW but nvidia-smi reads back %s", bus, watts,
                      now[0]["limit"] if now else "nothing")
            return False
        return True


class DryRunIO(RealIO):
    """Real readings, simulated power limits: logs what it would set and pretends it did."""

    def __init__(self):
        self.virtual = {}

    def read_gpus(self):
        gpus = RealIO.read_gpus(self)
        for g in gpus:
            if g["bus"] in self.virtual:
                g["limit"] = self.virtual[g["bus"]]
        return gpus

    def set_limit(self, bus, watts):
        log.info("DRY-RUN  would run: nvidia-smi -i %s -pl %d", bus, watts)
        self.virtual[bus] = float(watts)
        return True


def boot_id():
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return None


class Guard:
    def __init__(self, io, state_path=None, verbose=False, heartbeat_s=HEARTBEAT_S):
        self.io = io
        self.state_path = state_path
        self.verbose = verbose
        self.heartbeat_s = heartbeat_s
        self.held = {}          # bus -> {"orig", "set", "last_step", "cool_since", "floor_logged"}
        self.disabled = False
        self.last_warn = {}
        self.last_heartbeat = None
        self._load()

    # -- persistence (survives a crash + Restart=on-failure, discarded after a reboot) --
    def _load(self):
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                doc = json.load(f)
            if doc.get("boot_id") != boot_id():
                log.info("STATE    discarding held cards from a previous boot: %s", sorted(doc.get("held", {})))
                return
            self.held = doc.get("held", {})
            if self.held:
                log.info("STATE    resuming hold on %s", ", ".join(
                    "%s (guard set %dW, remembered %dW)" % (b, h["set"], h["orig"]) for b, h in self.held.items()))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            log.warning("WARN     ignoring unreadable state file %s: %s", self.state_path, e)
            self.held = {}

    def _save(self):
        if not self.state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"boot_id": boot_id(), "held": self.held}, f)
            os.replace(tmp, self.state_path)
        except OSError as e:
            log.warning("WARN     could not write state file: %s", e)

    # -- helpers --
    def warn(self, key, now, msg, *args):
        last = self.last_warn.get(key)
        if last is None or now - last >= ERROR_LOG_EVERY_S:
            self.last_warn[key] = now
            log.warning("WARN     " + msg, *args)

    @staticmethod
    def _t(t):
        return "hotspot=%sC vram=%sC" % t if t else "hotspot=n/a vram=n/a"

    def _break_cool_streaks(self):
        for h in self.held.values():
            h["cool_since"] = None

    def _read_temps_safe(self):
        try:
            return parse_temps(self.io.read_temps())
        except Exception:
            return {}

    # -- one poll --
    def poll(self, now):
        if self.io.flag_exists():
            if not self.disabled:
                self.disabled = True
                log.info("DISABLED flag %s present: taking no action%s", DISABLE_FLAG,
                         " (handing back held cards)" if self.held else "")
            if self.held:
                self.release_all(now, "disabled by flag file")
            return
        if self.disabled:
            self.disabled = False
            log.info("ENABLED  flag %s removed: guarding again", DISABLE_FLAG)

        try:
            temps = parse_temps(self.io.read_temps())
        except Exception as e:
            self.warn("gputemps", now, "no usable gputemps reading, skipping poll: %s", e)
            self._break_cool_streaks()
            return
        try:
            gpus = self.io.read_gpus()
        except Exception as e:
            self.warn("nvidia-smi", now, "nvidia-smi query failed, skipping poll: %s", e)
            self._break_cool_streaks()
            return

        seen = set()
        hottest = None
        for g in gpus:
            bus = g["bus"]
            t = temps.get(g["index"])
            if not guarded(g):
                if self.verbose:
                    log.info("POLL     idx%d bus=%s %s: excluded (not guarded)", g["index"], bus, g["name"])
                continue
            seen.add(bus)
            h = self.held.get(bus)
            if h is not None and (g["limit"] is None or abs(g["limit"] - h["set"]) > 0.5):
                log.info("OVERRIDE bus=%s idx%d %s cap %sW -> %sW changed outside the guard (guard set %dW); "
                         "forgetting this card, not touching its cap", bus, g["index"], self._t(t),
                         "%d" % h["set"], _w(g["limit"]), h["set"])
                del self.held[bus]
                self._save()
                continue
            if t is None:
                self.warn("missing:" + bus, now, "bus=%s idx%d: no plausible hotspot/VRAM reading, no action",
                          bus, g["index"])
                if h is not None:
                    h["cool_since"] = None
                continue
            peak = max(t)
            hottest = t if hottest is None else (max(hottest[0], t[0]), max(hottest[1], t[1]))
            action = self._decide(g, t, peak, h, now)
            if self.verbose:
                log.info("POLL     idx%d bus=%s %s cap=%sW %s -> %s", g["index"], bus, g["name"],
                         g["limit"], self._t(t), action)

        for bus in self.held:
            if bus not in seen:
                self.warn("gone:" + bus, now, "held card bus=%s is not in the nvidia-smi listing", bus)

        if self.heartbeat_s and (self.last_heartbeat is None or now - self.last_heartbeat >= self.heartbeat_s):
            self.last_heartbeat = now
            log.info("OK       guarding %d card(s), hottest hotspot %sC / vram %sC, holding %d%s", len(seen),
                     hottest[0] if hottest else "n/a", hottest[1] if hottest else "n/a", len(self.held),
                     (": " + ", ".join("%s at %dW (remembered %dW)" % (b, h["set"], h["orig"])
                                       for b, h in self.held.items())) if self.held else "")

    def _decide(self, g, t, peak, h, now):
        bus = g["bus"]
        if peak >= TRIP_C:
            if h is None:
                return self._step(g, t, now, None)
            h["cool_since"] = None
            if now - h["last_step"] >= STEP_INTERVAL_S:
                return self._step(g, t, now, h)
            return "hot, held at %dW (next step in %ds)" % (h["set"], STEP_INTERVAL_S - (now - h["last_step"]))
        if h is None:
            return "no action"
        if peak < RESTORE_C:
            if h["cool_since"] is None:
                h["cool_since"] = now
            if now - h["cool_since"] >= RESTORE_HOLD_S:
                if self.io.set_limit(bus, h["orig"]):
                    log.info("RESTORE  bus=%s idx%d %s cap %dW -> %dW (below %dC for %ds)", bus, g["index"],
                             self._t(t), h["set"], h["orig"], RESTORE_C, now - h["cool_since"])
                    del self.held[bus]
                    self._save()
                    return "restored"
                return "restore failed, retrying"
            return "cool, held at %dW (restore in %ds)" % (h["set"], RESTORE_HOLD_S - (now - h["cool_since"]))
        h["cool_since"] = None
        return "warm, held at %dW" % h["set"]

    def _step(self, g, t, now, h):
        bus = g["bus"]
        cur = h["set"] if h else g["limit"]
        if cur is None:
            self.warn("nolimit:" + bus, now, "bus=%s idx%d %s hot but its power limit is unreadable, no action",
                      bus, g["index"], self._t(t))
            return "no action (limit unreadable)"
        new = max(FLOOR_W, cur - STEP_W)
        if new >= cur:
            if h is None:
                self.warn("floor:" + bus, now, "AT-FLOOR bus=%s idx%d %s cap %dW already at/below the %dW floor, "
                          "no action", bus, g["index"], self._t(t), cur, FLOOR_W)
            elif not h.get("floor_logged"):
                h["floor_logged"] = True
                log.warning("AT-FLOOR bus=%s idx%d %s cap %dW is the %dW floor, cannot lower further",
                            bus, g["index"], self._t(t), cur, FLOOR_W)
            return "at floor"
        if not self.io.set_limit(bus, new):
            return "set failed"
        if h is None:
            self.held[bus] = {"orig": cur, "set": new, "last_step": now, "cool_since": None, "floor_logged": False}
            kind = "TRIP    "
        else:
            h["set"], h["last_step"] = new, now
            kind = "STEP    "
        log.info("%s bus=%s idx%d %s cap %dW -> %dW (>= %dC; remembered cap %dW)", kind.strip().ljust(8), bus,
                 g["index"], self._t(t), cur, new, TRIP_C, self.held[bus]["orig"])
        self._save()
        return kind.strip().lower()

    def release_all(self, now, reason):
        """Hand every held card back to its remembered cap (flag file / service stop)."""
        try:
            gpus = {g["bus"]: g for g in self.io.read_gpus()}
        except Exception as e:
            self.warn("release", now, "cannot hand back held cards (%s): nvidia-smi failed: %s", reason, e)
            return
        temps = self._read_temps_safe()
        for bus in list(self.held):
            h, g = self.held[bus], gpus.get(bus)
            if g is None:
                self.warn("gone:" + bus, now, "held card bus=%s missing, cannot hand it back", bus)
                continue
            t = temps.get(g["index"])
            if g["limit"] is None or abs(g["limit"] - h["set"]) > 0.5:
                log.info("OVERRIDE bus=%s idx%d %s cap %sW changed outside the guard (guard set %dW); "
                         "forgetting this card, not touching its cap", bus, g["index"], self._t(t), _w(g["limit"]), h["set"])
                del self.held[bus]
            elif self.io.set_limit(bus, h["orig"]):
                log.info("RESTORE  bus=%s idx%d %s cap %dW -> %dW (%s)", bus, g["index"], self._t(t),
                         h["set"], h["orig"], reason)
                del self.held[bus]
        self._save()


# ---- self-test ---------------------------------------------------------------------------------

class FakeIO:
    def __init__(self, cards):
        # cards: list of (index, bus, name, limit)
        self.cards = [{"index": i, "bus": b, "name": n, "limit": float(l)} for i, b, n, l in cards]
        self.temps_raw = "{}"
        self.flag = False
        self.gpus_error = None
        self.temps_error = None

    def flag_exists(self):
        return self.flag

    def read_temps(self):
        if self.temps_error:
            raise DataError(self.temps_error)
        return self.temps_raw

    def read_gpus(self):
        if self.gpus_error:
            raise DataError(self.gpus_error)
        return [dict(c) for c in self.cards]

    def set_limit(self, bus, watts):
        for c in self.cards:
            if c["bus"] == bus:
                c["limit"] = float(watts)
                return True
        return False

    def cap(self, bus):
        return [c["limit"] for c in self.cards if c["bus"] == bus][0]


class ListHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


A, B, X = "00000000:01:00.0", "00000000:02:00.0", "00000000:21:00.0"
G3090, GOTHER = "NVIDIA GeForce RTX 3090", "NVIDIA RTX A6000"


def tj(readings):
    """{index: (hotspot, vram)} -> gputemps JSON."""
    return json.dumps({"timestamp": 0, "gpus": [{"index": i, "core": 30, "junction": h, "vram": v}
                                                 for i, (h, v) in sorted(readings.items())]})


def selftest():
    global MATCH, EXCLUDE_BUSES
    configure(os.devnull)
    MATCH, EXCLUDE_BUSES = ["RTX 3090"], ["21:00.0"]
    cap = ListHandler()
    log.handlers[:] = [cap]
    log.setLevel(logging.INFO)
    rows, failures, samples = [], 0, {}

    def run(name, cards, steps, check, flag_steps=()):
        """steps: list of (t, temps) where temps is a dict, a raw string, or a callable(io)."""
        nonlocal failures
        io = FakeIO(cards)
        guard = Guard(io, heartbeat_s=None)
        cap.lines = []
        trace = []
        for t, temps in steps:
            io.temps_error = io.gpus_error = None
            if callable(temps):
                temps(io)
            elif isinstance(temps, dict):
                io.temps_raw = tj(temps)
            else:
                io.temps_raw = temps
            before = len(cap.lines)
            guard.poll(float(t))
            acts = [l.split()[0] for l in cap.lines[before:]]
            if acts:
                trace.append("t=%d %s" % (t, "+".join(acts)))
        ok, caps = check(io, guard, cap.lines)
        failures += 0 if ok else 1
        rows.append((name, "; ".join(trace) or "(none)", caps, "PASS" if ok else "FAIL"))
        samples[name] = list(cap.lines)
        return io, cap.lines

    two = [(0, A, G3090, 210), (1, X, GOTHER, 300)]

    def caps(io):
        return " ".join("%s=%d" % (c["bus"][9:11], c["limit"]) for c in io.cards)

    def want(expected, needs=(), forbids=()):
        def check(io, guard, lines):
            got = {c["bus"]: c["limit"] for c in io.cards}
            ok = all(got[b] == w for b, w in expected.items())
            ok = ok and all(any(l.startswith(n) for l in lines) for n in needs)
            ok = ok and not any(l.startswith(f) for l in lines for f in forbids)
            return ok, caps(io)
        return check

    run("1 trip at hotspot 94", two, [(0, {0: (94, 80), 1: (40, 40)})],
        want({A: 190, X: 300}, needs=["TRIP"]))
    run("2 trip at VRAM 94 (hotspot 70)", two, [(0, {0: (70, 94), 1: (40, 40)})],
        want({A: 190}, needs=["TRIP"]))
    run("3 no trip at 93/93", two, [(0, {0: (93, 93), 1: (40, 40)}), (60, {0: (93, 93), 1: (40, 40)})],
        want({A: 210}, forbids=["TRIP", "STEP"]))
    hot = {0: (96, 90), 1: (40, 40)}
    run("4 second step only after 60 s", two, [(0, hot), (30, hot), (55, hot), (60, hot)],
        want({A: 170}, needs=["TRIP", "STEP"]))
    run("5 one step per 60 s, not per poll", two, [(0, hot), (5, hot), (10, hot), (15, hot)],
        want({A: 190}, forbids=["STEP"]))
    run("6 floor at 150 (210 -> 190 -> 170 -> 150, stays)", two,
        [(0, hot), (60, hot), (120, hot), (180, hot), (240, hot), (300, hot)],
        want({A: 150}, needs=["AT-FLOOR"]))
    run("7 160 W card trips to 150, not 140", [(0, A, G3090, 160)], [(0, {0: (95, 95)}), (60, {0: (95, 95)})],
        want({A: 150}, needs=["TRIP", "AT-FLOOR"]))
    run("8 never raises a card below the floor (140 W, hot)", [(0, A, G3090, 140)],
        [(0, {0: (99, 99)}), (60, {0: (99, 99)})], want({A: 140}, needs=["WARN"], forbids=["TRIP", "STEP"]))
    cool = {0: (84, 84), 1: (40, 40)}
    run("9 restore after 5 min below 85", two,
        [(0, hot), (10, cool)] + [(10 + 5 * k, cool) for k in range(1, 60)] + [(310, cool)],
        want({A: 210}, needs=["TRIP", "RESTORE"]))
    run("10 no restore at 4 min 55 s below 85", two,
        [(0, hot)] + [(10 + 5 * k, cool) for k in range(0, 60)],
        want({A: 190}, forbids=["RESTORE"]))
    run("11 no restore at 86 (10 min)", two,
        [(0, hot)] + [(10 + 5 * k, {0: (86, 80), 1: (40, 40)}) for k in range(0, 120)],
        want({A: 190}, forbids=["RESTORE"]))
    run("12 cool streak broken by one 86 poll", two,
        [(0, hot), (10, cool), (200, cool), (205, {0: (80, 86), 1: (40, 40)}), (210, cool), (400, cool)],
        want({A: 190}, forbids=["RESTORE"]))
    run("13 restore after 2 steps returns the ORIGINAL cap", two,
        [(0, hot), (60, hot), (70, cool), (370, cool)],
        want({A: 210}, needs=["TRIP", "STEP", "RESTORE"]))

    def user_sets_200(io):
        io.cards[0]["limit"] = 200.0
        io.temps_raw = tj({0: (84, 84), 1: (40, 40)})
    run("14 manual override: user sets 200 W while held", two,
        [(0, hot), (10, user_sets_200), (20, cool), (400, cool), (800, cool)],
        want({A: 200}, needs=["TRIP", "OVERRIDE"], forbids=["RESTORE"]))

    def set_flag(on, temps):
        def f(io):
            io.flag = on
            io.temps_raw = tj(temps)
        return f
    run("15 flag file: held card handed back, no action while flagged", two,
        [(0, hot), (10, set_flag(True, hot)), (15, set_flag(True, hot)), (80, set_flag(True, {0: (99, 99), 1: (40, 40)}))],
        want({A: 210}, needs=["TRIP", "DISABLED", "RESTORE"], forbids=["STEP"]))
    run("16 flag removed: guarding again", two,
                        [(0, set_flag(True, hot)), (5, set_flag(True, hot)), (10, set_flag(False, hot))],
                        want({A: 190}, needs=["DISABLED", "ENABLED", "TRIP"]))
    run("17 DISABLED logged once across many flagged polls", two,
        [(5 * k, set_flag(True, hot)) for k in range(20)],
        lambda io, g, lines: (sum(l.startswith("DISABLED") for l in lines) == 1 and io.cap(A) == 210, caps(io)))
    run("18 card not in GUARD_MATCH ignored even at 120/120", two, [(0, {0: (40, 40), 1: (120, 120)}), (60, {0: (40, 40), 1: (120, 120)})],
        want({A: 210, X: 300}, forbids=["TRIP", "STEP"]))
    run("19 excluded bus ignored even if the name matches", [(0, A, G3090, 210), (1, X, G3090, 300)],
        [(0, {0: (40, 40), 1: (120, 120)})], want({X: 300}, forbids=["TRIP"]))

    def match_all(io):
        global MATCH
        MATCH = []
        io.temps_raw = tj({0: (96, 90), 1: (96, 90), 2: (96, 90)})
    run("19b empty GUARD_MATCH guards every card except excluded buses",
        [(0, A, G3090, 210), (1, B, GOTHER, 300), (2, X, GOTHER, 300)], [(0, match_all)],
        want({A: 190, B: 280, X: 300}, needs=["TRIP"]))
    MATCH = ["RTX 3090"]
    for i, (label, raw) in enumerate([
            ("not JSON", "garbage{{"), ("empty output", ""), ("truncated JSON", '{"gpus":[{"index":0,"junc'),
            ("no gpus list", '{"timestamp":1}'), ("gpus not a list", '{"gpus":"hot"}'),
            ("hotspot null", '{"gpus":[{"index":0,"junction":null,"vram":99}]}'),
            ("hotspot string", '{"gpus":[{"index":0,"junction":"99","vram":99}]}'),
            ("hotspot 255 (sensor junk)", '{"gpus":[{"index":0,"junction":255,"vram":99}]}'),
            ("VRAM missing", '{"gpus":[{"index":0,"junction":99}]}')]):
        run("20%s garbage: %s" % ("abcdefghi"[i], label), two, [(0, raw), (60, raw)],
            want({A: 210}, needs=["WARN"], forbids=["TRIP", "STEP"]))

    def gputemps_fails(io):
        io.temps_error = "gputemps exited 1: NVML init failed"
    run("21 gputemps exits non-zero", two, [(0, gputemps_fails), (60, gputemps_fails)],
        want({A: 210}, needs=["WARN"], forbids=["TRIP"]))

    def smi_fails(io):
        io.temps_raw = tj(hot)
        io.gpus_error = "nvidia-smi timed out"
    run("22 nvidia-smi query fails while hot", two, [(0, smi_fails), (60, smi_fails)],
        want({A: 210}, needs=["WARN"], forbids=["TRIP"]))
    run("23 garbage warnings rate-limited (60 bad polls = 1 line)", two, [(5 * k, "garbage") for k in range(60)],
        lambda io, g, lines: (sum(l.startswith("WARN") for l in lines) == 1, caps(io)))
    run("24 a garbage poll breaks the cool streak", two,
        [(0, hot), (10, cool), (200, "garbage"), (205, cool), (400, cool)],
        want({A: 190}, forbids=["RESTORE"]))

    def reshuffle(io):
        io.cards = [{"index": 0, "bus": B, "name": G3090, "limit": 210.0},
                    {"index": 1, "bus": A, "name": G3090, "limit": io.cap(A)}]
        io.temps_raw = tj({0: (50, 50), 1: (96, 90)})
    run("25 index order changes: card followed by bus id", [(0, A, G3090, 210), (1, B, G3090, 210)],
        [(0, {0: (96, 90), 1: (50, 50)}), (60, reshuffle)],
        want({A: 170, B: 210}, needs=["TRIP", "STEP"]))
    run("26 two hot cards tracked independently", [(0, A, G3090, 210), (1, B, G3090, 210)],
        [(0, {0: (96, 90), 1: (50, 50)}), (30, {0: (96, 90), 1: (95, 50)}), (60, {0: (96, 90), 1: (95, 50)})],
        want({A: 170, B: 190}))

    print("%-62s | %-62s | %-14s | %s" % ("scenario", "actions (t = seconds)", "caps after", "result"))
    print("-" * 150)
    for r in rows:
        print("%-62s | %-62s | %-14s | %s" % r)
    print("-" * 150)
    print("%d scenarios, %d failed" % (len(rows), failures))
    for name in samples:
        if name.split()[0] in ("13", "14", "15"):
            print()
            print("log lines, scenario %s:" % name)
            for line in samples[name]:
                print("  " + line)
    return 1 if failures else 0


# ---- service loop ------------------------------------------------------------------------------

def setup_logging(logfile):
    log.setLevel(logging.INFO)
    log.handlers[:] = []
    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(out)
    if logfile:
        fh = logging.handlers.WatchedFileHandler(logfile)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the synthetic decision-table tests and exit")
    ap.add_argument("--dry-run", action="store_true", help="real readings, never set a power limit, stdout only")
    ap.add_argument("--verbose", action="store_true", help="log every card on every poll")
    ap.add_argument("--config", default=CONFIG_FILE, help="config file (default %(default)s)")
    ap.add_argument("--polls", type=int, default=0, help="stop after N polls (0 = run forever)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    configure(args.config)
    setup_logging(None if args.dry_run else LOG_FILE)
    io = DryRunIO() if args.dry_run else RealIO()
    guard = Guard(io, state_path=None if args.dry_run else STATE_FILE, verbose=args.verbose)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    log.info("START    %sTRIP_C=%d RESTORE_C=%d RESTORE_HOLD_S=%d STEP_W=%d STEP_INTERVAL_S=%d FLOOR_W=%d POLL_S=%d "
             "guarding %s, excluded buses %s, disable flag %s", "DRY-RUN " if args.dry_run else "",
             TRIP_C, RESTORE_C, RESTORE_HOLD_S, STEP_W, STEP_INTERVAL_S, FLOOR_W, POLL_S, MATCH or "every card",
             EXCLUDE_BUSES or "none", DISABLE_FLAG)
    n = 0
    while not stop.is_set():
        try:
            guard.poll(time.time())
        except Exception as e:  # a bug in one poll must not take the guard down
            guard.warn("exception", time.time(), "poll raised %s: %s", type(e).__name__, e)
        n += 1
        if args.polls and n >= args.polls:
            break
        stop.wait(POLL_S)
    if guard.held:
        guard.release_all(time.time(), "service stopping")
    log.info("STOP     exiting%s", ", still holding %s" % sorted(guard.held) if guard.held else "")


if __name__ == "__main__":
    main()
