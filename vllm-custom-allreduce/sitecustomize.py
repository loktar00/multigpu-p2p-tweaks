# Force vLLM's custom all-reduce on more than two PCIe-only GPUs.
#
# vLLM only enables its custom all-reduce for world size > 2 when every GPU pair is connected
# by NVLink (CudaPlatform.is_fully_connected). With a P2P-enabled driver, PCIe peers work fine,
# so this file makes is_fully_connected() return True. vLLM's own P2P test (_can_p2p) still runs
# afterwards and disables custom all-reduce if peer access does not actually work.
#
# Optional: CA_MAX_BYTES=<n> sends only all-reduces of at most n bytes through the custom
# kernel; larger ones (batched decode, prefill chunks) stay on NCCL. Unset or 0 = no extra cap.
#
# Use: put this directory first on the vLLM server's PYTHONPATH. Python imports it at startup
# in the API server and in every worker. Nothing in the vLLM install is modified.
#
# MIT License, see LICENSE at the repository root.
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CA_MAX = int(os.environ.get("CA_MAX_BYTES", "0") or 0)


def _log(msg):
    print("[force-ca] " + msg, file=sys.stderr, flush=True)


def _run_shadowed_sitecustomize():
    # Python only imports the first sitecustomize on sys.path. Run the one this file hides
    # (Debian/Ubuntu ship one) so its behaviour is kept.
    for d in sys.path:
        if not d or os.path.abspath(d) == _HERE:
            continue
        path = os.path.join(d, "sitecustomize.py")
        if os.path.isfile(path):
            try:
                spec = importlib.util.spec_from_file_location("_shadowed_sitecustomize", path)
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
            except Exception as e:
                _log("shadowed %s raised %s: %s" % (path, type(e).__name__, e))
            return


def _patch_platform(mod):
    patched = []
    for cname in ("CudaPlatformBase", "NvmlCudaPlatform", "NonNvmlCudaPlatform"):
        cls = getattr(mod, cname, None)
        if cls is not None and hasattr(cls, "is_fully_connected"):
            cls.is_fully_connected = classmethod(lambda c, ids: True)
            patched.append(cname)
    if patched:
        _log("is_fully_connected -> True (%s)" % ", ".join(patched))
    else:
        _log("WARNING: no CUDA platform class with is_fully_connected found; nothing patched")


def _patch_ca(mod):
    cls = getattr(mod, "CustomAllreduce", None)
    if cls is None or not hasattr(cls, "should_custom_ar"):
        _log("WARNING: CustomAllreduce.should_custom_ar not found; CA_MAX_BYTES ignored")
        return
    orig = cls.should_custom_ar

    def should_custom_ar(self, inp):
        if inp.numel() * inp.element_size() > _CA_MAX:
            return False
        return orig(self, inp)

    cls.should_custom_ar = should_custom_ar
    _log("custom all-reduce capped at %d bytes" % _CA_MAX)


_TARGETS = {"vllm.platforms.cuda": _patch_platform}
if _CA_MAX:
    _TARGETS["vllm.distributed.device_communicators.custom_all_reduce"] = _patch_ca


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = _TARGETS.get(name)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(mod):
            orig_exec(mod)
            patch(mod)

        spec.loader.exec_module = exec_module
        return spec


_run_shadowed_sitecustomize()
sys.meta_path.insert(0, _PatchOnImport())
