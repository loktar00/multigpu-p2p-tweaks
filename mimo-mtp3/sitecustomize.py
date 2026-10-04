# Run all three of MiMo-V2's MTP modules in vLLM.
#
# MiMo-V2.6-Flash ships 3 MTP modules (num_nextn_predict_layers = 3), but vLLM's
# mimo_v2_mtp.py hard-codes _MIMO_V2_{PRO,FLASH}_NUM_MTP_LAYERS = 1 and builds one module, so
# num_speculative_tokens > 1 runs module 0 again for every draft position. This file:
#
#   1. sets both constants to 3 and builds min(num_speculative_tokens, 3) modules, so k = 2 or 3
#      runs vLLM's MultiModuleMTPSpeculator with module i for draft position i;
#   2. feeds every module the TARGET model's hidden state instead of the previous module's
#      output, which is how SGLang runs MiMo-V2. Done by returning (module output, input hidden),
#      the (logits_hidden, feedback_hidden) contract the speculator already supports. With vLLM's
#      default chaining the second module accepted 0.10; with the target hidden state, 0.81;
#   3. adds MiMoV2MTP.get_top_tokens so speculative_config use_local_argmax_reduction works.
#
# Env: MIMO_MTP_LAYERS (default 3), MIMO_MTP_FEEDBACK=target (default) or chain (vLLM's way).
#
# Use: put this directory first on the vLLM server's PYTHONPATH. Stack other overlays after it:
#   PYTHONPATH=/opt/vllm-mimo-mtp3:/opt/vllm-force-ca
# Nothing in the vLLM install is modified.
#
# MIT License, see LICENSE at the repository root.
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NMOD = int(os.environ.get("MIMO_MTP_LAYERS", "3") or 3)
_FEEDBACK = os.environ.get("MIMO_MTP_FEEDBACK", "target")


def _log(msg):
    print("[mimo-mtp3] " + msg, file=sys.stderr, flush=True)


def _run_next_sitecustomize():
    # Python only imports the first sitecustomize on sys.path; run the next one after this
    # directory so overlays stack.
    dirs = [os.path.abspath(d or os.curdir) for d in sys.path]
    if _HERE not in dirs:
        return
    for i in range(dirs.index(_HERE) + 1, len(dirs)):
        path = os.path.join(dirs[i], "sitecustomize.py")
        if os.path.isfile(path):
            try:
                spec = importlib.util.spec_from_file_location("_sitecustomize_%d" % i, path)
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
            except Exception as e:
                _log("%s raised %s: %s" % (path, type(e).__name__, e))
            return


def _patch_mtp(mod):
    need = ("MiMoV2MultiTokenPredictor", "MiMoV2MTP", "MiMoV2MTPLayer", "_MiMoV2MTPLayers",
            "VocabParallelEmbedding", "LogitsProcessor", "maybe_prefix")
    missing = [n for n in need if not hasattr(mod, n)]
    if missing:
        _log("WARNING: %s not found in mimo_v2_mtp; nothing patched" % ", ".join(missing))
        return

    import torch.nn as nn

    mod._MIMO_V2_PRO_NUM_MTP_LAYERS = _NMOD
    mod._MIMO_V2_FLASH_NUM_MTP_LAYERS = _NMOD

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        spec_cfg = vllm_config.speculative_config
        assert spec_cfg is not None
        n = max(1, min(int(spec_cfg.num_speculative_tokens), _NMOD))
        self.num_mtp_layers = n
        self.embed_tokens = mod.VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.mtp = mod._MiMoV2MTPLayers(config=config, num_mtp_layers=n,
                                        quant_config=vllm_config.quant_config,
                                        prefix=mod.maybe_prefix(prefix, "mtp.layers"))
        self.logits_processor = mod.LogitsProcessor(config.vocab_size)
        # A single module (k = 1) goes through the plain MTP speculator, which expects a tensor.
        for layer in self.mtp.layers.values():
            layer._mtp3_feedback = n > 1 and _FEEDBACK == "target"
        _log("MTP modules: %d, feedback: %s" % (n, _FEEDBACK))

    mod.MiMoV2MultiTokenPredictor.__init__ = __init__

    def get_top_tokens(self, hidden_states):
        return self.model.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    mod.MiMoV2MTP.get_top_tokens = get_top_tokens

    orig_forward = mod.MiMoV2MTPLayer.forward

    def forward(self, inputs_embeds, positions, previous_hidden_states):
        out = orig_forward(self, inputs_embeds, positions, previous_hidden_states)
        if getattr(self, "_mtp3_feedback", False):
            return out, previous_hidden_states
        return out

    mod.MiMoV2MTPLayer.forward = forward


_TARGETS = {"vllm.model_executor.models.mimo_v2_mtp": _patch_mtp}


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = _TARGETS.get(name)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(m):
            orig_exec(m)
            patch(m)

        spec.loader.exec_module = exec_module
        return spec


_run_next_sitecustomize()
sys.meta_path.insert(0, _PatchOnImport())
