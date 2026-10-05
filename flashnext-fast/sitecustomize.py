# Decode speedups for Qwen3.8-Flash-Next (vLLM "qwen4_exp" model) on RTX 3090s.
#
# Each part has its own switch (all off by default):
#
#   FN_SKINNY=1      every unquantized bf16 Linear and the LM head of the target and the MTP
#                    drafter runs a tuned Triton kernel (fn86_skinny.py, configs.json) for
#                    M <= FN_SKINNY_MAXM tokens (default 16; a verify step is 5, a draft 1).
#                    Larger M and shapes not in the table stay on cuBLAS. vLLM's own low-latency
#                    GEMM for this model only runs on sm_103, so on sm_86 these went to cuBLAS.
#   FN_DRAFT_I8=1    (needs FN_SKINNY=1) the MTP drafter's own bf16 linears (>= 1 MB) and a
#                    private copy of its LM head get int8 weight-only copies (per-row scales),
#                    used where the int8 kernel measured faster (configs_i8.json). Draft only:
#                    the target verifies every draft token in bf16, so output can't change.
#   FN_TOPK_GATHER=K target logits are exchanged between TP ranks as each rank's top-K
#                    (value, id) pairs instead of an all-gather of the full vocab, only for
#                    batches where that is exact (see _patch_runner). 64 was measured.
#                    FN_TOPK_CHECK=1 also runs the full gather and counts mismatches.
#   FN_GDN_SPLIT=1   vllm._custom_ops.fused_gdn_decode_post_conv_mtp -> gdnsplit/ (same kernel,
#                    128 value rows split over 8 CTAs per head). Bit-identical. Built with
#                    torch.utils.cpp_extension on first use, into gdnsplit/build.
#   FN_ASYNC_H2D=1   the short-conv attention metadata builder's `<x>_cpu.to(device)` copies
#                    become async_tensor_h2d (pinned, non_blocking). Pageable copies
#                    synchronized the host with the GPU on every step.
#
# Always on: Qwen4ExpMTP.get_top_tokens, so speculative_config use_local_argmax_reduction
# works (drafts all-gather one (value, id) pair per rank instead of full-vocab logits).
#
# Use: put this directory first on the vLLM server's PYTHONPATH. Stack other overlays after it:
#   PYTHONPATH=/opt/vllm-flashnext-fast:/opt/vllm-force-ca
# Nothing in the vLLM install is modified.
#
# MIT License, see LICENSE at the repository root.
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SKINNY = os.environ.get("FN_SKINNY", "0") == "1"
_MAXM = int(os.environ.get("FN_SKINNY_MAXM", "16"))
_DRAFT_I8 = os.environ.get("FN_DRAFT_I8", "0") == "1"
_TOPK = int(os.environ.get("FN_TOPK_GATHER", "0") or 0)
_TOPK_CHECK = os.environ.get("FN_TOPK_CHECK", "0") == "1"
_GDN_SPLIT = os.environ.get("FN_GDN_SPLIT", "0") == "1"
_ASYNC_H2D = os.environ.get("FN_ASYNC_H2D", "0") == "1"


def _log(msg):
    print("[flashnext-fast] " + msg, file=sys.stderr, flush=True)


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


if _DRAFT_I8 and not _SKINNY:
    _log("WARNING: FN_DRAFT_I8 needs FN_SKINNY=1; int8 drafter off")
    _DRAFT_I8 = False


# --- FN_SKINNY / FN_DRAFT_I8 -------------------------------------------------------------------

def _patch_llg(mod):
    if not hasattr(mod, "enable_qwen4_exp_low_latency_gemm"):
        _log("WARNING: enable_qwen4_exp_low_latency_gemm not found; skinny GEMMs off")
        return
    import torch
    import torch.nn.functional as F
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (ParallelLMHead,
                                                                     UnquantizedEmbeddingMethod)
    from vllm.utils.torch_utils import direct_register_custom_op
    sys.path.insert(0, _HERE)
    import fn86_skinny
    with open(os.path.join(_HERE, "configs.json")) as f:
        for r in json.load(f):
            fn86_skinny.CONFIGS[(r["N"], r["K"], r["BM"])] = tuple(r["cfg"])
    _log("skinny table: %d (N,K,M-bucket) entries" % len(fn86_skinny.CONFIGS))

    def _op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= _MAXM and x.is_contiguous() and weight.is_contiguous():
            return fn86_skinny.skinny_gemm(x, weight)
        if x.shape[0] <= _MAXM:
            key = ("noncontig", weight.shape[0], x.shape[1], x.shape[0], x.is_contiguous(),
                   weight.is_contiguous())
            if key not in fn86_skinny._MISS:
                fn86_skinny._MISS.add(key)
                _log("skinny SKIP %s" % (key,))
        return F.linear(x, weight)

    def _fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x.new_empty((x.shape[0], weight.shape[0]))

    direct_register_custom_op(op_name="fn86_skinny_gemm", op_func=_op, fake_impl=_fake)

    def _op_i8(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # int8 only where it measured faster (configs_i8.json); otherwise the bf16 path on w
        x = x.contiguous()
        if (q.shape[0], x.shape[1], fn86_skinny.bucket(x.shape[0])) in fn86_skinny.CONFIGS_I8:
            return fn86_skinny.skinny_i8(x, q, s)
        return _op(x, w)

    def _fake_i8(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x.new_empty((x.shape[0], q.shape[0]))

    direct_register_custom_op(op_name="fn86_skinny_i8", op_func=_op_i8, fake_impl=_fake_i8)
    with open(os.path.join(_HERE, "configs_i8.json")) as f:
        for r in json.load(f):
            fn86_skinny.CONFIGS_I8[(r["N"], r["K"], r["BM"])] = tuple(r["cfg"])

    class _Apply:
        def apply(self, layer, x, bias=None):
            q8 = getattr(layer, "_i8_q", None)
            if q8 is not None and bias is None and x.dtype == torch.bfloat16 \
                    and x.numel() // x.shape[-1] <= _MAXM:
                shp = x.shape
                y = torch.ops.vllm.fn86_skinny_i8(x.reshape(-1, shp[-1]), q8, layer._i8_s, layer.weight)
                return y.reshape(*shp[:-1], y.shape[-1])
            if bias is None and x.dtype == torch.bfloat16 and layer.weight.dtype == torch.bfloat16:
                shp = x.shape
                y = torch.ops.vllm.fn86_skinny_gemm(x.reshape(-1, shp[-1]), layer.weight)
                return y.reshape(*shp[:-1], y.shape[-1])
            return super().apply(layer, x, bias)

    class SkinnyLinear(_Apply, UnquantizedLinearMethod):
        pass

    class SkinnyEmbed(_Apply, UnquantizedEmbeddingMethod):
        pass

    orig_enable = mod.enable_qwen4_exp_low_latency_gemm

    def enable(module, dtype):
        if dtype != torch.bfloat16:
            return orig_enable(module, dtype)
        n = 0
        for child in module.modules():
            w = getattr(child, "weight", None)
            if w is None or w.dim() != 2:
                continue
            if isinstance(child, LinearBase) and type(child.quant_method) is UnquantizedLinearMethod:
                child.quant_method = SkinnyLinear()
            elif isinstance(child, ParallelLMHead) and type(child.quant_method) is UnquantizedEmbeddingMethod:
                child.quant_method = SkinnyEmbed()
            else:
                continue
            n += 1
            if w.is_cuda:
                fn86_skinny.prealloc(w.device, w.shape[0], w.shape[1])
        _log("skinny GEMM on %d bf16 linears (M <= %d)" % (n, _MAXM))

    mod.enable_qwen4_exp_low_latency_gemm = enable
    _log("skinny GEMMs on (FN_SKINNY=1)")


def _build_draft_i8(self):
    """int8 weight-only copies of the drafter's own bf16 linears (>= 1 MB) and a private int8 copy
    of the LM head for get_top_tokens. Modules under self.model are the MTP layer and its mixers,
    never the target's."""
    import fn86_skinny
    import torch
    from vllm.model_executor.layers.linear import LinearBase
    n, nbytes = 0, 0
    for mod in self.model.modules():
        w = getattr(mod, "weight", None)
        if isinstance(mod, LinearBase) and w is not None and w.dim() == 2 and w.dtype == torch.bfloat16 \
                and w.numel() * 2 >= (1 << 20):
            mod._i8_q, mod._i8_s = fn86_skinny.quant_i8(w.data)
            n += 1
            nbytes += w.numel()
    self._lm_i8 = fn86_skinny.quant_i8(self.lm_head.weight.data)
    _log("draft int8: %d linears (%.0f MB int8) + lm_head %s"
         % (n, nbytes / 1e6, tuple(self.lm_head.weight.shape)))


# --- local-argmax drafts (always on) -----------------------------------------------------------

def _patch_mtp(mod):
    if not hasattr(mod, "Qwen4ExpMTP"):
        _log("WARNING: Qwen4ExpMTP not found; nothing patched")
        return
    import torch

    def get_top_tokens(self, hidden_states):
        if not hasattr(self, "_lm_i8"):
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
        from vllm.distributed import (get_tensor_model_parallel_world_size,
                                      tensor_model_parallel_all_gather)
        hs = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = torch.ops.vllm.fn86_skinny_i8(hs, self._lm_i8[0], self._lm_i8[1], self.lm_head.weight)
        si = self.lm_head.shard_indices
        if si.num_org_vocab_padding > 0:
            logits[..., -si.num_org_vocab_padding:] = -float("inf")
        vals, idx = logits.max(dim=-1)
        gid = idx + si.org_vocab_start_index
        tp = get_tensor_model_parallel_world_size()
        if tp == 1:
            return gid
        pair = torch.stack([vals.float(), gid.float()], dim=-1)
        g = tensor_model_parallel_all_gather(pair, dim=-1).view(hs.shape[0], tp, 2)
        best = g[:, :, 0].argmax(dim=-1, keepdim=True)
        return g[:, :, 1].gather(dim=-1, index=best).squeeze(-1).to(torch.int64)

    mod.Qwen4ExpMTP.get_top_tokens = get_top_tokens
    _log("Qwen4ExpMTP.get_top_tokens added (use_local_argmax_reduction)")

    if _DRAFT_I8:
        orig_load = mod.Qwen4ExpMTP.load_weights

        def load_weights(self, weights):
            out = orig_load(self, weights)
            _build_draft_i8(self)
            return out

        mod.Qwen4ExpMTP.load_weights = load_weights
        _log("int8 drafter on (FN_DRAFT_I8=1)")


# --- FN_TOPK_GATHER ----------------------------------------------------------------------------

_TOPK_ON = [False]
_CHK = [0, 0, 0]


def _patch_logits(mod):
    """When the model runner marks the batch eligible, the target's vocab-parallel logits are
    exchanged as each rank's top-K (value, id) pairs and rebuilt as a full-vocab tensor that is
    -inf outside the candidates. Exact for greedy, and for top_k <= K-16 sampling (+top_p,
    temperature, <= 16 min-tokens stop masks): the union of per-rank top-K holds the global top-K."""
    import torch
    from vllm.distributed import tensor_model_parallel_all_gather
    cls = mod.LogitsProcessor
    orig = cls._get_logits

    def _get_logits(self, hidden_states, lm_head, embedding_bias, skip_gather=False):
        if not _TOPK_ON[0] or skip_gather or lm_head.tp_size == 1 or self.logits_as_input:
            return orig(self, hidden_states, lm_head, embedding_bias, skip_gather)
        local = self._apply_head(lm_head, hidden_states, embedding_bias)
        si = lm_head.shard_indices
        if si.num_org_vocab_padding > 0:
            local[..., -si.num_org_vocab_padding:] = -float("inf")
        n = local.shape[0]
        v, i = local.topk(_TOPK, dim=-1)
        pair = torch.cat([v.float(), (i + si.org_vocab_start_index).float()], dim=-1)
        g = tensor_model_parallel_all_gather(pair, dim=-1).view(n, lm_head.tp_size, 2, _TOPK)
        vals = g[:, :, 0, :].reshape(n, -1)
        ids = g[:, :, 1, :].reshape(n, -1).to(torch.int64)
        full = torch.full((n, self.org_vocab_size), -float("inf"), dtype=local.dtype, device=local.device)
        full.scatter_(1, ids, vals.to(local.dtype))
        if _TOPK_CHECK:
            ref = orig(self, hidden_states, lm_head, embedding_bias, skip_gather)
            kk = _TOPK - 16
            a = torch.equal(ref.argmax(-1), full.argmax(-1))
            b = torch.equal(ref.topk(kk, dim=-1).values, full.topk(kk, dim=-1).values)
            _CHK[0] += 1
            _CHK[1] += int(not a)
            _CHK[2] += int(not b)
            if _CHK[0] % 500 == 0:
                _log("topk check: %d calls, argmax mismatches %d, top-%d value mismatches %d"
                     % (_CHK[0], _CHK[1], kk, _CHK[2]))
        return full

    cls._get_logits = _get_logits
    _log("top-%d logits exchange on (FN_TOPK_GATHER)" % _TOPK)


_ELIGIBLE_ERR = [False]


def _patch_runner(mod):
    import numpy as np
    cls = getattr(mod, "GPUModelRunner", None)
    if cls is None or not hasattr(cls, "sample"):
        _log("WARNING: GPUModelRunner.sample not found; top-k exchange never used")
        return
    orig_sample = cls.sample

    def _eligible(self, input_batch, grammar_output):
        # Anything that needs logits outside the top-K falls back to the full gather.
        if grammar_output is not None or getattr(self, "batch_sharder", None) is not None:
            return False
        sm = self.sampler
        if sm is None:
            return False
        idx = input_batch.idx_mapping_np
        if sm.get_logprobs_dims(idx) is not None:
            return False
        lb = sm.logit_bias_state
        if np.any(lb.num_allowed_token_ids.np[idx] != 0) or np.any(lb.num_logit_bias.np[idx] != 0) \
                or np.any(lb.num_stop_token_ids.np[idx] > 16):
            return False
        if np.any(sm.penalties_state.use_penalty[idx]) or np.any(sm.bad_words_state.num_bad_words.np[idx] != 0):
            return False
        tb = sm.thinking_budget_state
        if tb.enabled and np.any(tb.use_thinking_budget[idx]):
            return False
        st = sm.sampling_states
        if np.any(st.min_p.np[idx] != 0.0):
            return False
        temp = st.temperature.np[idx]
        topk = st.top_k.np[idx]
        ok = (temp == 0.0) | ((topk >= 1) & (topk <= _TOPK - 16))
        return bool(np.all(ok))

    def eligible(self, input_batch, grammar_output):
        try:
            return _eligible(self, input_batch, grammar_output)
        except AttributeError as e:
            # Sampler state laid out differently in this vLLM: always use the full gather.
            if not _ELIGIBLE_ERR[0]:
                _ELIGIBLE_ERR[0] = True
                _log("WARNING: sampler state not recognised (%s); top-k exchange off" % e)
            return False

    def sample(self, hidden_states, input_batch, grammar_output):
        _TOPK_ON[0] = eligible(self, input_batch, grammar_output)
        try:
            return orig_sample(self, hidden_states, input_batch, grammar_output)
        finally:
            _TOPK_ON[0] = False

    cls.sample = sample


# --- FN_GDN_SPLIT ------------------------------------------------------------------------------

def _patch_custom_ops(mod):
    if not hasattr(mod, "fused_gdn_decode_post_conv_mtp"):
        _log("WARNING: fused_gdn_decode_post_conv_mtp not found; GDN split off")
        return
    sys.path.insert(0, os.path.join(_HERE, "gdnsplit"))
    import gdn_split_ext
    mod.fused_gdn_decode_post_conv_mtp = gdn_split_ext.fused_gdn_decode_post_conv_mtp
    _log("GDN MTP decode -> V-split kernel (FN_GDN_SPLIT=1)")


# --- FN_ASYNC_H2D ------------------------------------------------------------------------------

_H2D_RE = re.compile(r"\b(\w+_cpu)\.to\((\w+)\.device\)")


class _SourceLoader(importlib.machinery.SourceFileLoader):
    """Loads a module from its own file with edited source text."""

    def __init__(self, fullname, path, source):
        super().__init__(fullname, path)
        self._source = source

    def get_code(self, fullname):
        return compile(self._source, self.path, "exec", dont_inherit=True)


def _async_h2d_spec(name, path):
    spec = importlib.machinery.PathFinder.find_spec(name, path)
    if spec is None or not spec.origin:
        return None
    with open(spec.origin) as f:
        src = f.read()
    new, n = _H2D_RE.subn(r"async_tensor_h2d(\1, \2.device)", src)
    if n == 0 or "import async_tensor_h2d" not in src:
        _log("WARNING: no pageable copies found in %s; FN_ASYNC_H2D does nothing" % spec.origin)
        return None
    _log("%s: %d host-to-device copies made async" % (name, n))
    return importlib.util.spec_from_file_location(name, spec.origin,
                                                  loader=_SourceLoader(name, spec.origin, new))


# --- import hook -------------------------------------------------------------------------------

_TARGETS = {"vllm.models.qwen4_exp.nvidia.mtp": _patch_mtp}
if _SKINNY:
    _TARGETS["vllm.models.qwen4_exp.nvidia.low_latency_gemm"] = _patch_llg
if _TOPK:
    _TARGETS["vllm.model_executor.layers.logits_processor"] = _patch_logits
    _TARGETS["vllm.v1.worker.gpu.model_runner"] = _patch_runner
if _GDN_SPLIT:
    _TARGETS["vllm._custom_ops"] = _patch_custom_ops

_SHORT_CONV = "vllm.v1.attention.backends.short_conv_attn"


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == _SHORT_CONV and _ASYNC_H2D:
            return _async_h2d_spec(name, path)
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


_run_next_sitecustomize()
sys.meta_path.insert(0, _PatchOnImport())
