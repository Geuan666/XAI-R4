#!/usr/bin/env python3
"""
Tool-call circuit locating pipeline for Qwen3-1.7B.

This script implements the TODO plan end-to-end:
- E0 baseline metrics
- E1 causal tracing heatmaps (state/mlp/attn)
- E2 window restoration curves (mlp vs attn)
- E3 modified graph interventions (sever MLP / sever Attn)
- E4 head-level CT + AP
- E5 L7H14 probe
- E6 final circuit nodes/edges + figure
- Appendix figures: lineplot with CI, case examples

Outputs are written to reports/ and figs/ with mandatory filenames.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging


# ---------- Global style (paper-like, consistent across all figs) ----------
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 180,
        "savefig.dpi": 220,
    }
)


@dataclass
class SampleInfo:
    q: int
    clean_text: str
    corr_text: str
    meta: dict
    seq_len: int
    trigger_idx: int
    prompt_end_idx: int
    assistant_close_idx: int
    tool_instr_idx: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_mean_ci(
    arr: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap mean and CI along axis 0."""
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = arr[idx].mean(axis=1)
    mean = arr.mean(axis=0)
    lo = np.quantile(boot, alpha / 2, axis=0)
    hi = np.quantile(boot, 1 - alpha / 2, axis=0)
    return mean, lo, hi


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def softmax_prob_tool(logits: torch.Tensor, tool_id: int) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    return probs[..., tool_id]


def get_span_idx_in_segment(meta: dict, key: str, segment_name: str) -> Optional[int]:
    segs = meta["clean"]["segments"]
    seg = next((s for s in segs if s["name"] == segment_name), None)
    if seg is None:
        return None
    lo, hi = seg["token_start"], seg["token_end_exclusive"]
    spans = meta["clean"]["key_spans"].get(key, [])
    for sp in spans:
        t = sp["token_start"]
        if lo <= t < hi:
            return t
    spans_corr = meta["corrupted"]["key_spans"].get("State", [])
    for sp in spans_corr:
        t = sp["token_start"]
        if lo <= t < hi:
            return t
    return None


def get_segment_start(meta: dict, name: str) -> Optional[int]:
    for s in meta["clean"]["segments"]:
        if s["name"] == name:
            return s["token_start"]
    return None


def get_segment_end_minus1(meta: dict, name: str) -> Optional[int]:
    for s in meta["clean"]["segments"]:
        if s["name"] == name:
            return s["token_end_exclusive"] - 1
    return None


def build_sample_info(root: Path, q: int) -> SampleInfo:
    meta = json.loads((root / f"pair/meta-q{q}.json").read_text())
    clean_text = (root / f"pair/prompt-clean-q{q}.txt").read_text()
    corr_text = (root / f"pair/prompt-corrupted-q{q}.txt").read_text()
    seq_len = int(meta["clean"]["num_tokens"])

    trig = get_span_idx_in_segment(meta, "Write", "user_instruction")
    if trig is None:
        trig = get_segment_start(meta, "user_instruction")
    if trig is None:
        trig = 0

    assistant_close = get_segment_start(meta, "assistant_think_close")
    if assistant_close is None:
        assistant_close = seq_len - 2

    tool_instr = get_segment_start(meta, "tool_call_instruction")
    if tool_instr is None:
        tool_instr = get_segment_start(meta, "tool_call_example")
    if tool_instr is None:
        tool_instr = max(0, int(0.35 * (seq_len - 1)))

    return SampleInfo(
        q=q,
        clean_text=clean_text,
        corr_text=corr_text,
        meta=meta,
        seq_len=seq_len,
        trigger_idx=int(trig),
        prompt_end_idx=seq_len - 1,
        assistant_close_idx=int(assistant_close),
        tool_instr_idx=int(tool_instr),
    )


class QwenCircuitTracer:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except Exception:
            pass
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.n_layers = int(self.model.config.num_hidden_layers)
        self.n_heads = int(self.model.config.num_attention_heads)
        self.hidden = int(self.model.config.hidden_size)
        self.head_dim = self.hidden // self.n_heads

        ids = self.tokenizer.encode("<tool_call>", add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError("<tool_call> must be a single token for this pipeline.")
        self.tool_id = ids[0]

    def encode(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"].to(self.device)

    @torch.no_grad()
    def p_tool(self, input_ids: torch.Tensor) -> np.ndarray:
        out = self.model(input_ids=input_ids)
        p = softmax_prob_tool(out.logits[:, -1, :], self.tool_id)
        return p.detach().float().cpu().numpy()

    @torch.no_grad()
    def capture_component(self, input_ids: torch.Tensor, component: str) -> Dict[int, torch.Tensor]:
        """Capture full-sequence activation for each layer (batch size must be 1)."""
        caches: Dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(layer: int):
            def _hook(_m, _inp, out):
                if component == "attn":
                    t = out[0] if isinstance(out, tuple) else out
                else:
                    t = out
                caches[layer] = t[0].detach().clone()

            return _hook

        for l in range(self.n_layers):
            if component == "state":
                mod = self.model.model.layers[l]
            elif component == "mlp":
                mod = self.model.model.layers[l].mlp
            elif component == "attn":
                mod = self.model.model.layers[l].self_attn
            else:
                raise ValueError(component)
            hooks.append(mod.register_forward_hook(make_hook(l)))

        _ = self.model(input_ids=input_ids)
        for h in hooks:
            h.remove()
        return caches

    @torch.no_grad()
    def capture_o_proj_input(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Capture self_attn.o_proj input for each layer (shape: [seq, hidden])."""
        caches: Dict[int, torch.Tensor] = {}
        hooks = []

        def make_pre(layer: int):
            def _pre(_m, inp):
                x = inp[0]
                caches[layer] = x[0].detach().clone()

            return _pre

        for l in range(self.n_layers):
            mod = self.model.model.layers[l].self_attn.o_proj
            hooks.append(mod.register_forward_pre_hook(make_pre(l)))

        _ = self.model(input_ids=input_ids)
        for h in hooks:
            h.remove()
        return caches

    @torch.no_grad()
    def patch_token_layer_grid(
        self,
        base_ids: torch.Tensor,
        source_cache: Dict[int, torch.Tensor],
        component: str,
        token_positions: Sequence[int],
        p_base: float,
        batch_size: int,
    ) -> np.ndarray:
        """Return IE grid [n_tokens, n_layers] for single sample."""
        n_tok = len(token_positions)
        grid = np.zeros((n_tok, self.n_layers), dtype=np.float32)

        for l in range(self.n_layers):
            for s in range(0, n_tok, batch_size):
                chunk_pos = list(token_positions[s : s + batch_size])
                bsz = len(chunk_pos)
                inp = base_ids.repeat(bsz, 1)
                src = source_cache[l]

                if component == "state":
                    mod = self.model.model.layers[l]
                elif component == "mlp":
                    mod = self.model.model.layers[l].mlp
                elif component == "attn":
                    mod = self.model.model.layers[l].self_attn
                else:
                    raise ValueError(component)

                def _patch(_m, _inp, out):
                    if component == "attn":
                        t0 = out[0].clone()
                        for bi, pos in enumerate(chunk_pos):
                            t0[bi, pos, :] = src[pos, :].to(t0.dtype)
                        return (t0, out[1]) if isinstance(out, tuple) else t0
                    t = out.clone()
                    for bi, pos in enumerate(chunk_pos):
                        t[bi, pos, :] = src[pos, :].to(t.dtype)
                    return t

                h = mod.register_forward_hook(_patch)
                probs = self.p_tool(inp)
                h.remove()
                grid[s : s + bsz, l] = probs - p_base

        return grid

    @torch.no_grad()
    def patch_token_layer_grid_with_sever(
        self,
        base_ids: torch.Tensor,
        source_state_cache: Dict[int, torch.Tensor],
        token_positions: Sequence[int],
        p_base: float,
        sever_component: str,
        sever_cache: Dict[int, torch.Tensor],
        batch_size: int,
    ) -> np.ndarray:
        """
        Restore state at (token, layer), while severing downstream component path.
        sever_component in {"mlp", "attn"}
        """
        n_tok = len(token_positions)
        grid = np.zeros((n_tok, self.n_layers), dtype=np.float32)

        for l in range(self.n_layers):
            for s in range(0, n_tok, batch_size):
                chunk_pos = list(token_positions[s : s + batch_size])
                bsz = len(chunk_pos)
                inp = base_ids.repeat(bsz, 1)
                hooks = []

                src = source_state_cache[l]

                def _patch_state(_m, _inp, out):
                    t = out.clone()
                    for bi, pos in enumerate(chunk_pos):
                        t[bi, pos, :] = src[pos, :].to(t.dtype)
                    return t

                hooks.append(self.model.model.layers[l].register_forward_hook(_patch_state))

                for ll in range(l + 1, self.n_layers):
                    target = sever_cache[ll]
                    if sever_component == "mlp":
                        mod = self.model.model.layers[ll].mlp

                        def _mk_mlp_hook(tgt: torch.Tensor):
                            def _h(_m, _inp, out):
                                t = out.clone()
                                for bi, pos in enumerate(chunk_pos):
                                    t[bi, pos, :] = tgt[pos, :].to(t.dtype)
                                return t

                            return _h

                        hooks.append(mod.register_forward_hook(_mk_mlp_hook(target)))
                    elif sever_component == "attn":
                        mod = self.model.model.layers[ll].self_attn

                        def _mk_attn_hook(tgt: torch.Tensor):
                            def _h(_m, _inp, out):
                                if isinstance(out, tuple):
                                    t0 = out[0].clone()
                                    for bi, pos in enumerate(chunk_pos):
                                        t0[bi, pos, :] = tgt[pos, :].to(t0.dtype)
                                    return (t0, out[1])
                                t = out.clone()
                                for bi, pos in enumerate(chunk_pos):
                                    t[bi, pos, :] = tgt[pos, :].to(t.dtype)
                                return t

                            return _h

                        hooks.append(mod.register_forward_hook(_mk_attn_hook(target)))
                    else:
                        raise ValueError(sever_component)

                probs = self.p_tool(inp)
                for h in hooks:
                    h.remove()
                grid[s : s + bsz, l] = probs - p_base

        return grid

    @torch.no_grad()
    def patch_window_curve(
        self,
        base_ids: torch.Tensor,
        source_cache: Dict[int, torch.Tensor],
        component: str,
        token_idx: int,
        p_base: float,
        window: int,
    ) -> np.ndarray:
        """AIE curve over layer centers for one token using window restoration."""
        curve = np.zeros(self.n_layers, dtype=np.float32)
        half = window // 2

        for center in range(self.n_layers):
            lo = max(0, center - half)
            hi = min(self.n_layers, lo + window)
            lo = max(0, hi - window)

            hooks = []
            for l in range(lo, hi):
                src = source_cache[l]
                if component == "mlp":
                    mod = self.model.model.layers[l].mlp

                    def _mk_mlp_hook(src_layer: torch.Tensor):
                        def _h(_m, _inp, out):
                            t = out.clone()
                            t[0, token_idx, :] = src_layer[token_idx, :].to(t.dtype)
                            return t

                        return _h

                    hooks.append(mod.register_forward_hook(_mk_mlp_hook(src)))
                elif component == "attn":
                    mod = self.model.model.layers[l].self_attn

                    def _mk_attn_hook(src_layer: torch.Tensor):
                        def _h(_m, _inp, out):
                            if isinstance(out, tuple):
                                t0 = out[0].clone()
                                t0[0, token_idx, :] = src_layer[token_idx, :].to(t0.dtype)
                                return (t0, out[1])
                            t = out.clone()
                            t[0, token_idx, :] = src_layer[token_idx, :].to(t.dtype)
                            return t

                        return _h

                    hooks.append(mod.register_forward_hook(_mk_attn_hook(src)))
                else:
                    raise ValueError(component)

            probs = self.p_tool(base_ids)
            for h in hooks:
                h.remove()
            curve[center] = float(probs[0] - p_base)

        return curve

    @torch.no_grad()
    def patch_head_heatmap(
        self,
        base_ids: torch.Tensor,
        source_o_proj_in: Dict[int, torch.Tensor],
        token_idx: int,
        p_base: float,
        batch_size_heads: int = 16,
    ) -> np.ndarray:
        """Return IE heatmap [n_layers, n_heads] by patching o_proj input per head."""
        hm = np.zeros((self.n_layers, self.n_heads), dtype=np.float32)

        for l in range(self.n_layers):
            src = source_o_proj_in[l]
            heads = list(range(self.n_heads))
            for s in range(0, self.n_heads, batch_size_heads):
                chunk = heads[s : s + batch_size_heads]
                bsz = len(chunk)
                inp = base_ids.repeat(bsz, 1)

                def _pre(_m, inps):
                    x = inps[0].clone()
                    for bi, h in enumerate(chunk):
                        a = h * self.head_dim
                        b = (h + 1) * self.head_dim
                        x[bi, token_idx, a:b] = src[token_idx, a:b].to(x.dtype)
                    return (x,)

                hook = self.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(_pre)
                probs = self.p_tool(inp)
                hook.remove()
                hm[l, s : s + bsz] = probs - p_base

        return hm

    @torch.no_grad()
    def joint_head_mlp_effect_from_vectors(
        self,
        corr_ids: torch.Tensor,
        clean_o_trigger: np.ndarray,
        clean_mlp_prompt_end: np.ndarray,
        head_token_idx: int,
        mlp_token_idx: int,
        head_layer: int,
        head_idx: int,
        mlp_layer: int,
        p_corr: float,
    ) -> Tuple[float, float, float]:
        """Return (IE_head, IE_mlp, IE_joint) on one sample using cached trigger vectors."""

        def run_case(use_head: bool, use_mlp: bool) -> float:
            hooks = []
            if use_head:
                a = head_idx * self.head_dim
                b = (head_idx + 1) * self.head_dim
                src_head_vec = torch.as_tensor(
                    clean_o_trigger[head_layer, a:b], device=self.device
                )

                def _pre(_m, inps):
                    x = inps[0].clone()
                    x[0, head_token_idx, a:b] = src_head_vec.to(x.dtype)
                    return (x,)

                hooks.append(
                    self.model.model.layers[head_layer].self_attn.o_proj.register_forward_pre_hook(_pre)
                )

            if use_mlp:
                src_mlp_vec = torch.as_tensor(clean_mlp_prompt_end[mlp_layer], device=self.device)

                def _h(_m, _inp, out):
                    t = out.clone()
                    t[0, mlp_token_idx, :] = src_mlp_vec.to(t.dtype)
                    return t

                hooks.append(self.model.model.layers[mlp_layer].mlp.register_forward_hook(_h))

            p = float(self.p_tool(corr_ids)[0])
            for h in hooks:
                h.remove()
            return p

        p_head = run_case(True, False)
        p_mlp = run_case(False, True)
        p_joint = run_case(True, True)
        return p_head - p_corr, p_mlp - p_corr, p_joint - p_corr


class Pipeline:
    def __init__(
        self,
        root: Path,
        model_path: str,
        token_bins: int,
        batch_size: int,
        window_size: int,
        max_samples: Optional[int],
        seed: int,
        force: bool,
        early_radius: int,
        late_radius: int,
    ) -> None:
        self.root = root
        self.model_path = model_path
        self.legacy_token_bins = token_bins
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_samples = max_samples
        self.seed = seed
        self.force = force
        self.early_radius = early_radius
        self.late_radius = late_radius

        self.early_offsets = list(range(-self.early_radius, self.early_radius + 1))
        self.late_offsets = list(range(-self.late_radius, 1))
        self.token_row_labels = [f"T{d:+d}" for d in self.early_offsets] + [
            f"E{d:+d}" for d in self.late_offsets
        ]
        self.token_bins = len(self.token_row_labels)
        self.trigger_row = self.early_offsets.index(0)
        self.end_row = len(self.early_offsets) + self.late_offsets.index(0)

        self.reports = self.root / "reports"
        self.figs = self.root / "figs"
        self.cache = self.reports / "cache"
        ensure_dirs(self.reports, self.figs, self.cache)

        self.df = pd.read_csv(self.root / "pair/first_token_len_eval_qwen3_1.7b.csv")
        self.df["q"] = self.df["q"].astype(int)

        self.S_all = self.df["q"].tolist()
        self.S_strict = self.df[
            (self.df["clean_top1"] == "<tool_call>") & (self.df["corr_top1"] != "<tool_call>")
        ]["q"].tolist()
        self.S_ambiguous = self.df[
            (self.df["clean_top1"] == "<tool_call>") & (self.df["corr_top1"] == "<tool_call>")
        ]["q"].tolist()
        self.S_fail = self.df[
            (self.df["clean_top1"] != "<tool_call>") & (self.df["corr_top1"] != "<tool_call>")
        ]["q"].tolist()

        if self.max_samples is not None:
            self.S_strict = self.S_strict[: self.max_samples]

        self.tracer = QwenCircuitTracer(model_path=model_path, device="cuda")
        self.layers = self.tracer.n_layers
        self.heads = self.tracer.n_heads

    def _anchor_token_positions(self, info: SampleInfo, seq_len: int) -> np.ndarray:
        pos = []
        for d in self.early_offsets:
            pos.append(info.trigger_idx + d)
        for d in self.late_offsets:
            pos.append(info.prompt_end_idx + d)
        arr = np.array(pos, dtype=np.int32)
        arr = np.clip(arr, 0, seq_len - 1)
        return arr

    def _cache_file(self, q: int) -> Path:
        return self.cache / f"q{q}.npz"

    def _save_sample_cache(self, q: int, data: Dict[str, np.ndarray]) -> None:
        np.savez_compressed(self._cache_file(q), **data)

    def _load_sample_cache(self, q: int) -> Optional[Dict[str, np.ndarray]]:
        p = self._cache_file(q)
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}

    def run_sample(self, info: SampleInfo) -> Dict[str, np.ndarray]:
        if (not self.force) and self._cache_file(info.q).exists():
            cached = self._load_sample_cache(info.q)
            assert cached is not None
            return cached

        clean_ids = self.tracer.encode(info.clean_text)
        corr_ids = self.tracer.encode(info.corr_text)

        if clean_ids.shape != corr_ids.shape:
            raise RuntimeError(f"Token length mismatch for q{info.q}")
        seq_len = int(clean_ids.shape[1])

        token_pos = self._anchor_token_positions(info, seq_len)

        p_clean = float(self.tracer.p_tool(clean_ids)[0])
        p_corr = float(self.tracer.p_tool(corr_ids)[0])
        te = p_clean - p_corr
        te_denom = max(te, 1e-6)

        clean_state = self.tracer.capture_component(clean_ids, "state")
        clean_mlp = self.tracer.capture_component(clean_ids, "mlp")
        clean_attn = self.tracer.capture_component(clean_ids, "attn")
        corr_mlp = self.tracer.capture_component(corr_ids, "mlp")
        corr_attn = self.tracer.capture_component(corr_ids, "attn")

        ie_state = self.tracer.patch_token_layer_grid(
            base_ids=corr_ids,
            source_cache=clean_state,
            component="state",
            token_positions=token_pos,
            p_base=p_corr,
            batch_size=self.batch_size,
        )
        ie_mlp = self.tracer.patch_token_layer_grid(
            base_ids=corr_ids,
            source_cache=clean_mlp,
            component="mlp",
            token_positions=token_pos,
            p_base=p_corr,
            batch_size=self.batch_size,
        )
        ie_attn = self.tracer.patch_token_layer_grid(
            base_ids=corr_ids,
            source_cache=clean_attn,
            component="attn",
            token_positions=token_pos,
            p_base=p_corr,
            batch_size=self.batch_size,
        )

        # E2: window restoration at key tokens
        key_tokens = np.array([
            info.trigger_idx,
            info.assistant_close_idx,
            info.prompt_end_idx,
            info.tool_instr_idx,
        ], dtype=np.int32)
        e2_mlp = np.zeros((len(key_tokens), self.layers), dtype=np.float32)
        e2_attn = np.zeros((len(key_tokens), self.layers), dtype=np.float32)
        for i, t in enumerate(key_tokens.tolist()):
            e2_mlp[i] = self.tracer.patch_window_curve(
                base_ids=corr_ids,
                source_cache=clean_mlp,
                component="mlp",
                token_idx=t,
                p_base=p_corr,
                window=self.window_size,
            )
            e2_attn[i] = self.tracer.patch_window_curve(
                base_ids=corr_ids,
                source_cache=clean_attn,
                component="attn",
                token_idx=t,
                p_base=p_corr,
                window=self.window_size,
            )

        # E3: modified graph intervention (state restoration + downstream sever)
        e3_sever_mlp = self.tracer.patch_token_layer_grid_with_sever(
            base_ids=corr_ids,
            source_state_cache=clean_state,
            token_positions=[info.trigger_idx],
            p_base=p_corr,
            sever_component="mlp",
            sever_cache=corr_mlp,
            batch_size=1,
        )[0]
        e3_sever_attn = self.tracer.patch_token_layer_grid_with_sever(
            base_ids=corr_ids,
            source_state_cache=clean_state,
            token_positions=[info.trigger_idx],
            p_base=p_corr,
            sever_component="attn",
            sever_cache=corr_attn,
            batch_size=1,
        )[0]
        # baseline state curve at trigger for comparison
        trig_grid_idx = int(self.trigger_row)
        e3_base_curve = ie_state[trig_grid_idx]

        # E4: head-level CT/AP at trigger token
        clean_o_in = self.tracer.capture_o_proj_input(clean_ids)
        corr_o_in = self.tracer.capture_o_proj_input(corr_ids)
        ct_head = self.tracer.patch_head_heatmap(
            base_ids=corr_ids,
            source_o_proj_in=clean_o_in,
            token_idx=info.trigger_idx,
            p_base=p_corr,
            batch_size_heads=min(8, self.heads),
        )

        ap_drop = self.tracer.patch_head_heatmap(
            base_ids=clean_ids,
            source_o_proj_in=corr_o_in,
            token_idx=info.trigger_idx,
            p_base=p_clean,
            batch_size_heads=min(8, self.heads),
        )
        ap_head = -ap_drop  # convert drop to positive contribution score

        clean_o_trigger = np.stack(
            [
                clean_o_in[l][info.trigger_idx].detach().float().cpu().numpy()
                for l in range(self.layers)
            ],
            axis=0,
        ).astype(np.float16)
        clean_mlp_trigger = np.stack(
            [
                clean_mlp[l][info.trigger_idx].detach().float().cpu().numpy()
                for l in range(self.layers)
            ],
            axis=0,
        ).astype(np.float16)
        clean_mlp_prompt_end = np.stack(
            [
                clean_mlp[l][info.prompt_end_idx].detach().float().cpu().numpy()
                for l in range(self.layers)
            ],
            axis=0,
        ).astype(np.float16)

        # E5 probe: L7H14 token sweep (causal tracing style)
        probe_layer = 7
        probe_head = 14
        probe_tok_pos = token_pos
        probe_vals = np.zeros(len(probe_tok_pos), dtype=np.float32)
        if probe_layer < self.layers and probe_head < self.heads:
            src = clean_o_in[probe_layer]
            a = probe_head * self.tracer.head_dim
            b = (probe_head + 1) * self.tracer.head_dim
            for i, t in enumerate(probe_tok_pos.tolist()):
                def _pre(_m, inps):
                    x = inps[0].clone()
                    x[0, t, a:b] = src[t, a:b].to(x.dtype)
                    return (x,)

                h = self.tracer.model.model.layers[probe_layer].self_attn.o_proj.register_forward_pre_hook(_pre)
                p = float(self.tracer.p_tool(corr_ids)[0])
                h.remove()
                probe_vals[i] = p - p_corr

        out = {
            "q": np.array([info.q], dtype=np.int32),
            "seq_len": np.array([seq_len], dtype=np.int32),
            "token_pos": token_pos.astype(np.int32),
            "key_tokens": key_tokens.astype(np.int32),
            "p_clean": np.array([p_clean], dtype=np.float32),
            "p_corr": np.array([p_corr], dtype=np.float32),
            "trigger_idx": np.array([info.trigger_idx], dtype=np.int32),
            "prompt_end_idx": np.array([info.prompt_end_idx], dtype=np.int32),
            "TE": np.array([te], dtype=np.float32),
            "TE_denom": np.array([te_denom], dtype=np.float32),
            "corr_input_ids": corr_ids[0].detach().cpu().numpy().astype(np.int32),
            "ie_state": ie_state.astype(np.float32),
            "ie_mlp": ie_mlp.astype(np.float32),
            "ie_attn": ie_attn.astype(np.float32),
            "e2_mlp": e2_mlp.astype(np.float32),
            "e2_attn": e2_attn.astype(np.float32),
            "e3_base_curve": e3_base_curve.astype(np.float32),
            "e3_sever_mlp": e3_sever_mlp.astype(np.float32),
            "e3_sever_attn": e3_sever_attn.astype(np.float32),
            "ct_head": ct_head.astype(np.float32),
            "ap_head": ap_head.astype(np.float32),
            "clean_o_trigger": clean_o_trigger,
            "clean_mlp_trigger": clean_mlp_trigger,
            "clean_mlp_prompt_end": clean_mlp_prompt_end,
            "probe_token_pos": probe_tok_pos.astype(np.int32),
            "probe_vals": probe_vals.astype(np.float32),
        }

        self._save_sample_cache(info.q, out)
        return out

    def run_all_samples(self) -> List[Dict[str, np.ndarray]]:
        all_q = self.S_strict
        outputs = []
        for idx, q in enumerate(all_q, start=1):
            info = build_sample_info(self.root, q)
            out = self.run_sample(info)
            outputs.append(out)
            if idx % 5 == 0 or idx == len(all_q):
                print(f"[progress] {idx}/{len(all_q)} strict samples processed")
        return outputs

    def _stack(self, arrs: List[np.ndarray]) -> np.ndarray:
        return np.stack(arrs, axis=0)

    def aggregate_and_report(self, outs: List[Dict[str, np.ndarray]]) -> None:
        q_list = [int(o["q"][0]) for o in outs]
        p_clean = np.array([float(o["p_clean"][0]) for o in outs], dtype=np.float32)
        p_corr = np.array([float(o["p_corr"][0]) for o in outs], dtype=np.float32)
        te = np.array([float(o["TE"][0]) for o in outs], dtype=np.float32)
        te_denom = np.array([float(o["TE_denom"][0]) for o in outs], dtype=np.float32)

        ie_state = self._stack([o["ie_state"] for o in outs])
        ie_mlp = self._stack([o["ie_mlp"] for o in outs])
        ie_attn = self._stack([o["ie_attn"] for o in outs])

        n_ie_state = ie_state / te_denom[:, None, None]
        n_ie_mlp = ie_mlp / te_denom[:, None, None]
        n_ie_attn = ie_attn / te_denom[:, None, None]

        # E0 baseline + set-level stats
        e0_rows = []
        for set_name, qs in [
            ("S_all", self.S_all),
            ("S_strict", self.S_strict),
            ("S_ambiguous", self.S_ambiguous),
            ("S_fail", self.S_fail),
        ]:
            sub = self.df[self.df["q"].isin(qs)].copy()
            clean_hit = float((sub["clean_top1"] == "<tool_call>").mean())
            corr_hit = float((sub["corr_top1"] == "<tool_call>").mean())

            if set_name == "S_strict":
                ate = float(te.mean())
                _, lo, hi = bootstrap_mean_ci(te[:, None], n_boot=1000, seed=self.seed)
                ate_lo = float(lo[0])
                ate_hi = float(hi[0])
            else:
                ate = np.nan
                ate_lo = np.nan
                ate_hi = np.nan

            e0_rows.append(
                {
                    "set": set_name,
                    "n": int(len(sub)),
                    "clean_top1_hit_rate": clean_hit,
                    "corr_top1_hit_rate": corr_hit,
                    "ATE": ate,
                    "ATE_ci_low": ate_lo,
                    "ATE_ci_high": ate_hi,
                }
            )

        e0_df = pd.DataFrame(e0_rows)
        e0_df.to_csv(self.reports / "E0_baseline_metrics.csv", index=False)

        te_df = pd.DataFrame({"q": q_list, "P_clean": p_clean, "P_corr": p_corr, "TE": te})
        te_df.to_csv(self.reports / "TE_distribution_S_strict.csv", index=False)

        # Heatmap aggregates
        aie_state = ie_state.mean(axis=0)
        aie_mlp = ie_mlp.mean(axis=0)
        aie_attn = ie_attn.mean(axis=0)

        anie_state = n_ie_state.mean(axis=0)
        anie_mlp = n_ie_mlp.mean(axis=0)
        anie_attn = n_ie_attn.mean(axis=0)

        # Save npz for reproducibility
        np.savez_compressed(
            self.reports / "aggregate_strict_metrics.npz",
            q=np.array(q_list, dtype=np.int32),
            p_clean=p_clean,
            p_corr=p_corr,
            TE=te,
            AIE_state=aie_state,
            AIE_mlp=aie_mlp,
            AIE_attn=aie_attn,
            nIE_state=anie_state,
            nIE_mlp=anie_mlp,
            nIE_attn=anie_attn,
            ie_state_all=ie_state,
            ie_mlp_all=ie_mlp,
            ie_attn_all=ie_attn,
        )

        # E2 aggregate
        e2_mlp = self._stack([o["e2_mlp"] for o in outs])  # [N,4,L]
        e2_attn = self._stack([o["e2_attn"] for o in outs])
        e2_n_mlp = e2_mlp / te_denom[:, None, None]
        e2_n_attn = e2_attn / te_denom[:, None, None]
        np.savez_compressed(
            self.reports / "E2_window_curves.npz",
            e2_mlp=e2_mlp,
            e2_attn=e2_attn,
            e2_n_mlp=e2_n_mlp,
            e2_n_attn=e2_n_attn,
        )

        # E3 aggregate
        e3_base = self._stack([o["e3_base_curve"] for o in outs])
        e3_mlp = self._stack([o["e3_sever_mlp"] for o in outs])
        e3_attn = self._stack([o["e3_sever_attn"] for o in outs])
        np.savez_compressed(
            self.reports / "E3_modified_graph_curves.npz",
            base=e3_base,
            sever_mlp=e3_mlp,
            sever_attn=e3_attn,
        )

        # E4 aggregate
        ct_head = self._stack([o["ct_head"] for o in outs])  # [N,L,H]
        ap_head = self._stack([o["ap_head"] for o in outs])
        ct_head_aie = ct_head.mean(axis=0)
        ap_head_aie = ap_head.mean(axis=0)
        np.savez_compressed(
            self.reports / "E4_head_metrics.npz",
            ct_head=ct_head,
            ap_head=ap_head,
            ct_head_aie=ct_head_aie,
            ap_head_aie=ap_head_aie,
        )

        # E5 aggregate
        probe_vals = self._stack([o["probe_vals"] for o in outs])
        probe_pos = outs[0]["probe_token_pos"]
        np.savez_compressed(
            self.reports / "E5_L7H14_probe.npz",
            probe_vals=probe_vals,
            probe_token_pos=probe_pos,
        )

        # ---------- Plot helpers ----------
        def plot_heatmap(
            mat: np.ndarray,
            out_path: Path,
            title: str,
            xlabel: str,
            ylabel: str,
            clip_percentile: float = 99.0,
            x_ticks: Optional[List[int]] = None,
            x_ticklabels: Optional[List[str]] = None,
            y_ticks: Optional[List[int]] = None,
            y_ticklabels: Optional[List[str]] = None,
            y_split_after: Optional[int] = None,
        ) -> None:
            v = np.percentile(np.abs(mat), clip_percentile)
            v = max(v, 1e-6)
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v)

            fig, ax = plt.subplots(figsize=(8.0, 4.8))
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if x_ticks is not None:
                ax.set_xticks(x_ticks)
                if x_ticklabels is not None:
                    ax.set_xticklabels(x_ticklabels)
            if y_ticks is not None:
                ax.set_yticks(y_ticks)
                if y_ticklabels is not None:
                    ax.set_yticklabels(y_ticklabels)
            if y_split_after is not None:
                ax.axhline(y_split_after + 0.5, color="black", linewidth=1.0, alpha=0.55)
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("AIE: P_restore - P_corr")
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)

        # E1 figures (paper-style token organization: trigger-window + end-window)
        token_yticks: List[int] = []
        token_ylabs: List[str] = []
        early_show = sorted(set([-self.early_radius, -6, 0, 6, self.early_radius]))
        for d in early_show:
            if d in self.early_offsets:
                token_yticks.append(self.early_offsets.index(d))
                token_ylabs.append(f"T{d:+d}")
        late_show = sorted(set([-self.late_radius, -6, 0]))
        base_late = len(self.early_offsets)
        for d in late_show:
            if d in self.late_offsets:
                token_yticks.append(base_late + self.late_offsets.index(d))
                token_ylabs.append(f"E{d:+d}")
        layer_xticks = list(range(0, self.layers, max(1, self.layers // 7)))

        plot_heatmap(
            aie_state,
            self.figs / "ct_state_heatmap.png",
            "Causal Tracing State Heatmap (S_strict, AIE)",
            "Layer",
            "Anchored Token Position",
            x_ticks=layer_xticks,
            y_ticks=token_yticks,
            y_ticklabels=token_ylabs,
            y_split_after=len(self.early_offsets) - 1,
        )
        plot_heatmap(
            aie_mlp,
            self.figs / "ct_mlp_heatmap.png",
            "Causal Tracing MLP Heatmap (S_strict, AIE)",
            "Layer",
            "Anchored Token Position",
            x_ticks=layer_xticks,
            y_ticks=token_yticks,
            y_ticklabels=token_ylabs,
            y_split_after=len(self.early_offsets) - 1,
        )
        plot_heatmap(
            aie_attn,
            self.figs / "ct_attn_heatmap.png",
            "Causal Tracing Attn Heatmap (S_strict, AIE)",
            "Layer",
            "Anchored Token Position",
            x_ticks=layer_xticks,
            y_ticks=token_yticks,
            y_ticklabels=token_ylabs,
            y_split_after=len(self.early_offsets) - 1,
        )

        # E2 + state line plots with CI (paper-style: early trigger site vs late end site)
        state_trigger = ie_state[:, self.trigger_row, :]
        state_end = ie_state[:, self.end_row, :]
        mlp_trigger = e2_mlp[:, 0, :]
        mlp_end = e2_mlp[:, 2, :]
        attn_trigger = e2_attn[:, 0, :]
        attn_end = e2_attn[:, 2, :]

        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.1), sharey=True)
        panel_specs = [
            ("State (single restore)", state_trigger, state_end),
            ("MLP (window restore)", mlp_trigger, mlp_end),
            ("Attn (window restore)", attn_trigger, attn_end),
        ]
        for pi, (title, early_arr, late_arr) in enumerate(panel_specs):
            ax = axes[pi]
            mean_e, lo_e, hi_e = bootstrap_mean_ci(early_arr, n_boot=1000, seed=self.seed + 10 + pi)
            mean_l, lo_l, hi_l = bootstrap_mean_ci(late_arr, n_boot=1000, seed=self.seed + 20 + pi)
            ax.plot(mean_e, label="trigger (T+0)", linewidth=2.1, color="#9c2d2d")
            ax.fill_between(np.arange(self.layers), lo_e, hi_e, alpha=0.2, color="#9c2d2d")
            ax.plot(mean_l, label="prompt_end (E+0)", linewidth=2.1, color="#1f4e79")
            ax.fill_between(np.arange(self.layers), lo_l, hi_l, alpha=0.2, color="#1f4e79")
            ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("Layer")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
        axes[0].set_ylabel("AIE")
        fig.tight_layout()
        fig.savefig(self.figs / "ct_lineplot_with_ci.png")
        plt.close(fig)

        # Statistical significance summary for line-plot claims.
        sig_rows = []
        for name, arr in [
            ("state_trigger", state_trigger),
            ("state_end", state_end),
            ("mlp_trigger", mlp_trigger),
            ("mlp_end", mlp_end),
            ("attn_trigger", attn_trigger),
            ("attn_end", attn_end),
        ]:
            m = arr.mean(axis=0)
            l_peak = int(np.argmax(m))
            vals = arr[:, l_peak]
            m1, lo1, hi1 = bootstrap_mean_ci(vals[:, None], n_boot=1500, seed=self.seed + 101)
            sig_rows.append(
                {
                    "metric": name,
                    "peak_layer": l_peak,
                    "peak_AIE_mean": float(m1[0]),
                    "peak_AIE_ci_low": float(lo1[0]),
                    "peak_AIE_ci_high": float(hi1[0]),
                }
            )

        diff_early = mlp_trigger.max(axis=1) - attn_trigger.max(axis=1)
        de_m, de_lo, de_hi = bootstrap_mean_ci(diff_early[:, None], n_boot=1500, seed=self.seed + 102)
        sig_rows.append(
            {
                "metric": "early_mlp_minus_attn_peak",
                "peak_layer": -1,
                "peak_AIE_mean": float(de_m[0]),
                "peak_AIE_ci_low": float(de_lo[0]),
                "peak_AIE_ci_high": float(de_hi[0]),
            }
        )

        diff_late = attn_end.max(axis=1) - mlp_end.max(axis=1)
        dl_m, dl_lo, dl_hi = bootstrap_mean_ci(diff_late[:, None], n_boot=1500, seed=self.seed + 103)
        sig_rows.append(
            {
                "metric": "late_attn_minus_mlp_peak",
                "peak_layer": -1,
                "peak_AIE_mean": float(dl_m[0]),
                "peak_AIE_ci_low": float(dl_lo[0]),
                "peak_AIE_ci_high": float(dl_hi[0]),
            }
        )
        pd.DataFrame(sig_rows).to_csv(self.reports / "lineplot_significance.csv", index=False)

        # E3 modified graph intervention
        base_mean, base_lo, base_hi = bootstrap_mean_ci(e3_base, n_boot=1000, seed=self.seed)
        mlp_mean, mlp_lo, mlp_hi = bootstrap_mean_ci(e3_mlp, n_boot=1000, seed=self.seed + 1)
        attn_mean, attn_lo, attn_hi = bootstrap_mean_ci(e3_attn, n_boot=1000, seed=self.seed + 2)

        fig, ax = plt.subplots(figsize=(8.4, 4.3))
        xs = np.arange(self.layers)
        ax.plot(xs, base_mean, label="restore-state (baseline)", linewidth=2.2)
        ax.fill_between(xs, base_lo, base_hi, alpha=0.15)
        ax.plot(xs, mlp_mean, label="restore-state + sever-MLP", linewidth=2.0)
        ax.fill_between(xs, mlp_lo, mlp_hi, alpha=0.15)
        ax.plot(xs, attn_mean, label="restore-state + sever-Attn", linewidth=2.0)
        ax.fill_between(xs, attn_lo, attn_hi, alpha=0.15)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AIE at trigger token")
        ax.set_title("Modified Graph Intervention (S_strict)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.figs / "modified_graph_intervention.png")
        plt.close(fig)

        auc_base = e3_base.sum(axis=1)
        auc_mlp = e3_mlp.sum(axis=1)
        auc_attn = e3_attn.sum(axis=1)
        m0, lo0, hi0 = bootstrap_mean_ci(auc_base[:, None], n_boot=1000, seed=self.seed)
        m1, lo1, hi1 = bootstrap_mean_ci(auc_mlp[:, None], n_boot=1000, seed=self.seed + 1)
        m2, lo2, hi2 = bootstrap_mean_ci(auc_attn[:, None], n_boot=1000, seed=self.seed + 2)
        pd.DataFrame(
            [
                {
                    "condition": "baseline_restore_state",
                    "AUC_mean": float(m0[0]),
                    "AUC_ci_low": float(lo0[0]),
                    "AUC_ci_high": float(hi0[0]),
                },
                {
                    "condition": "sever_MLP",
                    "AUC_mean": float(m1[0]),
                    "AUC_ci_low": float(lo1[0]),
                    "AUC_ci_high": float(hi1[0]),
                },
                {
                    "condition": "sever_Attn",
                    "AUC_mean": float(m2[0]),
                    "AUC_ci_low": float(lo2[0]),
                    "AUC_ci_high": float(hi2[0]),
                },
            ]
        ).to_csv(self.reports / "modified_graph_metrics.csv", index=False)

        # Pairwise contrasts for modified graph significance.
        delta_b_mlp = auc_base - auc_mlp
        delta_b_attn = auc_base - auc_attn
        delta_mlp_attn = auc_mlp - auc_attn
        d1, d1_lo, d1_hi = bootstrap_mean_ci(delta_b_mlp[:, None], n_boot=1000, seed=self.seed + 11)
        d2, d2_lo, d2_hi = bootstrap_mean_ci(delta_b_attn[:, None], n_boot=1000, seed=self.seed + 12)
        d3, d3_lo, d3_hi = bootstrap_mean_ci(delta_mlp_attn[:, None], n_boot=1000, seed=self.seed + 13)
        pd.DataFrame(
            [
                {
                    "contrast": "baseline_minus_sever_MLP",
                    "mean": float(d1[0]),
                    "ci_low": float(d1_lo[0]),
                    "ci_high": float(d1_hi[0]),
                },
                {
                    "contrast": "baseline_minus_sever_Attn",
                    "mean": float(d2[0]),
                    "ci_low": float(d2_lo[0]),
                    "ci_high": float(d2_hi[0]),
                },
                {
                    "contrast": "sever_MLP_minus_sever_Attn",
                    "mean": float(d3[0]),
                    "ci_low": float(d3_lo[0]),
                    "ci_high": float(d3_hi[0]),
                },
            ]
        ).to_csv(self.reports / "modified_graph_significance.csv", index=False)

        # E4 head figures
        head_xticks = list(range(0, self.heads, max(1, self.heads // 8)))
        head_ylabs = [str(x) for x in head_xticks]
        layer_yticks = list(range(0, self.layers, max(1, self.layers // 7)))

        plot_heatmap(
            ct_head_aie,
            self.figs / "ct_head_heatmap.png",
            "Head-Level Causal Tracing (S_strict, AIE)",
            "Head Index",
            "Layer",
            x_ticks=head_xticks,
            x_ticklabels=head_ylabs,
            y_ticks=layer_yticks,
            y_ticklabels=[str(x) for x in layer_yticks],
        )
        plot_heatmap(
            ap_head_aie,
            self.figs / "ap_head_heatmap.png",
            "Head-Level Activation Patching (S_strict, AIE-like)",
            "Head Index",
            "Layer",
            x_ticks=head_xticks,
            x_ticklabels=head_ylabs,
            y_ticks=layer_yticks,
            y_ticklabels=[str(x) for x in layer_yticks],
        )

        # E5 L7H14 probe figure
        probe_mean, probe_lo, probe_hi = bootstrap_mean_ci(probe_vals, n_boot=1000, seed=self.seed)
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        ax.plot(np.arange(len(probe_mean)), probe_mean, linewidth=2.2, color="#9c2d2d")
        ax.fill_between(np.arange(len(probe_mean)), probe_lo, probe_hi, alpha=0.2, color="#9c2d2d")
        ax.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)
        ax.set_title("L7H14 Probe: Token Sweep (S_strict)")
        ax.set_xlabel("Relative Token Bin")
        ax.set_ylabel("AIE")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.figs / "L7H14_probe.png")
        plt.close(fig)

        # E6 final circuit nodes + edges
        # Nodes from top heads (ct+ap) and top mlp layers (trigger row in aie_mlp)
        # Prefer heads that are causally positive in CT; AP is only a tie-breaker.
        head_score = np.where(
            ct_head_aie > 0.0,
            ct_head_aie + 0.25 * np.maximum(ap_head_aie, 0.0),
            -1e9,
        )
        flat_idx = np.argsort(head_score.reshape(-1))[::-1]
        top_head_k = 8
        head_nodes = []
        used = set()
        for idx in flat_idx:
            l = int(idx // self.heads)
            h = int(idx % self.heads)
            name = f"L{l}H{h}"
            if name in used:
                continue
            used.add(name)
            head_nodes.append((name, l, h, float(ct_head_aie[l, h]), float(ap_head_aie[l, h])))
            if len(head_nodes) >= top_head_k:
                break

        # MLP nodes are selected by effect near output position (last token bin),
        # where the final first-token decision is directly read out.
        output_bin = self.token_bins - 1
        mlp_layer_score = aie_mlp[output_bin]
        top_mlp_k = 6
        mlp_layers = np.argsort(mlp_layer_score)[::-1][:top_mlp_k]

        node_rows = []
        for name, l, h, ctv, apv in head_nodes:
            node_rows.append(
                {
                    "node": name,
                    "type": "head",
                    "layer": l,
                    "head": h,
                    "AIE_ct": ctv,
                    "AIE_ap": apv,
                }
            )
        for l in mlp_layers.tolist():
            node_rows.append(
                {
                    "node": f"MLP{l}",
                    "type": "mlp",
                    "layer": int(l),
                    "head": np.nan,
                    "AIE_ct": float(aie_mlp[output_bin, l]),
                    "AIE_ap": np.nan,
                }
            )
        nodes_df = pd.DataFrame(node_rows)
        nodes_df.to_csv(self.reports / "final_circuit_nodes.csv", index=False)

        # Pairwise path validation (joint intervention) on top 3x3 for compute balance.
        # Head is restored at trigger token; MLP is restored at prompt-end token.
        edge_rows = []
        edge_heads = head_nodes[:3]
        edge_mlps = mlp_layers[:3].tolist()
        out_by_q = {int(o["q"][0]): o for o in outs}

        # Build edge-time cache (handles old cache format by lazy recompute once/sample).
        edge_cache: Dict[int, Dict[str, np.ndarray]] = {}
        print(f"[E6] preparing per-sample vectors for {len(q_list)} samples")
        for qi, q in enumerate(q_list, start=1):
            s = out_by_q[q]
            trigger_idx = int(s["trigger_idx"][0])
            prompt_end_idx = int(s["prompt_end_idx"][0]) if "prompt_end_idx" in s else int(s["seq_len"][0] - 1)
            corr_ids = s["corr_input_ids"].astype(np.int32)
            p_corr = float(s["p_corr"][0])

            if "clean_o_trigger" in s:
                clean_o_trigger = s["clean_o_trigger"]
            else:
                info = build_sample_info(self.root, q)
                clean_ids_t = self.tracer.encode(info.clean_text)
                clean_o = self.tracer.capture_o_proj_input(clean_ids_t)
                clean_o_trigger = np.stack(
                    [
                        clean_o[l][trigger_idx].detach().float().cpu().numpy()
                        for l in range(self.layers)
                    ],
                    axis=0,
                ).astype(np.float16)

            if "clean_mlp_prompt_end" in s:
                clean_mlp_prompt_end = s["clean_mlp_prompt_end"]
            else:
                info = build_sample_info(self.root, q)
                clean_ids_t = self.tracer.encode(info.clean_text)
                clean_m = self.tracer.capture_component(clean_ids_t, "mlp")
                clean_mlp_prompt_end = np.stack(
                    [
                        clean_m[l][prompt_end_idx].detach().float().cpu().numpy()
                        for l in range(self.layers)
                    ],
                    axis=0,
                ).astype(np.float16)

            edge_cache[q] = {
                "corr_ids": corr_ids,
                "p_corr": np.array([p_corr], dtype=np.float32),
                "trigger_idx": np.array([trigger_idx], dtype=np.int32),
                "prompt_end_idx": np.array([prompt_end_idx], dtype=np.int32),
                "clean_o_trigger": clean_o_trigger,
                "clean_mlp_prompt_end": clean_mlp_prompt_end,
            }
            if qi % 20 == 0 or qi == len(q_list):
                print(f"[E6]   preparation {qi}/{len(q_list)}")

        total_pairs = sum(1 for _, hl, _, _, _ in edge_heads for ml in edge_mlps if ml >= hl)
        pair_idx = 0
        for h_name, h_layer, h_idx, _, _ in edge_heads:
            for m_layer in edge_mlps:
                if m_layer < h_layer:
                    continue
                pair_idx += 1
                print(f"[E6] edge {pair_idx}/{total_pairs}: {h_name} -> MLP{m_layer}")
                ie_head_all = []
                ie_mlp_all = []
                ie_joint_all = []
                for qi, q in enumerate(q_list, start=1):
                    ec = edge_cache[q]
                    corr_ids_t = torch.from_numpy(ec["corr_ids"]).to(self.tracer.device).unsqueeze(0)
                    p_corr = float(ec["p_corr"][0])
                    trigger_idx = int(ec["trigger_idx"][0])
                    prompt_end_idx = int(ec["prompt_end_idx"][0])
                    ie_h, ie_m, ie_j = self.tracer.joint_head_mlp_effect_from_vectors(
                        corr_ids=corr_ids_t,
                        clean_o_trigger=ec["clean_o_trigger"],
                        clean_mlp_prompt_end=ec["clean_mlp_prompt_end"],
                        head_token_idx=trigger_idx,
                        mlp_token_idx=prompt_end_idx,
                        head_layer=h_layer,
                        head_idx=h_idx,
                        mlp_layer=int(m_layer),
                        p_corr=p_corr,
                    )
                    ie_head_all.append(ie_h)
                    ie_mlp_all.append(ie_m)
                    ie_joint_all.append(ie_j)
                    if qi % 20 == 0 or qi == len(q_list):
                        print(f"[E6]   samples {qi}/{len(q_list)} processed")

                ie_head_all = np.array(ie_head_all, dtype=np.float32)
                ie_mlp_all = np.array(ie_mlp_all, dtype=np.float32)
                ie_joint_all = np.array(ie_joint_all, dtype=np.float32)
                path_gain = ie_joint_all - np.maximum(ie_head_all, ie_mlp_all)
                mean_gain, lo_gain, hi_gain = bootstrap_mean_ci(
                    path_gain[:, None], n_boot=500, seed=self.seed + 7
                )
                edge_rows.append(
                    {
                        "source": h_name,
                        "target": f"MLP{m_layer}",
                        "path_gain_mean": float(mean_gain[0]),
                        "path_gain_ci_low": float(lo_gain[0]),
                        "path_gain_ci_high": float(hi_gain[0]),
                    }
                )

        edges_df = pd.DataFrame(edge_rows)
        if not edges_df.empty:
            edges_df = edges_df.sort_values("path_gain_mean", ascending=False)
        edges_df.to_csv(self.reports / "final_circuit_edges.csv", index=False)

        # final circuit figure
        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        ax.set_title("Final Candidate Circuit (S_strict)")
        ax.set_xlabel("Layer")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["MLP", "Head"])
        ax.set_xlim(-0.5, self.layers - 0.5)
        ax.set_ylim(-0.6, 1.6)
        ax.grid(axis="x", alpha=0.15)

        head_pos = {}
        for name, l, h, ctv, apv in head_nodes:
            ax.scatter([l], [1], s=85, c=[ctv + apv], cmap="RdBu_r", vmin=-0.1, vmax=0.1, edgecolor="k", linewidth=0.4)
            ax.text(l + 0.05, 1.05, name, fontsize=8)
            head_pos[name] = (l, 1)

        mlp_pos = {}
        for l in mlp_layers.tolist():
            n = f"MLP{l}"
            val = float(aie_mlp[output_bin, l])
            ax.scatter([l], [0], s=95, c=[val], cmap="RdBu_r", vmin=-0.1, vmax=0.1, edgecolor="k", linewidth=0.4)
            ax.text(l + 0.05, -0.12, n, fontsize=8)
            mlp_pos[n] = (l, 0)

        if not edges_df.empty:
            use_edges = edges_df[edges_df["path_gain_mean"] > 0].head(10)
            for _, r in use_edges.iterrows():
                s = r["source"]
                t = r["target"]
                if s in head_pos and t in mlp_pos:
                    x0, y0 = head_pos[s]
                    x1, y1 = mlp_pos[t]
                    lw = 0.8 + 6.0 * max(0.0, float(r["path_gain_mean"]))
                    ax.annotate(
                        "",
                        xy=(x1, y1 + 0.02),
                        xytext=(x0, y0 - 0.02),
                        arrowprops=dict(arrowstyle="->", color="#333333", lw=lw, alpha=0.7),
                    )

        sm = cm.ScalarMappable(norm=Normalize(vmin=-0.1, vmax=0.1), cmap="RdBu_r")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Node AIE score")
        fig.tight_layout()
        fig.savefig(self.figs / "final_circuit.png")
        plt.close(fig)

        # case examples: 5 typical + 5 abnormal
        # typical: strict with high TE; abnormal: fail set top5 by low clean prob from quick pass
        te_rank = np.argsort(te)[::-1]
        typical_q = [q_list[i] for i in te_rank[:5]]

        # Build fail examples from cached if available; fallback to strict tail
        fail_q = self.S_fail[:5]
        if len(fail_q) < 5:
            fail_q = [q_list[i] for i in np.argsort(te)[:5]]

        # load per-sample state maps from cache for chosen q
        panel_q = typical_q + fail_q
        fig, axes = plt.subplots(2, 5, figsize=(15.2, 5.6), sharex=True, sharey=True)
        for i, q in enumerate(panel_q):
            ax = axes[i // 5, i % 5]
            c = self._load_sample_cache(q)
            if c is None:
                continue
            mat = c["ie_state"]
            v = np.percentile(np.abs(mat), 98)
            v = max(v, 1e-6)
            im = ax.imshow(
                mat,
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v),
            )
            ax.set_title(f"q{q}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Case Trace Examples: 5 Typical + 5 Abnormal (state IE)")
        fig.tight_layout(rect=[0, 0.0, 1, 0.94])
        fig.savefig(self.figs / "case_trace_examples.png")
        plt.close(fig)

        # captions / method notes (rule 6.5)
        caption_lines = [
            "# Figure Captions and Statistical Notes",
            "",
            "- `ct_state_heatmap.png`: set=S_strict, metric=AIE, intervention=restoration on state in corrupted run, token-axis=anchored windows [T-12..T+12] + [E-12..E+0], clip=|value|<=99th percentile, stats=mean over samples.",
            "- `ct_mlp_heatmap.png`: set=S_strict, metric=AIE, intervention=restoration on MLP output, token-axis=anchored windows [T-12..T+12] + [E-12..E+0], clip=|value|<=99th percentile, stats=mean over samples.",
            "- `ct_attn_heatmap.png`: set=S_strict, metric=AIE, intervention=restoration on Attn output, token-axis=anchored windows [T-12..T+12] + [E-12..E+0], clip=|value|<=99th percentile, stats=mean over samples.",
            "- `ct_lineplot_with_ci.png`: set=S_strict, metric=AIE, intervention=state single-restore + MLP/Attn window restoration (window=10), normalize=no, stats=mean with 95% bootstrap CI.",
            "- `modified_graph_intervention.png`: set=S_strict, metric=AIE, intervention=restore-state with sever-MLP / sever-Attn, normalize=no, stats=mean with 95% bootstrap CI.",
            "- `ct_head_heatmap.png`: set=S_strict, metric=AIE, intervention=head-level restoration at trigger token, clip=|value|<=99th percentile, stats=mean over samples.",
            "- `ap_head_heatmap.png`: set=S_strict, metric=AIE-like (clean-drop), intervention=head-level activation patching (corr->clean) at trigger token, clip=|value|<=99th percentile, stats=mean over samples.",
            "- `L7H14_probe.png`: set=S_strict, metric=AIE, intervention=restore L7H14 over token sweep, normalize=no, stats=mean with 95% bootstrap CI.",
            "- `final_circuit.png`: set=S_strict, metric=node=AIE, edge=joint path gain, intervention=conditional restoration, normalize=no, stats=edge CI from bootstrap.",
            "- `case_trace_examples.png`: sets=5 typical from S_strict + 5 abnormal from S_fail/tail, metric=IE state map, intervention=single-point restoration.",
        ]
        (self.reports / "figure_captions.md").write_text("\n".join(caption_lines))

        # summary report
        _, ate_lo_arr, ate_hi_arr = bootstrap_mean_ci(te[:, None], n_boot=1000, seed=self.seed)
        summary = {
            "n_strict_used": len(q_list),
            "token_bins": self.token_bins,
            "token_axis": "anchored_windows",
            "early_window": f"T{self.early_offsets[0]:+d}..T{self.early_offsets[-1]:+d}",
            "late_window": f"E{self.late_offsets[0]:+d}..E{self.late_offsets[-1]:+d}",
            "layers": self.layers,
            "heads": self.heads,
            "P_clean_mean": float(np.mean(p_clean)),
            "P_corr_mean": float(np.mean(p_corr)),
            "ATE_mean": float(np.mean(te)),
            "ATE_ci95": [float(ate_lo_arr[0]), float(ate_hi_arr[0])],
            "AIE_state_peak": float(np.max(aie_state)),
            "AIE_mlp_peak": float(np.max(aie_mlp)),
            "AIE_attn_peak": float(np.max(aie_attn)),
            "note": "ATE_ci95 is [low, high] for S_strict.",
        }
        (self.reports / "summary_metrics.json").write_text(json.dumps(summary, indent=2))

        # sanity checks + diagnostic log
        checks = []
        checks.append(("ATE_positive", float(np.mean(te)) > 0.05))
        checks.append(("clean_gt_corr", float(np.mean(p_clean)) > float(np.mean(p_corr))))
        checks.append(("state_peak_positive", float(np.max(aie_state)) > 0.01))
        checks.append(("mlp_peak_positive", float(np.max(aie_mlp)) > 0.005))
        checks.append(("attn_peak_positive", float(np.max(aie_attn)) > 0.005))
        with (self.reports / "self_check_log.md").open("w") as f:
            f.write("# Self Check Log\n\n")
            for k, v in checks:
                f.write(f"- {k}: {'PASS' if v else 'FAIL'}\n")
            f.write("\n")
            if all(v for _, v in checks):
                f.write("All coarse sanity checks passed.\n")
            else:
                f.write("Some sanity checks failed; inspect aggregate metrics and rerun with refined setup.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/root/data/R4")
    parser.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--token-bins", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--early-radius", type=int, default=12)
    parser.add_argument("--late-radius", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    pipe = Pipeline(
        root=Path(args.root),
        model_path=args.model_path,
        token_bins=args.token_bins,
        batch_size=args.batch_size,
        window_size=args.window_size,
        max_samples=args.max_samples,
        seed=args.seed,
        force=args.force,
        early_radius=args.early_radius,
        late_radius=args.late_radius,
    )

    print(f"S_strict size used: {len(pipe.S_strict)}")
    outs = pipe.run_all_samples()
    pipe.aggregate_and_report(outs)
    print("Pipeline done.")


if __name__ == "__main__":
    main()
