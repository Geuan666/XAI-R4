#!/usr/bin/env python3
"""Appendix 4.1 noise robustness for tool-call causal tracing."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_mean_ci(arr: np.ndarray, n_boot: int = 800, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = arr[idx].mean(axis=1)
    mean = arr.mean(axis=0)
    lo = np.quantile(boot, 0.025, axis=0)
    hi = np.quantile(boot, 0.975, axis=0)
    return mean, lo, hi


def get_trigger_idx(meta: dict) -> int:
    seg = None
    for s in meta["clean"]["segments"]:
        if s["name"] == "user_instruction":
            seg = s
            break
    if seg is None:
        return 0
    lo, hi = seg["token_start"], seg["token_end_exclusive"]
    for sp in meta["clean"]["key_spans"].get("Write", []):
        t = sp["token_start"]
        if lo <= t < hi:
            return int(t)
    for sp in meta["corrupted"]["key_spans"].get("State", []):
        t = sp["token_start"]
        if lo <= t < hi:
            return int(t)
    return int(lo)


class NoiseTracer:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except Exception:
            pass

        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.layers = int(self.model.config.num_hidden_layers)

        tid = self.tok.encode("<tool_call>", add_special_tokens=False)
        assert len(tid) == 1
        self.tool_id = int(tid[0])

        emb_w = self.model.model.embed_tokens.weight.detach().float()
        self.emb_std_global = float(emb_w.std().item())
        self.emb_var_diag = emb_w.var(dim=0).to(self.device)

    def encode(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)

    @torch.no_grad()
    def p_tool(self, ids: torch.Tensor) -> float:
        out = self.model(input_ids=ids)
        p = torch.softmax(out.logits[:, -1, :].float(), dim=-1)[:, self.tool_id]
        return float(p[0].item())

    @torch.no_grad()
    def capture_clean_state(self, ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        caches: Dict[int, torch.Tensor] = {}
        hooks = []

        def mk(l: int):
            def _h(_m, _inp, out):
                caches[l] = out[0].detach().clone()

            return _h

        for l in range(self.layers):
            hooks.append(self.model.model.layers[l].register_forward_hook(mk(l)))
        _ = self.model(input_ids=ids)
        for h in hooks:
            h.remove()
        return caches

    def make_noise_vec(self, noise_type: str, hidden_size: int, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        if noise_type == "gaussian":
            sigma = 3.0 * self.emb_std_global
            return torch.randn(hidden_size, generator=g, device=self.device) * sigma

        if noise_type == "cov_diag_gaussian":
            # Diagonal covariance match (practical approximation).
            sigma_diag = torch.sqrt(self.emb_var_diag + 1e-8) * 3.0
            return torch.randn(hidden_size, generator=g, device=self.device) * sigma_diag

        if noise_type == "uniform":
            sigma = 3.0 * self.emb_std_global
            return (torch.rand(hidden_size, generator=g, device=self.device) * 2.0 - 1.0) * sigma

        raise ValueError(noise_type)

    @torch.no_grad()
    def p_noisy(
        self,
        ids: torch.Tensor,
        trigger_idx: int,
        noise_vec: torch.Tensor,
        layer_patch: Optional[Tuple[int, torch.Tensor]] = None,
    ) -> float:
        hooks = []

        def emb_hook(_m, _inp, out):
            t = out.clone()
            t[0, trigger_idx, :] = t[0, trigger_idx, :] + noise_vec.to(t.dtype)
            return t

        hooks.append(self.model.model.embed_tokens.register_forward_hook(emb_hook))

        if layer_patch is not None:
            layer, clean_vec = layer_patch

            def state_hook(_m, _inp, out):
                t = out.clone()
                t[0, trigger_idx, :] = clean_vec.to(t.dtype)
                return t

            hooks.append(self.model.model.layers[layer].register_forward_hook(state_hook))

        p = self.p_tool(ids)
        for h in hooks:
            h.remove()
        return p


def main() -> None:
    root = Path("/root/data/R4")
    reports = root / "reports"
    figs = root / "figs"
    reports.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    set_seed(123)

    df = pd.read_csv(root / "pair/first_token_len_eval_qwen3_1.7b.csv")
    strict_q = (
        df[(df["clean_top1"] == "<tool_call>") & (df["corr_top1"] != "<tool_call>")]["q"].astype(int).tolist()
    )

    tracer = NoiseTracer(model_path="/root/data/Qwen/Qwen3-1.7B", device="cuda")
    noise_types = ["gaussian", "cov_diag_gaussian", "uniform"]

    N = len(strict_q)
    L = tracer.layers
    curves = np.zeros((N, len(noise_types), L), dtype=np.float32)
    p_noise = np.zeros((N, len(noise_types)), dtype=np.float32)

    for i, q in enumerate(strict_q, start=1):
        meta = json.loads((root / f"pair/meta-q{q}.json").read_text())
        clean_text = (root / f"pair/prompt-clean-q{q}.txt").read_text()
        ids = tracer.encode(clean_text)
        trigger_idx = get_trigger_idx(meta)

        clean_state = tracer.capture_clean_state(ids)

        for ni, nt in enumerate(noise_types):
            noise = tracer.make_noise_vec(
                nt,
                hidden_size=tracer.model.config.hidden_size,
                seed=10_000 * (ni + 1) + q,
            )
            p0 = tracer.p_noisy(ids, trigger_idx, noise, layer_patch=None)
            p_noise[i - 1, ni] = p0
            for l in range(L):
                pr = tracer.p_noisy(ids, trigger_idx, noise, layer_patch=(l, clean_state[l][trigger_idx]))
                curves[i - 1, ni, l] = pr - p0

        if i % 10 == 0 or i == N:
            print(f"[noise] {i}/{N} samples")

    np.savez_compressed(
        reports / "noise_robustness_curves.npz",
        q=np.array(strict_q, dtype=np.int32),
        curves=curves,
        p_noise=p_noise,
        noise_types=np.array(noise_types),
    )

    natural = np.load(reports / "E3_modified_graph_curves.npz")["base"]  # [N,L]

    rows = []
    nat_mean = natural.mean(axis=0)
    nat_peak_layer = int(np.argmax(nat_mean))
    nat_peak_val = float(np.max(nat_mean))

    for ni, nt in enumerate(noise_types):
        arr = curves[:, ni, :]
        m = arr.mean(axis=0)
        peak_l = int(np.argmax(m))
        peak_v = float(np.max(m))
        rows.append(
            {
                "noise_type": nt,
                "peak_layer": peak_l,
                "peak_AIE": peak_v,
                "mean_AIE": float(m.mean()),
                "same_sign_as_natural_peak": bool((peak_v >= 0) == (nat_peak_val >= 0)),
                "peak_layer_delta_vs_natural": int(peak_l - nat_peak_layer),
            }
        )

    rob_df = pd.DataFrame(rows)
    rob_df.to_csv(reports / "noise_robustness.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    x = np.arange(L)

    m_nat, lo_nat, hi_nat = bootstrap_mean_ci(natural, seed=123)
    ax.plot(x, m_nat, label="natural corrupted", linewidth=2.3, color="#1f4e79")
    ax.fill_between(x, lo_nat, hi_nat, alpha=0.18, color="#1f4e79")

    colors = ["#9c2d2d", "#2f7d32", "#7a3e9d"]
    for ni, nt in enumerate(noise_types):
        m, lo, hi = bootstrap_mean_ci(curves[:, ni, :], seed=200 + ni)
        ax.plot(x, m, label=nt, linewidth=2.0, color=colors[ni])
        ax.fill_between(x, lo, hi, alpha=0.16, color=colors[ni])

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title("Noise Robustness (S_strict, trigger-token state restoration)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AIE")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "noise_robustness_curves.png")
    plt.close(fig)

    # Append caption notes
    cap = root / "reports/figure_captions.md"
    extra = (
        "\n- `noise_robustness_curves.png`: set=S_strict, metric=AIE, intervention=trigger-token state restoration under "
        "noise corruption (gaussian/cov-diag/uniform) and natural corrupted baseline, normalize=no, stats=mean with 95% bootstrap CI.\n"
    )
    if cap.exists():
        txt = cap.read_text()
        if "noise_robustness_curves.png" not in txt:
            cap.write_text(txt + extra)

    print("noise robustness done")


if __name__ == "__main__":
    main()
