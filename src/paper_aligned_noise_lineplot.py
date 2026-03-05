#!/usr/bin/env python3
"""Paper-aligned noisy corruption line-plot experiment for S_strict."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sys

sys.path.append("/root/data/R4/src")
from tool_call_circuit_pipeline import QwenCircuitTracer, build_sample_info, bootstrap_mean_ci  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_user_instruction_span(meta: dict) -> Tuple[int, int]:
    for s in meta["clean"]["segments"]:
        if s["name"] == "user_instruction":
            return int(s["token_start"]), int(s["token_end_exclusive"])
    return 0, 1


class NoiseLineRunner:
    def __init__(self, model_path: str, seed: int) -> None:
        self.tracer = QwenCircuitTracer(model_path=model_path, device="cuda")
        self.seed = seed
        self.layers = self.tracer.n_layers
        emb_std = self.tracer.model.model.embed_tokens.weight.detach().float().std().item()
        self.noise_sigma = float(3.0 * emb_std)

    @torch.no_grad()
    def p_with_noise(
        self,
        ids: torch.Tensor,
        span_positions: Sequence[int],
        noise: torch.Tensor,
        extra_hooks: List[torch.utils.hooks.RemovableHandle],
    ) -> float:
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        def emb_hook(_m, _inp, out):
            t = out.clone()
            for i, pos in enumerate(span_positions):
                t[:, pos, :] = t[:, pos, :] + noise[i].to(t.dtype)
            return t

        hooks.append(self.tracer.model.model.embed_tokens.register_forward_hook(emb_hook))
        hooks.extend(extra_hooks)
        p = float(self.tracer.p_tool(ids)[0])
        for h in hooks:
            h.remove()
        return p

    def state_curve_noisy(
        self,
        ids: torch.Tensor,
        span_positions: Sequence[int],
        noise: torch.Tensor,
        clean_state: Dict[int, torch.Tensor],
        token_idx: int,
        p_base: float,
    ) -> np.ndarray:
        curve = np.zeros(self.layers, dtype=np.float32)
        for l in range(self.layers):
            src = clean_state[l]

            def patch(_m, _inp, out):
                t = out.clone()
                t[0, token_idx, :] = src[token_idx, :].to(t.dtype)
                return t

            h = self.tracer.model.model.layers[l].register_forward_hook(patch)
            p = self.p_with_noise(ids, span_positions, noise, [h])
            curve[l] = p - p_base
        return curve

    def window_curve_noisy(
        self,
        ids: torch.Tensor,
        span_positions: Sequence[int],
        noise: torch.Tensor,
        clean_cache: Dict[int, torch.Tensor],
        component: str,
        token_idx: int,
        p_base: float,
        window: int,
    ) -> np.ndarray:
        curve = np.zeros(self.layers, dtype=np.float32)
        half = window // 2

        for center in range(self.layers):
            lo = max(0, center - half)
            hi = min(self.layers, lo + window)
            lo = max(0, hi - window)
            hs: List[torch.utils.hooks.RemovableHandle] = []

            for l in range(lo, hi):
                src = clean_cache[l]
                if component == "mlp":
                    mod = self.tracer.model.model.layers[l].mlp

                    def mk(src_layer: torch.Tensor):
                        def _h(_m, _inp, out):
                            t = out.clone()
                            t[0, token_idx, :] = src_layer[token_idx, :].to(t.dtype)
                            return t

                        return _h

                    hs.append(mod.register_forward_hook(mk(src)))
                elif component == "attn":
                    mod = self.tracer.model.model.layers[l].self_attn

                    def mk(src_layer: torch.Tensor):
                        def _h(_m, _inp, out):
                            if isinstance(out, tuple):
                                t0 = out[0].clone()
                                t0[0, token_idx, :] = src_layer[token_idx, :].to(t0.dtype)
                                return (t0, out[1])
                            t = out.clone()
                            t[0, token_idx, :] = src_layer[token_idx, :].to(t.dtype)
                            return t

                        return _h

                    hs.append(mod.register_forward_hook(mk(src)))
                else:
                    raise ValueError(component)

            p = self.p_with_noise(ids, span_positions, noise, hs)
            curve[center] = p - p_base
        return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/root/data/R4")
    ap.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    ap.add_argument("--n-noise", type=int, default=3)
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-samples", type=int, default=120)
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(args.root)

    df = pd.read_csv(root / "pair/first_token_len_eval_qwen3_1.7b.csv")
    strict_q = (
        df[(df["clean_top1"] == "<tool_call>") & (df["corr_top1"] != "<tool_call>")]["q"].astype(int).tolist()
    )[: args.max_samples]

    runner = NoiseLineRunner(args.model_path, args.seed)

    N = len(strict_q)
    L = runner.layers

    state_trigger = np.zeros((N, L), dtype=np.float32)
    state_end = np.zeros((N, L), dtype=np.float32)
    mlp_trigger = np.zeros((N, L), dtype=np.float32)
    mlp_end = np.zeros((N, L), dtype=np.float32)
    attn_trigger = np.zeros((N, L), dtype=np.float32)
    attn_end = np.zeros((N, L), dtype=np.float32)
    p_noise = np.zeros((N,), dtype=np.float32)

    for i, q in enumerate(strict_q, start=1):
        info = build_sample_info(root, q)
        clean_ids = runner.tracer.encode(info.clean_text)
        clean_state = runner.tracer.capture_component(clean_ids, "state")
        clean_mlp = runner.tracer.capture_component(clean_ids, "mlp")
        clean_attn = runner.tracer.capture_component(clean_ids, "attn")

        s0, s1 = get_user_instruction_span(info.meta)
        span_positions = list(range(s0, s1))
        if len(span_positions) == 0:
            span_positions = [info.trigger_idx]

        st_t_acc = []
        st_e_acc = []
        mlp_t_acc = []
        mlp_e_acc = []
        attn_t_acc = []
        attn_e_acc = []
        p_acc = []

        for r in range(args.n_noise):
            g = torch.Generator(device=runner.tracer.device)
            g.manual_seed(args.seed * 100000 + q * 100 + r)
            noise = torch.randn(
                (len(span_positions), runner.tracer.hidden),
                generator=g,
                device=runner.tracer.device,
            ) * runner.noise_sigma

            p_corr = runner.p_with_noise(clean_ids, span_positions, noise, extra_hooks=[])
            p_acc.append(p_corr)

            st_t_acc.append(
                runner.state_curve_noisy(
                    clean_ids,
                    span_positions,
                    noise,
                    clean_state,
                    token_idx=info.trigger_idx,
                    p_base=p_corr,
                )
            )
            st_e_acc.append(
                runner.state_curve_noisy(
                    clean_ids,
                    span_positions,
                    noise,
                    clean_state,
                    token_idx=info.prompt_end_idx,
                    p_base=p_corr,
                )
            )

            mlp_t_acc.append(
                runner.window_curve_noisy(
                    clean_ids,
                    span_positions,
                    noise,
                    clean_mlp,
                    component="mlp",
                    token_idx=info.trigger_idx,
                    p_base=p_corr,
                    window=args.window_size,
                )
            )
            mlp_e_acc.append(
                runner.window_curve_noisy(
                    clean_ids,
                    span_positions,
                    noise,
                    clean_mlp,
                    component="mlp",
                    token_idx=info.prompt_end_idx,
                    p_base=p_corr,
                    window=args.window_size,
                )
            )

            attn_t_acc.append(
                runner.window_curve_noisy(
                    clean_ids,
                    span_positions,
                    noise,
                    clean_attn,
                    component="attn",
                    token_idx=info.trigger_idx,
                    p_base=p_corr,
                    window=args.window_size,
                )
            )
            attn_e_acc.append(
                runner.window_curve_noisy(
                    clean_ids,
                    span_positions,
                    noise,
                    clean_attn,
                    component="attn",
                    token_idx=info.prompt_end_idx,
                    p_base=p_corr,
                    window=args.window_size,
                )
            )

        state_trigger[i - 1] = np.mean(np.stack(st_t_acc), axis=0)
        state_end[i - 1] = np.mean(np.stack(st_e_acc), axis=0)
        mlp_trigger[i - 1] = np.mean(np.stack(mlp_t_acc), axis=0)
        mlp_end[i - 1] = np.mean(np.stack(mlp_e_acc), axis=0)
        attn_trigger[i - 1] = np.mean(np.stack(attn_t_acc), axis=0)
        attn_end[i - 1] = np.mean(np.stack(attn_e_acc), axis=0)
        p_noise[i - 1] = float(np.mean(p_acc))

        if i % 10 == 0 or i == N:
            print(f"[paper-noise] {i}/{N} samples")

    reports = root / "reports"
    figs = root / "figs"
    reports.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    np.savez_compressed(
        reports / "paper_noise_lineplot_curves.npz",
        q=np.array(strict_q, dtype=np.int32),
        state_trigger=state_trigger,
        state_end=state_end,
        mlp_trigger=mlp_trigger,
        mlp_end=mlp_end,
        attn_trigger=attn_trigger,
        attn_end=attn_end,
        p_noise=p_noise,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), sharey=True)
    panel_specs = [
        ("State (single restore)", state_trigger, state_end),
        ("MLP (window restore)", mlp_trigger, mlp_end),
        ("Attn (window restore)", attn_trigger, attn_end),
    ]
    for pi, (title, early_arr, late_arr) in enumerate(panel_specs):
        ax = axes[pi]
        me, le, he = bootstrap_mean_ci(early_arr, n_boot=1000, seed=args.seed + 10 + pi)
        ml, ll, hl = bootstrap_mean_ci(late_arr, n_boot=1000, seed=args.seed + 20 + pi)
        ax.plot(me, label="trigger (T+0)", linewidth=2.1, color="#9c2d2d")
        ax.fill_between(np.arange(L), le, he, alpha=0.2, color="#9c2d2d")
        ax.plot(ml, label="prompt_end (E+0)", linewidth=2.1, color="#1f4e79")
        ax.fill_between(np.arange(L), ll, hl, alpha=0.2, color="#1f4e79")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Layer")
        ax.set_title(title)
        ax.legend(frameon=False)
    axes[0].set_ylabel("AIE (noisy corruption baseline)")
    fig.tight_layout()
    fig.savefig(figs / "ct_lineplot_with_ci_paper_noise.png", dpi=220)
    plt.close(fig)

    def peak_row(name: str, arr: np.ndarray) -> dict:
        m = arr.mean(axis=0)
        l = int(np.argmax(m))
        v = arr[:, l]
        mm, ll, hh = bootstrap_mean_ci(v[:, None], n_boot=1500, seed=args.seed + 77)
        return {
            "metric": name,
            "peak_layer": l,
            "peak_AIE_mean": float(mm[0]),
            "peak_AIE_ci_low": float(ll[0]),
            "peak_AIE_ci_high": float(hh[0]),
        }

    rows = [
        peak_row("state_trigger", state_trigger),
        peak_row("state_end", state_end),
        peak_row("mlp_trigger", mlp_trigger),
        peak_row("mlp_end", mlp_end),
        peak_row("attn_trigger", attn_trigger),
        peak_row("attn_end", attn_end),
    ]

    de = mlp_trigger.max(axis=1) - attn_trigger.max(axis=1)
    dm, dl, dh = bootstrap_mean_ci(de[:, None], n_boot=1500, seed=args.seed + 88)
    rows.append(
        {
            "metric": "early_mlp_minus_attn_peak",
            "peak_layer": -1,
            "peak_AIE_mean": float(dm[0]),
            "peak_AIE_ci_low": float(dl[0]),
            "peak_AIE_ci_high": float(dh[0]),
        }
    )

    dlv = attn_end.max(axis=1) - mlp_end.max(axis=1)
    dm, dl, dh = bootstrap_mean_ci(dlv[:, None], n_boot=1500, seed=args.seed + 89)
    rows.append(
        {
            "metric": "late_attn_minus_mlp_peak",
            "peak_layer": -1,
            "peak_AIE_mean": float(dm[0]),
            "peak_AIE_ci_low": float(dl[0]),
            "peak_AIE_ci_high": float(dh[0]),
        }
    )

    pd.DataFrame(rows).to_csv(reports / "paper_noise_lineplot_significance.csv", index=False)
    pd.DataFrame(
        {
            "n_samples": [N],
            "n_noise_repeats": [args.n_noise],
            "noise_sigma": [runner.noise_sigma],
            "p_noise_mean": [float(np.mean(p_noise))],
            "p_noise_std": [float(np.std(p_noise))],
        }
    ).to_csv(reports / "paper_noise_lineplot_summary.csv", index=False)

    print("paper-noise lineplot done")


if __name__ == "__main__":
    main()
