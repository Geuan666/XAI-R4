# Paper Alignment Review

## What Was Changed
- Reimplemented token-axis organization to anchored windows: `T-12..T+12` and `E-12..E+0` (paper-style early/late site view).
- Fixed modified-graph sever logic to freeze downstream module outputs only at the intervened token, not the full sequence.
- Added explicit line-plot significance reports (`lineplot_significance.csv`, `modified_graph_significance.csv`).
- Added paper-aligned noisy corruption control (`ct_lineplot_with_ci_paper_noise.png`) using 3σ Gaussian on `user_instruction` span with 3 repeats/sample.

## Main Significance (Natural Corruption, S_strict=120)
- ATE = 0.7412 (95% CI: 0.7161 to 0.7686).
- Early-site contrast (MLP - Attn peak): -0.4810 (95% CI: -0.5224 to -0.4414).
- Late-site contrast (Attn - MLP peak): 0.2504 (95% CI: 0.2199 to 0.2800).
- Modified graph baseline-sever_MLP delta: 2.4616 (95% CI: 2.1851 to 2.7480).
- Modified graph baseline-sever_Attn delta: 2.4748 (95% CI: 2.2362 to 2.7234).
- Sever_MLP - Sever_Attn: 0.0132 (95% CI: -0.0777 to 0.0991).

## Paper-Aligned Noisy Control (S_strict=120, 3 repeats/sample)
- Early-site contrast (MLP - Attn peak): -0.0303 (95% CI: -0.0415 to -0.0194).
- Late-site contrast (Attn - MLP peak): -0.0021 (95% CI: -0.0037 to -0.0004).
- Interpretation: noisy control narrows the early-site gap versus natural corruption, but does not invert it to MLP dominance in this task.

## Conclusion
- Current implementation quality now matches paper-style organization and significance reporting.
- Remaining divergence from ROME qualitative pattern appears to be task-mechanism difference, not a plotting or hook bug.