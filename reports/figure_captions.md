# Figure Captions and Statistical Notes

- `ct_state_heatmap.png`: set=S_strict, metric=AIE, intervention=restoration on state in corrupted run, token-axis=anchored windows [T-12..T+12] + [E-12..E+0], clip=|value|<=99th percentile, stats=mean over samples.
- `ct_mlp_heatmap.png`: set=S_strict, metric=AIE, intervention=restoration on MLP output, token-axis=anchored windows [T-12..T+12] + [E-12..E+0], clip=|value|<=99th percentile, stats=mean over samples.
- `ct_attn_heatmap.png`: set=S_strict, metric=AIE, intervention=restoration on Attn output, token-axis=anchored windows [T-12..T+12] + [E-12..E+0], clip=|value|<=99th percentile, stats=mean over samples.
- `ct_lineplot_with_ci.png`: set=S_strict, metric=AIE, intervention=state single-restore + MLP/Attn window restoration (window=10), normalize=no, stats=mean with 95% bootstrap CI.
- `modified_graph_intervention.png`: set=S_strict, metric=AIE, intervention=restore-state with sever-MLP / sever-Attn, normalize=no, stats=mean with 95% bootstrap CI.
- `ct_head_heatmap.png`: set=S_strict, metric=AIE, intervention=head-level restoration at trigger token, clip=|value|<=99th percentile, stats=mean over samples.
- `ap_head_heatmap.png`: set=S_strict, metric=AIE-like (clean-drop), intervention=head-level activation patching (corr->clean) at trigger token, clip=|value|<=99th percentile, stats=mean over samples.
- `L7H14_probe.png`: set=S_strict, metric=AIE, intervention=restore L7H14 over token sweep, normalize=no, stats=mean with 95% bootstrap CI.
- `final_circuit.png`: set=S_strict, metric=node=AIE, edge=joint path gain, intervention=conditional restoration, normalize=no, stats=edge CI from bootstrap.
- `case_trace_examples.png`: sets=5 typical from S_strict + 5 abnormal from S_fail/tail, metric=IE state map, intervention=single-point restoration.
- `ct_lineplot_with_ci_paper_noise.png`: set=S_strict, metric=AIE, intervention=paper-aligned noisy corruption on user_instruction span (3σ Gaussian, 3 repeats/sample) with state single-restore + MLP/Attn window restoration, normalize=no, stats=mean with 95% bootstrap CI.
