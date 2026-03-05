# Tool-Call Causal Tracing TODO (从 ROME 论文迁移)
> 模型：`/root/data/Qwen/Qwen3-1.7B`
## 0. 范围与目标

### 0.1 我们要回答的问题
在你的任务中，模型是否调用工具由**首个输出 token 是否为 `<tool_call>`**决定。  
我们要定位模型内部到底是哪些组件/路径在做这个决策（模块级或电路级）。

### 0.2 复现范围（严格对齐论文定位部分）
- 只做 ROME 论文里的 `Locating` / `Finding`（因果追踪与电路定位）。
- 不做参数编辑（ROME 写入）、不做编辑应用评测。
- 目标是把论文中定位部分的关键图、关键结论，在本项目上重做一遍。

### 0.3 论文方法在本项目的映射
- 论文中的事实对象 token `o`，映射为本项目的目标 token：`<tool_call>`。
- 论文中的 clean/corrupted/restoration 三次运行，映射为：
  - clean run: `prompt-clean-q*.txt`
  - corrupted run: `prompt-corrupted-q*.txt`（主设定，优先）
  - corrupted-with-restoration run: 在 corrupted 前向中，把指定层/位置/组件替换成 clean 对应激活
- 论文中的 “subject token” 映射为本任务的**触发词位置**（`user_instruction` 段开头，典型为 `Write` vs `State`）。

---

## 1. 数据与样本集定义

### 1.1 数据来源
- 全量配对数据：`pair/`
  - `prompt-clean-q{n}.txt`
  - `prompt-corrupted-q{n}.txt`
  - `meta-q{n}.json`（含 token 对齐、segment、key spans）
- 单样本调试：`sample/`（建议用 `q85` 做单样本管线验证）
- 现有首 token 统计：`pair/first_token_len_eval_qwen3_1.7b.csv`

### 1.2 评估集合（必须固定）
- `S_all`: 全部 164 条。
- `S_strict`: `clean_top1=<tool_call>` 且 `corr_top1!=<tool_call>`（当前 120 条，主分析集合）。
- `S_ambiguous`: clean 和 corrupt 都是 `<tool_call>`（当前 27 条，鲁棒性分析）。
- `S_fail`: clean 和 corrupt 都不是 `<tool_call>`（当前 17 条，失败模式分析）。

说明：主图、主结论优先基于 `S_strict`，其余集合用于附录/鲁棒性。

---

## 2. 核心指标（按论文定义改写）

记目标 token 为 `t_call = "<tool_call>"`，只看**首个生成位置**概率。

- `P_clean(q) = P(y1=t_call | x_clean^q)`
- `P_corr(q)  = P(y1=t_call | x_corr^q)`
- `P_restore(q,i,l,c)`：在 corrupted run 中恢复组件 `c`（state/MLP/attn/head）后，首 token 为 `t_call` 的概率

### 2.1 Total Effect / Indirect Effect
- `TE(q) = P_clean(q) - P_corr(q)`
- `IE(q,i,l,c) = P_restore(q,i,l,c) - P_corr(q)`
- `ATE = mean_q[TE(q)]`
- `AIE(i,l,c) = mean_q[IE(q,i,l,c)]`

### 2.2 归一化指标（用于跨样本比较）
- `nIE(q,i,l,c) = IE(q,i,l,c) / max(TE(q), eps)`
- 报告均值时同时给 `AIE` 与 `mean(nIE)`。

### 2.3 统计规则
- 所有均值都报告 95% CI（bootstrap, sample 级重采样）。
- 图注必须注明：样本集合、是否做 clip/normalize、CI 计算方式。

---

## 3. 实验主线（照论文定位部分执行）

## 3.1 E0: 基线确认（必须先做）
- 在 `S_all` 和 `S_strict` 上统计：
  - clean/corrupt 首 token 命中率
  - `ATE`（基于概率差）
  - `TE` 分布（直方图）
- 通过标准：`ATE` 显著 > 0，且 `S_strict` 的 clean-corr gap 明显。

## 3.2 E1: Causal Tracing 主热力图（对应论文主图）
- 对每个样本 `q`：
  - 记录 clean 全部中间激活。
  - 在 corrupted 跑中，对每个 `(token i, layer l)` 恢复单点 hidden state，算 `IE`。
- 聚合得到 `AIE_state[token, layer]` 热图。
- 目标：观察是否出现类似论文的“早期关键位点 + 晚期关键位点”。

输出：
- `figs/ct_state_heatmap.png`（全状态）
- `figs/ct_mlp_heatmap.png`（MLP贡献）
- `figs/ct_attn_heatmap.png`（Attn贡献）

## 3.3 E2: MLP vs Attention 分解（照论文窗口恢复法）
- 按论文做法：对 MLP/Attn 用连续层窗口恢复（默认窗口 10 层，中心在 `l*`）。
- 计算 `AIE_mlp(token,l*)`、`AIE_attn(token,l*)`。
- 对比结论：触发词位置的中层 MLP 是否占主导，临近输出位置的 attn 是否增强。

## 3.4 E3: Modified Graph Intervention（照论文“切断路径”实验）
- 两组干预：
  - sever MLP：冻结后续 MLP 输出为 corrupted 基线值
  - sever Attn：冻结后续 attn 输出为 corrupted 基线值
- 在两种图上重复 E1/E2，比较 AIE 变化。
- 目标：验证“早期因果效应是否依赖中层 MLP 路径”。

输出：
- `figs/modified_graph_intervention.png`
- `reports/modified_graph_metrics.csv`

## 3.5 E4: Head 级别定位（项目特化，服务最终电路）
- 在 attn 模块内部做 head-level tracing/patching，计算每个 head 的因果贡献。
- 分别产出：
  - `figs/ct_head_heatmap.png`（Causal Tracing head 热图）
  - `figs/ap_head_heatmap.png`（Activation Patching head 热图）
- 从 top-k head + top-k MLP 构建候选电路子图。

## 3.6 E5: 单组件探针与路径验证
- 对入选组件逐个做 probe 图（例如 `L7H14`、`MLP12`）：
  - 单点恢复曲线（layer/token sweep）
  - 对 `P(y1=t_call)` 的提升分布
- 文件命名示例：`figs/L7H14_probe.png`

## 3.7 E6: 最终电路图
- 节点：关键 head 与关键 MLP。
- 边：基于路径补丁/条件恢复得到的有效信息通路。
- 输出：
  - `figs/final_circuit.png`
  - `reports/final_circuit_nodes.csv`
  - `reports/final_circuit_edges.csv`

---

## 4. 附录复现实验（定位部分）

## 4.1 噪声/腐化规则鲁棒性（对应论文附录）
- 除“自然 corrupted prompt”外，再做：
  - embedding 高斯噪声（主设定建议 `3*sigma_t`）
  - 协方差匹配高斯噪声
  - uniform 噪声（`[-3sigma, 3sigma]`）
- 检查结论是否稳定（AIE 峰位与符号是否一致）。

## 4.2 线图 + CI（对应论文 line plot）
- 把热图中关键 token 的 layer 曲线单独画出，并给 95% CI。
- 输出：
  - `figs/ct_lineplot_with_ci.png`

## 4.3 典型/异常个例（对应论文个例图）
- 选 5 个“典型成功”+ 5 个“异常模式”样本。
- 展示 token-layer 因果图 + 文本上下文，分析触发词是否为唯一决定因素。
- 输出：
  - `figs/case_trace_examples.png`

---

## 5. 输出图与论文图对应关系（定位部分）

- 论文 `info-flow-maps` / `avg-1000-traces`：
  - 本项目对应 `ct_state_heatmap.png` + `ct_mlp_heatmap.png` + `ct_attn_heatmap.png`
- 论文 `modified-graph-intervention`：
  - 本项目对应 `modified_graph_intervention.png`
- 论文 `appendix-lineplot-causal-trace`：
  - 本项目对应 `ct_lineplot_with_ci.png`
- 论文附录个例：
  - 本项目对应 `case_trace_examples.png`
- 项目新增（为电路落地）：
  - `ct_head_heatmap.png`
  - `ap_head_heatmap.png`
  - `L7H14_probe.png`
  - `final_circuit.png`

---

## 6. 规则（强制）

## 6.1 命名规范（R1，强制）
- 注意力头：`L{layer}H{head}`，例：`L7H14`
- MLP：`MLP{layer}`，例：`MLP12`
- 其他结点（如 residual）：`RESID_L{layer}`（如确实需要，电路中不出现）
- 文件命名（强制保留）：
  - `ap_head_heatmap.png`
  - `ct_head_heatmap.png`
  - `L7H14_probe.png`
  - `final_circuit.png`

## 6.2 指标命名规范（新增强制）
- 总效应：`TE`, `ATE`
- 间接效应：`IE`, `AIE`
- 归一化间接效应：`nIE`
- 概率指标统一前缀：`P_`，如 `P_clean`, `P_corr`, `P_restore`
- 任何报告/表格禁止混用同义缩写（如 `effect`, `score`）替代以上主指标名

## 6.3 颜色与对比（R2，强制）
- 所有热力图使用红/蓝发散色系（`RdBu` 或同类）。
- 0 值必须映射到中心浅色（白/近白）。
- 正负贡献必须一眼区分。
- 若做 clip 或归一化，必须在图注写明（例如 “values clipped to 1st-99th percentile”）。

## 6.4 图排版与风格（R3，强制）
- 字体统一；字号层级：标题 > 轴标签 > tick。
- 留白充足，不挤。
- 统一色条范围和单位，便于图间比较。
- 同类图保持相同坐标方向（layer 轴、token 轴一致）。

## 6.5 图注模板（强制）
每张图注至少包含：
- 样本集（`S_strict`/`S_all` 等）
- 指标（`AIE` / `nIE` / `TE`）
- 干预方式（restoration/sever-MLP/sever-Attn）
- 是否做 clip/normalize
- 统计方式（mean ± 95% CI）

---

## 7. 执行顺序（建议）

1. 单样本打通（`sample/q85`）：确保 hook/restore 和指标计算正确。  
2. 批量跑 `S_strict`：先出 `ct_state/mlp/attn` 三张主热图。  
3. 跑 modified graph 干预：验证“中层 MLP 路径”结论。  
4. 跑 head-level：产出 `ct_head_heatmap.png` 与 `ap_head_heatmap.png`。  
5. 组件探针与最终电路图：`L7H14_probe.png`、`final_circuit.png`。  

---

## 8. 完成标准（Definition of Done）

- 主实验图、附录关键图、项目新增电路图全部生成并可复现。
- 至少复现出以下定性结论之一：
  - 存在稳定的“关键触发位点”（token-layer 局部峰值）。
  - MLP 与 Attn 在不同位置承担不同作用，且可被改图干预区分。
  - head 级别存在可组合成稳定决策电路的子集。
- 全部命名、指标、画图规则满足第 6 节强制规范。
