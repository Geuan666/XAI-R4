# AGENT GUIDE (Tool-Call Circuit Project)

## 1. Project Goal
- 目标：定位模型内部“是否输出 `<tool_call>` 作为首 token”的决策机制，找到关键模块与电路路径。
- 范围：只做定位与寻找（causal tracing / activation patching / circuit discovery）。
- 非目标：不做参数编辑（ROME rewrite）、不做编辑后应用评测。

## 2. Task Definition
- 对每个样本 `q`，有一对提示词：
  - `pair/prompt-clean-q{q}.txt`（期望首 token 为 `<tool_call>`）
  - `pair/prompt-corrupted-q{q}.txt`（期望首 token 非 `<tool_call>`）
- 关注唯一主指标：`P(y1 = "<tool_call>")`。
- 主分析集合优先使用：`clean_top1=<tool_call> 且 corr_top1!=<tool_call>` 的样本（当前 120 条）。

## 3. Key Folders
- `pair/`: 主数据集（clean/corrupt prompts + `meta-q*.json` 对齐信息 + 首 token统计CSV）。
- `sample/`: 单样本调试入口（推荐 `q85` 先打通整条实验链）。
- `src/`: 实验脚本与分析代码（hook、tracing、patching、统计）。
- `figs/`: 所有论文风格图像输出目录。
- `reports/`: 表格、日志、结论汇总（CSV/MD）。
- `Locating and Editing Factual Associations in GPT/`: ROME 论文源码与参考图（用于对齐实验与作图风格）。

## 4. Environment and Compute
- 模型：`/root/data/Qwen/Qwen3-1.7B`
- Python 环境：`base`。
- 计算优先级：优先使用 GPU（4090 24G）。
- 若显存不足：不要降级关键实验到低质量设置；等待资源释放后继续运行。
- 绘图风格与论文中图像一致
- 只在你的工作目录下读写，不要关注其他代码