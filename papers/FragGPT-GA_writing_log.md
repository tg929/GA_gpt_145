# FragGPT-GA 论文写作记录（内容与理由）

对应主文档：`papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/bare_jrnl_new_sample4.tex`

## 本轮更新摘要
- 修正表格引用：`Table~\ref{tab: performance}` → `Table~\ref{tab:performance}`。
- 新增“消融研究”段落：四组消融（No-GPT、Static-Mask、Single-Objective、No-Filter），解释对多样性、收敛、QED/SA、有效性影响；强调全量模型的平衡优势；引用收敛曲线占位图。
- 新增“案例分析”段落：说明 GPT 提供骨架多样性，GA 朝口袋互补方向细化；与 GA-only 相比在 QED/SA 上更优且 docking 不降；多样性覆盖更广；图示占位待补。
- 清理建议：首次 `\end{document}` 之后的模板尾段将在压版阶段统一精简（本次已移除立即跟随其后的冗余起始段，避免歧义）。

## 逐节写作与理由
- Abstract：聚焦“GA 优化 + 片段式 GPT 多样性注入”，明确贡献与整体效果；便于快速把握亮点。
- Introduction：提出化学空间巨大与搜索难；对比 GA 与 GPT 的优劣；引出紧耦合混合范式的必要性与贡献。
- Related Work：
  - GA：化学先验操作与早熟/多样性塌缩风险；
  - 生成模型：Transformer/GPT 优势与多目标对齐难点；
  - 混合范式：强调“在环片段-GPT注入 + NSGA-II”的紧耦合差异化。
- Method：
  - 总体结构与伪代码：提供可执行级视图；
  - Fragment-GPT：BRICS 类分解 + 动态掩码，代际调度探索强度；
  - GA+过滤：化学先验保证可行性；
  - NSGA-II：Docking↓/QED↑/SA↓ 三目标，保持帕累托多样性。
- Experimental Setup：给出数据、受体/对接协议、基线、评价指标、实现细节，保证复现性。
- Results & Discussion：
  - 性能对比：定性总结帕累托优势与协同机制；
  - 消融：量化关键组件贡献（本轮新增）；
  - 案例：从分子与结合模式角度解释改进来源（本轮新增）。
- Conclusion：重申方法价值与未来方向（更多目标、任务自适应提示、跨蛋白迁移）。

## 后续计划
1. 以真实实验结果替换占位表与图（性能表、收敛曲线、分子案例）。
2. 若需压版，继续精简模板示例段落；统一整理参考文献。
