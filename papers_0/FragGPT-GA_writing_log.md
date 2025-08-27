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

## 表格优化与消融实验扩展

### Table II & Table III 拆分 (最新实施)
- **改动内容**: 将原来的并列子表拆分为两个完全独立的表格
  - Table II: 对接分数比较 (`tab:docking_scores`)，包含TOP-100、TOP-10、TOP-1指标
  - Table III: 多样性和药物相似性指标 (`tab:diversity_metrics`)，包含Nov、Div、QED、SA指标
- **原因**: 用户要求分别命名为Table 2和Table 3，更符合学术规范且便于引用

### 消融实验：选择策略比较 (新增)
- **新增内容**: 在Ablation Studies部分添加了选择策略对比子节
- **包含表格**: Table IV (`tab:selection_ablation`) - 消融实验：选择策略比较
- **三种策略及公式**:
  1. **单目标选择**: $S_{\text{single}}(m) = -\text{DockingScore}(m)$
  2. **多目标选择**: NSGA-II with $\mathbf{f}(m) = [-\text{DockingScore}(m), \text{QED}(m), -\text{SA}(m)]^T$
  3. **综合评分**: $S_{\text{comp}}(m) = \alpha \cdot \frac{|\text{DockingScore}(m)|}{|\text{DS}_{\max}|} + \beta \cdot \text{QED}(m) + \gamma \cdot \frac{\text{SA}_{\max} - \text{SA}(m)}{\text{SA}_{\max}}$

- **评估指标表示**: 全文统一使用$S(\cdot)$符号表示各评估指标值
- **实验数据**: 采用用户提供的真实实验数据，包含三种策略的性能对比
- **分析重点**: 
  - 单目标选择：对接分数好但药物相似性差 ($S(\text{QED}) = 0.436$, $S(\text{SA}) = 3.145$)
  - 多目标选择：最平衡的性能 ($S(\text{QED}) = 0.764$, $S(\text{SA}) = 2.014$)
  - 综合评分：中等性能 ($S(\text{QED}) = 0.579$, $S(\text{SA}) = 2.645$)

## 后续计划
1. 以真实实验结果替换剩余占位表与图（收敛曲线、分子案例）。
2. 若需压版，继续精简模板示例段落；统一整理参考文献。
