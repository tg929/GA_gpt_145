# FragEvo：FragMLM（片段GPT）+ GA + Selection 的混合分子进化框架

> 中文版文档；English version: `README.md`

本项目实现了一个面向药物分子优化的“生成式模型 + 遗传算法（GA）+ 对接评估 + 选择策略”的端到端工作流。核心目标是在给定受体（蛋白）约束下，通过迭代进化不断产生并筛选更优分子。

---

## 1. 整体思路（你关心的 3 个关键模块）

### 1.1 FragMLM / Fragment GPT（`fragmlm/`）
**作用：提供“跳出局部最优”的新候选**。流程中先把父代分子分解为片段序列，再对末尾片段做掩码，仅保留前缀片段作为条件，让 GPT 在片段级别“续写”，最后把片段重构为完整分子 SMILES。

- 片段分解与掩码：`datasets/decompose/demo_frags.py`
  - 典型输出格式（每行一个条件序列）：`[BOS]frag1[SEP]frag2[SEP]... [SEP]`
  - 支持**动态掩码**：早期代掩码更多片段（探索），后期代掩码更少片段（利用/精修）
- GPT 批量生成入口：`fragmlm/generate_all.py`
  - 默认加载权重：`fragmlm/weights/dpo_0_400.pt`
  - 输出：`*.smi`（每行一个生成的 SMILES）

### 1.2 GA（Genetic Algorithm，`operations/`）
**作用：基于现有分子进行“可控局部搜索”**。本项目使用 AutoGrow 风格算子实现交叉/突变，并可选过滤。

- 交叉：`operations/crossover/crossover_demo_finetune.py`
- 突变：`operations/mutation/mutation_demo_finetune.py`
- 过滤：`operations/filter/filter_demo.py`
- 评估（对接）：`operations/docking/docking_demo_finetune.py`

### 1.3 Selection（选择策略，`operations/selecting/`）
**作用：从“父代 + 子代（含 GPT/GA 产物）”的合并池中选出下一代父代**（适者生存）。

支持三类选择：

1) **单目标（只看 docking score）**：`operations/selecting/molecular_selection.py`  
   - Rank / Roulette / Tournament 三种 selector  
2) **多目标（NSGA-II）**：`operations/selecting/selecting_multi_demo.py`  
   - 目标默认：Docking 最小化、QED 最大化、SA 最小化  
   - 指标计算带缓存：`utils/chem_metrics.py`（缓存文件默认写到运行目录根部）  
3) **RAG-score（组合指标）**：`operations/selecting/selecting_rag_score.py`  
   - 评分函数：`y = DS_hat * QED * SA_hat`（细节见脚本注释）  
   - 对应工作流入口：`FragEvo_rag.py`

---

## 2. 端到端工作流（项目流程）

项目内置两条主流程：

### 2.1 纯 GA 流程（baseline）
- 主入口：`fragevo/GA_main.py`
- 核心执行器：`operations/operations_execute_demo.py`

每代流程（概念）：
1. 初始种群去重 + 对接（Gen0）
2. 从父代提取 SMILES
3. 交叉 + 过滤
4. 突变 + 过滤
5. 子代对接评估
6. Selection：从（父代+子代）合并池中选出下一代父代
7. 对下一代父代做一次评估汇总（Top1/Top10/Top100/Novelty/Diversity/QED/SA）

### 2.2 FragEvo 混合流程（FragMLM + GA）
- 主入口：`FragEvo_main.py`
- 核心执行器：`operations/operations_execute_fragevo_demo.py`

与 GA baseline 的差异：每代在 GA 之前插入 **“分解+掩码 → GPT 生成”**，并把 GPT 产物加入 GA 输入池：

```
父代(纯SMILES)
  └─ 分解&掩码（datasets/decompose/demo_frags.py）
      └─ GPT生成（fragmlm/generate_all.py）
          └─ 合并池 = 父代 + GPT
              └─ 交叉/突变 → 对接 → 选择 → 下一代
```

---

## 3. 代码结构（快速定位）

建议从这几处读代码即可理解全链路：

- 工作流入口
  - `FragEvo_main.py`：混合流程入口（支持多受体并行/串行）
  - `fragevo/GA_main.py`：纯 GA 流程入口
  - `FragEvo_rag.py`：RAG-score 选择策略入口（只替换 selection 阶段）
- 三大模块
  - FragMLM：`fragmlm/generate_all.py`
  - GA：`operations/crossover/*`, `operations/mutation/*`, `operations/filter/*`
  - Selection：`operations/selecting/*`
- 对接评估
  - `operations/docking/docking_demo_finetune.py`
  - 受体/盒子参数：`fragevo/*.json` 的 `receptors` 块
- 指标评估（生成每代 `generation_k_evaluation.txt`）
  - `operations/scoring/scoring_demo.py`
  - 汇总多受体统计：`operations/stating/statistics_output_demo.py`

---

## 4. 环境与依赖

### 4.1 Python 环境（推荐 Conda）
项目提供 Conda 环境文件：`fragevo.yml`

```bash
conda env create -f fragevo.yml
conda activate fragevo
```

建议额外确认这些包可用（部分系统环境可能未自动带上）：
```bash
pip install -U psutil tqdm openpyxl
```

### 4.2 对接工具链
对接模块依赖（仓库内已包含/引用）：
- MGLTools：`mgltools_x86_64Linux2_1.5.6/`
- AutoDock Vina（或 QVina2）：`autogrow/docking/docking_executables/...`
- OpenBabel（Conda 安装）

如遇到 “Permission denied / Exec format error”，通常需要确保可执行权限，例如：
```bash
chmod +x autogrow/docking/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina
```

### 4.3 GPU（可选）
FragMLM 生成支持 GPU；如果不满足 CUDA 环境将自动回退 CPU（会慢很多）。

当前工作流默认不显式传 `--device` 给 `fragmlm/generate_all.py`，最简单的 GPU 选择方式是：
```bash
export CUDA_VISIBLE_DEVICES=0
```

---

## 5. 数据与配置

### 5.1 初始种群（SMILES）
格式：每行一个 SMILES（可带额外列，但第一列必须是 SMILES）。

仓库内示例：
- `datasets/initial_population/my_initial_population.smi`
- `datasets/source_compounds/naphthalene_smiles.smi`

默认情况下，`fragevo/config_example.json` 与 `fragevo/config_fragevo.json` 都指向 `datasets/initial_population/my_initial_population.smi`。如需替换初始种群，请修改配置中的 `workflow.initial_population_file`。

### 5.2 受体与对接盒子
受体配置在 JSON 的 `receptors` 块：
- `default_receptor`：不传 `--receptor` 时使用
- `target_list`：`--all_receptors` 会遍历其 key

每个受体需要：
- `file`：受体 PDB/PDBQT 路径（相对项目根目录）
- `center_x/y/z` 与 `size_x/y/z`：对接盒子参数

---

## 6. 运行方式（可复现命令）

下面命令都假设你在项目根目录执行。

### 6.1 纯 GA baseline

1) 单受体（默认受体）
```bash
python fragevo/GA_main.py --config fragevo/config_example.json --output_dir GA_output_demo
```

2) 指定受体（受体名需在 `config_example.json -> receptors.target_list` 中）
```bash
python fragevo/GA_main.py --config fragevo/config_example.json --receptor 4r6e --output_dir GA_output_demo
```

3) 跑所有受体（`target_list`）
```bash
python fragevo/GA_main.py --config fragevo/config_example.json --all_receptors --output_dir GA_output_all
```

### 6.2 FragEvo 混合流程（FragMLM + GA）

1) 单受体（默认受体/或用 `--receptor` 指定）
```bash
python FragEvo_main.py --config fragevo/config_fragevo.json --receptor parp1 --output_dir FragEvo_output_demo
```

2) 跑所有受体（可在配置 `performance` 中控制是否并行）
```bash
python FragEvo_main.py --config fragevo/config_fragevo.json --all_receptors --output_dir FragEvo_output_all
```

### 6.3 RAG-score 选择策略（可选）
入口：`FragEvo_rag.py`  
说明：该入口会复用 FragEvo 流程，仅替换 Selection 阶段为 `operations/selecting/selecting_rag_score.py`。

`fragevo/config_fragevo_rag.json` 当前包含绝对路径（如 `/data1/...`），对外复现前请先改成相对路径或你的本地路径。

```bash
python FragEvo_rag.py --config fragevo/config_fragevo_rag.json --receptor parp1 --output_dir FragEvo_output_rag
```

---

## 7. 输出结果说明

以 `--output_dir FragEvo_output_demo`、`--receptor parp1` 为例，输出目录结构大致为：

```
FragEvo_output_demo/
  parp1/
    execution_config_snapshot.json
    chem_metric_cache.json                  # 多目标选择用（QED/SA缓存）
    generation_0/
      initial_population_docked.smi
    generation_1/
      current_parent_smiles.smi
      masked_fragments.smi
      gpt_generated/gpt_generated_molecules.smi
      crossover_filtered.smi
      mutation_filtered.smi
      offspring_docked.smi
      generation_1_evaluation.txt
    ...
```

其中你在 IDE 打开的 `generation_10_evaluation.txt` 即由：
`operations/scoring/scoring_demo.py` 生成，包含：
- Docking Score：Top1 / Top10 mean / Top100 mean
- Novelty：相对初始种群的新颖性
- Diversity：Top100 多样性
- QED / SA：Top100 平均值

---

## 8. 结果汇总（多受体/多代）

对一个包含多个受体输出的目录（例如 `FragEvo_output_all/`），可用统计脚本汇总为 Excel：

```bash
python operations/stating/statistics_output_demo.py --output_dir FragEvo_output_all --excel_output all_statistics.xlsx
```

---

## 9. 复现建议（强烈推荐）

1) **先跑小规模 smoke test**：把配置里的 `max_generations`、`number_of_crossovers`、`number_of_mutants`、`n_select` 调小，确认整条链路（生成→对接→选择→评估）可跑通。  
2) **对接是最慢的部分**：优先调低 `docking_exhaustiveness`、`docking_num_modes` 做快速验证。  
3) **并行策略**：
   - 受体间并行：`performance.parallel_processing` + `max_workers`
   - 受体内并行（对接等）：`performance.number_of_processors`（`-1` 表示自动）

---

## 10. 常见问题排查

- **对接报错/无输出**：优先检查受体文件路径与盒子参数是否正确；再检查 Vina/QVina2 可执行文件权限。  
- **MGLTools 相关报错**：确认 `mgltools_x86_64Linux2_1.5.6/bin/pythonsh` 存在且可执行。  
- **GA 运行失败（初始种群找不到）**：检查 `workflow.initial_population_file` 指向的 `.smi` 是否存在。  
- **FragMLM 生成速度慢**：确认 PyTorch CUDA 可用，或通过 `CUDA_VISIBLE_DEVICES` 指定 GPU。  
