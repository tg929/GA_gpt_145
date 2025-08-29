#!/usr/bin/env python3
"""
绘制两个模型（AutoGrow4.0 与 FragGPT-GA/Ours）在10种蛋白质上的迭代折线图：
- 每个子图对应一个蛋白质
- x 轴为代数（generation_n），y 轴为该代的 TOP-1（最优，数值最小）对接分数

数据目录结构（示例）：
  base_dir/
    autogrow/
      1iep/
        generation_0_ranked.smi
        generation_1_ranked.smi
        ...
    ours/
      1iep/
        generation_0.smi
        generation_2.smi
        ...

注意：
- autogrow 的 .smi 文件名为 generation_{n}_ranked.smi，分数字段位于第 5 列（索引 4），若列不足则取最后一列作为分数。
- ours 的 .smi 文件名为 generation_{n}.smi，分数字段位于第 2 列（索引 1），若列不足则取最后一列作为分数。
- generation_0 表示初始种群。

绘图风格参考 violin_plot_comparison.py 与论文风格（Times New Roman 字体等）。
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # 仅用于一致的风格设定；不强制使用其绘图 API


def parse_score_from_line(line: str, preferred_index: int) -> float:
    """
    从一行 .smi 记录中解析对接分数。
    优先使用 preferred_index 索引的列；若不足则回退到最后一列。
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line")
    # 使用任意空白分割，兼容空格/制表符
    parts = re.split(r"\s+", line)
    # 有些文件可能出现多余的空字段，过滤
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("No tokens parsed")
    idx = preferred_index if preferred_index < len(parts) else len(parts) - 1
    try:
        return float(parts[idx])
    except ValueError:
        # 回退：尝试从末尾向前寻找第一个可解析为 float 的字段
        for token in reversed(parts):
            try:
                return float(token)
            except ValueError:
                continue
        raise


def collect_generation_best_scores(protein_dir: Path, pattern: str, gen_regex: re.Pattern,
                                   score_col_index: int,
                                   allow_nested: bool = False) -> List[Tuple[int, float]]:
    """
    收集某个蛋白质目录下每一代的 TOP-1（最小）对接分数。

    Args:
        protein_dir: 该蛋白质的目录 Path
        pattern: 匹配文件的 glob 模式（例如 'generation_*_ranked.smi' 或 'generation_*.smi'）
        gen_regex: 用于从文件名中解析代数的正则（需包含一个捕获组返回代数）
        score_col_index: 期望的分数列索引（不足时回退到最后一列）

    Returns:
        List[(generation, best_score)]，按 generation 升序排列
    """
    results: List[Tuple[int, float]] = []
    files: List[Path] = sorted(protein_dir.glob(pattern))

    # 对于允许嵌套的情况（例如 ours: generation_n/generation_n.smi）
    if allow_nested:
        nested_dirs = sorted([d for d in protein_dir.glob('generation_*') if d.is_dir()])
        for d in nested_dirs:
            # 优先匹配与目录同名的 .smi（如 generation_10/generation_10.smi）
            candidate = d / f"{d.name}.smi"
            if candidate.exists():
                files.append(candidate)
            else:
                # 次选：该目录下任意 .smi（取一个）
                smi_files = sorted(d.glob('*.smi'))
                files.extend(smi_files)

    # 去重（同一路径可能被重复加入）
    unique_files = []
    seen = set()
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    for fpath in sorted(unique_files):
        m = gen_regex.match(fpath.name)
        if not m:
            # 若是嵌套路径，尝试从父目录名中解析
            if allow_nested and gen_regex.match(fpath.parent.name):
                try:
                    gen = int(gen_regex.match(fpath.parent.name).group(1))
                except Exception:
                    continue
            else:
                continue
        else:
            try:
                gen = int(m.group(1))
            except Exception:
                continue
        best_score = None
        try:
            with fpath.open('r') as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        score = parse_score_from_line(raw, preferred_index=score_col_index)
                    except Exception:
                        continue
                    if best_score is None or score < best_score:
                        best_score = score
        except Exception as e:
            print(f"Warning: failed reading {fpath}: {e}")
            continue
        if best_score is not None:
            results.append((gen, best_score))
    # 去重（若有重复代文件，以最小分数保留）
    agg: Dict[int, float] = {}
    for g, s in results:
        if g not in agg or s < agg[g]:
            agg[g] = s
    out = sorted(agg.items(), key=lambda x: x[0])
    return out


def main():
    base_dir = Path("/data1/ytg/medium_models/GA_gpt/gens_linewave_pare")
    model_dirs = {
        "AutoGrow4.0": base_dir / "autogrow",
        "FragGPT-GA": base_dir / "ours",
        "RGA": base_dir / "RGA",
    }

    # 期望的分数列索引（与 violin_plot_comparison.py 中保持一致）
    score_col_index_map = {
        "AutoGrow4.0": 4,  # 第5列
        "FragGPT-GA": 1,   # 第2列
        "RGA": 2,          # 第3列（预览结果文件：SMILES, ID, SCORE, [files]）
    }

    # 文件名与正则
    file_pattern_map = {
        "AutoGrow4.0": ("generation_*_ranked.smi", re.compile(r"^generation_(\d+)_ranked\.smi$"), False),
        "FragGPT-GA": ("generation_*.smi", re.compile(r"^generation_(\d+)"), True),
        "RGA": ("results_gen*_*.txt", re.compile(r"^results_gen(\d+)_.*\.txt$"), False),
    }

    # 统一的蛋白质列表（取两个模型目录交集，确保每个子图都有两条曲线或至少一条）
    proteins_sets: List[set] = []
    for model, mdir in model_dirs.items():
        if not mdir.exists():
            print(f"Warning: model dir not found: {mdir}")
            proteins_sets.append(set())
            continue
        subdirs = [d.name for d in sorted(mdir.iterdir()) if d.is_dir() and d.name != "__pycache__"]
        proteins_sets.append(set(subdirs))
    # 优先用并集，避免丢失任一模型的蛋白质；排序用于固定子图顺序
    proteins = sorted(set().union(*proteins_sets))
    if not proteins:
        print("No protein directories found under models.")
        sys.exit(1)

    # 设定绘图风格
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18

    # 颜色与模型顺序（与小提琴图保持视觉一致：Auto 绿色，Ours 蓝色）
    model_order = ["AutoGrow4.0", "RGA", "FragGPT-GA"]
    line_colors = {
        "AutoGrow4.0": "#C5E0B4",  # 浅绿（与小提琴图一致）
        "RGA": "#F4B6C2",         # 粉色（与论文描述一致）
        "FragGPT-GA": "#9DC3E6",   # 浅蓝（与小提琴图/论文一致）
    }

    # 创建 2x5 子图布局
    rows, cols = 2, 5
    total = rows * cols
    # 只取前 10 个蛋白质（若有更多）
    proteins = proteins[:total]

    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()

    # 收集并绘制
    for idx, protein in enumerate(proteins):
        ax = axes[idx]
        # 每个模型的代-最优分数
        for model in model_order:
            mdir = model_dirs[model]
            pattern, gen_re, nested = file_pattern_map[model]
            score_idx = score_col_index_map[model]
            pdir = mdir / protein
            if not pdir.exists():
                print(f"Warning: missing protein dir for {model}: {pdir}")
                continue
            gen_scores = collect_generation_best_scores(pdir, pattern, gen_re, score_idx, allow_nested=nested)
            if not gen_scores:
                print(f"Warning: no generation scores for {model}/{protein}")
                continue
            gens = [g for g, _ in gen_scores]
            scores = [s for _, s in gen_scores]

            ax.plot(
                gens,
                scores,
                marker='o',
                markersize=4.5,
                linewidth=2,
                color=line_colors[model],
                markerfacecolor=line_colors[model],
                markeredgecolor='black',
                markeredgewidth=0.5,
                label='Auto' if model == 'AutoGrow4.0' else 'Ours',
            )

        # 标题与轴设置
        ax.set_title(f"{protein.upper()}", fontsize=22, fontweight='normal', pad=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        # y 轴范围：根据当前子图已有曲线动态设定，留出边距
        y_min, y_max = None, None
        lines = ax.get_lines()
        for line in lines:
            ys = line.get_ydata()
            if ys is None or len(ys) == 0:
                continue
            cur_min, cur_max = float(np.min(ys)), float(np.max(ys))
            y_min = cur_min if y_min is None else min(y_min, cur_min)
            y_max = cur_max if y_max is None else max(y_max, cur_max)
        if y_min is not None and y_max is not None:
            pad = 0.5
            ax.set_ylim(y_min - pad, y_max + pad)

        # 固定横轴刻度为 1、10、20
        try:
            ax.set_xticks([1, 10, 20])
        except Exception:
            pass

        # 字体
        for label in ax.get_xticklabels():
            label.set_fontfamily('Times New Roman')
        for label in ax.get_yticklabels():
            label.set_fontfamily('Times New Roman')

        # 右上角 TOP1（全代最优）注释
        stats_text = []
        label_map = {"AutoGrow4.0": "Auto", "RGA": "RGA", "FragGPT-GA": "Ours"}
        for model in model_order:
            # 找到对应曲线的数据
            for line in ax.get_lines():
                if line.get_label() == label_map[model]:
                    ys = line.get_ydata()
                    if ys is not None and len(ys) > 0:
                        stats_text.append(f"{label_map[model]}: {np.min(ys):.1f}")
        if stats_text:
            ax.text(
                0.98,
                0.98,
                "\n".join(stats_text),
                transform=ax.transAxes,
                va='top',
                ha='right',
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            )

    # 关闭多余子图（如果蛋白质不足 10）
    for j in range(len(proteins), len(axes)):
        axes[j].axis('off')

    # 全局 y 轴标签
    fig.text(
        0.02,
        0.5,
        'Docking Score (kcal/mol)',
        rotation=90,
        va='center',
        ha='center',
        fontsize=28,
        fontfamily='Times New Roman',
    )

    # 全局 x 轴标签
    fig.text(
        0.5,
        0.01,
        'Generations',
        va='center',
        ha='center',
        fontsize=24,
        fontfamily='Times New Roman',
    )

    # 图例（全局）
    legend_elements = [
        plt.Line2D([0], [0], color=line_colors['AutoGrow4.0'], lw=3, marker='o', markersize=6,
                   markerfacecolor=line_colors['AutoGrow4.0'], markeredgecolor='black', markeredgewidth=0.5,
                   label='Auto'),
        plt.Line2D([0], [0], color=line_colors['RGA'], lw=3, marker='o', markersize=6,
                   markerfacecolor=line_colors['RGA'], markeredgecolor='black', markeredgewidth=0.5,
                   label='RGA'),
        plt.Line2D([0], [0], color=line_colors['FragGPT-GA'], lw=3, marker='o', markersize=6,
                   markerfacecolor=line_colors['FragGPT-GA'], markeredgecolor='black', markeredgewidth=0.5,
                   label='Ours'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.017),
        ncol=3,
        fontsize=20,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.06, left=0.06, right=0.98, hspace=0.25, wspace=0.3)

    # 保存图片
    out_dir = Path('/data1/ytg/medium_models/GA_gpt/Overleaf Projects/Enhancing Molecular Generation withFragGPT-Guided Genetic Algorithms')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'linewave_iterations.png'
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white', edgecolor='none')
    print(f"Saved figure to: {out_path}")

    # 交互环境下可显示
    # plt.show()


if __name__ == "__main__":
    main()


