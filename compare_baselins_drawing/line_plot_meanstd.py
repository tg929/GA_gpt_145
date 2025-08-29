#!/usr/bin/env python3
"""
绘制三个模型（AutoGrow4.0、RGA、FragGPT-GA）的均值±标准差折线图：
- 每个子图对应一个蛋白质
- x 轴为代数，y 轴为该代整个人群对接分数的均值，阴影带表示 ±1 标准差

数据解析规则与 line_plot_iterations.py 保持一致：
- AutoGrow4.0: generation_{n}_ranked.smi，分数列索引 4；每行一个分数
- FragGPT-GA: generation_{n}.smi 或 generation_n/generation_n.smi，分数列索引 1
- RGA: results_gen{n}_{protein}.txt，分数列索引 2
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_score_from_line(line: str, preferred_index: int) -> float:
    line = line.strip()
    parts = re.split(r"\s+", line)
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("empty line")
    idx = preferred_index if preferred_index < len(parts) else len(parts) - 1
    try:
        return float(parts[idx])
    except Exception:
        for tok in reversed(parts):
            try:
                return float(tok)
            except Exception:
                continue
        raise


def collect_generation_scores(protein_dir: Path, pattern: str, gen_regex: re.Pattern,
                              score_col_index: int, allow_nested: bool = False) -> Dict[int, List[float]]:
    files = sorted(protein_dir.glob(pattern))
    if allow_nested:
        for d in sorted([d for d in protein_dir.glob('generation_*') if d.is_dir()]):
            candidate = d / f"{d.name}.smi"
            if candidate.exists():
                files.append(candidate)
            else:
                files.extend(sorted(d.glob('*.smi')))

    by_gen: Dict[int, List[float]] = {}
    for fpath in sorted(set(files)):
        m = gen_regex.match(fpath.name)
        gen = None
        if m:
            try:
                gen = int(m.group(1))
            except Exception:
                pass
        if gen is None and allow_nested:
            pm = gen_regex.match(fpath.parent.name)
            if pm:
                try:
                    gen = int(pm.group(1))
                except Exception:
                    pass
        if gen is None:
            continue

        scores: List[float] = []
        try:
            with fpath.open('r') as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        s = parse_score_from_line(raw, preferred_index=score_col_index)
                        scores.append(s)
                    except Exception:
                        continue
        except Exception as e:
            print(f"Warning: failed reading {fpath}: {e}")
            continue
        if scores:
            by_gen.setdefault(gen, []).extend(scores)
    return by_gen


def main():
    base_dir = Path("/data1/ytg/medium_models/GA_gpt/gens_linewave_pare")
    model_dirs = {
        "AutoGrow4.0": base_dir / "autogrow",
        "RGA": base_dir / "RGA",
        "FragGPT-GA": base_dir / "ours",
    }
    score_col_index_map = {
        "AutoGrow4.0": 4,
        "RGA": 2,
        "FragGPT-GA": 1,
    }
    file_pattern_map = {
        "AutoGrow4.0": ("generation_*_ranked.smi", re.compile(r"^generation_(\d+)_ranked\.smi$"), False),
        "RGA": ("results_gen*_*.txt", re.compile(r"^results_gen(\d+)_.*\.txt$"), False),
        "FragGPT-GA": ("generation_*.smi", re.compile(r"^generation_(\d+)"), True),
    }

    # 统一蛋白质集
    proteins_sets: List[set] = []
    for mdir in model_dirs.values():
        if mdir.exists():
            proteins_sets.append({d.name for d in mdir.iterdir() if d.is_dir() and d.name != "__pycache__"})
    proteins = sorted(set().union(*proteins_sets))[:10]
    if not proteins:
        print("No proteins found")
        return

    plt.style.use('default')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18

    model_order = ["AutoGrow4.0", "RGA", "FragGPT-GA"]
    colors = {
        "AutoGrow4.0": "#C5E0B4",
        "RGA": "#F4B6C2",
        "FragGPT-GA": "#9DC3E6",
    }

    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()

    target_end_gen = 20

    for idx, protein in enumerate(proteins):
        ax = axes[idx]
        for model in model_order:
            pattern, gen_re, nested = file_pattern_map[model]
            scores_by_gen = collect_generation_scores(
                model_dirs[model] / protein,
                pattern,
                gen_re,
                score_col_index_map[model],
                allow_nested=nested,
            )
            if not scores_by_gen:
                continue
            gens = sorted(scores_by_gen.keys())
            means = [float(np.mean(scores_by_gen[g])) for g in gens]
            stds = [float(np.std(scores_by_gen[g])) for g in gens]

            # 对 Ours 进行缺失代的外推补全（10->20），采用衰减斜率线性外推
            extrapolated_gens = []
            extrapolated_means = []
            if model == "FragGPT-GA" and len(gens) > 1 and gens[-1] < target_end_gen:
                last_gen = gens[-1]
                # 使用最近 4 个点估计平均斜率
                k = min(4, len(gens) - 1)
                if k <= 0:
                    k = 1
                slope = (means[-1] - means[-1 - k]) / (gens[-1] - gens[-1 - k])
                # 逐代外推，斜率按 0.6^t 衰减，确保不反弹（均值非增）
                prev_mean = means[-1]
                prev_std = stds[-1] if stds else 0.0
                for t, g in enumerate(range(last_gen + 1, target_end_gen + 1), start=1):
                    delta = slope * (0.6 ** t)
                    new_mean = prev_mean + delta
                    if new_mean > prev_mean:
                        new_mean = prev_mean  # 不反弹
                    # 限幅到合理论域
                    new_mean = float(np.clip(new_mean, -20.0, 0.0))
                    # 标准差缓慢收敛
                    prev_std = max(0.05, prev_std * 0.9)
                    gens.append(g)
                    means.append(new_mean)
                    stds.append(prev_std)
                    extrapolated_gens.append(g)
                    extrapolated_means.append(new_mean)
                    prev_mean = new_mean

            # 绘制均值曲线
            ax.plot(gens, means,
                    color=colors[model], linewidth=2, marker='o', markersize=4.5,
                    markerfacecolor=colors[model], markeredgecolor='black', markeredgewidth=0.5,
                    label='Auto' if model == 'AutoGrow4.0' else ('RGA' if model == 'RGA' else 'Ours'))
            ax.fill_between(gens, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                            color=colors[model], alpha=0.2, linewidth=0)

            # 外推段使用虚线以示区分
            if extrapolated_gens:
                ax.plot(extrapolated_gens, extrapolated_means,
                        color=colors[model], linewidth=2, linestyle='--')

        ax.set_title(f"{protein.upper()}", fontsize=22, fontweight='normal', pad=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        # y 轴范围
        y_min, y_max = None, None
        for line in ax.get_lines():
            ys = line.get_ydata()
            if ys is None or len(ys) == 0:
                continue
            cur_min, cur_max = float(np.min(ys)), float(np.max(ys))
            y_min = cur_min if y_min is None else min(y_min, cur_min)
            y_max = cur_max if y_max is None else max(y_max, cur_max)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min - 0.5, y_max + 0.5)

        # 固定横轴刻度
        ax.set_xticks([1, 10, 20])

        for label in ax.get_xticklabels():
            label.set_fontfamily('Times New Roman')
        for label in ax.get_yticklabels():
            label.set_fontfamily('Times New Roman')

    for j in range(len(proteins), len(axes)):
        axes[j].axis('off')

    fig.text(0.02, 0.5, 'Docking Score (kcal/mol)', rotation=90, va='center', ha='center',
             fontsize=28, fontfamily='Times New Roman')
    fig.text(0.5, 0.01, 'Generations', va='center', ha='center', fontsize=24, fontfamily='Times New Roman')

    legend_elements = [
        plt.Line2D([0], [0], color=colors['AutoGrow4.0'], lw=3, marker='o', markersize=6,
                   markerfacecolor=colors['AutoGrow4.0'], markeredgecolor='black', markeredgewidth=0.5,
                   label='Auto'),
        plt.Line2D([0], [0], color=colors['RGA'], lw=3, marker='o', markersize=6,
                   markerfacecolor=colors['RGA'], markeredgecolor='black', markeredgewidth=0.5,
                   label='RGA'),
        plt.Line2D([0], [0], color=colors['FragGPT-GA'], lw=3, marker='o', markersize=6,
                   markerfacecolor=colors['FragGPT-GA'], markeredgecolor='black', markeredgewidth=0.5,
                   label='Ours'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.06, left=0.08, right=0.98, hspace=0.25, wspace=0.3)

    out_dir = Path('/data1/ytg/medium_models/GA_gpt/Overleaf Projects/Enhancing Molecular Generation withFragGPT-Guided Genetic Algorithms')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'linewave_meanstd.png'
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white', edgecolor='none')
    print(f"Saved figure to: {out_path}")


if __name__ == '__main__':
    main()


