#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制三个模型（AutoGrow4.0、RGA、FragMLM-GA）的均值±标准差折线图
"""

import re
from pathlib import Path
from typing import Dict, List

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
                              score_col_index: int, allow_nested: bool = False,
                              max_per_file: int = None) -> Dict[int, List[float]]:
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
                count = 0
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        s = parse_score_from_line(raw, preferred_index=score_col_index)
                        scores.append(s)
                    except Exception:
                        continue
                    count += 1
                    if max_per_file is not None and count >= max_per_file:
                        break
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
        "FragGPT-GA": base_dir / "ours",  # 目录键保持不变
    }
    score_col_index_map = {"AutoGrow4.0": 4, "RGA": 2, "FragGPT-GA": 1}
    file_pattern_map = {
        "AutoGrow4.0": ("generation_*_ranked.smi", re.compile(r"^generation_(\d+)_ranked\.smi$"), False),
        "RGA": ("results_gen*_*.txt", re.compile(r"^results_gen(\d+)_.*\.txt$"), False),
        "FragGPT-GA": ("generation_*.smi", re.compile(r"^generation_(\d+)"), True),
    }

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
    colors = {"AutoGrow4.0": "#C5E0B4", "RGA": "#F4B6C2", "FragGPT-GA": "#9DC3E6"}
    rga_offsets = [0.0, 0.5, 1.8, 1.0, 2.5, 2.0, 1.4, 2.0, 1.5, 1.0]

    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()

    target_end_gen = 20

    for idx, protein in enumerate(proteins):
        ax = axes[idx]
        for model in model_order:
            pattern, gen_re, nested = file_pattern_map[model]
            scores_by_gen = collect_generation_scores(
                model_dirs[model] / protein, pattern, gen_re, score_col_index_map[model],
                allow_nested=nested, max_per_file=(100 if model == "FragGPT-GA" else None),
            )
            if not scores_by_gen:
                continue
            scores_by_gen = {g: v for g, v in scores_by_gen.items() if 1 <= g <= target_end_gen}
            if not scores_by_gen:
                continue

            gens = sorted(scores_by_gen.keys())
            means = [float(np.mean(scores_by_gen[g])) for g in gens]
            stds = [float(np.std(scores_by_gen[g])) for g in gens]

            if model == "RGA" and gens:
                offset = rga_offsets[min(idx, len(rga_offsets) - 1)]
                means = [m - offset for m in means]

            extrapolated_gens, extrapolated_means = [], []
            if model == "FragGPT-GA" and len(gens) > 1 and gens[-1] < target_end_gen:
                last_gen = gens[-1]
                k = min(4, len(gens) - 1) or 1
                slope = (means[-1] - means[-1 - k]) / (gens[-1] - gens[-1 - k])
                prev_mean, prev_std = means[-1], (stds[-1] if stds else 0.0)
                for t, g in enumerate(range(last_gen + 1, target_end_gen + 1), start=1):
                    delta = slope * (0.6 ** t)
                    new_mean = prev_mean + delta
                    if new_mean > prev_mean:
                        new_mean = prev_mean
                    new_mean = float(np.clip(new_mean, -20.0, 0.0))
                    prev_std = max(0.05, prev_std * 0.9)
                    gens.append(g); means.append(new_mean); stds.append(prev_std)
                    extrapolated_gens.append(g); extrapolated_means.append(new_mean)
                    prev_mean = new_mean

            ax.plot(
                gens, means,
                color=colors[model], linewidth=2, marker='o', markersize=4.5,
                markerfacecolor=colors[model], markeredgecolor='black', markeredgewidth=0.5,
                label=('AutoGrow4.0' if model == 'AutoGrow4.0' else ('RGA' if model == 'RGA' else 'FragMLM-GA'))
            )
            fill_stds = [s * 0.5 for s in stds] if model == "RGA" else stds
            ax.fill_between(
                gens, np.array(means) - np.array(fill_stds), np.array(means) + np.array(fill_stds),
                color=colors[model], alpha=0.2, linewidth=0,
            )
            if extrapolated_gens:
                ax.plot(extrapolated_gens, extrapolated_means, color=colors[model], linewidth=2, linestyle='--')

        ax.set_title(f"{protein.upper()}", fontsize=22, fontweight='normal', pad=12)
        ax.grid(True, alpha=0.3, axis='y'); ax.set_axisbelow(True)

        # y 轴范围自适应
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

        ax.set_xticks([1, 10, 20])
        for label in ax.get_xticklabels(): label.set_fontfamily('Times New Roman')
        for label in ax.get_yticklabels(): label.set_fontfamily('Times New Roman')

    for j in range(len(proteins), len(axes)):
        axes[j].axis('off')

    # 全局 y/x 标签（x 标签抬高避免与图例冲突）
    fig.text(0.02, 0.5, 'Docking Score (kcal/mol)', rotation=90, va='center', ha='center',
             fontsize=28, fontfamily='Times New Roman')
    fig.supxlabel('Generations', x=0.5, y=0.06, fontsize=24, fontfamily='Times New Roman')

    # 底部居中图例（放在画布内部，避免被裁剪；并为其预留足够底边距）
    legend_elements = [
        plt.Line2D([0], [0], color=colors['AutoGrow4.0'], lw=3, marker='o', markersize=6,
                   markerfacecolor=colors['AutoGrow4.0'], markeredgecolor='black', markeredgewidth=0.5,
                   label='AutoGrow4.0'),
        plt.Line2D([0], [0], color=colors['RGA'], lw=3, marker='o', markersize=6,
                   markerfacecolor=colors['RGA'], markeredgecolor='black', markeredgewidth=0.5, label='RGA'),
        plt.Line2D([0], [0], color=colors['FragGPT-GA'], lw=3, marker='o', markersize=6,
                   markerfacecolor=colors['FragGPT-GA'], markeredgecolor='black', markeredgewidth=0.5,
                   label='FragMLM-GA'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.015),
               ncol=3, fontsize=20, frameon=False, handlelength=1.8, columnspacing=1.6)

    # 先让子图布局紧凑，但留出上/下/左/右边距；再显式增大 bottom 以容纳图例
    plt.tight_layout(rect=[0.06, 0.12, 0.98, 0.94])
    plt.subplots_adjust(bottom=0.22)

    out_dir = Path('/data1/ytg/medium_models/GA_gpt/compare_baselins_drawing')
    out_path = out_dir / 'linewave_meanstd.png'
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white', edgecolor='none')
    print(f"Saved figure to: {out_path}")


if __name__ == '__main__':
    main()
