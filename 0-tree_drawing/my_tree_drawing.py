#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 EvoMol 的 exploration_graph 函数，
对 GA-GPT 运行结果生成探索树和分子表图。
"""
import argparse
import sys
import csv
from collections import Counter
from pathlib import Path
from packaging import version

# 确保可以导入 EvoMol 包
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (SCRIPT_DIR.parent / "EvoMol-master").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 GA-GPT 输出的 pop.csv 与 removed_ind_act_history.csv 绘制探索树"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="包含 pop.csv 与 removed_ind_act_history.csv 的目录"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data1/ytg/medium_models/GA_gpt/0-tree_drawing/my_tree_drawing"),
        help="指定输出目录，将生成的图像复制到此处（默认: 0-tree_drawing/my_tree_drawing）"
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="dot",
        help="Graphviz 布局方式（默认 dot，可选 neato、fdp 等）"
    )
    parser.add_argument(
        "--prop",
        type=str,
        default="total",
        help="用于着色/排序的指标列（如 total、qed、sa、docking_score）"
    )
    parser.add_argument(
        "--neighbours_threshold",
        type=int,
        default=0,
        help="仅当节点出度大于该阈值或为精英节点时才显示标签（默认 0）"
    )
    parser.add_argument(
        "--draw_actions",
        action="store_true",
        help="在边上绘制操作标签"
    )
    parser.add_argument(
        "--draw_scores",
        action="store_true",
        help="在节点旁显示得分"
    )
    parser.add_argument(
        "--plot_labels",
        action="store_true",
        help="在节点上绘制索引标签"
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="不绘制分子图片，仅生成探索树"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="C",
        help="根节点的历史标识，默认 C（若使用 GPT 生成，可改为 GEN1-ROOT 等）"
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=20.0,
        help="绘图宽度（英寸，默认 20）"
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=12.0,
        help="绘图高度（英寸，默认 12）"
    )
    parser.add_argument(
        "--mol_size_inches",
        type=float,
        default=0.3,
        help="嵌入分子图片的边长（英寸，默认 0.3）"
    )
    parser.add_argument(
        "--mol_size_px",
        type=int,
        default=300,
        help="嵌入分子图片的像素尺寸（默认 300）"
    )
    parser.add_argument(
        "--legend_font_size",
        type=float,
        default=9.0,
        help="节点编号/边标签字号（默认 9）"
    )
    parser.add_argument(
        "--legend_offset_x",
        type=float,
        default=0.02,
        help="节点得分文字的 X 偏移（默认 0.02）"
    )
    parser.add_argument(
        "--legend_offset_y",
        type=float,
        default=0.02,
        help="节点得分文字的 Y 偏移（默认 0.02）"
    )
    parser.add_argument(
        "--mols_per_row",
        type=int,
        default=3,
        help="分子九宫格的每行分子数（默认 3）"
    )
    parser.add_argument(
        "--draw_n_mols",
        type=int,
        default=18,
        help="九宫格中最多绘制的分子数量（默认 18，负数表示全部）"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        help="节点着色使用的 colormap 名称（默认 inferno）"
    )
    parser.add_argument(
        "--graphviz_args",
        type=str,
        default="-Grankdir=LR -Gnodesep=0.5 -Granksep=0.9",
        help="传递给 Graphviz 的额外参数"
    )
    parser.add_argument(
        "--top_roots",
        type=int,
        default=5,
        help="高亮贡献最大的根节点数量，0 表示不高亮"
    )
    parser.add_argument(
        "--top_ops",
        type=int,
        default=3,
        help="在图中允许显示的操作类型数量，按最终种群出现频次排序，0 表示全部不显示"
    )
    parser.add_argument(
        "--pop_only",
        action="store_true",
        help="仅绘制 pop.csv 中的最终种群，不包含 removed_ind_act_history.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()

    if not model_path.is_dir():
        sys.exit(f"错误：目录不存在 -> {model_path}")

    pop_file = model_path / "pop.csv"
    removed_file = model_path / "removed_ind_act_history.csv"
    if not pop_file.is_file() or not removed_file.is_file():
        sys.exit(f"错误：未找到 pop.csv 或 removed_ind_act_history.csv 于 {model_path}")

    def load_histories(csv_path: Path):
        histories = []
        with csv_path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history = row.get("history_data")
                if history:
                    histories.append(history)
        return histories

    def extract_root(history: str) -> str:
        return history.split("|", 1)[0]

    def extract_ops(history: str):
        ops = []
        for token in history.split("|")[1:]:
            op = token.split("_")[0]
            op = op.split("-")[0]
            if op:
                ops.append(op)
        return ops

    pop_histories = load_histories(pop_file)

    highlight_roots = []
    if args.top_roots > 0:
        root_counter = Counter(extract_root(h) for h in pop_histories)
        highlight_roots = [root for root, _ in root_counter.most_common(args.top_roots)]

    allowed_ops = None
    if args.top_ops > 0:
        op_counter = Counter()
        for hist in pop_histories:
            op_counter.update(extract_ops(hist))
        allowed_ops = [op for op, _ in op_counter.most_common(args.top_ops)]

    if highlight_roots:
        print("高亮根节点:", highlight_roots)
    if allowed_ops is not None:
        print("显示的操作类型:", allowed_ops)

    # 确认 Matplotlib 版本兼容（EvoMol 推荐 3.5.x）
    import matplotlib
    from packaging import version
    if version.parse(matplotlib.__version__) >= version.parse("3.6"):
        sys.exit(
            f"检测到 Matplotlib {matplotlib.__version__}，请安装 3.5.x "
            "（建议: pip install 'matplotlib==3.5.1' 'numpy==1.23.5'）"
        )

    from evomol.plot_exploration import exploration_graph  # type: ignore

    exploration_graph(
        model_path=str(model_path),
        layout=args.layout,
        prop_to_study_key=args.prop,
        plot_images=not args.no_images,
        draw_scores=args.draw_scores,
        draw_actions=args.draw_actions,
        plot_labels=args.plot_labels,
        neighbours_threshold=args.neighbours_threshold,
        root_node=args.root,
        figsize=(args.fig_width, args.fig_height),
        mol_size_inches=args.mol_size_inches,
        mol_size_px=(args.mol_size_px, args.mol_size_px),
        legend_offset=(args.legend_offset_x, args.legend_offset_y),
        legends_font_size=args.legend_font_size,
        mols_per_row=args.mols_per_row,
        draw_n_mols=None if args.draw_n_mols < 0 else args.draw_n_mols,
        cmap=args.cmap,
        graphviz_args=args.graphviz_args,
        normalize_prop=args.prop.lower() == "sa",
        highlight_roots=highlight_roots,
        allowed_ops=allowed_ops,
        pop_only=args.pop_only
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = [model_path / "expl_tree.png"]
    if not args.no_images:
        generated_files.append(model_path / "mol_table.png")

    for src in generated_files:
        if src.exists():
            dst = output_dir / src.name
            dst.write_bytes(src.read_bytes())

    print(f"探索树与分子表已生成，原始文件位于：{model_path}")
    if generated_files:
        print(f"已复制至：{output_dir}")


if __name__ == "__main__":
    main()
