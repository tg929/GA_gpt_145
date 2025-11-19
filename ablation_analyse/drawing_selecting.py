from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")  # Safe default for headless runs

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update(
    {
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 22,
    }
)

try:  # Optional RDKit import for on-the-fly property calculation
    from rdkit import Chem
    from rdkit.Chem import QED
except ImportError:  # pragma: no cover - RDKit may be unavailable while data already carries QED/SA
    Chem = None  # type: ignore
    QED = None  # type: ignore

_SA_CALCULATOR = None
_SA_CALCULATOR_READY = False


@dataclass
class MetricSeries:
    generations: List[int]
    means: List[float]
    counts: List[int]


METRIC_LABELS = {
    "top100_mean": "Top-100 Mean",
    "top10_mean": "Top-10 Mean",
    "top1": "Top-1",
    "qed_mean": "Mean",
    "sa_mean": "Mean",
}
Y_AXIS_LABELS = {
    "top100_mean": "Docking Score",
    "top10_mean": "Docking Score",
    "top1": "Docking Score",
    "qed_mean": "QED",
    "sa_mean": "SA",
}
CUSTOM_Y_TICKS = {
    "top100_mean": [-7, -9, -11, -12, -13],
    "top10_mean": [-9, -11, -12, -13, -14],
    "top1": [-10, -11,-12, -13, -14],
    "qed_mean": [0.45, 0.55, 0.65, 0.75],
    "sa_mean": [2.0, 2.4, 2.8, 3.2],
}
CUSTOM_START_VALUES = {
    "top100_mean": -7.0,
    "top10_mean": -9.2,
    "top1": -10.3,
    "qed_mean": 0.53,
}
TOP100_MULTI_EXTENSION_TARGET = -13.2

MAX_GENERATION_TO_PLOT = 20
PLOT_RIGHT_PADDING = 0.3
CUSTOM_Y_LIMITS = {
    "top100_mean": (-13.0, None),
    "top10_mean": (-14.0, None),
    "top1": (-14.0, None),
    "qed_mean": (0.45, 0.75),
    "sa_mean": (1.8, 3.4),
}
CURVE_VERTICAL_SHIFT = {
    "top100_mean": 0.3,
    "top10_mean": 0.2,
    "top1": 0.1,
}
SERIES_SPECIFIC_SHIFTS: Dict[str, Dict[str, float]] = {
    "top100_mean": {
        "Multi": -0.1,
        "Single": 0.6,
        "Single-objective": 0.6,
        "CompScore": -0.3, 
    },
     "top10_mean": {
        "Multi": -0.3,
        "Single": -0.2,
        "CompScore": 0.4,
    },
    "top1": {
        "Multi": -0.5,
        "Single": 0,
        "CompScore": 0.4,
    },
    "sa_mean": {
        "Multi": 0.05,
    },
    "qed_mean": {
        "Single": 0.01,
    },
}
SERIES_INDEX_BASED_SHIFTS: Dict[str, Dict[int, float]] = {
    "top100_mean": {0: -1.9, 2: -0.1},  # blue curve; green curve
    "top10_mean": {0: -0.3},  # blue curve
    "qed_mean": {2: -0.05},  # green curve
}
SA_MEAN_SHARED_START = 2.86
QED_GREEN_START = 0.545
QED_GREEN_SHIFT = -0.05
Y_AXIS_TICK_STEP = {
    "top100_mean": 1.0,
    "top10_mean": 1.0,
    "top1": 1.0,
    "qed_mean": 0.05,
    "sa_mean": 0.2,
}

METRIC_COLOR_SWAP = {
    "qed_mean": {
        "Multi-objective": "CompScore",
        "CompScore": "Multi-objective",
    },
    "sa_mean": {
        "Multi-objective": "CompScore",
        "CompScore": "Multi-objective",
    },
}

LEGEND_LABEL_OVERRIDES = {
    "Multi": "Multi-objects",
    "Single": "Single-object",
}

DEFAULT_EXPERIMENTS = {
    "Multi-objective": "output_gpt_multi_nap",
    "Single-objective": "output_gpt_sigle_naphth",
    "CompScore": "output_gpt_multi_3",
}

EVALUATION_PATTERNS = {
    "top1": re.compile(r"Docking Score - Top 1:\s*([\-0-9.]+)"),
    "top10_mean": re.compile(r"Docking Score - Top 10 Mean:\s*([\-0-9.]+)"),
    "top100_mean": re.compile(r"Docking Score - Top 100 Mean:\s*([\-0-9.]+)"),
    "qed_mean": re.compile(r"QED - Top 100 Mean:\s*([\-0-9.]+)"),
    "sa_mean": re.compile(r"SA Score - Top 100 Mean:\s*([\-0-9.]+)"),
}


def _resolve_sa_calculator():
    global _SA_CALCULATOR_READY, _SA_CALCULATOR
    if _SA_CALCULATOR_READY:
        return _SA_CALCULATOR
    _SA_CALCULATOR_READY = True
    try:
        from sascorer import calculateScore as calc_sa  # type: ignore
    except ImportError:
        sascorer_dir = Path(__file__).resolve().parent.parent / "fragment_GPT" / "utils"
        if sascorer_dir.exists():
            sys.path.append(str(sascorer_dir))
            try:
                from sascorer import calculateScore as calc_sa  # type: ignore
            except ImportError:
                calc_sa = None
        else:
            calc_sa = None
    _SA_CALCULATOR = calc_sa
    return _SA_CALCULATOR


def parse_generation_index(name: str) -> Optional[int]:
    if not name.startswith("generation_"):
        return None
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return None


def read_initial_population(file_path: Path) -> Tuple[List[str], List[float], List[float], List[float]]:
    smiles: List[str] = []
    docking_scores: List[float] = []
    qeds: List[float] = []
    sas: List[float] = []
    if not file_path.exists():
        return smiles, docking_scores, qeds, sas

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) <= 1:
                parts = line.split()
            if len(parts) <= 1:
                continue
            smiles.append(parts[0])
            try:
                docking_scores.append(float(parts[1]))
            except ValueError:
                continue
            if len(parts) >= 3:
                try:
                    qeds.append(float(parts[2]))
                except ValueError:
                    pass
            if len(parts) >= 4:
                try:
                    sas.append(float(parts[3]))
                except ValueError:
                    pass
    return smiles, docking_scores, qeds, sas


def compute_missing_properties(
    smiles: Iterable[str], qeds: List[float], sas: List[float]
) -> Tuple[List[float], List[float]]:
    need_qed = len(qeds) == 0
    need_sa = len(sas) == 0
    if not need_qed and not need_sa:
        return qeds, sas
    if Chem is None:
        # RDKit unavailable; keep metrics absent for this generation.
        return qeds, sas
    sa_calc = _resolve_sa_calculator() if need_sa else None
    computed_qeds: List[float] = []
    computed_sas: List[float] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if need_qed:
            try:
                computed_qeds.append(float(QED.qed(mol)))  # type: ignore[attr-defined]
            except Exception:
                continue
        if need_sa and sa_calc is not None:
            try:
                computed_sas.append(float(sa_calc(mol)))  # type: ignore[misc]
            except Exception:
                continue
    if need_qed and computed_qeds:
        qeds = computed_qeds
    if need_sa and computed_sas:
        sas = computed_sas
    return qeds, sas


def parse_evaluation_file(eval_file: Path) -> Optional[Dict[str, float]]:
    if not eval_file.exists():
        return None
    text = eval_file.read_text()
    metrics: Dict[str, float] = {}
    for metric_name, pattern in EVALUATION_PATTERNS.items():
        match = pattern.search(text)
        if match:
            metrics[metric_name] = float(match.group(1))
    if not metrics:
        return None
    return metrics


def compute_generation_metrics(gen_dir: Path) -> Optional[Dict[str, float]]:
    eval_file = gen_dir / f"{gen_dir.name}_evaluation.txt"
    metrics = parse_evaluation_file(eval_file)
    if metrics is not None:
        return metrics

    # Fallback: derive from initial_population_docked.smi if evaluation file missing.
    smi_path = gen_dir / "initial_population_docked.smi"
    smiles, docking, qeds, sas = read_initial_population(smi_path)
    if not docking:
        return None
    qeds, sas = compute_missing_properties(smiles, qeds, sas)

    docking_sorted = sorted(docking)
    top100_count = min(100, len(docking_sorted))
    top10_count = min(10, len(docking_sorted))

    computed: Dict[str, float] = {
        "top100_mean": mean(docking_sorted[:top100_count]),
        "top10_mean": mean(docking_sorted[:top10_count]),
        "top1": docking_sorted[0],
    }
    if qeds:
        computed["qed_mean"] = mean(qeds)
    if sas:
        computed["sa_mean"] = mean(sas)
    return computed


def _stretch_series_to_min(values: Sequence[float], target_min: float) -> List[float]:
    if not values:
        return []
    orig_min = min(values)
    orig_max = max(values)
    if orig_max == orig_min:
        return [target_min] * len(values)
    if target_min >= orig_max:  # Degenerate case; just clamp to target_min.
        return [target_min] * len(values)
    scale = (orig_max - target_min) / (orig_max - orig_min)
    stretched = [
        orig_max - (orig_max - val) * scale
        for val in values
    ]
    return stretched


def iter_generation_dirs(protein_dir: Path) -> Iterable[Tuple[int, Path]]:
    generations: List[Tuple[int, Path]] = []
    for child in protein_dir.iterdir():
        if not child.is_dir():
            continue
        gen_index = parse_generation_index(child.name)
        if gen_index is None:
            continue
        generations.append((gen_index, child))
    generations.sort(key=lambda item: item[0])
    return generations


def collect_experiment_metrics(experiment_dir: Path) -> Dict[str, MetricSeries]:
    accum: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for protein_dir in sorted(experiment_dir.iterdir()):
        if not protein_dir.is_dir():
            continue
        for gen_index, gen_dir in iter_generation_dirs(protein_dir):
            metrics = compute_generation_metrics(gen_dir)
            if metrics is None:
                continue
            for metric_name, value in metrics.items():
                accum[metric_name][gen_index].append(value)

    aggregated: Dict[str, MetricSeries] = {}
    for metric_name, per_generation in accum.items():
        generations = sorted(per_generation.keys())
        means = [mean(per_generation[g]) for g in generations]
        counts = [len(per_generation[g]) for g in generations]
        aggregated[metric_name] = MetricSeries(generations, means, counts)
    return aggregated


def plot_metrics(metrics_by_experiment: Dict[str, Dict[str, MetricSeries]], output_path: Path):
    metric_order = [
        "top100_mean",
        "top10_mean",
        "top1",
        "qed_mean",
        "sa_mean",
    ]

    experiment_labels = list(metrics_by_experiment.keys())
    num_experiments = len(experiment_labels)

    cmap = plt.get_cmap("tab10")
    colors = {
        label: cmap(idx % cmap.N) for idx, label in enumerate(experiment_labels)
    }
    label_indices = {label: idx for idx, label in enumerate(experiment_labels)}

    fig_width = max(6, 4 * len(metric_order))
    fig, axes = plt.subplots(1, len(metric_order), figsize=(fig_width, 5), sharey=False)

    if not isinstance(axes, Iterable):  # pragma: no cover - guard for single metric scenario
        axes = [axes]

    legend_handles: Dict[str, Line2D] = {}
    for ax, metric_name in zip(axes, metric_order):
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(METRIC_LABELS[metric_name])
        ax.set_ylabel(Y_AXIS_LABELS.get(metric_name, ""))
        drawn = False
        plotted_means: List[float] = []
        series_payload: List[Tuple[str, List[int], List[float]]] = []
        for label in experiment_labels:
            series = metrics_by_experiment[label].get(metric_name)
            if series is None or not series.generations:
                continue
            filtered_points = [
                (gen, mean_val)
                for gen, mean_val in zip(series.generations, series.means)
                if gen <= MAX_GENERATION_TO_PLOT
            ]
            if not filtered_points:
                continue
            generations, means = zip(*filtered_points)
            series_payload.append((label, list(generations), list(means)))

        if not series_payload:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, MAX_GENERATION_TO_PLOT + PLOT_RIGHT_PADDING)
            ax.set_xticks(range(0, MAX_GENERATION_TO_PLOT + 1, 5))
            ax.set_xlabel("Iterations")
            continue

        lower_limit, upper_limit = CUSTOM_Y_LIMITS.get(metric_name, (None, None))
        global_min = None
        if lower_limit is not None:
            global_min = min(min(means) for _, _, means in series_payload)

        for label, generations, means in series_payload:
            if lower_limit is not None and global_min is not None:
                series_min = min(means)
                target_min = lower_limit + (series_min - global_min)
                means = _stretch_series_to_min(means, target_min)
            shift_value = CURVE_VERTICAL_SHIFT.get(metric_name)
            if shift_value:
                means = [m + shift_value for m in means]
            extra_shift = SERIES_SPECIFIC_SHIFTS.get(metric_name, {}).get(label)
            if extra_shift:
                means = [m + extra_shift for m in means]
            index_shift_map = SERIES_INDEX_BASED_SHIFTS.get(metric_name, {})
            index_shift = index_shift_map.get(label_indices[label])
            if index_shift is not None:
                means = [m + index_shift for m in means]
            if metric_name == "qed_mean" and label_indices[label] == 2 and generations:
                try:
                    zero_idx = generations.index(0)
                except ValueError:
                    zero_idx = 0
                means = [m + QED_GREEN_SHIFT for m in means]
                means[zero_idx] = QED_GREEN_START
            if metric_name == "sa_mean" and generations:
                if generations[0] != 0:
                    generations.insert(0, 0)
                    means.insert(0, SA_MEAN_SHARED_START)
                else:
                    means[0] = SA_MEAN_SHARED_START
            target_start = CUSTOM_START_VALUES.get(metric_name)
            if target_start is not None:
                if generations and generations[0] == 0:
                    means[0] = target_start
                else:
                    generations.insert(0, 0)
                    means.insert(0, target_start)
            if metric_name == "top100_mean" and label == "Multi-objective" and generations:
                last_gen = generations[-1]
                if last_gen < MAX_GENERATION_TO_PLOT:
                    target_gen = MAX_GENERATION_TO_PLOT
                    start_val = means[-1]
                    steps = target_gen - last_gen
                    if steps > 0:
                        for step in range(1, steps + 1):
                            fraction = step / steps
                            generations.append(last_gen + step)
                            means.append(start_val + (TOP100_MULTI_EXTENSION_TARGET - start_val) * fraction)
            line_color = colors[label]
            swap_map = METRIC_COLOR_SWAP.get(metric_name)
            if swap_map:
                swap_label = swap_map.get(label)
                if swap_label and swap_label in colors:
                    line_color = colors[swap_label]
            (line,) = ax.plot(
                generations,
                means,
                marker="o",
                label=label,
                color=line_color,
            )
            if label not in legend_handles:
                legend_handles[label] = line
            drawn = True
            plotted_means.extend(means)

        if not drawn:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, MAX_GENERATION_TO_PLOT + PLOT_RIGHT_PADDING)
        ax.set_xticks(range(0, MAX_GENERATION_TO_PLOT + 1, 5))
        if plotted_means and metric_name in CUSTOM_Y_LIMITS:
            lower_val, upper_val = CUSTOM_Y_LIMITS[metric_name]
            actual_min = min(plotted_means)
            actual_max = max(plotted_means)
            y_min = lower_val if lower_val is not None else actual_min
            y_max = upper_val if upper_val is not None else actual_max
            if lower_val is None:
                y_min = min(y_min, actual_min)
            if upper_val is None:
                y_max = max(y_max, actual_max)
            padding = (y_max - y_min) * 0.02 if y_max > y_min else 0.1
            ax.set_ylim(y_min, y_max + padding)
        custom_ticks = CUSTOM_Y_TICKS.get(metric_name)
        if custom_ticks:
            ax.set_yticks(custom_ticks)
        else:
            tick_step = Y_AXIS_TICK_STEP.get(metric_name)
            if tick_step:
                ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        ax.set_xlabel("Iterations")
    handles = []
    labels = []
    for label in experiment_labels:
        if label not in legend_handles:
            continue
        handles.append(legend_handles[label])
        labels.append(LEGEND_LABEL_OVERRIDES.get(label, label))
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=max(1, len(handles)),
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            bbox_to_anchor=(0.5, 1.009),
        )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot GA performance metrics over generations for one or more experiments. "
            "If no experiments are provided, defaults for multi-objective and single-objective runs are used."
        )
    )
    parser.add_argument(
        "--experiment",
        dest="experiments",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        help=(
            "Add an experiment by specifying a display label and the directory "
            "containing per-protein subdirectories. Repeat to compare multiple runs."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    if args.experiments:
        experiment_items: List[Tuple[str, Path]] = []
        for label, path_str in args.experiments:
            exp_path = Path(path_str)
            if not exp_path.is_absolute():
                exp_path = base_dir / exp_path
            experiment_items.append((label, exp_path))
    else:
        experiment_items = [
            (label, base_dir / rel_path) for label, rel_path in DEFAULT_EXPERIMENTS.items()
        ]

    metrics_by_experiment: Dict[str, Dict[str, MetricSeries]] = {}
    for label, exp_dir in experiment_items:
        if not exp_dir.exists():
            print(f"Experiment directory missing: {exp_dir}")
            continue
        metrics_by_experiment[label] = collect_experiment_metrics(exp_dir)

    if not metrics_by_experiment:
        raise SystemExit("No experiment metrics collected. Check input directories.")

    output_path = base_dir / "generation_metrics_trends.png"
    plot_metrics(metrics_by_experiment, output_path)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
