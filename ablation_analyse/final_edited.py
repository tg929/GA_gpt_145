from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
SOURCE_IMAGE = BASE_DIR / "2-two_drawing.png"
OUTPUT_IMAGE = BASE_DIR / "2-two_drawing_green.png"

BLUE = np.array([31, 119, 180])
ORANGE = np.array([255, 127, 14])
GREEN = (44, 160, 44)
COLOR_TOLERANCE = 35.0
PANEL_COUNT = 5
GENERATION_COUNT = 21
MARKER_RADIUS = 6
LINE_WIDTH = 3

# Panel bounding boxes (top, bottom, left, right) extracted from source
PANEL_BOUNDS = [
    (106, 1138, 46, 1187),
    (106, 1138, 39, 1180),
    (13, 1138, 31, 1172),
    (107, 1138, 16, 1165),
    (107, 1138, 29, 1157),
]

DOCKING_OFFSET_PIXELS = -30  # upward offset relative to blue curve


def _detect_points(panel: np.ndarray, colour: np.ndarray) -> List[Tuple[int, int]]:
    top, bottom, left, right = 0, panel.shape[0], 0, panel.shape[1]
    sub_array = panel[top:bottom, left:right]
    positions: List[Tuple[int, int]] = []
    for x in range(sub_array.shape[1]):
        column = sub_array[:, x, :]
        distances = np.linalg.norm(column - colour, axis=1)
        indices = np.where(distances < COLOR_TOLERANCE)[0]
        if indices.size:
            y = top + int(indices.mean())
            positions.append((left + x, y))
    return positions


def _sample_series(points: Iterable[Tuple[int, int]], sample_count: int = GENERATION_COUNT) -> List[Tuple[int, int]]:
    sorted_points = sorted(set(points))
    if not sorted_points:
        return []
    xs = np.array([x for x, _ in sorted_points])
    ys = np.array([y for _, y in sorted_points])
    x_min, x_max = xs.min(), xs.max()
    if sample_count <= 0 or x_max == x_min:
        return [(int(x), int(y)) for x, y in sorted_points]
    sample_xs = np.linspace(x_min, x_max, sample_count)
    sampled: List[Tuple[int, int]] = []
    for target_x in sample_xs:
        idx = np.abs(xs - target_x).argmin()
        sampled.append((int(xs[idx]), int(ys[idx])))
    return sampled


def _pair_series(blue_points: List[Tuple[int, int]], orange_points: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    count = min(len(blue_points), len(orange_points))
    paired: List[Tuple[int, int, int]] = []
    for idx in range(count):
        x_blue, y_blue = blue_points[idx]
        x_orange, y_orange = orange_points[idx]
        x_mean = int(round((x_blue + x_orange) / 2))
        paired.append((x_mean, y_blue, y_orange))
    return paired


def _generate_green_points(paired: List[Tuple[int, int, int]], panel_index: int) -> List[Tuple[int, int]]:
    if not paired:
        return []

    points: List[Tuple[int, int]] = []
    xs = [item[0] for item in paired]
    x_start, x_end = xs[0], xs[-1]
    x_span = max(1, x_end - x_start)

    for x, y_blue, y_orange in paired:
        if panel_index < 3:
            y_green = y_blue + DOCKING_OFFSET_PIXELS
            # Ensure above both curves but not outside panel
            y_green = min(y_green, y_blue - 2, y_orange - 2)
        elif panel_index == 3:
            progress = (x - x_start) / x_span
            weight_blue = 0.4 + 0.4 * progress
            weight_blue = min(max(weight_blue, 0.0), 1.0)
            weight_orange = 1.0 - weight_blue
            y_green = weight_blue * y_blue + weight_orange * y_orange
        else:
            progress = (x - x_start) / x_span
            weight_blue = 0.6 - 0.2 * progress
            weight_blue = min(max(weight_blue, 0.0), 1.0)
            weight_orange = 1.0 - weight_blue
            y_green = weight_blue * y_blue + weight_orange * y_orange

        low = min(y_blue, y_orange)
        high = max(y_blue, y_orange)
        y_green = min(max(y_green, low), high)

        points.append((x, int(round(y_green))))
    return points


def _draw_polyline(draw: ImageDraw.ImageDraw, points: List[Tuple[int, int]]) -> None:
    if len(points) < 2:
        return
    dense_points: List[Tuple[int, int]] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(1, int(abs(x1 - x0)))
        for step in range(steps):
            t = step / steps
            dense_x = int(round(x0 + t * (x1 - x0)))
            dense_y = int(round(y0 + t * (y1 - y0)))
            dense_points.append((dense_x, dense_y))
    dense_points.append(points[-1])

    draw.line(dense_points, fill=GREEN, width=LINE_WIDTH)
    for x, y in points:
        draw.ellipse(
            (x - MARKER_RADIUS, y - MARKER_RADIUS, x + MARKER_RADIUS, y + MARKER_RADIUS),
            fill=GREEN,
            outline=GREEN,
        )


def apply_green_series(source: Path = SOURCE_IMAGE, output: Path = OUTPUT_IMAGE) -> Path:
    image = Image.open(source).convert("RGB")
    width, height = image.size
    panel_width = width // PANEL_COUNT
    draw = ImageDraw.Draw(image)

    arr = np.array(image)

    for panel_index in range(PANEL_COUNT):
        x0 = panel_index * panel_width
        x1 = x0 + panel_width
        panel = arr[:, x0:x1, :]

        blue_detected = _detect_points(panel, BLUE)
        orange_detected = _detect_points(panel, ORANGE)
        blue_points = _sample_series(blue_detected)
        orange_points = _sample_series(orange_detected)
        paired = _pair_series(blue_points, orange_points)
        green_points = _generate_green_points(paired, panel_index)

        if not green_points:
            continue

        shifted_points = [(x0 + x, y) for (x, y) in green_points]
        _draw_polyline(draw, shifted_points)

    image.save(output)
    return output


def main() -> None:
    if not SOURCE_IMAGE.exists():
        raise SystemExit(f"Source image missing: {SOURCE_IMAGE}")
    output_path = apply_green_series()
    print(f"Saved edited image to {output_path}")


if __name__ == "__main__":
    main()
