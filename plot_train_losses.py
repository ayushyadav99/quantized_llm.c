#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


def parse_args():
    parser = argparse.ArgumentParser(description="Plot train_losses.csv emitted by train_gpt2cu.")
    parser.add_argument("loss_dump", help="Path to train_losses.csv, or an output log directory containing it.")
    parser.add_argument("-o", "--output", help="Output SVG path. Defaults to loss_plot.svg next to the dump.")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window for an extra smoothed curve.")
    parser.add_argument("--all-logs", action="store_true", help="When loss_dump is a directory, plot all train_losses*.csv files in it.")
    parser.add_argument("--title", default="Training loss", help="Plot title.")
    return parser.parse_args()


def resolve_loss_dumps(path, all_logs):
    path = Path(path)
    if not path.is_dir():
        return [path]
    if all_logs:
        return sorted(path.glob("train_losses*.csv"))
    return [path / "train_losses.csv"]


def label_from_path(path):
    label = path.stem
    if label.startswith("train_losses_"):
        label = label[len("train_losses_"):]
    elif label == "train_losses":
        label = "loss"
    return label


def read_losses(path):
    steps = []
    losses = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"step", "train_loss"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit(f"{path} is missing required CSV columns: step, train_loss")
        for row in reader:
            try:
                step = int(row["step"])
                loss = float(row["train_loss"])
            except ValueError:
                continue
            if math.isfinite(loss):
                steps.append(step)
                losses.append(loss)
    if not steps:
        raise SystemExit(f"No finite training losses found in {path}")
    return steps, losses


def moving_average(values, window):
    if window <= 1:
        return values[:]
    out = []
    running = 0.0
    queue = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        out.append(running / len(queue))
    return out


def ticks(lo, hi, count):
    if count <= 1 or lo == hi:
        return [lo]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def scale_points(xs, ys, x_min, x_max, y_min, y_max, left, top, width, height):
    x_span = x_max - x_min if x_max != x_min else 1.0
    y_span = y_max - y_min if y_max != y_min else 1.0
    points = []
    for x, y in zip(xs, ys):
        sx = left + (x - x_min) / x_span * width
        sy = top + height - (y - y_min) / y_span * height
        points.append(f"{sx:.2f},{sy:.2f}")
    return " ".join(points)


def write_svg(path, series, smooth_window, title):
    colors = [
        "#d64545",
        "#1769aa",
        "#2f855a",
        "#b7791f",
        "#805ad5",
        "#dd6b20",
        "#319795",
        "#d53f8c",
        "#4a5568",
        "#38a169",
    ]
    plot_series = []
    for idx, item in enumerate(series):
        ys = moving_average(item["losses"], smooth_window)
        plot_series.append({
            "label": item["label"],
            "steps": item["steps"],
            "losses": ys,
            "color": colors[idx % len(colors)],
        })

    all_steps = [step for item in plot_series for step in item["steps"]]
    all_losses = [loss for item in plot_series for loss in item["losses"]]
    x_min, x_max = min(all_steps), max(all_steps)
    y_min, y_max = min(all_losses), max(all_losses)
    y_pad = (y_max - y_min) * 0.08 if y_max != y_min else max(abs(y_max) * 0.08, 1.0)
    y_min -= y_pad
    y_max += y_pad

    svg_width, svg_height = 1100, 680
    left, right, top, bottom = 90, 170, 60, 75
    plot_width = svg_width - left - right
    plot_height = svg_height - top - bottom

    x_ticks = ticks(float(x_min), float(x_max), 6)
    y_ticks = ticks(y_min, y_max, 6)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2933; }",
        ".grid { stroke: #d9e2ec; stroke-width: 1; }",
        ".axis { stroke: #334e68; stroke-width: 1.5; }",
        ".curve { fill: none; stroke-width: 2.3; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{svg_width / 2:.0f}" y="30" text-anchor="middle" font-size="22" font-weight="700">{escape(title)}</text>',
    ]

    for tick in y_ticks:
        y = top + plot_height - (tick - y_min) / (y_max - y_min) * plot_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="13">{tick:.4g}</text>')

    for tick in x_ticks:
        x = left + (tick - x_min) / (x_max - x_min if x_max != x_min else 1.0) * plot_width
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle" font-size="13">{tick:.0f}</text>')

    parts.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
    ])

    for item in plot_series:
        points = scale_points(item["steps"], item["losses"], x_min, x_max, y_min, y_max, left, top, plot_width, plot_height)
        parts.append(f'<polyline class="curve" stroke="{item["color"]}" points="{points}"/>')

    legend_x = left + plot_width + 24
    legend_y = top + 20
    if smooth_window > 1:
        parts.append(f'<text x="{legend_x}" y="{legend_y - 18}" font-size="13">moving avg window={smooth_window}</text>')
    for idx, item in enumerate(plot_series):
        y = legend_y + idx * 24
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{item["color"]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{y + 5}" font-size="13">{escape(item["label"])}</text>')

    parts.extend([
        f'<text x="{left + plot_width / 2:.0f}" y="{svg_height - 22}" text-anchor="middle" font-size="15">step</text>',
        f'<text x="22" y="{top + plot_height / 2:.0f}" text-anchor="middle" font-size="15" transform="rotate(-90 22 {top + plot_height / 2:.0f})">training loss</text>',
        "</svg>",
    ])

    path.write_text("\n".join(parts) + "\n")


def write_single_svg(path, steps, losses, smooth_window, title):
    smoothed = moving_average(losses, smooth_window)
    x_min, x_max = min(steps), max(steps)
    y_values = losses + (smoothed if smooth_window > 1 else [])
    y_min, y_max = min(y_values), max(y_values)
    y_pad = (y_max - y_min) * 0.08 if y_max != y_min else max(abs(y_max) * 0.08, 1.0)
    y_min -= y_pad
    y_max += y_pad

    svg_width, svg_height = 1100, 680
    left, right, top, bottom = 90, 35, 60, 75
    plot_width = svg_width - left - right
    plot_height = svg_height - top - bottom

    loss_points = scale_points(steps, losses, x_min, x_max, y_min, y_max, left, top, plot_width, plot_height)
    smooth_points = scale_points(steps, smoothed, x_min, x_max, y_min, y_max, left, top, plot_width, plot_height)

    x_ticks = ticks(float(x_min), float(x_max), 6)
    y_ticks = ticks(y_min, y_max, 6)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2933; }",
        ".grid { stroke: #d9e2ec; stroke-width: 1; }",
        ".axis { stroke: #334e68; stroke-width: 1.5; }",
        ".loss { fill: none; stroke: #d64545; stroke-width: 2; }",
        ".smooth { fill: none; stroke: #1769aa; stroke-width: 2.5; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{svg_width / 2:.0f}" y="30" text-anchor="middle" font-size="22" font-weight="700">{escape(title)}</text>',
    ]

    for tick in y_ticks:
        y = top + plot_height - (tick - y_min) / (y_max - y_min) * plot_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="13">{tick:.4g}</text>')

    for tick in x_ticks:
        x = left + (tick - x_min) / (x_max - x_min if x_max != x_min else 1.0) * plot_width
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle" font-size="13">{tick:.0f}</text>')

    parts.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
        f'<polyline class="loss" points="{loss_points}"/>',
    ])
    if smooth_window > 1:
        parts.append(f'<polyline class="smooth" points="{smooth_points}"/>')
        parts.append(f'<text x="{left + plot_width - 10}" y="{top + 24}" text-anchor="end" font-size="14">raw loss, moving avg window={smooth_window}</text>')
    else:
        parts.append(f'<text x="{left + plot_width - 10}" y="{top + 24}" text-anchor="end" font-size="14">raw loss</text>')

    parts.extend([
        f'<text x="{left + plot_width / 2:.0f}" y="{svg_height - 22}" text-anchor="middle" font-size="15">step</text>',
        f'<text x="22" y="{top + plot_height / 2:.0f}" text-anchor="middle" font-size="15" transform="rotate(-90 22 {top + plot_height / 2:.0f})">training loss</text>',
        "</svg>",
    ])

    path.write_text("\n".join(parts) + "\n")


def main():
    args = parse_args()
    loss_dumps = resolve_loss_dumps(args.loss_dump, args.all_logs)
    missing = [path for path in loss_dumps if not path.exists()]
    if missing:
        raise SystemExit(f"Loss dump not found: {missing[0]}")
    if not loss_dumps:
        raise SystemExit(f"No train_losses*.csv files found in {args.loss_dump}")

    input_path = Path(args.loss_dump)
    if args.output:
        output = Path(args.output)
    elif input_path.is_dir():
        output = input_path / ("loss_plot_all.svg" if args.all_logs else "loss_plot.svg")
    else:
        output = loss_dumps[0].with_name("loss_plot.svg")

    series = []
    for loss_dump in loss_dumps:
        steps, losses = read_losses(loss_dump)
        series.append({"label": label_from_path(loss_dump), "steps": steps, "losses": losses})

    smooth_window = max(1, args.smooth)
    if len(series) == 1:
        write_single_svg(output, series[0]["steps"], series[0]["losses"], smooth_window, args.title)
    else:
        write_svg(output, series, smooth_window, args.title)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
