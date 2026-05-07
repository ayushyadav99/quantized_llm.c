#!/usr/bin/env python3
"""
Interactive experiment runner.
Runs training commands one by one, then generates a comparison plot.
"""

import subprocess
import sys
import csv
import math
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape


# ── SVG style catalogue ──────────────────────────────────────────────────────
COLORS = [
    "#d64545", "#1769aa", "#2f855a", "#b7791f",
    "#805ad5", "#dd6b20", "#319795", "#d53f8c",
]

# stroke-dasharray patterns: solid, dashed, dotted, dash-dot, long-dash
DASH_PATTERNS = ["none", "8,4", "2,4", "8,4,2,4", "16,4"]

MARKER_SYMBOLS = ["circle", "square", "triangle", "diamond", "cross"]

SMOOTH_WINDOW = 10  # default smoothing, no prompt needed


def moving_average(values, window):
    if window <= 1:
        return values[:]
    out, running, queue = [], 0.0, []
    for v in values:
        queue.append(v)
        running += v
        if len(queue) > window:
            running -= queue.pop(0)
        out.append(running / len(queue))
    return out


def read_losses(path, column="train_loss"):
    steps, losses = [], []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "step" not in reader.fieldnames:
            raise SystemExit(f"{path} is missing required CSV column: step")
        if column not in reader.fieldnames:
            raise SystemExit(f"{path} has no column '{column}' — available: {reader.fieldnames}")
        for row in reader:
            try:
                step = int(row["step"])
                loss = float(row[column])
            except (ValueError, KeyError):
                continue
            if math.isfinite(loss):
                steps.append(step)
                losses.append(loss)
    if not steps:
        raise SystemExit(f"No finite values found in column '{column}' of {path}")
    return steps, losses


def csv_has_column(path, column):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames is not None and column in reader.fieldnames


def ticks(lo, hi, count):
    if count <= 1 or lo == hi:
        return [lo]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def scale_x(x, x_min, x_max, left, width):
    span = x_max - x_min if x_max != x_min else 1.0
    return left + (x - x_min) / span * width


def scale_y(y, y_min, y_max, top, height):
    span = y_max - y_min if y_max != y_min else 1.0
    return top + height - (y - y_min) / span * height


def marker_path(symbol, cx, cy, r=5):
    if symbol == "circle":
        return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}"/>'
    if symbol == "square":
        h = r
        return f'<rect x="{cx-h:.2f}" y="{cy-h:.2f}" width="{2*h}" height="{2*h}"/>'
    if symbol == "triangle":
        pts = f"{cx:.2f},{cy-r:.2f} {cx-r:.2f},{cy+r:.2f} {cx+r:.2f},{cy+r:.2f}"
        return f'<polygon points="{pts}"/>'
    if symbol == "diamond":
        pts = f"{cx:.2f},{cy-r:.2f} {cx+r:.2f},{cy:.2f} {cx:.2f},{cy+r:.2f} {cx-r:.2f},{cy:.2f}"
        return f'<polygon points="{pts}"/>'
    if symbol == "cross":
        t = r * 0.35
        pts = (f"{cx-t:.2f},{cy-r:.2f} {cx+t:.2f},{cy-r:.2f} {cx+t:.2f},{cy-t:.2f} "
               f"{cx+r:.2f},{cy-t:.2f} {cx+r:.2f},{cy+t:.2f} {cx+t:.2f},{cy+t:.2f} "
               f"{cx+t:.2f},{cy+r:.2f} {cx-t:.2f},{cy+r:.2f} {cx-t:.2f},{cy+t:.2f} "
               f"{cx-r:.2f},{cy+t:.2f} {cx-r:.2f},{cy-t:.2f} {cx-t:.2f},{cy-t:.2f}")
        return f'<polygon points="{pts}"/>'
    return ""


def write_comparison_svg(output_path, series, smooth_window=1, loss_label="training loss"):
    """Write an interactive SVG with tooltips, distinct line styles and markers."""
    svg_w, svg_h = 1200, 720
    left, right, top, bottom = 90, 260, 60, 75
    plot_w = svg_w - left - right
    plot_h = svg_h - top - bottom

    all_steps, all_losses = [], []
    plot_series = []
    for idx, s in enumerate(series):
        ys = moving_average(s["losses"], smooth_window)
        plot_series.append({
            "label":  s["label"],
            "steps":  s["steps"],
            "losses": ys,
            "color":  COLORS[idx % len(COLORS)],
            "dash":   DASH_PATTERNS[idx % len(DASH_PATTERNS)],
            "marker": MARKER_SYMBOLS[idx % len(MARKER_SYMBOLS)],
        })
        all_steps.extend(s["steps"])
        all_losses.extend(ys)

    x_min, x_max = min(all_steps), max(all_steps)
    y_min, y_max = min(all_losses), max(all_losses)
    y_pad = (y_max - y_min) * 0.08 if y_max != y_min else max(abs(y_max) * 0.08, 1.0)
    y_min -= y_pad
    y_max += y_pad

    x_ticks = ticks(float(x_min), float(x_max), 7)
    y_ticks = ticks(y_min, y_max, 7)

    def marker_stride(n):
        return max(1, n // 20)

    title = f"Training Loss Comparison" if "train" in loss_label else "Validation Loss Comparison"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2933; }",
        ".grid { stroke: #d9e2ec; stroke-width: 1; }",
        ".axis { stroke: #334e68; stroke-width: 1.5; }",
        ".curve { fill: none; stroke-width: 2.5; }",
        ".marker { opacity: 0.85; }",
        ".tip { display: none; pointer-events: none; }",
        ".tip rect { fill: #1f2933; rx: 4; opacity: 0.88; }",
        ".tip text { fill: #fff; font-size: 12px; }",
        ".dot:hover + .tip { display: block; }",
        "</style>",
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{svg_w/2:.0f}" y="38" text-anchor="middle" '
        f'font-size="22" font-weight="700">{escape(title)}</text>',
    ]

    for tick in y_ticks:
        y = scale_y(tick, y_min, y_max, top, plot_h)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                     f'x2="{left+plot_w}" y2="{y:.2f}"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
                     f'font-size="12">{tick:.4g}</text>')

    for tick in x_ticks:
        x = scale_x(tick, x_min, x_max, left, plot_w)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" '
                     f'x2="{x:.2f}" y2="{top+plot_h}"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+22}" text-anchor="middle" '
                     f'font-size="12">{tick:.0f}</text>')

    parts += [
        f'<line class="axis" x1="{left}" y1="{top+plot_h}" '
        f'x2="{left+plot_w}" y2="{top+plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" '
        f'x2="{left}" y2="{top+plot_h}"/>',
    ]

    parts.append(f'<text x="{left+plot_w/2:.0f}" y="{svg_h-18}" '
                 f'text-anchor="middle" font-size="14">step</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2:.0f}" text-anchor="middle" '
                 f'font-size="14" transform="rotate(-90 18 {top+plot_h/2:.0f})">'
                 f'{escape(loss_label)}</text>')

    for s in plot_series:
        color  = s["color"]
        dash   = s["dash"]
        marker = s["marker"]
        steps  = s["steps"]
        losses = s["losses"]
        stride = marker_stride(len(steps))

        dash_attr = f'stroke-dasharray="{dash}"' if dash != "none" else ""

        pts = " ".join(
            f'{scale_x(sx, x_min, x_max, left, plot_w):.2f},'
            f'{scale_y(sy, y_min, y_max, top, plot_h):.2f}'
            for sx, sy in zip(steps, losses)
        )
        parts.append(
            f'<polyline class="curve" stroke="{color}" {dash_attr} points="{pts}"/>'
        )

        for i in range(0, len(steps), stride):
            sx = scale_x(steps[i], x_min, x_max, left, plot_w)
            sy = scale_y(losses[i], y_min, y_max, top, plot_h)
            tip_text = f"step {steps[i]}  loss {losses[i]:.4f}"
            tip_w = len(tip_text) * 7 + 12
            tip_x = min(sx + 8, left + plot_w - tip_w - 4)
            tip_y = max(sy - 30, top + 4)

            parts.append(
                f'<g class="dot" style="cursor:crosshair">'
                f'<g class="marker" fill="{color}" stroke="white" stroke-width="1">'
                f'{marker_path(marker, sx, sy, 5)}</g>'
                f'</g>'
                f'<g class="tip">'
                f'<rect x="{tip_x:.1f}" y="{tip_y:.1f}" width="{tip_w}" height="22" rx="4" '
                f'fill="#1f2933" opacity="0.88"/>'
                f'<text x="{tip_x+6:.1f}" y="{tip_y+15:.1f}" fill="white" font-size="12">'
                f'{escape(tip_text)}</text>'
                f'</g>'
            )

    lx = left + plot_w + 24
    ly = top + 10
    if smooth_window > 1:
        parts.append(f'<text x="{lx}" y="{ly}" font-size="11" fill="#555">'
                     f'smoothing window={smooth_window}</text>')
        ly += 18

    for idx, s in enumerate(plot_series):
        color  = s["color"]
        dash   = s["dash"]
        marker = s["marker"]
        label  = s["label"]
        y      = ly + idx * 32

        dash_attr = f'stroke-dasharray="{dash}"' if dash != "none" else ""
        parts.append(f'<line x1="{lx}" y1="{y+6}" x2="{lx+28}" y2="{y+6}" '
                     f'stroke="{color}" stroke-width="2.5" {dash_attr}/>')
        parts.append(f'<g fill="{color}" stroke="white" stroke-width="1">'
                     f'{marker_path(marker, lx+14, y+6, 5)}</g>')
        parts.append(f'<text x="{lx+36}" y="{y+11}" font-size="13">'
                     f'{escape(label)}</text>')

    parts.append("</svg>")
    Path(output_path).write_text("\n".join(parts) + "\n")


# ── Label extraction ─────────────────────────────────────────────────────────

def extract_label(cmd):
    parts = []

    oq = re.search(r'--optim_quant\s+(\S+)', cmd)
    if oq and oq.group(1) != "fp32":
        parts.append(oq.group(1).upper())

    ogs = re.search(r'--optim_group_size\s+(\S+)', cmd)
    if ogs:
        parts.append(f"gs={ogs.group(1)}")

    ptq = re.search(r'--ptq\s+1', cmd)
    pp  = re.search(r'--ptq_precision\s+(\S+)', cmd)
    if ptq:
        parts.append(f"ptq-{pp.group(1) if pp else 'int8'}")

    pgs = re.search(r'--ptq_group_size\s+(\S+)', cmd)
    if pgs:
        parts.append(f"pgs={pgs.group(1)}")

    w = re.search(r'-w\s+0', cmd)
    if w:
        parts.append("no-master")

    if not parts:
        oq2 = re.search(r'--optim_quant\s+(\S+)', cmd)
        parts.append(oq2.group(1) if oq2 else "fp32")

    return " | ".join(parts) if parts else "baseline"


def find_log_dir(cmd):
    """Return (log_dir, is_temp, final_cmd).
    If -o is present, use it as-is (is_temp=False).
    Otherwise create a temp dir and append -o <tmpdir> to cmd (is_temp=True).
    """
    m = re.search(r'-o\s+(\S+)', cmd)
    if m:
        return Path(m.group(1)), False, cmd
    tmp = Path(tempfile.mkdtemp(prefix="exp_logs_"))
    return tmp, True, cmd + f" -o {tmp}"


def auto_save_path(experiments):
    """Save SVG in the common parent of non-temp log dirs, else current dir."""
    fname = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
    persistent_dirs = [e["log_dir"] for e in experiments if not e["is_temp"]]
    if persistent_dirs:
        parents = [d.resolve().parent for d in persistent_dirs]
        if len(set(str(p) for p in parents)) == 1:
            return parents[0] / fname
    return Path(fname)


# ── Main interactive loop ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Experiment Runner")
    print("=" * 60)
    print("Enter training commands one at a time.")
    print("Type 'done' when finished to generate the comparison plot.\n")

    experiments = []
    run_num = 0

    while True:
        run_num += 1
        print(f"[Experiment {run_num}] Enter command (or 'done' to finish):")
        cmd = input("  > ").strip()

        if cmd.lower() in ("done", "no", "n", "quit", "exit", ""):
            if not experiments:
                print("No experiments run. Exiting.")
                sys.exit(0)
            break

        log_dir, is_temp, run_cmd = find_log_dir(cmd)
        csv_path = log_dir / "train_losses.csv"

        print(f"\n  Running: {run_cmd}\n" + "-" * 56)
        try:
            result = subprocess.run(run_cmd, shell=True)
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            if is_temp:
                shutil.rmtree(log_dir, ignore_errors=True)
            run_num -= 1
            continue

        if result.returncode != 0:
            print(f"  ⚠  Command exited with code {result.returncode}. Skipping this run.")
            if is_temp:
                shutil.rmtree(log_dir, ignore_errors=True)
            run_num -= 1
            continue

        if not csv_path.exists():
            print(f"  ⚠  Expected CSV not found at {csv_path}. Skipping.")
            if is_temp:
                shutil.rmtree(log_dir, ignore_errors=True)
            run_num -= 1
            continue

        label = extract_label(cmd)
        experiments.append({"cmd": cmd, "label": label, "csv": csv_path,
                             "log_dir": log_dir, "is_temp": is_temp})
        print(f"\n  Done. Label: '{label}'")

        print(f"\n[Experiment {run_num}] Add another? (Enter to continue, 'done' to finish):")
        ans = input("  > ").strip().lower()
        if ans in ("done", "no", "n", "quit", "exit"):
            break
        print()

    # ── One question: train or val? ──────────────────────────────────────────
    print("\n" + "=" * 60)

    # Check if any CSV has val_loss
    has_val = any(csv_has_column(e["csv"], "val_loss") for e in experiments)

    if has_val:
        print("Plot training loss or validation loss? [train/val, default: train]")
        choice = input("  > ").strip().lower()
        use_val = choice.startswith("v")
    else:
        use_val = False

    column     = "val_loss"    if use_val else "train_loss"
    loss_label = "validation loss" if use_val else "training loss"

    # ── Load data ────────────────────────────────────────────────────────────
    series = []
    for e in experiments:
        try:
            steps, losses = read_losses(e["csv"], column)
            series.append({"label": e["label"], "steps": steps, "losses": losses})
        except SystemExit as ex:
            print(f"  Skipping '{e['label']}': {ex}")

    if not series:
        print("No valid data to plot. Exiting.")
        sys.exit(1)

    # ── Auto-save ────────────────────────────────────────────────────────────
    output = auto_save_path(experiments)
    write_comparison_svg(output, series, smooth_window=SMOOTH_WINDOW, loss_label=loss_label)
    print(f"\n  Plot saved to: {output.resolve()}")
    print("  Open the SVG in a browser — hover over markers to see exact values.\n")

    # Clean up temp log dirs
    for e in experiments:
        if e["is_temp"]:
            shutil.rmtree(e["log_dir"], ignore_errors=True)


if __name__ == "__main__":
    main()
