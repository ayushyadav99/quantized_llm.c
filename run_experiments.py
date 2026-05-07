#!/usr/bin/env python3
"""
Interactive experiment runner with dropdown menus.
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


BINARY        = "./train_gpt2cu"
SMOOTH_WINDOW = 10
COMPARISON_DIR = Path("logs/comparison_graphs")

COLORS         = ["#d64545","#1769aa","#2f855a","#b7791f","#805ad5","#dd6b20","#319795","#d53f8c"]
DASH_PATTERNS  = ["none", "8,4", "2,4", "8,4,2,4", "16,4"]
MARKER_SYMBOLS = ["circle", "square", "triangle", "diamond", "cross"]


# ── Terminal menu helpers ─────────────────────────────────────────────────────

def pick(prompt, options, default_idx=0, labels=None):
    """Show a numbered list and return the chosen option."""
    display = labels or options
    print(f"\n{prompt}")
    for i, d in enumerate(display):
        marker = "  ← default" if i == default_idx else ""
        print(f"  [{i+1}] {d}{marker}")
    while True:
        raw = input("  > ").strip()
        if raw == "":
            return options[default_idx]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Please enter a number 1–{len(options)}")


def ask_int(prompt, default):
    """Ask for an integer, fall back to default on blank input."""
    raw = input(f"\n{prompt} [default {default}]: ").strip()
    return int(raw) if raw.isdigit() else default


# ── Experiment config builder ─────────────────────────────────────────────────

def build_experiment(exp_num, common_flags):
    """Interactively build one experiment's flags via dropdowns."""
    print(f"\n{'─'*56}")
    print(f"  Experiment {exp_num} settings")
    print(f"{'─'*56}")

    flags = list(common_flags)   # start from shared flags

    # Optimizer state format
    oq = pick(
        "Optimizer state format (--optim_quant):",
        ["fp32", "fp8", "int8", "int4"],
        default_idx=0,
        labels=["fp32  — full precision (baseline)",
                "fp8   — FP8 with COAT expansion",
                "int8  — INT8 with COAT expansion",
                "int4  — INT4 with COAT expansion"]
    )
    flags.append(f"--optim_quant {oq}")

    if oq != "fp32":
        gs = pick(
            "Optimizer group size (--optim_group_size):",
            ["32", "64", "128", "256", "512"],
            default_idx=2
        )
        flags.append(f"--optim_group_size {gs}")

    # Weight quantization (PTQ)
    ptq = pick(
        "Weight quantization / PTQ (--ptq):",
        ["0", "1"],
        default_idx=0,
        labels=["disabled (default)",
                "enabled"]
    )
    flags.append(f"--ptq {ptq}")

    if ptq == "1":
        pp = pick(
            "PTQ precision (--ptq_precision):",
            ["int8", "int4"],
            default_idx=0
        )
        flags.append(f"--ptq_precision {pp}")

        pgs = pick(
            "PTQ group size (--ptq_group_size):",
            ["32", "64", "128", "256"],
            default_idx=2
        )
        flags.append(f"--ptq_group_size {pgs}")

    # Master weights
    mw = pick(
        "Keep FP32 master weights?",
        ["1", "0"],
        default_idx=0,
        labels=["yes — keep master weights (default)",
                "no  — disable master weights (-w 0, saves memory)"]
    )
    if mw == "0":
        flags.append("-w 0")

    cmd = BINARY + " " + " ".join(flags)
    print(f"\n  Command: {cmd}")
    return cmd


# ── Data helpers ──────────────────────────────────────────────────────────────

def moving_average(values, window):
    if window <= 1:
        return values[:]
    out, running, queue = [], 0.0, []
    for v in values:
        queue.append(v); running += v
        if len(queue) > window: running -= queue.pop(0)
        out.append(running / len(queue))
    return out


def read_losses(path, column="train_loss"):
    steps, losses = [], []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "step" not in reader.fieldnames:
            raise SystemExit(f"{path}: missing 'step' column")
        if column not in reader.fieldnames:
            avail = [c for c in reader.fieldnames if "loss" in c.lower()]
            raise SystemExit(f"{path}: no column '{column}'. Available: {avail}")
        for row in reader:
            try:
                s, l = int(row["step"]), float(row[column])
            except (ValueError, KeyError):
                continue
            if math.isfinite(l):
                steps.append(s); losses.append(l)
    if not steps:
        raise SystemExit(f"No finite values in '{column}' of {path}")
    return steps, losses


def available_loss_columns(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        return [c for c in reader.fieldnames if "loss" in c.lower()]


# ── Label extraction ──────────────────────────────────────────────────────────

def extract_label(cmd):
    parts = []
    oq  = re.search(r'--optim_quant\s+(\S+)', cmd)
    ogs = re.search(r'--optim_group_size\s+(\S+)', cmd)
    ptq = re.search(r'--ptq\s+1', cmd)
    pp  = re.search(r'--ptq_precision\s+(\S+)', cmd)
    pgs = re.search(r'--ptq_group_size\s+(\S+)', cmd)

    if oq:  parts.append(oq.group(1).upper())
    if ogs: parts.append(f"gs={ogs.group(1)}")
    if ptq: parts.append(f"ptq-{pp.group(1) if pp else 'int8'}")
    if pgs: parts.append(f"pgs={pgs.group(1)}")
    if re.search(r'-w\s+0', cmd): parts.append("no-master")
    return " | ".join(parts) if parts else "baseline"


def make_run_cmd(cmd, exp_num):
    """Inject a temp -o dir if the command has none."""
    m = re.search(r'-o\s+(\S+)', cmd)
    if m:
        return Path(m.group(1)), False, cmd
    tmp = Path(tempfile.mkdtemp(prefix=f"exp{exp_num}_"))
    return tmp, True, cmd + f" -o {tmp}"


# ── SVG writer ────────────────────────────────────────────────────────────────

def ticks(lo, hi, count):
    if count <= 1 or lo == hi: return [lo]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]

def sx(x, xmin, xmax, left, w):
    return left + (x - xmin) / (xmax - xmin if xmax != xmin else 1) * w

def sy(y, ymin, ymax, top, h):
    return top + h - (y - ymin) / (ymax - ymin if ymax != ymin else 1) * h

def mpath(sym, cx, cy, r=5):
    if sym == "circle":   return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}"/>'
    if sym == "square":   return f'<rect x="{cx-r:.2f}" y="{cy-r:.2f}" width="{2*r}" height="{2*r}"/>'
    if sym == "triangle":
        return f'<polygon points="{cx:.2f},{cy-r:.2f} {cx-r:.2f},{cy+r:.2f} {cx+r:.2f},{cy+r:.2f}"/>'
    if sym == "diamond":
        return f'<polygon points="{cx:.2f},{cy-r:.2f} {cx+r:.2f},{cy:.2f} {cx:.2f},{cy+r:.2f} {cx-r:.2f},{cy:.2f}"/>'
    if sym == "cross":
        t = r * 0.35
        return (f'<polygon points="{cx-t:.2f},{cy-r:.2f} {cx+t:.2f},{cy-r:.2f} {cx+t:.2f},{cy-t:.2f} '
                f'{cx+r:.2f},{cy-t:.2f} {cx+r:.2f},{cy+t:.2f} {cx+t:.2f},{cy+t:.2f} '
                f'{cx+t:.2f},{cy+r:.2f} {cx-t:.2f},{cy+r:.2f} {cx-t:.2f},{cy+t:.2f} '
                f'{cx-r:.2f},{cy+t:.2f} {cx-r:.2f},{cy-t:.2f} {cx-t:.2f},{cy-t:.2f}"/>')
    return ""


def write_svg(output_path, series, smooth_window, loss_label):
    W, H = 1200, 720
    L, R, T, B = 90, 260, 60, 75
    PW, PH = W - L - R, H - T - B

    plot_series = []
    all_xs, all_ys = [], []
    for i, s in enumerate(series):
        ys = moving_average(s["losses"], smooth_window)
        plot_series.append({**s, "losses": ys,
                            "color": COLORS[i % len(COLORS)],
                            "dash":  DASH_PATTERNS[i % len(DASH_PATTERNS)],
                            "mark":  MARKER_SYMBOLS[i % len(MARKER_SYMBOLS)]})
        all_xs.extend(s["steps"]); all_ys.extend(ys)

    xmin, xmax = min(all_xs), max(all_xs)
    ymin, ymax = min(all_ys), max(all_ys)
    ypad = (ymax - ymin) * 0.08 if ymax != ymin else max(abs(ymax) * 0.08, 1.0)
    ymin -= ypad; ymax += ypad

    title = "Validation Loss Comparison" if "val" in loss_label else "Training Loss Comparison"
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<style>text{font-family:Arial,sans-serif;fill:#1f2933}"
        ".grid{stroke:#d9e2ec;stroke-width:1}.axis{stroke:#334e68;stroke-width:1.5}"
        ".curve{fill:none;stroke-width:2.5}.tip{display:none;pointer-events:none}"
        ".dot:hover+.tip{display:block}</style>",
        f'<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{W//2}" y="38" text-anchor="middle" font-size="22" font-weight="700">{escape(title)}</text>',
    ]

    for tick in ticks(ymin, ymax, 7):
        y = sy(tick, ymin, ymax, T, PH)
        p.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+PW}" y2="{y:.1f}"/>')
        p.append(f'<text x="{L-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{tick:.4g}</text>')

    for tick in ticks(float(xmin), float(xmax), 7):
        x = sx(tick, xmin, xmax, L, PW)
        p.append(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+PH}"/>')
        p.append(f'<text x="{x:.1f}" y="{T+PH+22}" text-anchor="middle" font-size="12">{tick:.0f}</text>')

    p += [
        f'<line class="axis" x1="{L}" y1="{T+PH}" x2="{L+PW}" y2="{T+PH}"/>',
        f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{T+PH}"/>',
        f'<text x="{L+PW//2}" y="{H-18}" text-anchor="middle" font-size="14">step</text>',
        f'<text x="18" y="{T+PH//2}" text-anchor="middle" font-size="14" '
        f'transform="rotate(-90 18 {T+PH//2})">{escape(loss_label)}</text>',
    ]

    for s in plot_series:
        da = f'stroke-dasharray="{s["dash"]}"' if s["dash"] != "none" else ""
        pts = " ".join(f'{sx(a,xmin,xmax,L,PW):.1f},{sy(b,ymin,ymax,T,PH):.1f}'
                       for a, b in zip(s["steps"], s["losses"]))
        p.append(f'<polyline class="curve" stroke="{s["color"]}" {da} points="{pts}"/>')

        stride = max(1, len(s["steps"]) // 20)
        for i in range(0, len(s["steps"]), stride):
            cx = sx(s["steps"][i], xmin, xmax, L, PW)
            cy = sy(s["losses"][i], ymin, ymax, T, PH)
            tip = f"step {s['steps'][i]}  loss {s['losses'][i]:.4f}"
            tw = len(tip) * 7 + 12
            tx = min(cx + 8, L + PW - tw - 4)
            ty = max(cy - 30, T + 4)
            p.append(
                f'<g class="dot" style="cursor:crosshair">'
                f'<g fill="{s["color"]}" stroke="white" stroke-width="1">{mpath(s["mark"],cx,cy,5)}</g></g>'
                f'<g class="tip">'
                f'<rect x="{tx:.1f}" y="{ty:.1f}" width="{tw}" height="22" rx="4" fill="#1f2933" opacity="0.88"/>'
                f'<text x="{tx+6:.1f}" y="{ty+15:.1f}" fill="white" font-size="12">{escape(tip)}</text></g>'
            )

    lx, ly = L + PW + 24, T + 10
    if smooth_window > 1:
        p.append(f'<text x="{lx}" y="{ly}" font-size="11" fill="#555">smoothing={smooth_window}</text>')
        ly += 18
    for i, s in enumerate(plot_series):
        y = ly + i * 32
        da = f'stroke-dasharray="{s["dash"]}"' if s["dash"] != "none" else ""
        p.append(f'<line x1="{lx}" y1="{y+6}" x2="{lx+28}" y2="{y+6}" stroke="{s["color"]}" stroke-width="2.5" {da}/>')
        p.append(f'<g fill="{s["color"]}" stroke="white" stroke-width="1">{mpath(s["mark"],lx+14,y+6,5)}</g>')
        p.append(f'<text x="{lx+36}" y="{y+11}" font-size="13">{escape(s["label"])}</text>')

    p.append("</svg>")
    Path(output_path).write_text("\n".join(p) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 56)
    print("  Experiment Runner")
    print("=" * 56)

    # Common flags asked once
    steps = ask_int("Training steps (-x)", default=200)
    common_flags = [f"-x {steps}"]

    # Collect experiments via dropdowns
    raw_experiments = []
    exp_num = 0
    while True:
        exp_num += 1
        cmd = build_experiment(exp_num, common_flags)
        raw_experiments.append(cmd)

        again = input("\n  Add another experiment? [y/n, default: n]: ").strip().lower()
        if not again.startswith("y"):
            break

    # Inject -o dirs
    experiments = []
    for i, cmd in enumerate(raw_experiments):
        log_dir, is_temp, run_cmd = make_run_cmd(cmd, i + 1)
        experiments.append({
            "run_cmd": run_cmd,
            "label":   extract_label(cmd),
            "log_dir": log_dir,
            "is_temp": is_temp,
            "csv":     log_dir / "train_losses.csv",
        })

    # Preview
    print("\n" + "=" * 56)
    print("  Will run:")
    for i, e in enumerate(experiments):
        print(f"  [{i+1}] {e['label']}")
        print(f"       {e['run_cmd']}")

    # Run all
    done = []
    for e in experiments:
        print(f"\n  Running [{e['label']}]\n" + "-" * 56)
        try:
            result = subprocess.run(e["run_cmd"], shell=True)
        except KeyboardInterrupt:
            print("\n  Interrupted — skipping.")
            if e["is_temp"]: shutil.rmtree(e["log_dir"], ignore_errors=True)
            continue
        if result.returncode != 0:
            print(f"  Exited {result.returncode} — skipping.")
            if e["is_temp"]: shutil.rmtree(e["log_dir"], ignore_errors=True)
            continue
        if not e["csv"].exists():
            print(f"  CSV not found — skipping.")
            if e["is_temp"]: shutil.rmtree(e["log_dir"], ignore_errors=True)
            continue
        print(f"\n  Done. Label: '{e['label']}'")
        done.append(e)

    if not done:
        print("No experiments completed. Exiting.")
        sys.exit(1)

    # Train or val?
    print("\n" + "=" * 56)
    loss_type = pick(
        "Which loss to plot?",
        ["train_loss", "val_loss"],
        default_idx=0,
        labels=["Training loss", "Validation loss"]
    )
    loss_label = "training loss" if loss_type == "train_loss" else "validation loss"

    # Load and plot
    series = []
    for e in done:
        try:
            steps_data, losses = read_losses(e["csv"], loss_type)
            series.append({"label": e["label"], "steps": steps_data, "losses": losses})
        except SystemExit as ex:
            print(f"  Skipping '{e['label']}': {ex}")

    if not series:
        print("No valid data. Exiting.")
        sys.exit(1)

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    output = COMPARISON_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
    write_svg(output, series, SMOOTH_WINDOW, loss_label)
    print(f"\n  Plot saved to: {output.resolve()}")
    print("  Open in a browser — hover over markers to see exact values.\n")

    for e in done:
        if e["is_temp"]: shutil.rmtree(e["log_dir"], ignore_errors=True)


if __name__ == "__main__":
    main()
