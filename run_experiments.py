#!/usr/bin/env python3
"""
Interactive experiment runner.
Enter commands one at a time, then generates a comparison plot.
"""

import subprocess
import sys
import csv
import math
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape


SMOOTH_WINDOW  = 10
COMPARISON_DIR = Path("logs/comparison_graphs")

COLORS         = ["#d64545","#1769aa","#2f855a","#b7791f","#805ad5","#dd6b20","#319795","#d53f8c"]
DASH_PATTERNS  = ["none","8,4","2,4","8,4,2,4","16,4"]
MARKER_SYMBOLS = ["circle","square","triangle","diamond","cross"]


# ── Label extraction ──────────────────────────────────────────────────────────

def extract_label(cmd):
    parts = []

    oq = re.search(r'--optim_quant\s+(\S+)', cmd)
    if oq:
        val = oq.group(1).upper()
        if val != "FP32":
            val = f"{val}(COAT)"
        parts.append(val)

        ogs = re.search(r'--optim_group_size\s+(\S+)', cmd)
        if ogs:
            parts.append(f"coat-gs={ogs.group(1)}")

    ptq = re.search(r'--ptq\s+1', cmd)
    pp  = re.search(r'--ptq_precision\s+(\S+)', cmd)
    if ptq:
        parts.append(f"ptq-{pp.group(1) if pp else 'int8'}")

    pgs = re.search(r'--ptq_group_size\s+(\S+)', cmd)
    if pgs:
        parts.append(f"ptq-gs={pgs.group(1)}")

    if re.search(r'-w\s+0', cmd):
        parts.append("no-master")

    return " | ".join(parts) if parts else "baseline"


# ── Temp dir injection ────────────────────────────────────────────────────────

def make_run_cmd(cmd, exp_num):
    """Return (log_dir, is_temp, final_cmd). Injects -o if missing."""
    m = re.search(r'-o\s+(\S+)', cmd)
    if m:
        return Path(m.group(1)), False, cmd
    tmp = Path(tempfile.mkdtemp(prefix=f"exp{exp_num}_"))
    return tmp, True, cmd + f" -o {tmp}"


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


# ── SVG writer ────────────────────────────────────────────────────────────────

def ticks(lo, hi, count):
    if count <= 1 or lo == hi: return [lo]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]

def _sx(x, xmin, xmax, L, W):
    return L + (x - xmin) / (xmax - xmin if xmax != xmin else 1) * W

def _sy(y, ymin, ymax, T, H):
    return T + H - (y - ymin) / (ymax - ymin if ymax != ymin else 1) * H

def mpath(sym, cx, cy, r=5):
    if sym == "circle":
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}"/>'
    if sym == "square":
        return f'<rect x="{cx-r:.1f}" y="{cy-r:.1f}" width="{2*r}" height="{2*r}"/>'
    if sym == "triangle":
        return (f'<polygon points="{cx:.1f},{cy-r:.1f} '
                f'{cx-r:.1f},{cy+r:.1f} {cx+r:.1f},{cy+r:.1f}"/>')
    if sym == "diamond":
        return (f'<polygon points="{cx:.1f},{cy-r:.1f} {cx+r:.1f},{cy:.1f} '
                f'{cx:.1f},{cy+r:.1f} {cx-r:.1f},{cy:.1f}"/>')
    if sym == "cross":
        t = r * 0.35
        return (f'<polygon points="{cx-t:.1f},{cy-r:.1f} {cx+t:.1f},{cy-r:.1f} '
                f'{cx+t:.1f},{cy-t:.1f} {cx+r:.1f},{cy-t:.1f} {cx+r:.1f},{cy+t:.1f} '
                f'{cx+t:.1f},{cy+t:.1f} {cx+t:.1f},{cy+r:.1f} {cx-t:.1f},{cy+r:.1f} '
                f'{cx-t:.1f},{cy+t:.1f} {cx-r:.1f},{cy+t:.1f} {cx-r:.1f},{cy-t:.1f} '
                f'{cx-t:.1f},{cy-t:.1f}"/>')
    return ""


def write_svg(output_path, series, smooth_window, loss_label):
    W, H = 1200, 720
    L, R, T, B = 90, 260, 60, 75
    PW, PH = W - L - R, H - T - B

    plot_series, all_xs, all_ys = [], [], []
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
        y = _sy(tick, ymin, ymax, T, PH)
        p += [f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+PW}" y2="{y:.1f}"/>',
              f'<text x="{L-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{tick:.4g}</text>']

    for tick in ticks(float(xmin), float(xmax), 7):
        x = _sx(tick, xmin, xmax, L, PW)
        p += [f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+PH}"/>',
              f'<text x="{x:.1f}" y="{T+PH+22}" text-anchor="middle" font-size="12">{tick:.0f}</text>']

    p += [
        f'<line class="axis" x1="{L}" y1="{T+PH}" x2="{L+PW}" y2="{T+PH}"/>',
        f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{T+PH}"/>',
        f'<text x="{L+PW//2}" y="{H-18}" text-anchor="middle" font-size="14">step</text>',
        f'<text x="18" y="{T+PH//2}" text-anchor="middle" font-size="14" '
        f'transform="rotate(-90 18 {T+PH//2})">{escape(loss_label)}</text>',
    ]

    for s in plot_series:
        da = f'stroke-dasharray="{s["dash"]}"' if s["dash"] != "none" else ""
        pts = " ".join(f'{_sx(a,xmin,xmax,L,PW):.1f},{_sy(b,ymin,ymax,T,PH):.1f}'
                       for a, b in zip(s["steps"], s["losses"]))
        p.append(f'<polyline class="curve" stroke="{s["color"]}" {da} points="{pts}"/>')

        stride = max(1, len(s["steps"]) // 20)
        for i in range(0, len(s["steps"]), stride):
            cx = _sx(s["steps"][i], xmin, xmax, L, PW)
            cy = _sy(s["losses"][i], ymin, ymax, T, PH)
            tip = f"step {s['steps'][i]}  loss {s['losses'][i]:.4f}"
            tw = len(tip) * 7 + 12
            tx = min(cx + 8, L + PW - tw - 4)
            ty = max(cy - 30, T + 4)
            p.append(
                f'<g class="dot" style="cursor:crosshair">'
                f'<g fill="{s["color"]}" stroke="white" stroke-width="1">{mpath(s["mark"],cx,cy,5)}</g></g>'
                f'<g class="tip">'
                f'<rect x="{tx:.1f}" y="{ty:.1f}" width="{tw}" height="22" rx="4" '
                f'fill="#1f2933" opacity="0.88"/>'
                f'<text x="{tx+6:.1f}" y="{ty+15:.1f}" fill="white" font-size="12">'
                f'{escape(tip)}</text></g>'
            )

    lx, ly = L + PW + 24, T + 10
    if smooth_window > 1:
        p.append(f'<text x="{lx}" y="{ly}" font-size="11" fill="#555">smoothing={smooth_window}</text>')
        ly += 18
    for i, s in enumerate(plot_series):
        y = ly + i * 32
        da = f'stroke-dasharray="{s["dash"]}"' if s["dash"] != "none" else ""
        p += [
            f'<line x1="{lx}" y1="{y+6}" x2="{lx+28}" y2="{y+6}" '
            f'stroke="{s["color"]}" stroke-width="2.5" {da}/>',
            f'<g fill="{s["color"]}" stroke="white" stroke-width="1">{mpath(s["mark"],lx+14,y+6,5)}</g>',
            f'<text x="{lx+36}" y="{y+11}" font-size="13">{escape(s["label"])}</text>',
        ]

    p.append("</svg>")
    Path(output_path).write_text("\n".join(p) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def ask(prompt, default=""):
    """input() wrapper that handles EOF gracefully instead of crashing."""
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        return input().strip()
    except EOFError:
        return default


def main():
    print("=" * 56)
    print("  Experiment Runner")
    print("=" * 56)
    print("Enter training commands one at a time.")
    print("Type 'done' when finished.\n")

    experiments = []
    exp_num = 0

    while True:
        exp_num += 1
        cmd = ask(f"[Experiment {exp_num}] Command (or 'done'):\n  > ")

        if cmd.lower() in ("done", "no", "n", "quit", "exit", ""):
            if not experiments:
                print("No experiments run. Exiting.")
                sys.exit(0)
            break

        log_dir, is_temp, run_cmd = make_run_cmd(cmd, exp_num)
        csv_path = log_dir / "train_losses.csv"

        print(f"\n  Running: {run_cmd}\n" + "-" * 56)
        try:
            # stdin=None inherits terminal so training output is live;
            # the training binary does not read stdin so this is safe.
            result = subprocess.run(run_cmd, shell=True, stdin=open(os.devnull))
        except KeyboardInterrupt:
            print("\n  Interrupted — skipping.")
            if is_temp: shutil.rmtree(log_dir, ignore_errors=True)
            exp_num -= 1
            continue

        print("\n" + "=" * 56)
        print("  TRAINING COMPLETE")
        print("=" * 56)

        if result.returncode != 0:
            print(f"  Command exited with code {result.returncode} — skipping.")
            if is_temp: shutil.rmtree(log_dir, ignore_errors=True)
            exp_num -= 1
            continue

        if not csv_path.exists():
            print(f"  CSV not found at {csv_path} — skipping.")
            if is_temp: shutil.rmtree(log_dir, ignore_errors=True)
            exp_num -= 1
            continue

        label = extract_label(cmd)
        experiments.append({"cmd": cmd, "label": label,
                             "csv": csv_path, "log_dir": log_dir, "is_temp": is_temp})
        print(f"  Label: '{label}'")

        ans = ask("\n  Add another experiment? [y/n, default: n]: ")
        if not ans.lower().startswith("y"):
            break
        print()

    # Ask train or val
    print("\n" + "=" * 56)
    print("  Which loss to plot?")
    print("  [1] Training loss  (default)")
    print("  [2] Validation loss")
    print("=" * 56)
    choice = ask("  > ")
    use_val    = choice.strip() == "2"
    loss_label = "validation loss" if use_val else "training loss"

    # Load data
    series = []
    for e in experiments:
        try:
            if use_val:
                val_csv = e["log_dir"] / "val_losses.csv"
                if not val_csv.exists():
                    print(f"  Skipping '{e['label']}': val_losses.csv not found — rebuild first")
                    continue
                steps, losses = read_losses(val_csv, "val_loss")
            else:
                steps, losses = read_losses(e["csv"], "train_loss")
            series.append({"label": e["label"], "steps": steps, "losses": losses})
        except SystemExit as ex:
            print(f"  Skipping '{e['label']}': {ex}")

    if not series:
        print("No valid data. Exiting.")
        sys.exit(1)

    # Save
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    output = COMPARISON_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
    write_svg(output, series, SMOOTH_WINDOW, loss_label)

    print("\n" + "=" * 56)
    print(f"  Graph saved to:")
    print(f"  {output.resolve()}")
    print("  Open in a browser — hover markers to see exact values.")
    print("=" * 56 + "\n")

    for e in experiments:
        if e["is_temp"]: shutil.rmtree(e["log_dir"], ignore_errors=True)


if __name__ == "__main__":
    main()
