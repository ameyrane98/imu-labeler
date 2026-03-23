"""Generate an animated demo GIF showcasing IMU Labeler.

Run:  python docs/create_demo_gif.py
Output: docs/screenshots/demo.gif
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import medfilt

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SAMPLE_CSV = os.path.join(PROJECT_DIR, "examples", "sample_data", "sample_imu.csv")
OUTPUT_GIF = os.path.join(SCRIPT_DIR, "screenshots", "demo.gif")

os.makedirs(os.path.dirname(OUTPUT_GIF), exist_ok=True)

# ── Load data ────────────────────────────────────────────────────
if not os.path.exists(SAMPLE_CSV):
    print(f"Sample data not found. Run: python examples/generate_sample_data.py")
    sys.exit(1)

df = pd.read_csv(SAMPLE_CSV)
t = df["timestamp"].values
accel_norm = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2).values
accel_norm = medfilt(accel_norm, kernel_size=3)

# ── Demo scenario ────────────────────────────────────────────────
# Simulate: zoom in, pan, add annotations, show labels
LABELS_CFG = {
    "w": {"name": "Walking",  "color": "#4ecdc4"},
    "r": {"name": "Running",  "color": "#e76f51"},
    "s": {"name": "Sitting",  "color": "#a78bfa"},
    "t": {"name": "Standing", "color": "#f4a261"},
}

# Pre-defined annotations that appear during the demo
DEMO_ANNOTATIONS = [
    {"start": 10, "stop": 18, "label": "w", "appear_frame": 18},
    {"start": 25, "stop": 35, "label": "r", "appear_frame": 30},
    {"start": 45, "stop": 52, "label": "s", "appear_frame": 42},
    {"start": 60, "stop": 70, "label": "t", "appear_frame": 55},
    {"start": 80, "stop": 88, "label": "w", "appear_frame": 68},
]

# Camera keyframes: (frame, xlim_start, xlim_end)
# Start zoomed out, zoom into activity, pan, zoom out
TOTAL_FRAMES = 80
CAMERA_KEYFRAMES = [
    (0,   0, 120),     # full view
    (10,  0, 120),     # hold
    (22,  5, 45),      # zoom into first annotations
    (38,  20, 60),     # pan right
    (52,  40, 80),     # pan to middle
    (65,  50, 95),     # pan further
    (73,  0, 120),     # zoom back out
    (80,  0, 120),     # hold
]

def interp_camera(frame):
    """Interpolate camera position from keyframes."""
    for i in range(len(CAMERA_KEYFRAMES) - 1):
        f0, x0s, x0e = CAMERA_KEYFRAMES[i]
        f1, x1s, x1e = CAMERA_KEYFRAMES[i + 1]
        if f0 <= frame <= f1:
            t_val = (frame - f0) / max(1, f1 - f0)
            # Smooth easing
            t_val = t_val * t_val * (3 - 2 * t_val)
            return x0s + (x1s - x0s) * t_val, x0e + (x1e - x0e) * t_val
    return CAMERA_KEYFRAMES[-1][1], CAMERA_KEYFRAMES[-1][2]


def fmt_mmss(x, _=None):
    x = abs(x)
    return f"{int(x)//60:02d}:{int(x)%60:02d}"


# ── Build figure ─────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 3.5), facecolor="#0f0f1a", dpi=72)
gs = gridspec.GridSpec(2, 1, figure=fig,
                       left=0.05, right=0.95, top=0.90, bottom=0.08,
                       hspace=0.12, height_ratios=[0.82, 0.18])

# Title
fig.text(0.5, 0.96, "IMU LABELER", ha="center", va="top",
         color="#00d4ff", fontsize=18, fontweight="bold")
fig.text(0.5, 0.925, "Interactive annotation tool for wearable sensor data",
         ha="center", va="top", color="#888899", fontsize=10)

# Signal plot
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor("#13131f")
for sp in ax.spines.values():
    sp.set_edgecolor("#333355")
ax.tick_params(colors="#aaaacc", labelsize=8)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_mmss))
ax.grid(True, color="#1e1e35", linewidth=0.5, zorder=1)

line, = ax.plot(t, accel_norm, color="#00d4ff", linewidth=0.9, alpha=0.9, zorder=2)
ax.set_ylabel("Accel Norm (m/s²)", color="#aaaacc", fontsize=9)
ax.set_xlabel("Time (s)", color="#aaaacc", fontsize=9)

ymin, ymax = accel_norm.min(), accel_norm.max()
pad = (ymax - ymin) * 0.1
ax.set_ylim(ymin - pad, ymax + pad)

# Playhead
playhead = ax.axvline(0, color="#ff4444", linewidth=1.5, zorder=10)

# Status bar
ax_st = fig.add_subplot(gs[1, 0])
ax_st.set_facecolor("#0f0f1a")
ax_st.axis("off")

status_text = ax_st.text(
    0.01, 0.7, "Mode: browse   [A] annotate   [S] save   [Space] play   [H] help",
    color="#88aaff", fontsize=9, va="center", transform=ax_st.transAxes)
count_text = ax_st.text(
    0.01, 0.25, "Annotations: 0",
    color="#aaaacc", fontsize=9, va="center", transform=ax_st.transAxes)

# Legend in status bar
for i, (key, info) in enumerate(LABELS_CFG.items()):
    ax_st.add_patch(plt.Rectangle(
        (0.55 + i * 0.15, 0.15), 0.03, 0.7,
        transform=ax_st.transAxes, color=info["color"], alpha=0.8))
    ax_st.text(0.59 + i * 0.15, 0.5, f"{key} = {info['name']}",
               color="#ccccee", fontsize=8, va="center", transform=ax_st.transAxes)

# ── Annotation state ─────────────────────────────────────────────
ann_artists = []
visible_annotations = []

# Selection highlight (appears briefly before annotation is confirmed)
select_highlight = None


def update(frame):
    # Camera
    xs, xe = interp_camera(frame)
    ax.set_xlim(xs, xe)

    # Playhead — sweep across visible area
    ph_x = xs + (xe - xs) * ((frame * 2) % 100) / 100
    playhead.set_xdata([ph_x, ph_x])

    # Title update
    title_parts = ["[accel] norm"]
    title_parts.append(f"  |  filter: median(k=3)")
    if any(a["appear_frame"] - 8 <= frame <= a["appear_frame"] for a in DEMO_ANNOTATIONS):
        title_parts.append("  |  ANNOTATING...")
    ax.set_title("  ".join(title_parts), color="#e0e0e0", fontsize=9, pad=4)

    # Show annotations as they "appear"
    n_visible = 0
    for ann in DEMO_ANNOTATIONS:
        if frame >= ann["appear_frame"] and ann not in visible_annotations:
            visible_annotations.append(ann)
            color = LABELS_CFG[ann["label"]]["color"]
            patch = ax.axvspan(ann["start"], ann["stop"],
                               alpha=0.22, color=color, zorder=3)
            ls = ax.axvline(ann["start"], color=color, lw=1.3, alpha=0.85, zorder=4)
            le = ax.axvline(ann["stop"], color=color, lw=1.3, alpha=0.85, zorder=4)
            mid = (ann["start"] + ann["stop"]) / 2
            ylim = ax.get_ylim()
            ytop = ylim[1] - (ylim[1] - ylim[0]) * 0.04
            txt = ax.text(mid, ytop, ann["label"], color=color, fontsize=11,
                          ha="center", va="top", fontweight="bold", zorder=6,
                          bbox=dict(boxstyle="round,pad=0.2",
                                    fc="#0f0f1a", ec=color, alpha=0.85))
            ann_artists.append((patch, ls, le, txt))

        # Show selection highlight just before annotation appears
        if ann["appear_frame"] - 8 <= frame < ann["appear_frame"] and ann not in visible_annotations:
            global select_highlight
            if select_highlight is not None:
                select_highlight.remove()
            select_highlight = ax.axvspan(ann["start"], ann["stop"],
                                          alpha=0.15, color="#ffffff", zorder=4)
            status_text.set_text(
                f"Mode: ANNOTATE   selecting {fmt_mmss(ann['start'])} -> {fmt_mmss(ann['stop'])}   "
                f"press {ann['label']} to label")
            status_text.set_color("#ffaa00")
            break
        elif frame == ann["appear_frame"]:
            if select_highlight is not None:
                select_highlight.remove()
                select_highlight = None
            name = LABELS_CFG[ann["label"]]["name"]
            status_text.set_text(f"Added [{ann['label']}] {name}  {fmt_mmss(ann['start'])} -> {fmt_mmss(ann['stop'])}")
            status_text.set_color("#6bcb77")
            break
    else:
        if frame > 5 and frame % 40 > 15:
            status_text.set_text("Mode: browse   [A] annotate   [S] save   [Space] play   [H] help")
            status_text.set_color("#88aaff")

    count_text.set_text(f"Annotations: {len(visible_annotations)}")

    return [line, playhead, status_text, count_text]


# ── Render ───────────────────────────────────────────────────────
print(f"Rendering {TOTAL_FRAMES} frames...")
anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=80, blit=False)
anim.save(OUTPUT_GIF, writer=PillowWriter(fps=10))
print(f"Done! Saved to: {OUTPUT_GIF}")

# Also get file size
size_mb = os.path.getsize(OUTPUT_GIF) / (1024 * 1024)
print(f"File size: {size_mb:.1f} MB")
plt.close()
