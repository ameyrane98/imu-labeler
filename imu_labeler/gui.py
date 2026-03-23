"""Interactive annotation GUI — generalized for any IMU / time-series data."""

import csv
import os
import re

import matplotlib
matplotlib.use("TkAgg")

import cv2
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import matplotlib.ticker as mticker
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.widgets import SpanSelector, Slider, Button

from .utils import fmt_mmss, fmt_mmss_short, apply_filter


# ══════════════════════════════════════════════════════════════════
#  SENSOR AUTO-DETECTION
# ══════════════════════════════════════════════════════════════════

def _detect_sensor_groups(df):
    """Auto-detect groups of (x, y, z) sensor columns.

    Returns a dict like:
        {"index_accel": ["index_accel_x", "index_accel_y", "index_accel_z"],
         "wrist_gyro":  ["wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z"]}
    """
    groups = {}
    cols = list(df.columns)
    seen = set()

    # Pattern: prefix_x, prefix_y, prefix_z  (or _X _Y _Z)
    for col in cols:
        if col in seen:
            continue
        for suffix in ("_x", "_y", "_z", "_X", "_Y", "_Z"):
            if col.endswith(suffix):
                prefix = col[: -len(suffix)]
                x_col = prefix + suffix[0:1].lower().join(suffix[1:])  # normalize
                # Try to find all three axes
                candidates = []
                for ax in ("x", "y", "z"):
                    for case in (ax, ax.upper()):
                        c = prefix + "_" + case
                        if c in cols:
                            candidates.append(c)
                            break
                if len(candidates) == 3:
                    groups[prefix] = candidates
                    seen.update(candidates)
                break

    # If nothing found, try accel_x / accel_y / accel_z style
    if not groups:
        for prefix in ("accel", "acc", "gyro", "mag", "accelerometer", "gyroscope"):
            candidates = []
            for ax in ("x", "y", "z"):
                for pattern in (f"{prefix}_{ax}", f"{prefix}{ax}", f"{prefix.upper()}_{ax.upper()}"):
                    if pattern in cols:
                        candidates.append(pattern)
                        break
            if len(candidates) == 3:
                groups[prefix] = candidates

    return groups


def _detect_timestamp_col(df):
    """Auto-detect the best timestamp column for display."""
    # Prefer already-processed columns
    for c in ("Video_Sync_Time", "Resampled_Timestamp", "New_Timestamp",
              "UTC_Timestamp", "timestamp", "Timestamp", "time", "Time"):
        if c in df.columns:
            return c
    for c in df.columns:
        if "time" in c.lower():
            return c
    return None


def _compute_norm(df, cols, filter_type, filter_k):
    """Compute filtered magnitude from 3-axis columns."""
    raw = np.sqrt(df[cols[0]] ** 2 + df[cols[1]] ** 2 + df[cols[2]] ** 2).values
    return apply_filter(raw, filter_type, filter_k)


# ══════════════════════════════════════════════════════════════════
#  HELP TEXT
# ══════════════════════════════════════════════════════════════════

def _build_help_lines(labels_cfg):
    lines = [
        ("IMU LABELER — HELP", 11, "#00d4ff", True),
        ("", 7, "#aaaacc", False),
        ("NAVIGATION", 10, "#ffd93d", True),
        ("  Hover mouse over plot  ->  scrubs video to that time", 9, "#ccccee", False),
        ("  Scroll wheel on plot   ->  zoom in / out", 9, "#ccccee", False),
        ("  <- / -> Arrow keys     ->  jump +/-5 seconds", 9, "#ccccee", False),
        ("  Drag seek slider       ->  jump to any time", 9, "#ccccee", False),
        ("", 7, "#aaaacc", False),
        ("PLAYBACK", 10, "#ffd93d", True),
        ("  [Space]                ->  Play / Pause", 9, "#ccccee", False),
        ("  Speed buttons          ->  change playback rate", 9, "#ccccee", False),
        ("", 7, "#aaaacc", False),
        ("ANNOTATING", 10, "#ffd93d", True),
        ("  Press [A]              ->  enter Annotate Mode", 9, "#ccccee", False),
        ("  Then drag on the plot  ->  highlight a span", 9, "#ccccee", False),
        ("  Or: press [  to set start, then ] to set stop", 9, "#ccccee", False),
    ]
    # Show label keys
    label_keys = []
    for key, info in labels_cfg.items():
        name = info.get("name", key)
        label_keys.append(f"    {key} = {name}")
    if label_keys:
        keys_str = "  Press " + " / ".join(labels_cfg.keys()) + " to assign label"
        lines.append((keys_str, 9, "#ccccee", False))
        for lk in label_keys:
            lines.append((lk, 9, "#6bcb77", False))
    lines += [
        ("  Right-click colored box->  delete annotation", 9, "#ccccee", False),
        ("  Double-click color box ->  change label", 9, "#ccccee", False),
        ("  [Ctrl+Z]               ->  undo last action", 9, "#ccccee", False),
        ("", 7, "#aaaacc", False),
        ("SAVING", 10, "#ffd93d", True),
        ("  [S]                    ->  manual save", 9, "#ccccee", False),
        ("  Auto-save              ->  happens on every add/delete", 9, "#ccccee", False),
        ("", 7, "#aaaacc", False),
        ("Press [H] to close this panel", 9, "#ff6b6b", True),
    ]
    return lines


# ══════════════════════════════════════════════════════════════════
#  ANNOTATOR
# ══════════════════════════════════════════════════════════════════

class Annotator:
    def __init__(self, csv_path, video_path, output_file,
                 labels_cfg, ref_labels_file=None,
                 filter_type="median", filter_k=3,
                 speed_options=None, sensor_groups=None,
                 timestamp_column=None):

        self.output_file = output_file
        self.labels_cfg = labels_cfg
        self.label_keys = list(labels_cfg.keys())
        self.label_colors = {k: v.get("color", "#ffffff") for k, v in labels_cfg.items()}
        self.label_names = {k: v.get("name", k) for k, v in labels_cfg.items()}
        self.filter_type = filter_type
        self.filter_k = filter_k
        self.speed_options = speed_options or [0.5, 1.0, 2.0, 4.0]

        self.help_lines = _build_help_lines(labels_cfg)

        # ── Load sensor data ──────────────────────────────────────
        df = pd.read_csv(csv_path)
        self.df = df

        # Detect sensor groups
        if sensor_groups:
            # User-specified: dict of name -> [col_x, col_y, col_z]
            self.sensor_groups = sensor_groups
        else:
            self.sensor_groups = _detect_sensor_groups(df)

        if not self.sensor_groups:
            # Fallback: just use all numeric columns as a single signal
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude timestamp-like columns
            numeric_cols = [c for c in numeric_cols if "time" not in c.lower()]
            if numeric_cols:
                self.sensor_groups = {"signal": numeric_cols[:1]}  # just first numeric col
                self._single_signal = True
            else:
                raise ValueError("No numeric sensor columns found in the CSV!")
        else:
            self._single_signal = False

        self.sensor_names = list(self.sensor_groups.keys())
        self.active_sensor = self.sensor_names[0]

        # Detect timestamps
        ts_col = timestamp_column or _detect_timestamp_col(df)
        if ts_col and ts_col in df.columns:
            self.t_num = df[ts_col].values.astype(float)
        else:
            # Use row index as time
            self.t_num = np.arange(len(df), dtype=float)

        # Display time
        if "Video_Sync_Time" in df.columns:
            self.t_display = df["Video_Sync_Time"].values.astype(float)
            self.x_label = "Video Sync Time (s)"
        elif "Resampled_Timestamp" in df.columns:
            t = df["Resampled_Timestamp"].values.astype(float)
            self.t_display = t - t[0]
            self.x_label = "Time (s)"
        else:
            self.t_display = self.t_num - self.t_num[0]
            self.x_label = "Time (s)"

        self.ref_labels = self._load_reference_labels(ref_labels_file)

        # ── State ─────────────────────────────────────────────────
        self.annotate_mode = False
        self.pending_start = None
        self.pending_vline = None
        self._last_mouse_x = self.t_display[0]
        self._help_visible = False
        self._help_artists = []
        self.ann_list = []
        self.ann_artists = []
        self._undo_stack = []
        self.is_playing = False
        self.playhead_x = self.t_display[0]
        self.play_speed = 1.0

        # ── OpenCV ────────────────────────────────────────────────
        self.cap = None
        if video_path and os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.vid_dur = self.total_frames / self.fps
            print(f"[video] {video_path}  {self.fps:.1f} fps  {self.vid_dur:.1f}s")
        else:
            self.fps = 30.0
            self.total_frames = 0
            self.vid_dur = 0.0
            if video_path:
                print(f"[video] WARNING: not found: {video_path}")
            else:
                print("[video] No video file — IMU-only mode")

        # ── Figure ────────────────────────────────────────────────
        has_video = self.cap is not None

        if has_video:
            self.fig = plt.figure(figsize=(20, 9), facecolor="#0f0f1a")
            gs = gridspec.GridSpec(
                3, 1, figure=self.fig,
                left=0.03, right=0.97, top=0.97, bottom=0.03,
                hspace=0.08, height_ratios=[0.45, 0.37, 0.18],
            )
            vid_row, plot_row, ctrl_row = 0, 1, 2
        else:
            self.fig = plt.figure(figsize=(20, 7), facecolor="#0f0f1a")
            gs = gridspec.GridSpec(
                2, 1, figure=self.fig,
                left=0.03, right=0.97, top=0.97, bottom=0.03,
                hspace=0.08, height_ratios=[0.70, 0.30],
            )
            plot_row, ctrl_row = 0, 1

        self.fig.canvas.manager.set_window_title("IMU Labeler")

        # Video panel
        if has_video:
            self.ax_vid = self.fig.add_subplot(gs[vid_row, 0])
            self.ax_vid.set_facecolor("#000000")
            self.ax_vid.set_xticks([])
            self.ax_vid.set_yticks([])
            self.ax_vid.set_title(
                "Video  |  hover over plot to scrub  |  press [H] for help",
                color="#aaaacc", fontsize=9, pad=3,
            )
            for sp in self.ax_vid.spines.values():
                sp.set_edgecolor("#333355")
            self._vid_im = self.ax_vid.imshow(
                np.zeros((480, 640, 3), dtype=np.uint8),
                aspect="auto", interpolation="bilinear",
            )
            self._vid_txt = self.ax_vid.text(
                0.01, 0.97, "", color="#00ffff", fontsize=9,
                transform=self.ax_vid.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="#000000", alpha=0.6),
            )
        else:
            self.ax_vid = None

        # Signal plot
        self.ax = self.fig.add_subplot(gs[plot_row, 0])
        self.ax.set_facecolor("#13131f")
        for sp in self.ax.spines.values():
            sp.set_edgecolor("#333355")
        self.ax.tick_params(colors="#aaaacc", labelsize=8)
        self.ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_mmss_short))
        self.ax.grid(True, color="#1e1e35", linewidth=0.5, zorder=1)

        # Controls strip
        gs_bot = gridspec.GridSpecFromSubplotSpec(
            2, 6, subplot_spec=gs[ctrl_row, 0], hspace=0.35, wspace=0.25,
        )

        # Play button
        ax_play = self.fig.add_subplot(gs_bot[0, 0])
        self.btn_play = Button(ax_play, "Play", color="#1a1a2e", hovercolor="#2a2a4e")
        self.btn_play.label.set_color("#00d4ff")
        self.btn_play.label.set_fontsize(9)
        self.btn_play.on_clicked(self._toggle_play)

        # Speed buttons
        self._speed_btns = []
        for si, spd in enumerate(self.speed_options):
            ax_s = self.fig.add_subplot(gs_bot[1, si])
            clr = "#003050" if spd == 1.0 else "#1a1a2e"
            btn = Button(ax_s, f"{spd}x", color=clr, hovercolor="#2a4060")
            btn.label.set_color("#00d4ff")
            btn.label.set_fontsize(8)
            btn.on_clicked(lambda e, s=spd: self._set_speed(s))
            self._speed_btns.append((spd, btn))

        # Help button
        ax_help = self.fig.add_subplot(gs_bot[1, 4])
        self.btn_help = Button(ax_help, "[H] Help", color="#1a1a2e", hovercolor="#2a2a4e")
        self.btn_help.label.set_color("#ffd93d")
        self.btn_help.label.set_fontsize(8)
        self.btn_help.on_clicked(lambda e: self._toggle_help())

        # Seek slider
        ax_seek = self.fig.add_subplot(gs_bot[0, 1:])
        self.slider_seek = Slider(
            ax_seek, "", self.t_display[0], self.t_display[-1],
            valinit=self.t_display[0], color="#00d4ff", track_color="#1e1e35",
        )
        self.slider_seek.label.set_color("#aaaacc")
        self.slider_seek.valtext.set_color("#aaaacc")
        self.slider_seek.valtext.set_text(fmt_mmss_short(self.t_display[0]))
        self._slider_updating = False
        self.slider_seek.on_changed(self._on_seek)

        # Sensor radio (only if multiple sensor groups)
        if len(self.sensor_names) > 1:
            ax_radio = self.fig.add_subplot(gs_bot[1, 4:])
            ax_radio.set_facecolor("#1a1a2e")
            self.radio = mwidgets.RadioButtons(
                ax_radio, self.sensor_names, active=0, activecolor="#00d4ff",
            )
            for lbl in self.radio.labels:
                lbl.set_color("#ccccee")
                lbl.set_fontsize(9)
            self.radio.on_clicked(self._on_sensor_change)
        else:
            self.radio = None

        # Status bar
        ax_st = self.fig.add_subplot(gs_bot[1, 5:])
        ax_st.set_facecolor("#0f0f1a")
        ax_st.axis("off")
        mode_label = "IMU-only" if not has_video else "browse"
        keys_hint = " / ".join(self.label_keys)
        self.status = ax_st.text(
            0.01, 0.65,
            f"Mode: {mode_label}   [A] annotate   [S] save   [Space] play   [H] help",
            color="#88aaff", fontsize=8, va="center", transform=ax_st.transAxes,
        )
        self._count_txt = ax_st.text(
            0.01, 0.2, "Annotations: 0",
            color="#aaaacc", fontsize=8, va="center", transform=ax_st.transAxes,
        )

        # Signal line + playhead
        (self.line,) = self.ax.plot(
            [], [], color="#00d4ff", linewidth=0.9, alpha=0.9, zorder=2,
        )
        self.playhead_line = self.ax.axvline(
            self.playhead_x, color="#ff4444", linewidth=1.5, zorder=10,
        )
        self.playhead_line.set_visible(False)

        self._ref_artists = []
        self._update_plot()
        self._load_existing()

        # Events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        self.span = SpanSelector(
            self.ax, self._on_span_select, "horizontal",
            useblit=True, props=dict(alpha=0.2, facecolor="white"),
            interactive=False,
        )
        self.span.active = False

        self.anim = animation.FuncAnimation(
            self.fig, self._on_animate, interval=33, cache_frame_data=False,
        )

        self._refresh_video_frame(self.playhead_x)
        plt.show()
        if self.cap:
            self.cap.release()

    # ── Reference labels ──────────────────────────────────────────
    def _load_reference_labels(self, ref_file):
        if not ref_file or not os.path.exists(ref_file):
            return []
        ldf = pd.read_csv(ref_file)
        ref = []
        # Try common column names for start/stop
        start_col = next((c for c in ldf.columns if "start" in c.lower()), None)
        stop_col = next((c for c in ldf.columns if "stop" in c.lower() or "end" in c.lower()), None)
        label_col = next((c for c in ldf.columns if c.lower() in ("label", "activity", "class", "category")), None)
        if start_col and stop_col:
            for _, row in ldf.iterrows():
                entry = {
                    "start_d": float(row[start_col]),
                    "stop_d": float(row[stop_col]),
                    "activity": str(row[label_col]) if label_col else "?",
                }
                ref.append(entry)
            print(f"[labels] {len(ref)} reference label(s) loaded")
        return ref

    # ── Plot ──────────────────────────────────────────────────────
    def _on_sensor_change(self, label):
        self.active_sensor = label
        self._update_plot()

    def _update_plot(self):
        cols = self.sensor_groups[self.active_sensor]
        if len(cols) == 3 and not self._single_signal:
            y = _compute_norm(self.df, cols, self.filter_type, self.filter_k)
            ylabel = f"{self.active_sensor} norm"
        else:
            y = self.df[cols[0]].values.astype(float)
            y = apply_filter(y, self.filter_type, self.filter_k)
            ylabel = cols[0]

        self.line.set_data(self.t_display, y)
        cur = self.ax.get_xlim()
        if cur == (0.0, 1.0) or cur == (self.t_display[0], self.t_display[-1]):
            self.ax.set_xlim(self.t_display[0], self.t_display[-1])
        ymin, ymax = y.min(), y.max()
        pad = (ymax - ymin) * 0.1 or 1
        self.ax.set_ylim(ymin - pad, ymax + pad)

        keys_hint = " / ".join(self.label_keys)
        self.ax.set_title(
            f"[{self.active_sensor}] {ylabel}  |  "
            f"filter: {self.filter_type}(k={self.filter_k})  |  "
            f"[A] annotate   right-click=delete   dbl-click=edit   [Ctrl+Z]=undo",
            color="#e0e0e0", fontsize=9, pad=4,
        )
        self.ax.set_xlabel(self.x_label, color="#aaaacc", fontsize=9)
        self.ax.set_ylabel(ylabel, color="#aaaacc", fontsize=9)

        for a in self._ref_artists:
            a.remove()
        self._ref_artists = []
        for ref in self.ref_labels:
            span = self.ax.axvspan(
                ref["start_d"], ref["stop_d"],
                alpha=0.12, color="#ffff00", zorder=1,
            )
            mid = (ref["start_d"] + ref["stop_d"]) / 2
            ylim = self.ax.get_ylim()
            txt = self.ax.text(
                mid, ylim[1] - (ylim[1] - ylim[0]) * 0.01,
                ref["activity"], color="#cccc00", fontsize=7,
                ha="center", va="top", zorder=3, style="italic",
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="#0f0f1a", ec="#888800", alpha=0.7),
            )
            self._ref_artists.extend([span, txt])
        self.fig.canvas.draw_idle()

    # ── Video ─────────────────────────────────────────────────────
    def _refresh_video_frame(self, xval):
        if not self.cap or xval is None:
            return
        frame_idx = max(0, min(int(xval * self.fps), self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._vid_im.set_data(rgb)
            self._vid_im.set_extent([0, rgb.shape[1], rgb.shape[0], 0])
            self.ax_vid.set_xlim(0, rgb.shape[1])
            self.ax_vid.set_ylim(rgb.shape[0], 0)
            self._vid_txt.set_text(
                f"  {fmt_mmss_short(xval)}  |  frame {frame_idx}"
                f"  |  speed {self.play_speed}x"
            )
            self.ax_vid.set_xticks([])
            self.ax_vid.set_yticks([])
        if not self._slider_updating:
            self._slider_updating = True
            self.slider_seek.set_val(
                np.clip(xval, self.t_display[0], self.t_display[-1])
            )
            self.slider_seek.valtext.set_text(fmt_mmss_short(xval))
            self._slider_updating = False

    # ── Animation ─────────────────────────────────────────────────
    def _on_animate(self, _frame):
        if not self.is_playing:
            return
        self.playhead_x += self.play_speed / self.fps
        if self.playhead_x > self.t_display[-1]:
            self.playhead_x = self.t_display[0]
        self.playhead_line.set_xdata([self.playhead_x, self.playhead_x])
        self._refresh_video_frame(self.playhead_x)
        self.fig.canvas.draw_idle()

    # ── Speed ─────────────────────────────────────────────────────
    def _set_speed(self, spd):
        self.play_speed = spd
        for s, btn in self._speed_btns:
            c = "#003050" if s == spd else "#1a1a2e"
            btn.color = c
            btn.ax.set_facecolor(c)
        self.fig.canvas.draw_idle()

    # ── Seek ──────────────────────────────────────────────────────
    def _on_seek(self, val):
        if self._slider_updating:
            return
        self.playhead_x = float(val)
        self.playhead_line.set_visible(True)
        self.playhead_line.set_xdata([self.playhead_x, self.playhead_x])
        self._refresh_video_frame(self.playhead_x)
        self.fig.canvas.draw_idle()

    def _seek_to(self, xval):
        self.playhead_line.set_visible(True)
        self.playhead_line.set_xdata([xval, xval])
        self._refresh_video_frame(xval)
        self.fig.canvas.draw_idle()

    # ── Play / Pause ──────────────────────────────────────────────
    def _toggle_play(self, _event=None):
        self.is_playing = not self.is_playing
        self.playhead_line.set_visible(True)
        if self.is_playing:
            self.btn_play.label.set_text("Pause")
            self.status.set_text(f"PLAYING {self.play_speed}x   [Space]=pause")
            self.status.set_color("#00ff00")
        else:
            self.btn_play.label.set_text("Play")
            self.status.set_text("PAUSED   [Space]=play   [A]=annotate")
            self.status.set_color("#ffaa00")
        self.fig.canvas.draw_idle()

    # ── Help ──────────────────────────────────────────────────────
    def _toggle_help(self):
        help_ax = self.ax_vid if self.ax_vid else self.ax
        if self._help_visible:
            for a in self._help_artists:
                a.remove()
            self._help_artists = []
            self._help_visible = False
        else:
            overlay = help_ax.add_patch(
                plt.Rectangle(
                    (0, 0), 1, 1,
                    transform=help_ax.transAxes,
                    color="#0a0a18", alpha=0.93, zorder=20,
                )
            )
            self._help_artists.append(overlay)
            y_pos = 0.97
            for text, size, color, bold in self.help_lines:
                t = help_ax.text(
                    0.03, y_pos, text, fontsize=size, color=color,
                    fontweight="bold" if bold else "normal",
                    transform=help_ax.transAxes,
                    va="top", ha="left", zorder=21,
                )
                self._help_artists.append(t)
                y_pos -= 0.034
            self._help_visible = True
        self.fig.canvas.draw_idle()

    # ── Motion ────────────────────────────────────────────────────
    def _on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._last_mouse_x = event.xdata
        if not self.is_playing:
            self.playhead_x = event.xdata
            self.playhead_line.set_xdata([self.playhead_x, self.playhead_x])
            self._refresh_video_frame(event.xdata)

    # ── Scroll zoom ───────────────────────────────────────────────
    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        scale = 1 / 1.2 if event.button == "up" else 1.2
        xlim = self.ax.get_xlim()
        xd = event.xdata
        w = (xlim[1] - xlim[0]) * scale
        r = (xlim[1] - xd) / (xlim[1] - xlim[0])
        self.ax.set_xlim([xd - w * (1 - r), xd + w * r])
        self.fig.canvas.draw_idle()

    # ── Keyboard ──────────────────────────────────────────────────
    def _on_key(self, event):
        k = event.key
        if k == " ":
            self._toggle_play()
            return
        if k == "right":
            self.playhead_x = min(self.playhead_x + 5.0, self.t_display[-1])
            self._seek_to(self.playhead_x)
            return
        if k == "left":
            self.playhead_x = max(self.playhead_x - 5.0, self.t_display[0])
            self._seek_to(self.playhead_x)
            return
        if k == "ctrl+z":
            self._undo()
            return
        if k == "h":
            self._toggle_help()
            return
        if k == "a":
            self.annotate_mode = not self.annotate_mode
            self.pending_start = None
            if self.pending_vline:
                self.pending_vline.remove()
                self.pending_vline = None
            if self.annotate_mode:
                keys_hint = " / ".join(self.label_keys)
                self.status.set_text(
                    f"Mode: ANNOTATE   drag / [ ] keys   "
                    f"press {keys_hint} to label   right-click=delete"
                )
                self.status.set_color("#ffaa00")
                self.span.active = True
            else:
                self.status.set_text(
                    "Mode: browse   [A] annotate   [S] save   "
                    "[Space] play   [H] help"
                )
                self.status.set_color("#88aaff")
                self.span.active = False
            self.fig.canvas.draw_idle()
            return
        if k == "s":
            self._save()
            return
        if k == "[":
            xpos = self._last_mouse_x
            self.annotate_mode = True
            self.span.active = True
            self.pending_start = xpos
            if self.pending_vline:
                self.pending_vline.remove()
            self.pending_vline = self.ax.axvline(
                xpos, color="#6bcb77", linewidth=1.5,
                linestyle="--", alpha=0.9, zorder=5,
            )
            self.status.set_text(
                f"start = {fmt_mmss(xpos)}   <- press ] to set stop"
            )
            self.fig.canvas.draw_idle()
            return
        if k == "]" and self.pending_start is not None:
            s, e = sorted([self.pending_start, self._last_mouse_x])
            self.pending_start = None
            if self.pending_vline:
                self.pending_vline.remove()
                self.pending_vline = None
            self._ask_label(s, e)

    # ── Click ─────────────────────────────────────────────────────
    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.button in (2, 3) and self.annotate_mode:
            self._remove_at(event.xdata)
        elif event.button == 1 and event.dblclick:
            self._edit_at(event.xdata)

    def _on_span_select(self, xmin, xmax):
        if not self.annotate_mode or abs(xmax - xmin) < 0.01:
            return
        self._ask_label(xmin, xmax)

    # ── Label picker ──────────────────────────────────────────────
    def _ask_label(self, start_d, stop_d, editing_idx=None):
        was_playing = self.is_playing
        self.is_playing = False
        preview = self.ax.axvspan(
            start_d, stop_d, alpha=0.15, color="#ffffff", zorder=4,
        )
        keys_hint = " / ".join(self.label_keys)
        action = (
            "Change label:"
            if editing_idx is not None
            else f"Range: {fmt_mmss(start_d)} -> {fmt_mmss(stop_d)}"
        )
        self.status.set_text(
            f"{action}   press  {keys_hint}  to label   Esc=cancel"
        )
        self.status.set_color("#ffffff")

        btn_axes, btn_store = [], []
        for i, lbl in enumerate(self.label_keys):
            bax = self.fig.add_axes([0.40 + i * 0.12, 0.36, 0.10, 0.20])
            color = self.label_colors[lbl]
            btn = mwidgets.Button(bax, lbl, color=color, hovercolor="#ffffff")
            btn.label.set_fontsize(22)
            btn.label.set_fontweight("bold")
            btn.label.set_color("#111111")
            btn_axes.append(bax)
            btn_store.append(btn)
        chosen = [None]

        def cleanup():
            preview.remove()
            for bax in btn_axes:
                self.fig.delaxes(bax)
            self.fig.canvas.mpl_disconnect(cid_key)
            for cid in btn_cids:
                self.fig.canvas.mpl_disconnect(cid)
            self.fig.canvas.draw_idle()

        def on_key(ev):
            if ev.key in self.label_keys:
                chosen[0] = ev.key
            elif ev.key == "escape":
                chosen[0] = None
            else:
                return
            cleanup()
            _finish()

        def make_cb(lbl):
            def cb(ev):
                chosen[0] = lbl
                cleanup()
                _finish()
            return cb

        cid_key = self.fig.canvas.mpl_connect("key_press_event", on_key)
        btn_cids = [
            btn.on_clicked(make_cb(lbl))
            for btn, lbl in zip(btn_store, self.label_keys)
        ]
        self.fig.canvas.draw_idle()

        def _finish():
            if chosen[0]:
                lbl = chosen[0]
                if editing_idx is not None:
                    old = self.ann_list[editing_idx]
                    self._undo_stack.append(
                        ("edit_undo", dict(old), dict(self.ann_artists[editing_idx]))
                    )
                    for art in self.ann_artists[editing_idx].values():
                        art.remove()
                    self.ann_list.pop(editing_idx)
                    self.ann_artists.pop(editing_idx)
                    self._add_annotation(
                        old["start_d"], old["stop_d"],
                        old["start"], old["stop"], lbl,
                    )
                    self.status.set_text(f"Label changed to [{lbl}]")
                else:
                    sr = float(np.interp(start_d, self.t_display, self.t_num))
                    er = float(np.interp(stop_d, self.t_display, self.t_num))
                    self._add_annotation(start_d, stop_d, sr, er, lbl)
                    self.status.set_text(
                        f"Added [{lbl}]  {fmt_mmss(start_d)} -> {fmt_mmss(stop_d)}"
                    )
                self.status.set_color("#88aaff")
            else:
                self.status.set_text("Cancelled.")
                self.status.set_color("#88aaff")
            self.is_playing = was_playing
            self.fig.canvas.draw_idle()

    # ── Edit at click ─────────────────────────────────────────────
    def _edit_at(self, xval):
        for i, ann in enumerate(self.ann_list):
            if ann["start_d"] <= xval <= ann["stop_d"]:
                self._ask_label(ann["start_d"], ann["stop_d"], editing_idx=i)
                return

    # ── Add annotation ────────────────────────────────────────────
    def _add_annotation(self, start_d, stop_d, start_raw, stop_raw, label,
                        _skip_save=False):
        color = self.label_colors.get(label, "#ffffff")
        patch = self.ax.axvspan(start_d, stop_d, alpha=0.22, color=color, zorder=3)
        line_s = self.ax.axvline(start_d, color=color, lw=1.3, alpha=0.85, zorder=4)
        line_e = self.ax.axvline(stop_d, color=color, lw=1.3, alpha=0.85, zorder=4)
        mid = (start_d + stop_d) / 2
        ylim = self.ax.get_ylim()
        ytop = ylim[1] - (ylim[1] - ylim[0]) * 0.04
        txt = self.ax.text(
            mid, ytop, label, color=color, fontsize=10,
            ha="center", va="top", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="#0f0f1a", ec=color, alpha=0.85),
        )
        ann_data = {
            "start_d": start_d, "stop_d": stop_d,
            "start": start_raw, "stop": stop_raw, "label": label,
        }
        art_data = {"patch": patch, "line_s": line_s, "line_e": line_e, "txt": txt}
        self.ann_list.append(ann_data)
        self.ann_artists.append(art_data)
        if not _skip_save:
            self._undo_stack.append(("add", ann_data, art_data))
        self._count_txt.set_text(f"Annotations: {len(self.ann_list)}")
        self.fig.canvas.draw_idle()
        print(
            f"[add] {label}  {fmt_mmss(start_d)} -> {fmt_mmss(stop_d)}"
            f"  (raw: {start_raw:.4f} -> {stop_raw:.4f})"
        )
        if not _skip_save:
            self._save()

    # ── Delete ────────────────────────────────────────────────────
    def _remove_at(self, xval):
        for i, ann in enumerate(self.ann_list):
            if ann["start_d"] <= xval <= ann["stop_d"]:
                self._undo_stack.append(("delete", dict(ann)))
                for art in self.ann_artists[i].values():
                    art.remove()
                self.ann_list.pop(i)
                self.ann_artists.pop(i)
                self._count_txt.set_text(f"Annotations: {len(self.ann_list)}")
                self.fig.canvas.draw_idle()
                print(f"[delete] annotation #{i}")
                self._save()
                return

    # ── Undo ──────────────────────────────────────────────────────
    def _undo(self):
        if not self._undo_stack:
            self.status.set_text("Nothing to undo!")
            self.status.set_color("#ff6b6b")
            self.fig.canvas.draw_idle()
            return
        action = self._undo_stack.pop()
        if action[0] == "add":
            _, ann_data, _ = action
            idx = next(
                (i for i, a in enumerate(self.ann_list) if a is ann_data), None,
            )
            if idx is not None:
                for art in self.ann_artists[idx].values():
                    art.remove()
                self.ann_list.pop(idx)
                self.ann_artists.pop(idx)
        elif action[0] == "delete":
            _, ann_data = action
            self._add_annotation(
                ann_data["start_d"], ann_data["stop_d"],
                ann_data["start"], ann_data["stop"],
                ann_data["label"], _skip_save=True,
            )
        elif action[0] == "edit_undo":
            _, old_ann, _ = action
            if self.ann_list:
                last = len(self.ann_list) - 1
                for art in self.ann_artists[last].values():
                    art.remove()
                self.ann_list.pop(last)
                self.ann_artists.pop(last)
            self._add_annotation(
                old_ann["start_d"], old_ann["stop_d"],
                old_ann["start"], old_ann["stop"],
                old_ann["label"], _skip_save=True,
            )
        self._count_txt.set_text(f"Annotations: {len(self.ann_list)}")
        self.status.set_text("Undone")
        self.status.set_color("#6bcb77")
        self.fig.canvas.draw_idle()
        self._save()

    # ── Save ──────────────────────────────────────────────────────
    def _save(self):
        rows = sorted(self.ann_list, key=lambda a: a["start"])
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["start", "stop", "label"])
            w.writeheader()
            for r in rows:
                w.writerow({
                    "start": r["start"], "stop": r["stop"], "label": r["label"],
                })
        msg = f"Saved {len(rows)} annotation(s) -> {self.output_file}"
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()
        print(f"[save] {msg}")

    # ── Load on startup ───────────────────────────────────────────
    def _load_existing(self):
        if not os.path.exists(self.output_file):
            return
        with open(self.output_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            sr = float(r["start"])
            er = float(r["stop"])
            sd = float(np.interp(sr, self.t_num, self.t_display))
            ed = float(np.interp(er, self.t_num, self.t_display))
            self._add_annotation(sd, ed, sr, er, r["label"], _skip_save=True)
        print(f"[load] {len(rows)} existing annotation(s)")
