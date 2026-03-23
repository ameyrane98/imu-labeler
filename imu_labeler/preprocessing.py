"""Timestamp resampling and video stretching."""

import json
import os
import re
import subprocess

import numpy as np
import pandas as pd


def resample_timestamps(csv_path, out_csv, window=100, json_path=None,
                        status_cb=None):
    """Smooth jittery sensor timestamps using windowed-mean averaging."""
    if status_cb:
        status_cb("Reading sensor CSV...")
    data = pd.read_csv(csv_path)

    # Find timestamp column
    ts_col = _find_timestamp_col(data)
    if ts_col is None:
        if status_cb:
            status_cb("No timestamp column found — skipping resampling")
        data.to_csv(out_csv, index=False)
        return

    if status_cb:
        status_cb("Smoothing timestamps (windowed mean)...")
    delta_t = data[ts_col].diff().dropna()
    n = len(delta_t)
    new_deltas = []
    for i in range(0, n, window):
        chunk = delta_t.iloc[i : i + window]
        mean_dt = chunk.mean()
        new_deltas.extend([mean_dt] * len(chunk))

    coeff = delta_t.sum() / sum(new_deltas) if sum(new_deltas) > 0 else 1.0
    corrected = [dt * coeff for dt in new_deltas]

    t0 = data[ts_col].iloc[0]
    new_ts = [t0]
    for dt in corrected:
        new_ts.append(new_ts[-1] + dt)
    data["Resampled_Timestamp"] = new_ts[: len(data)]

    # Video sync alignment
    video_stime = None
    if json_path and os.path.exists(json_path):
        with open(json_path, "r") as f:
            meta = json.load(f)
        video_stime = meta.get("video1_stime") or meta.get("video_start_time")

    if video_stime is not None:
        data["Video_Sync_Time"] = data["Resampled_Timestamp"] - video_stime
        if status_cb:
            status_cb(f"Video sync aligned (offset: {video_stime:.2f})")

    if status_cb:
        status_cb(f"Saving -> {os.path.basename(out_csv)}")
    data.to_csv(out_csv, index=False)
    print(f"[resample] {len(data):,} rows coeff={coeff:.6f} -> {out_csv}")


def stretch_video(video_path, json_path, out_vid, fps=30, crf=18,
                  status_cb=None, progress_cb=None):
    """Time-stretch video to match sensor session duration via FFmpeg."""
    os.makedirs(os.path.dirname(out_vid) or ".", exist_ok=True)

    with open(json_path, "r") as f:
        meta = json.load(f)
    etime_key = next(
        (k for k in ("video1_etime", "video_end_time") if k in meta), None
    )
    stime_key = next(
        (k for k in ("video1_stime", "video_start_time") if k in meta), None
    )
    if not etime_key or not stime_key:
        raise ValueError(
            "JSON must contain video start/end time keys "
            "(video1_stime/video1_etime or video_start_time/video_end_time)"
        )
    target_dur = meta[etime_key] - meta[stime_key]

    cmd_probe = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    orig_dur = float(subprocess.check_output(cmd_probe).decode().strip())
    stretch = target_dur / orig_dur
    total_frames = int(orig_dur * fps * stretch)

    if status_cb:
        status_cb(f"{orig_dur:.1f}s -> {target_dur:.1f}s (x{stretch:.4f})")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"setpts={stretch}*PTS,fps={fps}",
        "-fps_mode", "cfr",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "veryfast",
        out_vid,
    ]
    proc = subprocess.Popen(
        cmd, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
    )

    frame_pat = re.compile(r"frame=\s*(\d+)")
    time_pat = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
    for line in proc.stderr:
        fm = frame_pat.search(line)
        tm = time_pat.search(line)
        if fm and progress_cb:
            fr = int(fm.group(1))
            pct = min(100, fr * 100 // max(1, total_frames))
            elapsed = ""
            if tm:
                h, m, s = int(tm.group(1)), int(tm.group(2)), float(tm.group(3))
                elapsed = f"  {h:02d}:{m:02d}:{int(s):02d} elapsed"
            progress_cb(fr, total_frames, pct, elapsed)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed (code {proc.returncode})")
    print(f"[video] Done -> {out_vid}")


def _find_timestamp_col(df):
    """Auto-detect the timestamp column."""
    candidates = [
        "UTC_Timestamp", "utc_timestamp",
        "timestamp", "Timestamp", "TIMESTAMP",
        "time", "Time", "TIME",
        "epoch", "unix_time",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: first column with 'time' in the name
    for c in df.columns:
        if "time" in c.lower():
            return c
    return None
