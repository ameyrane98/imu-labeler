"""Application orchestration — file discovery, preprocessing, GUI launch."""

import os
import re
import sys
import time
import threading

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from .preprocessing import resample_timestamps, stretch_video
from .gui import Annotator


def _find_csv(data_dir):
    """Find the first sensor CSV in data_dir."""
    for f in sorted(os.listdir(data_dir)):
        fl = f.lower()
        if fl.endswith(".csv") and "resample" not in fl and "annotation" not in fl:
            return os.path.join(data_dir, f)
    return None


def _find_video(data_dir):
    """Find the first video file in data_dir."""
    for f in sorted(os.listdir(data_dir)):
        fl = f.lower()
        if fl.endswith((".mp4", ".avi", ".mov", ".mkv")):
            return os.path.join(data_dir, f)
    return None


def _find_json(data_dir):
    """Find a JSON metadata file in data_dir."""
    for f in sorted(os.listdir(data_dir)):
        if f.lower().endswith(".json"):
            return os.path.join(data_dir, f)
    return None


def run(cfg):
    """Main entry: discover files, preprocess, launch annotator."""
    data_dir = cfg["data_dir"]

    # --- Resolve files ---
    csv_file = cfg.get("csv_file") or _find_csv(data_dir)
    if csv_file is None:
        print(f"Error: No sensor CSV found in '{data_dir}'.")
        print("Use --csv to specify the path directly.")
        sys.exit(1)

    video_file = cfg.get("video_file") or _find_video(data_dir)
    json_file = cfg.get("video_json") or _find_json(data_dir)
    ref_labels = cfg.get("reference_labels_file")
    output_file = cfg.get("output_file") or os.path.join(data_dir, "annotations.csv")

    print(f"[imu-labeler]")
    print(f"  CSV:    {csv_file}")
    print(f"  Video:  {video_file or '(none — IMU-only mode)'}")
    print(f"  JSON:   {json_file or '(none)'}")
    print(f"  Output: {output_file}")

    # --- Preprocessing ---
    resampled_csv = csv_file
    stretched_vid = video_file

    if cfg.get("resample_timestamps", True):
        stem = os.path.splitext(os.path.basename(csv_file))[0]
        resampled_csv = os.path.join(data_dir, stem + "_resampled.csv")
        if not os.path.exists(resampled_csv):
            print("\n[preprocessing] Resampling timestamps...")
            resample_timestamps(
                csv_file, resampled_csv,
                window=cfg.get("resample_window", 100),
                json_path=json_file,
                status_cb=lambda msg: print(f"  {msg}"),
            )
        else:
            print(f"  Resampled CSV already exists: {os.path.basename(resampled_csv)}")

    if video_file and json_file:
        out_dir = os.path.join(data_dir, "processed_videos")
        vbase = os.path.basename(video_file)
        stretched_vid = os.path.join(out_dir, f"STRETCHED_{vbase}")
        if not os.path.exists(stretched_vid):
            print("\n[preprocessing] Stretching video...")
            stretch_video(
                video_file, json_file, stretched_vid,
                fps=cfg.get("video_fps", 30),
                crf=cfg.get("video_crf", 18),
                status_cb=lambda msg: print(f"  {msg}"),
                progress_cb=lambda fr, tot, pct, el: print(
                    f"\r  frame {fr:,}/{tot:,}  {pct}%{el}", end="", flush=True
                ),
            )
            print()  # newline after progress
        else:
            print(f"  Stretched video exists: {os.path.basename(stretched_vid)}")
    elif video_file and not json_file:
        # Use video directly without stretching
        stretched_vid = video_file

    # --- Launch GUI ---
    print("\n[launching annotator]")
    Annotator(
        csv_path=resampled_csv,
        video_path=stretched_vid,
        output_file=output_file,
        labels_cfg=cfg.get("labels", {}),
        ref_labels_file=ref_labels,
        filter_type=cfg.get("filter_type", "median"),
        filter_k=cfg.get("filter_k", 3),
        speed_options=cfg.get("speed_options", [0.5, 1.0, 2.0, 4.0]),
        sensor_groups=cfg.get("sensor_groups"),
        timestamp_column=cfg.get("timestamp_column"),
    )
