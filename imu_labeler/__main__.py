"""Entry point: ``python -m imu_labeler`` or ``imu-labeler`` CLI."""

import argparse
import os
import sys

from .config import load_config, DEFAULT_CONFIG
from .app import run


def main():
    parser = argparse.ArgumentParser(
        prog="imu-labeler",
        description="Interactive GUI for labeling IMU / time-series sensor data.",
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=".",
        help="Directory containing sensor CSV (and optionally video + JSON). Default: current dir.",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to a YAML config file. If omitted, uses config.yaml in data_dir or built-in defaults.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to the sensor CSV file directly (overrides auto-detection).",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to a video file to synchronize with sensor data.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video even if one is found.",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: '{data_dir}' is not a directory.")
        sys.exit(1)

    # Load config
    config_path = args.config
    if config_path is None:
        candidate = os.path.join(data_dir, "config.yaml")
        if os.path.exists(candidate):
            config_path = candidate

    cfg = load_config(config_path) if config_path else dict(DEFAULT_CONFIG)

    # CLI overrides
    if args.csv:
        cfg["csv_file"] = os.path.abspath(args.csv)
    if args.video:
        cfg["video_file"] = os.path.abspath(args.video)
    if args.no_video:
        cfg["video_file"] = None

    cfg["data_dir"] = data_dir
    run(cfg)


if __name__ == "__main__":
    main()
