"""Configuration loading and defaults."""

import copy
import yaml

DEFAULT_CONFIG = {
    # --- Labels ---
    "labels": {
        "w": {"name": "Walking",  "color": "#4ecdc4", "key": "w"},
        "r": {"name": "Running",  "color": "#e76f51", "key": "r"},
        "s": {"name": "Sitting",  "color": "#a78bfa", "key": "s"},
        "t": {"name": "Standing", "color": "#f4a261", "key": "t"},
    },

    # --- Sensor columns ---
    # Auto-detect if not specified. Set to a list like ["accel_x", "accel_y", "accel_z"]
    # to manually choose columns. The tool computes magnitude = sqrt(x^2 + y^2 + z^2).
    "sensor_groups": None,  # auto-detect

    # --- Timestamp column ---
    "timestamp_column": None,  # auto-detect: looks for UTC_Timestamp, timestamp, time, etc.

    # --- Signal processing ---
    "filter_type": "median",   # none | median | moving | gaussian
    "filter_k": 3,

    # --- Video ---
    "video_file": None,        # auto-detected from data_dir if not set
    "video_json": None,        # JSON with video timing metadata (optional)
    "video_fps": 30,
    "video_crf": 18,

    # --- Playback ---
    "speed_options": [0.5, 1.0, 2.0, 4.0],

    # --- Preprocessing ---
    "resample_timestamps": True,
    "resample_window": 100,

    # --- Reference labels ---
    "reference_labels_file": None,

    # --- Output ---
    "output_file": None,  # defaults to <data_dir>/annotations.csv
}


def load_config(path):
    """Load config from YAML, merged over defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg.update(user_cfg)
    return cfg
