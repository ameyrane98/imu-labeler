# IMU Labeler

**Free, open-source GUI tool for labeling IMU and time-series sensor data** — with optional synchronized video playback.

Built for researchers and engineers who work with wearable sensors (accelerometers, gyroscopes, magnetometers) and need a fast, keyboard-driven way to annotate activities in their data.

## Features

- **Interactive signal visualization** — plot accelerometer/gyroscope magnitude with zoom, pan, and real-time scrubbing
- **Synchronized video playback** — hover over the signal plot to scrub the video to that exact timestamp
- **Configurable labels** — define your own activity categories, colors, and keyboard shortcuts via YAML
- **Auto-detect sensor columns** — automatically finds `(x, y, z)` column groups in your CSV
- **Keyboard-driven workflow** — drag to select, press a key to label. Undo, delete, edit with shortcuts
- **Auto-save** — annotations are saved on every add/delete/edit
- **Timestamp resampling** — smooths jittery sensor timestamps using windowed-mean averaging
- **Video time-stretching** — automatically aligns video duration to sensor session via FFmpeg
- **Reference label overlay** — compare your annotations against a reference set
- **Signal filtering** — median, Gaussian, or moving-average filters built in
- **Works without video** — IMU-only mode for datasets without video

## Installation

```bash
pip install git+https://github.com/ameyrane98/imu-labeler.git
```

Or clone and install locally:

```bash
git clone https://github.com/ameyrane98/imu-labeler.git
cd imu-labeler
pip install -e .
```

### Requirements

- Python 3.9+
- FFmpeg (optional, for video stretching)
- Tk backend for matplotlib (`python3-tk` on Linux)

## Quick Start

### 1. Try with sample data

```bash
# Generate synthetic IMU data
python examples/generate_sample_data.py

# Launch the labeler
imu-labeler examples/sample_data/
```

### 2. Use with your own data

Point the tool at a directory containing your sensor CSV:

```bash
imu-labeler /path/to/your/data/
```

The tool auto-detects:
- The first `.csv` file as sensor data
- Any `.mp4` / `.avi` / `.mov` as the video
- Any `.json` as video timing metadata
- Sensor columns with `_x`, `_y`, `_z` suffixes

### 3. Customize labels

Create a `config.yaml` in your data directory:

```yaml
labels:
  w:
    name: Walking
    color: "#4ecdc4"
    key: w
  r:
    name: Running
    color: "#ff6b6b"
    key: r
  s:
    name: Standing
    color: "#ffd93d"
    key: s
  sit:
    name: Sitting
    color: "#6bcb77"
    key: t
```

## Controls

| Action | Key / Mouse |
|--------|------------|
| Play / Pause | `Space` |
| Jump +/- 5 sec | `Arrow keys` |
| Zoom | `Scroll wheel` |
| Toggle annotate mode | `A` |
| Select span | `Drag` on plot |
| Set start / stop | `[` then `]` |
| Assign label | Press label key (e.g., `d`, `e`, `v`) |
| Delete annotation | `Right-click` on colored span |
| Edit label | `Double-click` on colored span |
| Undo | `Ctrl+Z` |
| Save | `S` (also auto-saves) |
| Help | `H` |

## CLI Options

```
imu-labeler [data_dir] [options]

Arguments:
  data_dir          Directory with sensor CSV (default: current dir)

Options:
  -c, --config      Path to YAML config file
  --csv             Path to sensor CSV (overrides auto-detection)
  --video           Path to video file
  --no-video        Disable video even if one is found
```

## CSV Format

Your sensor CSV should have:

1. **A timestamp column** — named `timestamp`, `UTC_Timestamp`, `time`, or similar
2. **Sensor columns in (x, y, z) groups** — e.g., `accel_x`, `accel_y`, `accel_z`

Example:

```csv
timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
0.000,0.12,-0.34,9.78,1.2,-0.5,0.3
0.020,0.15,-0.31,9.81,1.1,-0.4,0.2
...
```

The tool computes the magnitude `sqrt(x^2 + y^2 + z^2)` for visualization.

## Output

Annotations are saved as CSV:

```csv
start,stop,label
1234567.1234,1234569.5678,w
1234572.0000,1234575.3456,r
```

## Background

This project grew out of a research annotation pipeline for labeling wearable IMU sensor data from finger-mounted sensors synchronized with video recordings. After building the internal tool for our research, I generalized it into this open-source tool so anyone working with IMU data can benefit from a fast, purpose-built labeling workflow.

## License

MIT
