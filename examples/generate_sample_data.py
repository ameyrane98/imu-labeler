"""Generate synthetic IMU data for testing IMU Labeler.

Run: python examples/generate_sample_data.py
This creates examples/sample_data/sample_imu.csv with realistic-looking
accelerometer and gyroscope signals.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

DURATION = 120  # seconds
SAMPLE_RATE = 50  # Hz
N = DURATION * SAMPLE_RATE

t = np.linspace(0, DURATION, N)

# Simulate accelerometer — gravity + movement bursts
base_accel = 9.81
noise = np.random.normal(0, 0.3, N)

# Add activity bursts (simulating hand movements)
activity_signal = np.zeros(N)
burst_times = [(10, 18), (25, 35), (45, 52), (60, 70), (80, 88), (95, 108)]
for start, end in burst_times:
    mask = (t >= start) & (t <= end)
    activity_signal[mask] = np.sin(2 * np.pi * 3 * (t[mask] - start)) * 2.5

accel_x = noise + activity_signal * 0.7
accel_y = noise * 0.8 + activity_signal * 0.5
accel_z = base_accel + noise * 0.5 + activity_signal * 0.3

# Simulate gyroscope
gyro_x = np.random.normal(0, 5, N) + activity_signal * 20
gyro_y = np.random.normal(0, 5, N) + activity_signal * 15
gyro_z = np.random.normal(0, 3, N) + activity_signal * 10

# Add jitter to timestamps (realistic sensor behavior)
jitter = np.cumsum(np.random.normal(0, 0.0002, N))
timestamps = t + jitter

df = pd.DataFrame({
    "timestamp": timestamps,
    "accel_x": accel_x,
    "accel_y": accel_y,
    "accel_z": accel_z,
    "gyro_x": gyro_x,
    "gyro_y": gyro_y,
    "gyro_z": gyro_z,
})

out_dir = os.path.join(os.path.dirname(__file__), "sample_data")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "sample_imu.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} samples -> {out_path}")
print(f"\nTry it:  imu-labeler {out_dir}")
