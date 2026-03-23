"""Shared utilities."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


def fmt_mmss_short(x, _=None):
    x = abs(x)
    return f"{int(x) // 60:02d}:{int(x) % 60:02d}"


def fmt_mmss(x, _=None):
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}{int(x) // 60:02d}:{x - int(x) // 60 * 60:010.7f}"


def apply_filter(signal, filter_type, filter_k):
    if filter_type == "none":
        return signal
    if filter_type == "median":
        k = filter_k if filter_k % 2 == 1 else filter_k + 1
        return medfilt(signal, kernel_size=k)
    if filter_type == "moving":
        k = filter_k if filter_k % 2 == 1 else filter_k + 1
        return np.convolve(signal, np.ones(k) / k, mode="same")
    if filter_type == "gaussian":
        return gaussian_filter1d(signal, sigma=filter_k)
    raise ValueError(f"Unknown filter_type: '{filter_type}'")
