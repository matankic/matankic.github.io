#!/usr/bin/env python3
"""
Convert all *dB.wav files in this directory to:
- Numpy arrays (*.npy)
- JSON files (*.json) with {"sampleRate": int, "data": [floats...]}

Assumptions:
- WAV files are PCM 8/16/24/32-bit or IEEE float 32-bit
- If stereo/multi-channel, they are averaged to mono

Usage:
    python3 convert_wav_signals.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
import wave
from glob import glob

try:
    import numpy as np
except Exception as e:
    sys.stderr.write("This script requires numpy. Install via `python3 -m pip install numpy`\n")
    raise


def read_wav_mono_float32(path: str) -> tuple[float, np.ndarray]:
    with wave.open(path, 'rb') as wf:
        num_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()  # bytes per sample
        framerate = wf.getframerate()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sampwidth == 1:
        dtype = np.uint8
        arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sampwidth == 2:
        dtype = np.int16
        arr = np.frombuffer(raw, dtype=dtype).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit little-endian to int32
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        b = (a[:, 0].astype(np.uint32) |
             (a[:, 1].astype(np.uint32) << 8) |
             (a[:, 2].astype(np.uint32) << 16))
        # sign extend 24-bit
        mask = 1 << 23
        b = (b ^ mask) - mask
        arr = b.astype(np.float32) / (1 << 23)
    elif sampwidth == 4:
        # Could be int32 or float32; try interpreting as float32 first
        try:
            arr = np.frombuffer(raw, dtype=np.float32)
            # If values are all NaN or inf, fall back to int32
            if not np.isfinite(arr).all():
                raise ValueError
        except Exception:
            arr = (np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    if num_channels > 1:
        arr = arr.reshape(-1, num_channels).mean(axis=1)
    return float(framerate), arr.astype(np.float32)


def save_outputs(base_out: str, sr: float, data: np.ndarray) -> None:
    np.save(base_out + '.npy', data.astype(np.float32))
    out_json = {
        "sampleRate": int(sr),
        "data": data.astype(np.float32).tolist(),
    }
    with open(base_out + '.json', 'w') as f:
        json.dump(out_json, f)


def main() -> int:
    wav_files = sorted(glob('*dB.wav'))
    if not wav_files:
        print('No *dB.wav files found in', os.getcwd())
        return 1

    for wav_path in wav_files:
        try:
            sr, data = read_wav_mono_float32(wav_path)
            base = os.path.splitext(wav_path)[0]
            save_outputs(base, sr, data)
            print(f'Converted {wav_path} -> {base}.npy, {base}.json (sr={int(sr)}, N={len(data)})')
        except Exception as e:
            print(f'Failed to convert {wav_path}: {e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


