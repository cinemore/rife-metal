#!/usr/bin/env python3
"""
Per-stage activation diff. Reads our debug-dumped tensors from RIFE_DUMP_DIR and
compares each to the matching tensor in reference_activations.npz.

Usage:
    python tools/diff-activations.py /tmp/dump Tests/fixtures/reference_activations.npz
"""

import sys
from pathlib import Path

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(peak / np.sqrt(mse))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: diff-activations.py <dump-dir> <reference.npz>", file=sys.stderr)
        return 2

    dump_dir = Path(sys.argv[1])
    reference = np.load(sys.argv[2])

    pass_count = 0
    fail_count = 0
    for name in reference.files:
        ref = reference[name]
        ours_path = dump_dir / f"{name}.npy"
        if not ours_path.exists():
            print(f"{name:30s} MISSING in dump")
            fail_count += 1
            continue
        ours = np.load(ours_path)
        if ours.shape != ref.shape:
            print(f"{name:30s} shape {ours.shape} != {ref.shape}")
            fail_count += 1
            continue
        peak = float(np.max(np.abs(ref))) or 1.0
        max_err = float(np.max(np.abs(ours - ref)))
        p = psnr(ours, ref, peak)
        ok = p > 30.0
        marker = "PASS" if ok else "FAIL"
        print(f"{name:30s} max_abs_err={max_err:.4f}  PSNR={p:.1f} dB  {marker}")
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    print(f"\n{pass_count} pass, {fail_count} fail")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
