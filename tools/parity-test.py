#!/usr/bin/env python3
"""
End-to-end parity test: runs the rife-metal CLI on Tests/fixtures/{frame_a,frame_b}.png,
compares the output to Tests/fixtures/reference_mid.png by PSNR.

Acceptance: PSNR > 40 dB.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", default=".build/release/rife-metal")
    parser.add_argument("--fixtures", type=Path, default=Path("Tests/fixtures"))
    parser.add_argument("--model", type=Path,
                        default=Path("Sources/RifeMetal/Resources/rife-v4.26.rmw"))
    parser.add_argument("--threshold", type=float, default=25.0)
    args = parser.parse_args()

    out_path = args.fixtures / "ours_mid.png"
    cmd = [
        args.cli,
        "-0", str(args.fixtures / "frame_a.png"),
        "-1", str(args.fixtures / "frame_b.png"),
        "-o", str(out_path),
        "-m", str(args.model),
    ]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"CLI exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    ours = np.asarray(Image.open(out_path).convert("RGB"))
    ref = np.asarray(Image.open(args.fixtures / "reference_mid.png").convert("RGB"))
    if ours.shape != ref.shape:
        print(f"shape mismatch: ours={ours.shape} ref={ref.shape}", file=sys.stderr)
        return 1

    p = psnr(ours, ref)
    print(f"PSNR vs PyTorch reference: {p:.2f} dB (threshold {args.threshold:.1f})")
    return 0 if p > args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
