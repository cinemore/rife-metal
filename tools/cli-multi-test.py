#!/usr/bin/env python3
"""
End-to-end CLI multi-output test: runs `rife-metal --timesteps 0.33,0.67` on the fixtures
and asserts that the templated output paths exist, are valid PNGs, and have the expected
dimensions.

Acceptance: both files exist, both decode as 640x360 RGB PNGs.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", default=".build/release/rife-metal")
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--model", type=Path,
                        default=Path("Sources/RifeMetal/Resources/rife-v4.26.rmw"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp"))
    args = parser.parse_args()

    template = args.out_dir / "cli_multi_test.png"
    expected_paths = [
        args.out_dir / "cli_multi_test_t0.33.png",
        args.out_dir / "cli_multi_test_t0.67.png",
    ]
    # Clean any stale outputs from a prior run.
    for p in expected_paths + [template]:
        p.unlink(missing_ok=True)

    cmd = [
        args.cli,
        "-0", str(args.fixtures / "frame_a.png"),
        "-1", str(args.fixtures / "frame_b.png"),
        "-o", str(template),
        "-m", str(args.model),
        "--timesteps", "0.33,0.67",
    ]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"CLI exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    # The bare template path should NOT exist; only the templated ones.
    if template.exists():
        print(f"unexpected: bare template {template} was written", file=sys.stderr)
        return 1

    for p in expected_paths:
        if not p.exists():
            print(f"missing expected output: {p}", file=sys.stderr)
            return 1
        with Image.open(p) as im:
            if im.size != (640, 360):
                print(f"{p}: expected size 640x360, got {im.size}", file=sys.stderr)
                return 1
            if im.mode not in ("RGB", "RGBA"):
                print(f"{p}: expected RGB/RGBA mode, got {im.mode}", file=sys.stderr)
                return 1

    print(f"OK: {len(expected_paths)} outputs at expected paths and dimensions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
