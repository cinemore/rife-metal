#!/usr/bin/env python3
"""
Build rife-metal in release config, run the --bench-stream flag,
print results in a readable form.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli",      default=".build/release/rife-metal")
    parser.add_argument("--fixtures", type=Path, default=Path("Tests/fixtures"))
    parser.add_argument("--model",    type=Path,
                        default=Path("Sources/RifeMetal/Resources/rife-v4.26.rmw"))
    parser.add_argument("--tier",     default="hq",
                        choices=["hq", "balanced", "fast"])
    parser.add_argument("--iters",    type=int, default=30)
    args = parser.parse_args()

    subprocess.run(["swift", "build", "-c", "release"], check=True)

    print(f"running stream bench: tier={args.tier}, iters={args.iters}")
    cmd = [
        str(args.cli),
        "--bench-stream", str(args.iters),
        "-0", str(args.fixtures / "frame_a.png"),
        "-1", str(args.fixtures / "frame_b.png"),
        "-m", str(args.model),
        "-o", "/tmp/ignored.png",
        "--tier", args.tier,
    ]
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
