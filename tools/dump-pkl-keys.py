#!/usr/bin/env python3
"""
Dev helper: prints all keys in a Practical-RIFE flownet.pkl checkpoint, with shapes.

Usage:
    python3 tools/dump-pkl-keys.py --pytorch-checkpoint third_party/Practical-RIFE/train_log/flownet.pkl
"""

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-checkpoint", required=True, type=Path)
    args = parser.parse_args()

    state = torch.load(args.pytorch_checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}

    for k in sorted(state.keys()):
        v = state[k]
        if hasattr(v, "shape"):
            print(f"{k}\t{tuple(v.shape)}")
        else:
            print(f"{k}\t{type(v).__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
