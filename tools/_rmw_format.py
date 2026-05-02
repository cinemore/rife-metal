"""
Shared writer for the .rmw weight file format.

File layout:
  [0:4]   Magic "RMW1"
  [4:8]   Version uint32 LE = 1
  [8:12]  Header length uint32 LE
  [12:?]  Header UTF-8 JSON
  [?:?+p] Padding to 16-byte alignment
  [?:end] Concatenated tensor blob (fp16 bytes)

Header JSON:
  {
    "model": str,
    "ifblock_channels": [int, ...],
    "scale_list": [int, ...],
    "input_layout": "NHWC",
    "tensors": [
       {"name": str, "dtype": "f16", "shape": [int, ...], "offset": int, "size": int},
       ...
    ]
  }
"""

import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np


MAGIC = b"RMW1"
VERSION = 1


def write_rmw(
    out_path: Path,
    *,
    model: str,
    ifblock_channels: list[int],
    scale_list: list[int],
    tensors: Iterable[tuple[str, np.ndarray]],
) -> None:
    """Writes a .rmw file. 4D conv weights must already be permuted to HWIO order."""
    blob = bytearray()
    entries = []
    for name, array in tensors:
        if array.dtype != np.float16:
            array = array.astype(np.float16)
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        offset = len(blob)
        blob.extend(array.tobytes())
        entries.append({
            "name": name,
            "dtype": "f16",
            "shape": list(array.shape),
            "offset": offset,
            "size": int(array.nbytes),
        })

    header = {
        "model": model,
        "ifblock_channels": ifblock_channels,
        "scale_list": scale_list,
        "input_layout": "NHWC",
        "tensors": entries,
    }
    header_json = json.dumps(header, sort_keys=True).encode("utf-8")

    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        unpadded = 12 + len(header_json)
        pad = (16 - (unpadded % 16)) % 16
        f.write(b"\x00" * pad)
        f.write(blob)
