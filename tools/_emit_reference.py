"""
Runs the original Practical-RIFE PyTorch model on a fixed pair of frames and dumps:
  - reference_mid.png         : interpolated frame (uint8)
  - reference_activations.npz : per-stage activations as fp32

Frame fixtures (frame_a.png, frame_b.png) must already exist in the target directory.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as _F
from PIL import Image


def emit_reference(checkpoint_path: Path, fixtures_dir: Path) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    frame_a_path = fixtures_dir / "frame_a.png"
    frame_b_path = fixtures_dir / "frame_b.png"
    if not frame_a_path.exists() or not frame_b_path.exists():
        print(f"missing frame_a.png / frame_b.png in {fixtures_dir}; provide them first.",
              file=sys.stderr)
        return

    img_a = np.asarray(Image.open(frame_a_path).convert("RGB"), dtype=np.float32) / 255.0
    img_b = np.asarray(Image.open(frame_b_path).convert("RGB"), dtype=np.float32) / 255.0

    a = torch.from_numpy(img_a).permute(2, 0, 1).unsqueeze(0).contiguous()
    b = torch.from_numpy(img_b).permute(2, 0, 1).unsqueeze(0).contiguous()
    timestep = torch.tensor([0.5])

    # v4.26 IFBlock internal stride is 4× (conv0 stride=2 twice); combined with scale=8,
    # the model requires H and W to be multiples of 64.  Pad to the next multiple and
    # crop the output back to the original size.
    orig_h, orig_w = a.shape[2], a.shape[3]
    stride = 64
    pad_h = (stride - orig_h % stride) % stride
    pad_w = (stride - orig_w % stride) % stride
    if pad_h > 0 or pad_w > 0:
        a = _F.pad(a, (0, pad_w, 0, pad_h), mode="replicate")
        b = _F.pad(b, (0, pad_w, 0, pad_h), mode="replicate")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "Practical-RIFE"))
    try:
        from train_log.IFNet_HDv3 import IFNet  # noqa: E402
    except ImportError as e:
        print(f"cannot import IFNet from third_party/Practical-RIFE/: {e}", file=sys.stderr)
        print("Run the gdown setup in tools/convert-weights.py docstring.", file=sys.stderr)
        return

    model = IFNet()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    # Strip training-only submodules (teacher, caltime) that are commented out in the
    # inference IFNet but present in the training checkpoint.
    state = {k: v for k, v in state.items() if not k.startswith(("teacher.", "caltime."))}
    model.load_state_dict(state, strict=True)
    model.eval()

    activations = {}
    handles = []
    if hasattr(model, "block0"):
        # v4.26 stores blocks as separate attributes (block0..block4), not a ModuleList.
        # Hook each one for per-stage flow/mask debug dump.
        block_attrs = [f"block{i}" for i in range(5)]
        for i, name in enumerate(block_attrs):
            block = getattr(model, name)
            def make_hook(idx):
                def hook(module, inp, out):
                    if isinstance(out, tuple):
                        for k, t in enumerate(out):
                            activations[f"stage{idx}_out{k}"] = t.detach().cpu().numpy()
                return hook
            handles.append(block.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        # v4.26 has 5 IFBlocks; pass the matching 5-element scale list explicitly.
        # (The forward() default [8,4,2,1] is a 4-element v4.6 remnant.)
        result = model(torch.cat([a, b], dim=1), timestep, scale_list=[16, 8, 4, 2, 1], training=False)
        # v4.26 IFNet returns (flow_list, mask, merged) per IFNet_HDv3.py.
        if isinstance(result, tuple) and len(result) == 3:
            _, _, merged = result
        else:
            print(f"unexpected IFNet output shape: {type(result).__name__}", file=sys.stderr)
            return
        # merged is a list of 5 (warped0, warped1) tuples for stages 0..3 and a single
        # blended tensor for stage 4 (the final midframe). See IFNet_HDv3.py forward().
        mid = merged[4]
        if isinstance(mid, tuple):
            # Defensive: if v4.26 returns merged[4] as a tuple, take the blended tensor.
            mid = mid[0] if mid[0].shape[1] == 3 else mid[-1]

    for h in handles:
        h.remove()

    # Crop padding back to original frame size.
    mid = mid[:, :, :orig_h, :orig_w]
    mid_np = mid.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    Image.fromarray((mid_np * 255).astype(np.uint8)).save(fixtures_dir / "reference_mid.png")
    if activations:
        np.savez(fixtures_dir / "reference_activations.npz", **activations)
    print(f"emitted reference to {fixtures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-checkpoint", type=Path, required=True,
                        help="Path to flownet.pkl (Practical-RIFE v4.26 checkpoint)")
    parser.add_argument("--fixtures-dir", type=Path, default=Path("Tests/fixtures"),
                        help="Directory containing frame_a.png / frame_b.png (outputs written here)")
    args = parser.parse_args()
    emit_reference(args.pytorch_checkpoint, args.fixtures_dir)
