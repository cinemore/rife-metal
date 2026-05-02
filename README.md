# RifeMetal

Native Apple Silicon RIFE frame interpolation — Swift Package + CLI.
Uses MPSGraph + a custom Metal warp kernel; no Vulkan/MoltenVK dependency.

The package ships **Practical-RIFE v4.26** weights inline (~11 MB resource);
no model file management needed for typical use. See [THIRD-PARTY.md](./THIRD-PARTY.md)
for the upstream MIT attribution.

## Library usage

```swift
import RifeMetal

let interpolator = try RifeInterpolator(
    configuration: .bundled(qualityTier: .hq)   // uses the package's bundled weights
)

let midframe = try interpolator.interpolate(previous: frameA, current: frameB)
// or for arbitrary timesteps (e.g. 3x interpolation):
let frames = try interpolator.interpolate(previous: frameA, current: frameB,
                                           timesteps: [0.33, 0.67])
```

`RifeConfiguration.bundled(qualityTier:)` resolves the bundled `rife-v4.26.rmw`
via SwiftPM's per-target `Bundle.module`. Power users can pass a custom
`modelURL:` to load alternate weights — see "Custom weights" below.

## Quality tiers

- `.hq` — full-res IFNet inference. Highest quality.
- `.balanced` — half-resolution internal grid (UHD mode). ~4× cheaper than hq
  on large frames; sharpness preserved via final full-res warp+blend.
- `.fast` — quarter-resolution internal grid. Targets 4K 30fps real-time.

## CLI

```bash
swift build -c release

./.build/release/rife-metal \
  -0 frame_a.png -1 frame_b.png -o mid.png \
  -m Sources/RifeMetal/Resources/rife-v4.26.rmw

# Multi-frame (3x interpolation):
./.build/release/rife-metal \
  -0 frame_a.png -1 frame_b.png -o out.png \
  -m Sources/RifeMetal/Resources/rife-v4.26.rmw \
  --timesteps 0.33,0.67
# → out_t0.33.png, out_t0.67.png
```

## Custom weights (advanced / maintainers)

To regenerate the bundled `.rmw` from a different Practical-RIFE checkpoint
(e.g. a future v4.x or a `lite` variant), use the conversion tool:

```bash
# 1. Download a PyTorch checkpoint from Practical-RIFE
#    (https://github.com/hzwer/Practical-RIFE — Google Drive / 百度网盘 links)
#    and place at third_party/Practical-RIFE/train_log/flownet.pkl.

pip install -r tools/requirements.txt   # torch + numpy + Pillow

# 2. Convert PyTorch → .rmw
python tools/convert-weights.py \
  --pytorch-checkpoint third_party/Practical-RIFE/train_log/flownet.pkl \
  --out Sources/RifeMetal/Resources/rife-v4.26.rmw
```

The converter is currently v4.26-only (5 IFBlocks, encoder Head, ResConv with
`beta`, 8-channel feat propagation). Other architectures need converter updates.

---

## Validation

The bundled `.rmw` ships with parity verified against the upstream Practical-RIFE
v4.26 PyTorch reference. To re-verify locally:

```bash
swift test                  # 21 XCTests, including PSNR + multi-timestep + shape sanity
python3 tools/parity-test.py # CLI parity vs PyTorch v4.26 reference; expects 37+ dB on hq
python3 tools/cli-multi-test.py  # multi-output filename templating
```

The `tools/parity-test.py` script runs the CLI on `tests/fixtures/{frame_a,frame_b}.png`
and reports PSNR against `tests/fixtures/reference_mid.png` (PyTorch v4.26 midframe).
hq tier currently measures **37.43 dB**; threshold gates against gross numerical bugs.

For per-stage activation debugging on architecture-port work, set `RIFE_DUMP_DIR`
to dump per-stage `stage{0..N}_flow.npy` / `mask.npy` / `feat.npy` files; compare
against a PyTorch reference set regenerated via `tools/_emit_reference.py`. See
the v2.1 port spec for the methodology.

## Status & roadmap

Currently shipping: **Practical-RIFE v4.26** weights, multi-frame interpolation
(arbitrary timesteps via `interpolate(prev, curr, timesteps:)`), three quality
tiers (hq/balanced/fast), macOS 13+. iOS 16+ targets compile but have not been
runtime-validated on iOS devices.

Deferred: lite weight variants, HDR support (current pipeline is BGRA8
end-to-end), iOS device validation, caller-supplied multi-output buffer entry
(for IOSurface pool reuse).

## License

Apache License 2.0 — see [LICENSE](./LICENSE).

The bundled `rife-v4.26.rmw` weights are derived from Practical-RIFE (MIT) —
see [THIRD-PARTY.md](./THIRD-PARTY.md) for upstream attribution.
