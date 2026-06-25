# RifeMetal

[English](./README.md) | [简体中文](./README-zh.md)

原生 Apple Metal / MPSGraph RIFE 补帧库与命令行工具。它使用 MPSGraph 和自定义 Metal warp kernel，不依赖 Vulkan 或 MoltenVK。

该包内置 **Practical-RIFE v4.26** 权重（约 11 MB 资源），常规使用不需要额外管理模型文件。上游 MIT 许可归属见 [THIRD-PARTY.md](./THIRD-PARTY.md)。

## 演示

| 输入帧 A | 补帧结果（t=0.5） | 输入帧 B |
|:--:|:--:|:--:|
| ![Frame A](Tests/fixtures/frame_a.png) | ![Interpolated](Tests/fixtures/ours_mid.png) | ![Frame B](Tests/fixtures/frame_b.png) |

输入帧来自 [nihui/rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)（MIT）。中间帧由 RifeMetal 基于两张输入帧生成。

## 要求

- macOS 13+（iOS 16+ 目标可以编译，但尚未完成设备运行验证）
- Xcode 15.3+ / Swift 5.10+
- 推荐 Apple Silicon；Intel Mac 在 macOS 13-26 上受支持

## 安装

### Homebrew CLI

```bash
brew install cinemore/tap/rife-metal
```

Homebrew 包会安装与你 Mac CPU 架构匹配的原生命令行工具，并附带 `rife-v4.26.rmw` 权重。Apple Silicon Mac 会安装 `arm64` 包，Intel Mac 会安装 `x86_64` 包。如果没有传入 `--model`，Homebrew wrapper 会自动使用已安装的内置权重：

```bash
rife-metal -0 frame_a.png -1 frame_b.png -o mid.png
```

如需使用自定义权重，传入 `--model /path/to/file.rmw`。

### GitHub Release CLI

从 [latest release](https://github.com/cinemore/rife-metal/releases/latest) 下载适合你 Mac 的包：

- `rife-metal-macos-arm64.tar.gz`：Apple Silicon Mac
- `rife-metal-macos-x86_64.tar.gz`：Intel Mac
- `rife-metal-macos-universal.tar.gz`：需要同时覆盖两种架构时使用

```bash
tar -xzf rife-metal-macos-arm64.tar.gz
./rife-metal-macos-arm64/bin/rife-metal \
  -0 frame_a.png -1 frame_b.png -o mid.png \
  -m rife-metal-macos-arm64/share/rife-metal/rife-v4.26.rmw
```

### Swift Package Manager Library

在 `Package.swift` 中添加：

```swift
dependencies: [
    .package(url: "https://github.com/cinemore/rife-metal.git", branch: "main"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "RifeMetal", package: "rife-metal"),
        ]
    ),
]
```

在 Xcode 中可通过 **File -> Add Package Dependencies...** 添加仓库 URL。

## 库用法

```swift
import RifeMetal

let interpolator = try RifeInterpolator(
    configuration: .bundled(qualityTier: .hq)
)

let midframe: CGImage = try interpolator.interpolate(previous: frameA, current: frameB)

let frames: [CGImage] = try interpolator.interpolate(previous: frameA,
                                                     current:  frameB,
                                                     timesteps: [0.33, 0.67])
```

视频管线可使用 `CVPixelBuffer` 重载（BGRA8 / RGBA8）。

`RifeConfiguration.bundled(qualityTier:)` 会通过 SwiftPM 的 `Bundle.module` 解析内置 `rife-v4.26.rmw`。如果要加载其他权重，可以使用自定义 `modelURL:`：

```swift
let config = RifeConfiguration(modelURL: myWeightsURL, qualityTier: .balanced)
let interpolator = try RifeInterpolator(configuration: config)
```

## 质量档位

- `.hq`：全分辨率 IFNet 推理，质量最高
- `.balanced`：内部半分辨率网格（UHD mode），在大图上约节省 4 倍计算量，并通过最终全分辨率 warp + blend 保持锐度
- `.fast`：内部四分之一分辨率网格，目标是 4K 30fps 实时处理

## 从源码运行 CLI

```bash
swift build -c release

./.build/release/rife-metal \
  -0 frame_a.png -1 frame_b.png -o mid.png \
  -m Sources/RifeMetal/Resources/rife-v4.26.rmw

# 多帧输出（3x 补帧）：
./.build/release/rife-metal \
  -0 frame_a.png -1 frame_b.png -o out.png \
  -m Sources/RifeMetal/Resources/rife-v4.26.rmw \
  --timesteps 0.33,0.67
# -> out_t0.33.png, out_t0.67.png
```

命令行工具需要显式传入 `-m`，因为可执行目标不会继承库目标的 `Bundle.module` 资源解析。可以指向 `Sources/RifeMetal/Resources/` 中的内置 `.rmw`，也可以使用自定义模型文件。

## 自定义权重（高级 / 维护者）

如需从其他 Practical-RIFE checkpoint 重新生成内置 `.rmw`，使用转换工具：

```bash
# 1. 从 Practical-RIFE 下载 PyTorch checkpoint
#    (https://github.com/hzwer/Practical-RIFE)
#    并放到 third_party/Practical-RIFE/train_log/flownet.pkl

pip install -r tools/requirements.txt

# 2. 转换 PyTorch -> .rmw
python tools/convert-weights.py \
  --pytorch-checkpoint third_party/Practical-RIFE/train_log/flownet.pkl \
  --out Sources/RifeMetal/Resources/rife-v4.26.rmw
```

当前 converter 仅适配 v4.26 架构。其他网络结构需要更新转换逻辑。

## 验证

```bash
swift test
python3 tools/parity-test.py
python3 tools/cli-multi-test.py
```

`tools/parity-test.py` 会使用 CLI 处理 `Tests/fixtures/{frame_a,frame_b}.png`，并与 PyTorch v4.26 参考中间帧比较 PSNR。当前 hq 档位约为 37 dB，用于拦截明显数值错误。

如需调试架构移植中的中间激活，可设置 `RIFE_DUMP_DIR=/path/to/dir`，导出每个 stage 的 `flow.npy`、`mask.npy`、`feat.npy` 等文件，并与 PyTorch 参考结果比较。

## 状态与路线图

当前已发布：**Practical-RIFE v4.26** 权重、任意 timestep 的多帧补帧、三档质量模式（hq / balanced / fast）、macOS 13+。iOS 16+ 目标可以编译，但尚未在 iOS 设备上完成运行验证。

暂缓事项：lite 权重变体、HDR 支持（当前管线为 BGRA8 端到端）、iOS 设备验证、调用方提供多输出 buffer 的入口。

## 相关项目

- [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)：上游 PyTorch 实现，也是内置 v4.26 权重来源
- [nihui/rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)：基于 ncnn + Vulkan 的跨平台 RIFE 实现，也是演示输入帧来源

## 许可证

Apache License 2.0，见 [LICENSE](./LICENSE)。

内置 `rife-v4.26.rmw` 权重派生自 Practical-RIFE（MIT），上游归属见 [THIRD-PARTY.md](./THIRD-PARTY.md)。
