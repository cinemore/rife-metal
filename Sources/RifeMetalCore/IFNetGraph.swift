import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import CoreVideo

public enum IFNetError: Error {
    case dimensionsMismatch
    case textureAllocationFailed
    case commandBufferFailed(String)
    case invalidInternalScale(String)
}

/// Encapsulates the compiled IFNet executables and the runtime warp orchestration.
/// One instance is bound to a fixed (width, height, internalScale); rebuild for new
/// resolutions or new tiers.
public final class IFNetGraph {
    private var accumulationRounding: SIMD2<UInt32>?

    package let context: InferenceContext
    private let weights: WeightStore
    /// Full padded resolution (the output dims).
    private let width: Int
    private let height: Int
    /// Resolution at which IFNet stages run. Equal to (width, height) when internalScale == 1.
    private let internalWidth: Int
    private let internalHeight: Int
    private let internalScale: Double
    private let scaleList: [Int]
    // 仅 Balanced 融合裁剪；空阶段模型仍使用完整尺寸的转换回退。
    package var supportsCroppedOutput: Bool { internalScale == 0.5 && !scaleList.isEmpty }
    private let textureCache: CVMetalTextureCache

    /// Pre-allocated buffer/texture pool — eliminates per-call GPU heap traffic.
    private let pool: BufferPool

    /// Pairs an MPSGraphExecutable with the ordered list of placeholder tensors
    /// it was compiled against — required for deterministic input ordering at runtime.
    private struct CompiledStage {
        let executable: MPSGraphExecutable
        /// Placeholders in the exact order the feeds dict was built; mirrors the order
        /// in which runtime MPSGraphTensorData values must be supplied to encode(to:inputs:).
        let inputTensors: [MPSGraphTensor]
    }

    private let stageExecutables: [CompiledStage]

    /// Pre-compiled encoder Head (img → 4-ch encoder features); run once per input image.
    private let encoderExecutable: MPSGraphExecutable

    /// Pre-compiled element-wise add for two [1, internalH, internalW, 4] flow tensors (fp16).
    private let addFlowExecutable: MPSGraphExecutable
    /// Pre-compiled element-wise add for two [1, internalH, internalW, 1] mask tensors (fp16).
    private let addMaskExecutable: MPSGraphExecutable


    public init(context: InferenceContext,
                weights: WeightStore,
                width: Int,
                height: Int,
                internalScale: Double = 1.0) throws {
        guard internalScale > 0.0 && internalScale <= 1.0 else {
            throw IFNetError.invalidInternalScale("internalScale must be in (0, 1], got \(internalScale)")
        }
        let intW = Int((Double(width) * internalScale).rounded(.toNearestOrAwayFromZero))
        let intH = Int((Double(height) * internalScale).rounded(.toNearestOrAwayFromZero))
        guard intW >= 32, intH >= 32, intW % 32 == 0, intH % 32 == 0 else {
            throw IFNetError.invalidInternalScale(
                "internalScale=\(internalScale) with padded \(width)x\(height) yields internal " +
                "\(intW)x\(intH); must be >=32 and multiple of 32 (caller's padding multiple should " +
                "be 32/internalScale)."
            )
        }

        self.context = context
        self.weights = weights
        self.width = width
        self.height = height
        self.internalWidth = intW
        self.internalHeight = intH
        self.internalScale = internalScale
        self.scaleList = weights.header.scaleList

        var cache: CVMetalTextureCache?
        let err = CVMetalTextureCacheCreate(
            kCFAllocatorDefault, nil,
            context.device, nil, &cache
        )
        guard err == kCVReturnSuccess, let cache else {
            throw IFNetError.textureAllocationFailed
        }
        self.textureCache = cache

        // Allocate the persistent buffer/texture pool before compiling executables so that
        // any out-of-memory error surfaces early (at warmup, not mid-inference).
        let pool = try BufferPool(
            device: context.device,
            paddedW: width,
            paddedH: height,
            internalWidth: intW,
            internalHeight: intH,
            internalScale: internalScale
        )
        self.pool = pool
        if ProcessInfo.processInfo.environment["RIFE_VERBOSE"] == "1" {
            let heapStatus = pool.heap != nil ? "MTLHeap" : "per-buffer fallback"
            let heapMB = pool.heap.map { Double($0.size) / 1024 / 1024 } ?? 0
            let msg: String
            if pool.heap != nil {
                msg = "[pool] \(width)×\(height) intScale=\(internalScale) " +
                      "alloc=\(heapStatus) heapSize=\(String(format: "%.1f", heapMB)) MB\n"
            } else {
                msg = "[pool] \(width)×\(height) intScale=\(internalScale) alloc=\(heapStatus)\n"
            }
            FileHandle.standardError.write(Data(msg.utf8))
        }

        // Compile encoder Head (once per IFNetGraph instance).
        self.encoderExecutable = try Self.compileEncoderHead(
            context: context,
            weights: weights,
            width: intW,
            height: intH
        )

        var execs: [CompiledStage] = []
        for (i, scale) in weights.header.scaleList.enumerated() {
            let stage = try Self.compileStageExecutable(
                context: context,
                weights: weights,
                stageIndex: i,
                scale: scale,
                width: intW,
                height: intH
            )
            execs.append(stage)
        }
        self.stageExecutables = execs

        self.addFlowExecutable = Self.compileAddExecutable(
            context: context,
            shape: [1, NSNumber(value: intH), NSNumber(value: intW), 4],
            name: "addFlow"
        )
        self.addMaskExecutable = Self.compileAddExecutable(
            context: context,
            shape: [1, NSNumber(value: intH), NSNumber(value: intW), 1],
            name: "addMask"
        )
        // 编译后只探测一次真实后端的舍入；未知行为保留原 graph，不能按设备猜测。
        if internalScale <= 0.5,
           let flow = Self.probeAdditionRounding(context: context, executable: addFlowExecutable, width: intW, height: intH, channels: 4),
           let mask = Self.probeAdditionRounding(context: context, executable: addMaskExecutable, width: intW, height: intH, channels: 1) {
            accumulationRounding = SIMD2(flow, mask)
        }
    }

    private static func probeAdditionRounding(context: InferenceContext, executable: MPSGraphExecutable,
                                              width: Int, height: Int, channels: Int) -> UInt32? {
        #if arch(arm64)
        let count = width * height * channels
        guard let a = context.device.makeBuffer(length: count * 2, options: .storageModeShared),
              let b = context.device.makeBuffer(length: count * 2, options: .storageModeShared),
              let output = context.device.makeBuffer(length: count * 2, options: .storageModeShared),
              let rawCommand = context.commandQueue.makeCommandBuffer() else { return nil }
        let ap = a.contents().bindMemory(to: UInt16.self, capacity: count)
        let bp = b.contents().bindMemory(to: UInt16.self, capacity: count)
        // 输入及符号模式每 63486 个元素完整重复；只计算一个周期，仍校验全部输出。
        let patternCount = min(count, 0x7bff * 2)
        var evenPattern = [UInt16](repeating: 0, count: patternCount)
        var awayPattern = [UInt16](repeating: 0, count: patternCount)
        for i in 0..<patternCount {
            ap[i] = UInt16((i * 73) % 0x7bff) | (i.isMultiple(of: 2) ? 0x8000 : 0)
            bp[i] = UInt16((i * 193 + 79) % 0x7bff) | (i.isMultiple(of: 3) ? 0x8000 : 0)
            let exact = Float(Float16(bitPattern: ap[i])) + Float(Float16(bitPattern: bp[i]))
            let even = Float16(exact).bitPattern
            var away = even
            let magnitude = abs(Float(Float16(bitPattern: even)))
            if magnitude < abs(exact), even & 0x7fff < 0x7bff {
                let next = Float(Float16(bitPattern: (even & 0x7fff) + 1))
                if abs(exact) == (magnitude + next) / 2 { away = even + 1 }
            }
            evenPattern[i] = even
            awayPattern[i] = away
        }
        var offset = patternCount
        while offset < count {
            let length = min(patternCount, count - offset)
            ap.advanced(by: offset).update(from: ap, count: length)
            bp.advanced(by: offset).update(from: bp, count: length)
            offset += length
        }
        let shape: [NSNumber] = [1, NSNumber(value: height), NSNumber(value: width), NSNumber(value: channels)]
        let command = MPSCommandBuffer(commandBuffer: rawCommand)
        _ = executable.encode(to: command,
                              inputs: [MPSGraphTensorData(a, shape: shape, dataType: .float16), MPSGraphTensorData(b, shape: shape, dataType: .float16)],
                              results: [MPSGraphTensorData(output, shape: shape, dataType: .float16)], executionDescriptor: nil)
        command.commit()
        command.waitUntilCompleted()
        guard command.commandBuffer.status == .completed else { return nil }
        let values = output.contents().bindMemory(to: UInt16.self, capacity: count)
        func matches(_ pattern: [UInt16]) -> Bool {
            pattern.withUnsafeBufferPointer { expected in
                var offset = 0
                while offset < count {
                    let length = min(patternCount, count - offset)
                    if memcmp(values.advanced(by: offset), expected.baseAddress!, length * 2) != 0 { return false }
                    offset += length
                }
                return true
            }
        }
        return matches(evenPattern) ? 0 : matches(awayPattern) ? 1 : nil
        #else
        // Intel 的 Swift 工具链不支持这些 Float16 转换，保留原 MPSGraph 累加路径。
        return nil
        #endif
    }

    /// Compiles a tiny element-wise addition graph for a fixed shape, ready to encode onto a command buffer.
    private static func compileAddExecutable(context: InferenceContext,
                                              shape: [NSNumber],
                                              name: String) -> MPSGraphExecutable {
        let g = MPSGraph()
        let a = g.placeholder(shape: shape, dataType: .float16, name: "\(name).a")
        let b = g.placeholder(shape: shape, dataType: .float16, name: "\(name).b")
        let sum = g.addition(a, b, name: "\(name).sum")
        let feeds: [MPSGraphTensor: MPSGraphShapedType] = [
            a: MPSGraphShapedType(shape: shape, dataType: .float16),
            b: MPSGraphShapedType(shape: shape, dataType: .float16),
        ]
        return g.compile(with: context.mpsGraphDevice,
                          feeds: feeds,
                          targetTensors: [sum],
                          targetOperations: nil,
                          compilationDescriptor: nil)
    }

    // MARK: - Encoder Head compilation

    /// Compiles the v4.26 encoder Head subgraph: img [1, intH, intW, 3] → feat [1, intH, intW, 4].
    /// Architecture: cnn0(stride 2) → cnn1(stride 1) → cnn2(stride 1) → cnn3(ConvTranspose stride 2).
    /// Net spatial effect: 2× down then 2× up → same spatial size as input.
    private static func compileEncoderHead(context: InferenceContext,
                                            weights: WeightStore,
                                            width: Int,
                                            height: Int) throws -> MPSGraphExecutable {
        let g = MPSGraph()

        let img = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 3],
                                 dataType: .float16, name: "img")

        func loadConst(name: String) throws -> MPSGraphTensor {
            guard let entry = weights.header.tensors.first(where: { $0.name == name }) else {
                throw WeightStoreError.tensorNotFound(name)
            }
            let bytes = try weights.tensorBytes(named: name)
            return g.constant(bytes, shape: entry.shape.map(NSNumber.init), dataType: .float16)
        }

        // cnn0: Conv2d(3→16, k=3, s=2, p=1) + LeakyReLU(0.2)
        let w0 = try loadConst(name: "encode.cnn0.weight")
        let b0 = try loadConst(name: "encode.cnn0.bias")
        var x = g.rifeConv2D(input: img, weight: w0, bias: b0, stride: 2, padding: 1,
                              name: "encode.cnn0")
        x = g.rifeLeakyReLU(input: x, negativeSlope: 0.2, name: "encode.cnn0.act")

        // cnn1: Conv2d(16→16, k=3, s=1, p=1) + LeakyReLU(0.2)
        let w1 = try loadConst(name: "encode.cnn1.weight")
        let b1 = try loadConst(name: "encode.cnn1.bias")
        x = g.rifeConv2D(input: x, weight: w1, bias: b1, stride: 1, padding: 1,
                          name: "encode.cnn1")
        x = g.rifeLeakyReLU(input: x, negativeSlope: 0.2, name: "encode.cnn1.act")

        // cnn2: Conv2d(16→16, k=3, s=1, p=1) + LeakyReLU(0.2)
        let w2 = try loadConst(name: "encode.cnn2.weight")
        let b2 = try loadConst(name: "encode.cnn2.bias")
        x = g.rifeConv2D(input: x, weight: w2, bias: b2, stride: 1, padding: 1,
                          name: "encode.cnn2")
        x = g.rifeLeakyReLU(input: x, negativeSlope: 0.2, name: "encode.cnn2.act")

        // cnn3: ConvTranspose2d(16→4, k=4, s=2, p=1). No activation.
        let w3 = try loadConst(name: "encode.cnn3.weight")
        let b3 = try loadConst(name: "encode.cnn3.bias")
        let out = g.rifeConvTranspose2D(input: x, weight: w3, bias: b3, stride: 2, padding: 1,
                                         name: "encode.cnn3")

        let feeds: [MPSGraphTensor: MPSGraphShapedType] = [
            img: MPSGraphShapedType(shape: img.shape!, dataType: .float16),
        ]
        return g.compile(with: context.mpsGraphDevice,
                          feeds: feeds,
                          targetTensors: [out],
                          targetOperations: nil,
                          compilationDescriptor: nil)
    }

    // MARK: - Stage executable compilation

    private static func compileStageExecutable(context: InferenceContext,
                                                weights: WeightStore,
                                                stageIndex: Int,
                                                scale: Int,
                                                width: Int,
                                                height: Int) throws -> CompiledStage {
        let g = MPSGraph()
        let mpsDevice = context.mpsGraphDevice

        // v4.26 channel layout:
        //   Stage 0: cat(img0[3], img1[3], f0[4], f1[4], t[1]) = 15 ch
        //            IFBlock receives this 15-ch input; conv0 in_ch = 15.
        //   Stage i>0: cat(warped0[3], warped1[3], wf0[4], wf1[4], t[1], mask[1], feat[8]) = 24 ch
        //              IFBlock then concats downsampled prev_flow [4] inside build() → 28 ch.
        //
        // Placeholders are at IFNet *internal* resolution.

        let img0    = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 3],
                                     dataType: .float16, name: "img0")
        let img1    = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 3],
                                     dataType: .float16, name: "img1")
        let tTensor = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 1],
                                     dataType: .float16, name: "t")

        let blockInput: MPSGraphTensor
        let orderedInputs: [MPSGraphTensor]
        var prevFlowPlaceholder: MPSGraphTensor? = nil

        if stageIndex == 0 {
            let f0 = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 4],
                                    dataType: .float16, name: "f0")
            let f1 = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 4],
                                    dataType: .float16, name: "f1")
            // Concat order matches PyTorch: torch.cat((img0, img1, f0, f1, timestep), 1) in NCHW
            // which in NHWC is dim=3 concat: [img0, img1, f0, f1, t]
            blockInput = g.concatTensors([img0, img1, f0, f1, tTensor], dimension: 3,
                                          name: "stage0.concat")
            orderedInputs = [img0, img1, f0, f1, tTensor]
        } else {
            let warped0 = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 3],
                                         dataType: .float16, name: "warped0")
            let warped1 = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 3],
                                         dataType: .float16, name: "warped1")
            let wf0 = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 4],
                                     dataType: .float16, name: "wf0")
            let wf1 = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 4],
                                     dataType: .float16, name: "wf1")
            let prevMask = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 1],
                                          dataType: .float16, name: "prev_mask")
            let prevFeat = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 8],
                                          dataType: .float16, name: "prev_feat")
            let prevFlow = g.placeholder(shape: [1, NSNumber(value: height), NSNumber(value: width), 4],
                                          dataType: .float16, name: "prev_flow")
            prevFlowPlaceholder = prevFlow
            // Concat order matches PyTorch:
            //   torch.cat((warped_img0, warped_img1, wf0, wf1, timestep, mask, feat), 1)
            // (IFBlockBuilder concats prev_flow inside build(), after the spatial downsample.)
            blockInput = g.concatTensors([warped0, warped1, wf0, wf1, tTensor, prevMask, prevFeat],
                                          dimension: 3, name: "stage\(stageIndex).concat")
            // orderedInputs order matches the runtime allInputPairs table in run().
            orderedInputs = [img0, img1, tTensor, warped0, warped1, wf0, wf1, prevMask, prevFeat, prevFlow]
        }

        // Note: img0 / img1 placeholders are created for all stages but only fed for stage 0.
        // For stage i>0 they sit in orderedInputs so the feeds dict size is correct, but
        // they are not wired into blockInput — MPSGraph may or may not include them in the
        // final graph. We include them in feeds to keep the compile signature consistent.

        let blockBuilder = IFBlockBuilder(graph: g, weights: weights,
                                           prefix: "block\(stageIndex).")
        let (flow, mask, feat) = try blockBuilder.build(input: blockInput,
                                                         previousFlow: prevFlowPlaceholder,
                                                         scale: scale)

        let feeds = Dictionary(uniqueKeysWithValues: orderedInputs.map { tensor -> (MPSGraphTensor, MPSGraphShapedType) in
            (tensor, MPSGraphShapedType(shape: tensor.shape!, dataType: tensor.dataType))
        })

        let exec = g.compile(with: mpsDevice,
                              feeds: feeds,
                              targetTensors: [flow, mask, feat],
                              targetOperations: nil,
                              compilationDescriptor: nil)
        return CompiledStage(executable: exec, inputTensors: orderedInputs)
    }

    // MARK: - Inference orchestration

    /// Runs IFNet on two BGRA8 input pixel buffers, writes the result at the given `timestep`
    /// to the BGRA8 output buffer. `timestep ∈ (0, 1)`; `0.5` produces the midframe.
    ///
    /// All GPU work (input conversion, encoder Head, IFNet stages with per-stage warps at
    /// internal res, final-flow upsample, full-res warp, blend, BGRA writeout) is encoded onto
    /// a single MPSCommandBuffer which is committed once at the end.
    ///
    /// Intermediate buffers and textures are reused from the pre-allocated BufferPool — no
    /// MTLBuffer or MTLTexture allocations occur per call.
    public func run(previous: CVPixelBuffer,
                    current:  CVPixelBuffer,
                    output:   CVPixelBuffer,
                    timestep: Float = 0.5) throws {
        pool.setTimestep(timestep)
        let verbose = ProcessInfo.processInfo.environment["RIFE_VERBOSE"] == "1"
        let dumpDir = ProcessInfo.processInfo.environment["RIFE_DUMP_DIR"]
        let benchStart = verbose ? DispatchTime.now() : nil
        defer {
            if verbose, let start = benchStart {
                let elapsedNs = DispatchTime.now().uptimeNanoseconds &- start.uptimeNanoseconds
                let ms = Double(elapsedNs) / 1_000_000
                FileHandle.standardError.write(Data(String(format: "[bench] inference: %.2f ms\n", ms).utf8))
            }
        }
        let gpuTimingEnabled = verbose
        let queue = context.commandQueue
        let conv = context.conversionKernels
        let warp = context.warpKernel
        let p = pool  // shorthand alias

        let useInternalScale = internalScale != 1.0
        let intW = internalWidth
        let intH = internalHeight

        // 1. Pixel-buffer-backed MTLTextures for input/output (zero copy via IOSurface).
        let prevTex = try PixelBufferConvert.makeTexture(from: previous, textureCache: textureCache)
        let currTex = try PixelBufferConvert.makeTexture(from: current,  textureCache: textureCache)
        let outTex  = try PixelBufferConvert.makeTexture(from: output,   textureCache: textureCache)

        // 2. Pre-built MPSGraphTensorData wrappers around pool buffers.
        let flowShape:    [NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 4]
        let maskShape:    [NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 1]
        let intRGBShape:  [NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 3]
        let intFeat4Shape:[NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 4]
        let intFeat8Shape:[NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 8]

        let img0IntTD = MPSGraphTensorData(p.img0IntBuf, shape: intRGBShape,  dataType: .float16)
        let img1IntTD = MPSGraphTensorData(p.img1IntBuf, shape: intRGBShape,  dataType: .float16)
        let tTD       = MPSGraphTensorData(p.tBuf,       shape: maskShape,    dataType: .float16)

        // Encoder feature tensor-data descriptors.
        let f0TD = MPSGraphTensorData(p.f0Buf, shape: intFeat4Shape, dataType: .float16)
        let f1TD = MPSGraphTensorData(p.f1Buf, shape: intFeat4Shape, dataType: .float16)

        // Warped feature tensor-data (filled per-stage).
        let wf0TD = MPSGraphTensorData(p.stageWf0Buf, shape: intFeat4Shape, dataType: .float16)
        let wf1TD = MPSGraphTensorData(p.stageWf1Buf, shape: intFeat4Shape, dataType: .float16)

        // Flow/mask ping-pong accumulators.
        let flowAccTD_A = MPSGraphTensorData(p.flowAccBufA, shape: flowShape, dataType: .float16)
        let maskAccTD_A = MPSGraphTensorData(p.maskAccBufA, shape: maskShape, dataType: .float16)
        let flowAccTD_B = MPSGraphTensorData(p.flowAccBufB, shape: flowShape, dataType: .float16)
        let maskAccTD_B = MPSGraphTensorData(p.maskAccBufB, shape: maskShape, dataType: .float16)
        let flowRawTD   = MPSGraphTensorData(p.flowRawBuf,  shape: flowShape, dataType: .float16)
        let maskRawTD   = MPSGraphTensorData(p.maskRawBuf,  shape: maskShape, dataType: .float16)

        // Feat ping-pong accumulators.
        let featAccTD_A = MPSGraphTensorData(p.stageFeatBufA, shape: intFeat8Shape, dataType: .float16)
        let featAccTD_B = MPSGraphTensorData(p.stageFeatBufB, shape: intFeat8Shape, dataType: .float16)

        let stageWarped0TD = MPSGraphTensorData(p.stageWarped0Buf, shape: intRGBShape, dataType: .float16)
        let stageWarped1TD = MPSGraphTensorData(p.stageWarped1Buf, shape: intRGBShape, dataType: .float16)

        // 3. Single command buffer for the whole inference.
        guard let cb = queue.makeCommandBuffer() else {
            throw IFNetError.commandBufferFailed("makeCommandBuffer failed")
        }
        var mpsCB = MPSCommandBuffer(commandBuffer: cb)

        // Helper: dispatch a compute kernel grid covering the given dims.
        func dispatch2D(_ enc: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState,
                        gridW: Int, gridH: Int) {
            let w = pipeline.threadExecutionWidth
            let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
            let tg = MTLSize(width: w, height: h, depth: 1)
            let groups = MTLSize(
                width:  (gridW + w - 1) / w,
                height: (gridH + h - 1) / h,
                depth: 1
            )
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        }

        // 3a. Full-res BGRA → RGB packed buffer → expand to full-res RGBA texture (warp source).
        do {
            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("input convert: makeComputeCommandEncoder failed")
            }
            enc.setComputePipelineState(conv.bgraToRGB)
            enc.setTexture(prevTex, index: 0)
            enc.setBuffer(p.img0FullBuf, offset: 0, index: 0)
            dispatch2D(enc, conv.bgraToRGB, gridW: width, gridH: height)
            enc.setTexture(currTex, index: 0)
            enc.setBuffer(p.img1FullBuf, offset: 0, index: 0)
            dispatch2D(enc, conv.bgraToRGB, gridW: width, gridH: height)

            enc.setComputePipelineState(conv.rgbBufToRGBATex)
            enc.setBuffer(p.img0FullBuf, offset: 0, index: 0)
            enc.setTexture(p.srcTex0Full, index: 0)
            dispatch2D(enc, conv.rgbBufToRGBATex, gridW: width, gridH: height)
            enc.setBuffer(p.img1FullBuf, offset: 0, index: 0)
            enc.setTexture(p.srcTex1Full, index: 0)
            dispatch2D(enc, conv.rgbBufToRGBATex, gridW: width, gridH: height)
            enc.endEncoding()
        }

        // 3b. Internal-res RGB packed buffers + RGBA textures.
        if useInternalScale {
            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("downsample fused: makeComputeCommandEncoder failed")
            }
            enc.setComputePipelineState(conv.bgraDownsampleToRGB)
            var invScale: Float = Float(1.0 / internalScale)
            var outDim: SIMD2<UInt32> = SIMD2(UInt32(intW), UInt32(intH))
            enc.setBytes(&invScale, length: MemoryLayout<Float>.size, index: 1)
            enc.setBytes(&outDim, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            enc.setTexture(prevTex, index: 0)
            enc.setBuffer(p.img0IntBuf, offset: 0, index: 0)
            dispatch2D(enc, conv.bgraDownsampleToRGB, gridW: intW, gridH: intH)
            enc.setTexture(currTex, index: 0)
            enc.setBuffer(p.img1IntBuf, offset: 0, index: 0)
            dispatch2D(enc, conv.bgraDownsampleToRGB, gridW: intW, gridH: intH)
            enc.endEncoding()
        }
        do {
            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("internal expand: makeComputeCommandEncoder failed")
            }
            enc.setComputePipelineState(conv.rgbBufToRGBATex)
            enc.setBuffer(p.img0IntBuf, offset: 0, index: 0)
            enc.setTexture(p.srcTex0Int, index: 0)
            dispatch2D(enc, conv.rgbBufToRGBATex, gridW: intW, gridH: intH)
            enc.setBuffer(p.img1IntBuf, offset: 0, index: 0)
            enc.setTexture(p.srcTex1Int, index: 0)
            dispatch2D(enc, conv.rgbBufToRGBATex, gridW: intW, gridH: intH)
            enc.endEncoding()
        }

        // 3c. Encoder Head: run img0Int → f0Buf and img1Int → f1Buf.
        //     Encoder is run twice using the same compiled executable.
        encoderExecutable.encode(to: mpsCB,
                                  inputs: [img0IntTD],
                                  results: [f0TD],
                                  executionDescriptor: nil)
        encoderExecutable.encode(to: mpsCB,
                                  inputs: [img1IntTD],
                                  results: [f1TD],
                                  executionDescriptor: nil)

        // 3d. Expand f0Buf/f1Buf (4-ch fp16 buffers) → f0Tex/f1Tex (RGBA textures) for the
        //     per-stage feature warps. This runs once; the textures are read-only after this point.
        do {
            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("feat expand: makeComputeCommandEncoder failed")
            }
            enc.setComputePipelineState(conv.rgba4BufToRGBATex)
            enc.setBuffer(p.f0Buf, offset: 0, index: 0)
            enc.setTexture(p.f0Tex, index: 0)
            dispatch2D(enc, conv.rgba4BufToRGBATex, gridW: intW, gridH: intH)
            enc.setBuffer(p.f1Buf, offset: 0, index: 0)
            enc.setTexture(p.f1Tex, index: 0)
            dispatch2D(enc, conv.rgba4BufToRGBATex, gridW: intW, gridH: intH)
            enc.endEncoding()
        }

        // 4. Stages run at INTERNAL resolution.
        //
        // Ping-pong accumulator pattern for flow/mask (identical to v4.6):
        //   Stage 0: write directly into flowAccBufA / maskAccBufA (no add).
        //   Stage i>0: write raw into flowRawBuf / maskRawBuf; add(acc, raw) → opposite slot.
        //
        // Ping-pong for feat:
        //   Stage 0: write into stageFeatBufA.
        //   Stage 1: read from A → write into B.
        //   Stage 2: read from B → write into A. Etc.

        var accFlowTD: MPSGraphTensorData? = nil
        var accMaskTD: MPSGraphTensorData? = nil
        var accFlowBuf: MTLBuffer? = nil
        var accMaskBuf: MTLBuffer? = nil
        var accFeatTD: MPSGraphTensorData? = nil
        var warped0IntTD: MPSGraphTensorData? = nil
        var warped1IntTD: MPSGraphTensorData? = nil
        var accIsInSlotA = true
        var featIsInSlotA = true

        // Per-stage feat raw output slot (written by stage executable, then ping-ponged).
        // We reuse stageFeatBufA for the first stage's direct write and stageFeatBufB for the
        // alternating destination; raw feat is written directly into the destination slot.
        // (Unlike flow/mask, we don't need a separate "raw" buffer for feat because the
        //  feat is carried forward, not accumulated — each stage's feat completely replaces
        //  the previous stage's feat.)

        for i in 0..<scaleList.count {
            let stage = stageExecutables[i]

            // Determine which feat buffer is the output for this stage.
            let outFeatTD: MPSGraphTensorData
            if i == 0 {
                outFeatTD = featAccTD_A
            } else {
                outFeatTD = featIsInSlotA ? featAccTD_B : featAccTD_A
            }

            // Build the input table for this stage's executable.
            let allInputPairs: [(MPSGraphTensor, MPSGraphTensorData)]
            if i == 0 {
                // Stage 0 inputs: [img0, img1, f0, f1, t]
                allInputPairs = [
                    (stage.inputTensors[0], img0IntTD),   // img0
                    (stage.inputTensors[1], img1IntTD),   // img1
                    (stage.inputTensors[2], f0TD),         // f0
                    (stage.inputTensors[3], f1TD),         // f1
                    (stage.inputTensors[4], tTD),          // t
                ]
            } else {
                // Stage i>0 inputs: [img0, img1, t, warped0, warped1, wf0, wf1, prevMask, prevFeat, prevFlow]
                // (img0/img1 are in orderedInputs but not fed to blockInput for i>0 — they are
                //  still included in the feeds dict as they appear in orderedInputs.)
                allInputPairs = [
                    (stage.inputTensors[0], img0IntTD),    // img0 (unused in graph, satisfies feeds dict)
                    (stage.inputTensors[1], img1IntTD),    // img1 (unused in graph, satisfies feeds dict)
                    (stage.inputTensors[2], tTD),           // t
                    (stage.inputTensors[3], warped0IntTD!), // warped0
                    (stage.inputTensors[4], warped1IntTD!), // warped1
                    (stage.inputTensors[5], wf0TD),         // wf0
                    (stage.inputTensors[6], wf1TD),         // wf1
                    (stage.inputTensors[7], accMaskTD!),    // prevMask
                    (stage.inputTensors[8], accFeatTD!),    // prevFeat
                    (stage.inputTensors[9], accFlowTD!),    // prevFlow
                ]
            }

            let feedOrder = stage.executable.feedTensors ?? stage.inputTensors
            let inputs: [MPSGraphTensorData] = feedOrder.map { feedTensor in
                allInputPairs.first { $0.0 === feedTensor }!.1
            }

            if i == 0 {
                // Stage 0: write directly into accumulator slot A.
                _ = stage.executable.encode(to: mpsCB,
                                             inputs: inputs,
                                             results: [flowAccTD_A, maskAccTD_A, outFeatTD],
                                             executionDescriptor: nil)
                accFlowTD    = flowAccTD_A
                accMaskTD    = maskAccTD_A
                accFlowBuf   = p.flowAccBufA
                accMaskBuf   = p.maskAccBufA
                accFeatTD    = outFeatTD
                accIsInSlotA = true
                featIsInSlotA = true
            } else {
                // Stage 1+: write raw flow/mask into flowRawBuf/maskRawBuf, feat into outFeatTD.
                _ = stage.executable.encode(to: mpsCB,
                                             inputs: inputs,
                                             results: [flowRawTD, maskRawTD, outFeatTD],
                                             executionDescriptor: nil)

                guard let prevFlowTD = accFlowTD, let prevMaskTD = accMaskTD else {
                    throw IFNetError.dimensionsMismatch
                }

                let newFlowAccTD:  MPSGraphTensorData
                let newMaskAccTD:  MPSGraphTensorData
                let newFlowAccBuf: MTLBuffer
                let newMaskAccBuf: MTLBuffer
                if accIsInSlotA {
                    newFlowAccTD  = flowAccTD_B
                    newMaskAccTD  = maskAccTD_B
                    newFlowAccBuf = p.flowAccBufB
                    newMaskAccBuf = p.maskAccBufB
                } else {
                    newFlowAccTD  = flowAccTD_A
                    newMaskAccTD  = maskAccTD_A
                    newFlowAccBuf = p.flowAccBufA
                    newMaskAccBuf = p.maskAccBufA
                }

                _ = addFlowExecutable.encode(to: mpsCB,
                                              inputs: [prevFlowTD, flowRawTD],
                                              results: [newFlowAccTD],
                                              executionDescriptor: nil)
                _ = addMaskExecutable.encode(to: mpsCB,
                                              inputs: [prevMaskTD, maskRawTD],
                                              results: [newMaskAccTD],
                                              executionDescriptor: nil)

                accFlowTD    = newFlowAccTD
                accMaskTD    = newMaskAccTD
                accFlowBuf   = newFlowAccBuf
                accMaskBuf   = newMaskAccBuf
                accFeatTD    = outFeatTD
                accIsInSlotA = !accIsInSlotA
                featIsInSlotA = !featIsInSlotA
            }

            // Per-stage debug dump (gated by RIFE_DUMP_DIR).
            if let dumpDir = dumpDir {
                let device = context.device
                let intFlowBytes = intW * intH * 4 * 2
                let intMaskBytes = intW * intH * 1 * 2
                guard let flowStage = device.makeBuffer(length: intFlowBytes, options: .storageModeShared),
                      let maskStage = device.makeBuffer(length: intMaskBytes, options: .storageModeShared) else {
                    throw IFNetError.textureAllocationFailed
                }
                if let blitEnc = mpsCB.commandBuffer.makeBlitCommandEncoder() {
                    blitEnc.copy(from: accFlowBuf!, sourceOffset: 0,
                                 to: flowStage, destinationOffset: 0, size: intFlowBytes)
                    blitEnc.copy(from: accMaskBuf!, sourceOffset: 0,
                                 to: maskStage, destinationOffset: 0, size: intMaskBytes)
                    blitEnc.endEncoding()
                }
                mpsCB.commit()
                mpsCB.waitUntilCompleted()
                let flowStageTD = MPSGraphTensorData(flowStage, shape: flowShape, dataType: .float16)
                let maskStageTD = MPSGraphTensorData(maskStage, shape: maskShape, dataType: .float16)
                Self.dumpStageActivations(stageIndex: i,
                                          flow: flowStageTD,
                                          mask: maskStageTD,
                                          dumpDir: dumpDir)
                guard let cb2 = queue.makeCommandBuffer() else {
                    throw IFNetError.commandBufferFailed("dump: makeCommandBuffer failed")
                }
                mpsCB = MPSCommandBuffer(commandBuffer: cb2)
            }

            // Per-stage warp at INTERNAL res. Skip on the LAST stage — the final warp happens
            // at FULL res after we upsample the accumulated flow/mask.
            let isLastStage = (i == scaleList.count - 1)
            if isLastStage { continue }

            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("stage \(i) post-encoder failed")
            }

            // Split [1,intH,intW,4] flow buffer → 2 RG textures.
            enc.setComputePipelineState(conv.flowSplit)
            enc.setBuffer(accFlowBuf!, offset: 0, index: 0)
            enc.setTexture(p.stageFlowTex0, index: 0)
            enc.setTexture(p.stageFlowTex1, index: 1)
            dispatch2D(enc, conv.flowSplit, gridW: intW, gridH: intH)

            // Backward warps for images at internal res.
            warp.encode(into: enc, source: p.srcTex0Int, flow: p.stageFlowTex0, output: p.stageWarpedTex0)
            warp.encode(into: enc, source: p.srcTex1Int, flow: p.stageFlowTex1, output: p.stageWarpedTex1)

            // Pack RGBA texture → fp16 RGB buffer for next stage feeds (warped images).
            enc.setComputePipelineState(conv.rgbaTexToRGBBuf)
            enc.setTexture(p.stageWarpedTex0, index: 0)
            enc.setBuffer(p.stageWarped0Buf, offset: 0, index: 0)
            dispatch2D(enc, conv.rgbaTexToRGBBuf, gridW: intW, gridH: intH)
            enc.setTexture(p.stageWarpedTex1, index: 0)
            enc.setBuffer(p.stageWarped1Buf, offset: 0, index: 0)
            dispatch2D(enc, conv.rgbaTexToRGBBuf, gridW: intW, gridH: intH)

            // v4.26: Backward warps for 4-ch encoder features at internal res.
            // f0Tex/f1Tex were populated once (step 3d); stageWf0Tex/stageWf1Tex receive output.
            warp.encode(into: enc, source: p.f0Tex, flow: p.stageFlowTex0, output: p.stageWf0Tex)
            warp.encode(into: enc, source: p.f1Tex, flow: p.stageFlowTex1, output: p.stageWf1Tex)

            // Pack 4-ch warped feature textures → 4-ch buffers for next stage's MPSGraph feeds.
            enc.setComputePipelineState(conv.rgbaTexTo4ChBuf)
            enc.setTexture(p.stageWf0Tex, index: 0)
            enc.setBuffer(p.stageWf0Buf, offset: 0, index: 0)
            dispatch2D(enc, conv.rgbaTexTo4ChBuf, gridW: intW, gridH: intH)
            enc.setTexture(p.stageWf1Tex, index: 0)
            enc.setBuffer(p.stageWf1Buf, offset: 0, index: 0)
            dispatch2D(enc, conv.rgbaTexTo4ChBuf, gridW: intW, gridH: intH)

            enc.endEncoding()

            warped0IntTD = stageWarped0TD
            warped1IntTD = stageWarped1TD
        }

        // 5. Promote final flow/mask to FULL res.
        guard let _ = accFlowTD,
              let _ = accMaskTD,
              let finalFlowIntBuf = accFlowBuf else {
            // No stages ran — convert img0 directly to output as a fallback.
            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("fallback convert: makeComputeCommandEncoder failed")
            }
            enc.setComputePipelineState(conv.rgbToBGRA)
            enc.setBuffer(p.img0FullBuf, offset: 0, index: 0)
            enc.setTexture(outTex, index: 0)
            dispatch2D(enc, conv.rgbToBGRA, gridW: width, gridH: height)
            enc.endEncoding()
            mpsCB.commit()
            mpsCB.waitUntilCompleted()
            return
        }

        guard let resolvedMaskBuf = accMaskBuf else { throw IFNetError.dimensionsMismatch }

        // 6. Final full-res warp + blend + BGRA writeout.
        guard let finalEnc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
            throw IFNetError.commandBufferFailed("final warp encoder failed")
        }
        if useInternalScale {
            finalEnc.setComputePipelineState(conv.flowUpsampleSplit)
            finalEnc.setBuffer(finalFlowIntBuf, offset: 0, index: 0)
            var intDim: SIMD2<UInt32> = SIMD2(UInt32(intW), UInt32(intH))
            var valueScale: Float = Float(1.0 / internalScale)
            finalEnc.setBytes(&intDim, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
            finalEnc.setBytes(&valueScale, length: MemoryLayout<Float>.size, index: 2)
            finalEnc.setTexture(p.finalFlowTex0, index: 0)
            finalEnc.setTexture(p.finalFlowTex1, index: 1)
            dispatch2D(finalEnc, conv.flowUpsampleSplit, gridW: width, gridH: height)
        } else {
            finalEnc.setComputePipelineState(conv.flowSplit)
            finalEnc.setBuffer(finalFlowIntBuf, offset: 0, index: 0)
            finalEnc.setTexture(p.finalFlowTex0, index: 0)
            finalEnc.setTexture(p.finalFlowTex1, index: 1)
            dispatch2D(finalEnc, conv.flowSplit, gridW: width, gridH: height)
        }

        warp.encodeToBuffer(into: finalEnc,
                             source: p.srcTex0Full,
                             flow: p.finalFlowTex0,
                             output: p.finalWarped0Buf,
                             width: width, height: height)
        warp.encodeToBuffer(into: finalEnc,
                             source: p.srcTex1Full,
                             flow: p.finalFlowTex1,
                             output: p.finalWarped1Buf,
                             width: width, height: height)

        if useInternalScale {
            finalEnc.setComputePipelineState(conv.blendUpsampleMaskAndPack)
            finalEnc.setBuffer(p.finalWarped0Buf, offset: 0, index: 0)
            finalEnc.setBuffer(p.finalWarped1Buf, offset: 0, index: 1)
            finalEnc.setBuffer(resolvedMaskBuf, offset: 0, index: 2)
            var intDimM: SIMD2<UInt32> = SIMD2(UInt32(intW), UInt32(intH))
            finalEnc.setBytes(&intDimM, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
            finalEnc.setTexture(outTex, index: 0)
            dispatch2D(finalEnc, conv.blendUpsampleMaskAndPack, gridW: width, gridH: height)
        } else {
            finalEnc.setComputePipelineState(conv.blendAndPack)
            finalEnc.setBuffer(p.finalWarped0Buf, offset: 0, index: 0)
            finalEnc.setBuffer(p.finalWarped1Buf, offset: 0, index: 1)
            finalEnc.setBuffer(resolvedMaskBuf, offset: 0, index: 2)
            finalEnc.setTexture(outTex, index: 0)
            dispatch2D(finalEnc, conv.blendAndPack, gridW: width, gridH: height)
        }
        finalEnc.endEncoding()

        if gpuTimingEnabled {
            let cbRef = mpsCB.commandBuffer
            cbRef.addCompletedHandler { cb in
                let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000
                let kernelMs = (cb.kernelEndTime - cb.kernelStartTime) * 1000
                let line = String(format: "[bench] gpu=%.2f ms kernel=%.2f ms\n", gpuMs, kernelMs)
                FileHandle.standardError.write(Data(line.utf8))
            }
        }

        mpsCB.commit()
        mpsCB.waitUntilCompleted()
    }

    // MARK: - Stream encoding

    /// Single-frame encode: BGRA→RGB (full res) → optional downsample → encoder Head.
    /// Writes results into caller-provided buffers. No IFBlock stages run.
    /// Single MTLCommandBuffer, committed once + waitUntilCompleted.
    public func encodeFrame(frame:         CVPixelBuffer,
                            featBufOut:    MTLBuffer,
                            imgIntBufOut:  MTLBuffer,
                            imgFullBufOut: MTLBuffer) throws {
        let queue = context.commandQueue
        let conv = context.conversionKernels
        let useInternalScale = internalScale != 1.0
        let intW = internalWidth
        let intH = internalHeight

        let frameTex = try PixelBufferConvert.makeTexture(
            from: frame, textureCache: textureCache)

        guard let cb = queue.makeCommandBuffer() else {
            throw IFNetError.commandBufferFailed("makeCommandBuffer failed")
        }
        let mpsCB = MPSCommandBuffer(commandBuffer: cb)

        // Helper: dispatch a compute kernel grid covering the given dims.
        func dispatch2D(_ enc: MTLComputeCommandEncoder,
                        _ pipeline: MTLComputePipelineState,
                        gridW: Int, gridH: Int) {
            let w = pipeline.threadExecutionWidth
            let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
            let tg = MTLSize(width: w, height: h, depth: 1)
            let groups = MTLSize(
                width:  (gridW + w - 1) / w,
                height: (gridH + h - 1) / h,
                depth: 1)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        }

        // 1. Full-res BGRA → RGB packed buffer.
        do {
            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("encodeFrame BGRA→RGB: makeComputeCommandEncoder failed")
            }
            enc.setComputePipelineState(conv.bgraToRGB)
            enc.setTexture(frameTex, index: 0)
            enc.setBuffer(imgFullBufOut, offset: 0, index: 0)
            dispatch2D(enc, conv.bgraToRGB, gridW: width, gridH: height)
            enc.endEncoding()
        }

        // 2. Internal-res RGB packed buffer (downsample, only when scale != 1).
        if useInternalScale {
            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("encodeFrame downsample: makeComputeCommandEncoder failed")
            }
            enc.setComputePipelineState(conv.bgraDownsampleToRGB)
            var invScale: Float = Float(1.0 / internalScale)
            var outDim: SIMD2<UInt32> = SIMD2(UInt32(intW), UInt32(intH))
            enc.setBytes(&invScale, length: MemoryLayout<Float>.size, index: 1)
            enc.setBytes(&outDim, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            enc.setTexture(frameTex, index: 0)
            enc.setBuffer(imgIntBufOut, offset: 0, index: 0)
            dispatch2D(enc, conv.bgraDownsampleToRGB, gridW: intW, gridH: intH)
            enc.endEncoding()
        }
        // (when internalScale == 1, the caller passes the same MTLBuffer for
        //  imgIntBufOut and imgFullBufOut; step 1 already wrote it.)

        // 3. Encoder Head.
        let intRGBShape:   [NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 3]
        let intFeat4Shape: [NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 4]
        let imgIntTD = MPSGraphTensorData(imgIntBufOut, shape: intRGBShape,   dataType: .float16)
        let featTD   = MPSGraphTensorData(featBufOut,   shape: intFeat4Shape, dataType: .float16)
        encoderExecutable.encode(to: mpsCB,
                                  inputs: [imgIntTD],
                                  results: [featTD],
                                  executionDescriptor: nil)

        mpsCB.commit()
        mpsCB.waitUntilCompleted()
    }

    // MARK: - Stream inference

    /// Stream inference with a cached previous frame.
    /// prev*Buf are caller-provided buffers populated from a prior encodeFrame
    /// or runStream call (no encoder runs on prev). Encoder runs only on curr,
    /// writing to curr*Buf. All 5 IFBlock stages, final warp+blend, and BGRA
    /// writeout proceed normally. Single MTLCommandBuffer, committed once.
    public func runStream(curr:           CVPixelBuffer,
                          prevFeatBuf:    MTLBuffer, currFeatBuf:    MTLBuffer,
                          prevImgIntBuf:  MTLBuffer, currImgIntBuf:  MTLBuffer,
                          prevImgFullBuf: MTLBuffer, currImgFullBuf: MTLBuffer,
                          timesteps:      [Float],
                          outputs:        [CVPixelBuffer]) throws {
        precondition(timesteps.count == outputs.count,
                     "timesteps and outputs must have matching counts")
        // 输出可以裁剪，但不得超出完整推理网格。
        guard outputs.allSatisfy({ output in
            let w = CVPixelBufferGetWidth(output), h = CVPixelBufferGetHeight(output)
            return (w == width && h == height) || (supportsCroppedOutput && w <= width && h <= height)
        }) else {
            throw IFNetError.dimensionsMismatch
        }
        for t in timesteps {
            precondition(t > 0 && t < 1,
                         "timestep \(t) out of (0, 1)")
        }

        let verbose = ProcessInfo.processInfo.environment["RIFE_VERBOSE"] == "1"
        let dumpDir = ProcessInfo.processInfo.environment["RIFE_DUMP_DIR"]
        let benchStart = verbose ? DispatchTime.now() : nil
        defer {
            if verbose, let start = benchStart {
                let elapsedNs = DispatchTime.now().uptimeNanoseconds &- start.uptimeNanoseconds
                let ms = Double(elapsedNs) / 1_000_000
                FileHandle.standardError.write(Data(String(format: "[bench] stream inference: %.2f ms\n", ms).utf8))
            }
        }
        let gpuTimingEnabled = verbose
        let queue = context.commandQueue
        let conv = context.conversionKernels
        let warp = context.warpKernel
        let p = pool

        let useInternalScale = internalScale != 1.0
        let intW = internalWidth
        let intH = internalHeight

        // 1. Pixel-buffer-backed MTLTextures for curr input only.
        //    Outputs are bound per-timestep inside the multi-output loop.
        let currTex = try PixelBufferConvert.makeTexture(from: curr, textureCache: textureCache)

        // 2. Tensor data wrappers.
        let flowShape:    [NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 4]
        let maskShape:    [NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 1]
        let intRGBShape:  [NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 3]
        let intFeat4Shape:[NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 4]
        let intFeat8Shape:[NSNumber] = [1, NSNumber(value: intH), NSNumber(value: intW), 8]

        // Bind caller-provided prev/curr buffers for the prev/curr image and feat tensors.
        let img0IntTD = MPSGraphTensorData(prevImgIntBuf, shape: intRGBShape, dataType: .float16)
        let img1IntTD = MPSGraphTensorData(currImgIntBuf, shape: intRGBShape, dataType: .float16)
        let tTD       = MPSGraphTensorData(p.tBuf,        shape: maskShape,   dataType: .float16)

        let f0TD = MPSGraphTensorData(prevFeatBuf, shape: intFeat4Shape, dataType: .float16)
        let f1TD = MPSGraphTensorData(currFeatBuf, shape: intFeat4Shape, dataType: .float16)

        // Warped feature tensor-data (filled per-stage).
        let wf0TD = MPSGraphTensorData(p.stageWf0Buf, shape: intFeat4Shape, dataType: .float16)
        let wf1TD = MPSGraphTensorData(p.stageWf1Buf, shape: intFeat4Shape, dataType: .float16)

        // Flow/mask ping-pong accumulators.
        let flowAccTD_A = MPSGraphTensorData(p.flowAccBufA, shape: flowShape, dataType: .float16)
        let maskAccTD_A = MPSGraphTensorData(p.maskAccBufA, shape: maskShape, dataType: .float16)
        let flowAccTD_B = MPSGraphTensorData(p.flowAccBufB, shape: flowShape, dataType: .float16)
        let maskAccTD_B = MPSGraphTensorData(p.maskAccBufB, shape: maskShape, dataType: .float16)
        let flowRawTD   = MPSGraphTensorData(p.flowRawBuf,  shape: flowShape, dataType: .float16)
        let maskRawTD   = MPSGraphTensorData(p.maskRawBuf,  shape: maskShape, dataType: .float16)

        // Feat ping-pong accumulators.
        let featAccTD_A = MPSGraphTensorData(p.stageFeatBufA, shape: intFeat8Shape, dataType: .float16)
        let featAccTD_B = MPSGraphTensorData(p.stageFeatBufB, shape: intFeat8Shape, dataType: .float16)

        let stageWarped0TD = MPSGraphTensorData(p.stageWarped0Buf, shape: intRGBShape, dataType: .float16)
        let stageWarped1TD = MPSGraphTensorData(p.stageWarped1Buf, shape: intRGBShape, dataType: .float16)

        // 3. Single command buffer for the whole inference.
        guard let cb = queue.makeCommandBuffer() else {
            throw IFNetError.commandBufferFailed("makeCommandBuffer failed")
        }
        var mpsCB = MPSCommandBuffer(commandBuffer: cb)

        // Helper: dispatch a compute kernel grid covering the given dims.
        func dispatch2D(_ enc: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState,
                        gridW: Int, gridH: Int) {
            let w = pipeline.threadExecutionWidth
            let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
            let tg = MTLSize(width: w, height: h, depth: 1)
            let groups = MTLSize(
                width:  (gridW + w - 1) / w,
                height: (gridH + h - 1) / h,
                depth: 1
            )
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        }

        // 3a. Full-res BGRA → RGB packed buffer (curr only; prev was written by a prior call) +
        //     expand BOTH prev and curr full-res RGB buffers to RGBA textures (warp source).
        do {
            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("input convert: makeComputeCommandEncoder failed")
            }
            // BGRA → RGB on curr only.
            enc.setComputePipelineState(conv.bgraToRGB)
            enc.setTexture(currTex, index: 0)
            enc.setBuffer(currImgFullBuf, offset: 0, index: 0)
            dispatch2D(enc, conv.bgraToRGB, gridW: width, gridH: height)

            // RGB-buf → RGBA texture for BOTH prev and curr (pool textures are scratch,
            // refreshed every call regardless of which side the underlying buf came from).
            enc.setComputePipelineState(conv.rgbBufToRGBATex)
            enc.setBuffer(prevImgFullBuf, offset: 0, index: 0)
            enc.setTexture(p.srcTex0Full, index: 0)
            dispatch2D(enc, conv.rgbBufToRGBATex, gridW: width, gridH: height)
            enc.setBuffer(currImgFullBuf, offset: 0, index: 0)
            enc.setTexture(p.srcTex1Full, index: 0)
            dispatch2D(enc, conv.rgbBufToRGBATex, gridW: width, gridH: height)
            enc.endEncoding()
        }

        // 3b. Internal-res RGB packed buffer (downsample on curr only) +
        //     RGBA texture expansion for BOTH prev and curr internal-res buffers.
        if useInternalScale {
            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("downsample fused: makeComputeCommandEncoder failed")
            }
            enc.setComputePipelineState(conv.bgraDownsampleToRGB)
            var invScale: Float = Float(1.0 / internalScale)
            var outDim: SIMD2<UInt32> = SIMD2(UInt32(intW), UInt32(intH))
            enc.setBytes(&invScale, length: MemoryLayout<Float>.size, index: 1)
            enc.setBytes(&outDim, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            enc.setTexture(currTex, index: 0)
            enc.setBuffer(currImgIntBuf, offset: 0, index: 0)
            dispatch2D(enc, conv.bgraDownsampleToRGB, gridW: intW, gridH: intH)
            enc.endEncoding()
        }
        do {
            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("internal expand: makeComputeCommandEncoder failed")
            }
            enc.setComputePipelineState(conv.rgbBufToRGBATex)
            enc.setBuffer(prevImgIntBuf, offset: 0, index: 0)
            enc.setTexture(p.srcTex0Int, index: 0)
            dispatch2D(enc, conv.rgbBufToRGBATex, gridW: intW, gridH: intH)
            enc.setBuffer(currImgIntBuf, offset: 0, index: 0)
            enc.setTexture(p.srcTex1Int, index: 0)
            dispatch2D(enc, conv.rgbBufToRGBATex, gridW: intW, gridH: intH)
            enc.endEncoding()
        }

        // 3c. Encoder Head on CURR ONLY (prev's feat was computed on a previous call).
        encoderExecutable.encode(to: mpsCB,
                                  inputs: [img1IntTD],
                                  results: [f1TD],
                                  executionDescriptor: nil)

        // 3d. Expand prev/curr feat buffers (4-ch fp16) → f0Tex/f1Tex (RGBA textures) for
        //     per-stage feature warps. Both expansions run each call (pool textures are scratch).
        do {
            guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("feat expand: makeComputeCommandEncoder failed")
            }
            enc.setComputePipelineState(conv.rgba4BufToRGBATex)
            enc.setBuffer(prevFeatBuf, offset: 0, index: 0)
            enc.setTexture(p.f0Tex, index: 0)
            dispatch2D(enc, conv.rgba4BufToRGBATex, gridW: intW, gridH: intH)
            enc.setBuffer(currFeatBuf, offset: 0, index: 0)
            enc.setTexture(p.f1Tex, index: 0)
            dispatch2D(enc, conv.rgba4BufToRGBATex, gridW: intW, gridH: intH)
            enc.endEncoding()
        }

        // 4. Per-timestep: set t, run all 5 stages, then full-res warp+blend+writeout.
        //    Mirrors `interpolate(timesteps:)` which calls run() per timestep.
        //    Encoder Head + RGB conversions + texture expansions above are timestep-independent
        //    and run once. For multi-timestep we must commit the prep work first so the
        //    fresh cb in iteration 1+ sees populated pool textures.
        let multiTimestep = timesteps.count > 1
        if multiTimestep {
            mpsCB.commit()
            mpsCB.waitUntilCompleted()
        }

        for (outIdx, t) in timesteps.enumerated() {
            pool.setTimestep(t)
            if multiTimestep {
                guard let cb2 = queue.makeCommandBuffer() else {
                    throw IFNetError.commandBufferFailed("per-timestep makeCommandBuffer failed")
                }
                mpsCB = MPSCommandBuffer(commandBuffer: cb2)
            }

            var accFlowTD: MPSGraphTensorData? = nil
            var accMaskTD: MPSGraphTensorData? = nil
            var accFlowBuf: MTLBuffer? = nil
            var accMaskBuf: MTLBuffer? = nil
            var accFeatTD: MPSGraphTensorData? = nil
            var warped0IntTD: MPSGraphTensorData? = nil
            var warped1IntTD: MPSGraphTensorData? = nil
            var accIsInSlotA = true
            var featIsInSlotA = true

            for i in 0..<scaleList.count {
                let stage = stageExecutables[i]

                let outFeatTD: MPSGraphTensorData
                if i == 0 {
                    outFeatTD = featAccTD_A
                } else {
                    outFeatTD = featIsInSlotA ? featAccTD_B : featAccTD_A
                }

                let allInputPairs: [(MPSGraphTensor, MPSGraphTensorData)]
                if i == 0 {
                    allInputPairs = [
                        (stage.inputTensors[0], img0IntTD),   // img0 (prev)
                        (stage.inputTensors[1], img1IntTD),   // img1 (curr)
                        (stage.inputTensors[2], f0TD),         // f0 (prev feat)
                        (stage.inputTensors[3], f1TD),         // f1 (curr feat)
                        (stage.inputTensors[4], tTD),          // t
                    ]
                } else {
                    allInputPairs = [
                        (stage.inputTensors[0], img0IntTD),    // img0 (unused in graph)
                        (stage.inputTensors[1], img1IntTD),    // img1 (unused in graph)
                        (stage.inputTensors[2], tTD),           // t
                        (stage.inputTensors[3], warped0IntTD!), // warped0
                        (stage.inputTensors[4], warped1IntTD!), // warped1
                        (stage.inputTensors[5], wf0TD),         // wf0
                        (stage.inputTensors[6], wf1TD),         // wf1
                        (stage.inputTensors[7], accMaskTD!),    // prevMask
                        (stage.inputTensors[8], accFeatTD!),    // prevFeat
                        (stage.inputTensors[9], accFlowTD!),    // prevFlow
                    ]
                }

                let feedOrder = stage.executable.feedTensors ?? stage.inputTensors
                let inputs: [MPSGraphTensorData] = feedOrder.map { feedTensor in
                    allInputPairs.first { $0.0 === feedTensor }!.1
                }

                if i == 0 {
                    _ = stage.executable.encode(to: mpsCB,
                                                 inputs: inputs,
                                                 results: [flowAccTD_A, maskAccTD_A, outFeatTD],
                                                 executionDescriptor: nil)
                    accFlowTD    = flowAccTD_A
                    accMaskTD    = maskAccTD_A
                    accFlowBuf   = p.flowAccBufA
                    accMaskBuf   = p.maskAccBufA
                    accFeatTD    = outFeatTD
                    accIsInSlotA = true
                    featIsInSlotA = true
                } else {
                    _ = stage.executable.encode(to: mpsCB,
                                                 inputs: inputs,
                                                 results: [flowRawTD, maskRawTD, outFeatTD],
                                                 executionDescriptor: nil)

                    guard let prevFlowTD = accFlowTD, let prevMaskTD = accMaskTD else {
                        throw IFNetError.dimensionsMismatch
                    }

                    let newFlowAccTD:  MPSGraphTensorData
                    let newMaskAccTD:  MPSGraphTensorData
                    let newFlowAccBuf: MTLBuffer
                    let newMaskAccBuf: MTLBuffer
                    if accIsInSlotA {
                        newFlowAccTD  = flowAccTD_B
                        newMaskAccTD  = maskAccTD_B
                        newFlowAccBuf = p.flowAccBufB
                        newMaskAccBuf = p.maskAccBufB
                    } else {
                        newFlowAccTD  = flowAccTD_A
                        newMaskAccTD  = maskAccTD_A
                        newFlowAccBuf = p.flowAccBufA
                        newMaskAccBuf = p.maskAccBufA
                    }

                    if var rounding = accumulationRounding {
                        guard let encoder = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                            throw IFNetError.commandBufferFailed("flow/mask accumulation encoder failed")
                        }
                        encoder.setComputePipelineState(conv.accumulateFlowMask)
                        encoder.setBuffer(accFlowBuf!, offset: 0, index: 0)
                        encoder.setBuffer(p.flowRawBuf, offset: 0, index: 1)
                        encoder.setBuffer(newFlowAccBuf, offset: 0, index: 2)
                        encoder.setBuffer(accMaskBuf!, offset: 0, index: 3)
                        encoder.setBuffer(p.maskRawBuf, offset: 0, index: 4)
                        encoder.setBuffer(newMaskAccBuf, offset: 0, index: 5)
                        var dimensions = SIMD2<UInt32>(UInt32(intW), UInt32(intH))
                        encoder.setBytes(&dimensions, length: MemoryLayout<SIMD2<UInt32>>.size, index: 6)
                        encoder.setBytes(&rounding, length: MemoryLayout<SIMD2<UInt32>>.size, index: 7)
                        dispatch2D(encoder, conv.accumulateFlowMask, gridW: intW, gridH: intH)
                        encoder.endEncoding()
                    } else {
                        _ = addFlowExecutable.encode(to: mpsCB,
                                                      inputs: [prevFlowTD, flowRawTD],
                                                      results: [newFlowAccTD],
                                                      executionDescriptor: nil)
                        _ = addMaskExecutable.encode(to: mpsCB,
                                                      inputs: [prevMaskTD, maskRawTD],
                                                      results: [newMaskAccTD],
                                                      executionDescriptor: nil)
                    }

                    accFlowTD    = newFlowAccTD
                    accMaskTD    = newMaskAccTD
                    accFlowBuf   = newFlowAccBuf
                    accMaskBuf   = newMaskAccBuf
                    accFeatTD    = outFeatTD
                    accIsInSlotA = !accIsInSlotA
                    featIsInSlotA = !featIsInSlotA
                }

                if let dumpDir = dumpDir {
                    let device = context.device
                    let intFlowBytes = intW * intH * 4 * 2
                    let intMaskBytes = intW * intH * 1 * 2
                    guard let flowStage = device.makeBuffer(length: intFlowBytes, options: .storageModeShared),
                          let maskStage = device.makeBuffer(length: intMaskBytes, options: .storageModeShared) else {
                        throw IFNetError.textureAllocationFailed
                    }
                    if let blitEnc = mpsCB.commandBuffer.makeBlitCommandEncoder() {
                        blitEnc.copy(from: accFlowBuf!, sourceOffset: 0,
                                     to: flowStage, destinationOffset: 0, size: intFlowBytes)
                        blitEnc.copy(from: accMaskBuf!, sourceOffset: 0,
                                     to: maskStage, destinationOffset: 0, size: intMaskBytes)
                        blitEnc.endEncoding()
                    }
                    mpsCB.commit()
                    mpsCB.waitUntilCompleted()
                    let flowStageTD = MPSGraphTensorData(flowStage, shape: flowShape, dataType: .float16)
                    let maskStageTD = MPSGraphTensorData(maskStage, shape: maskShape, dataType: .float16)
                    Self.dumpStageActivations(stageIndex: i,
                                              flow: flowStageTD,
                                              mask: maskStageTD,
                                              dumpDir: dumpDir)
                    guard let cb2 = queue.makeCommandBuffer() else {
                        throw IFNetError.commandBufferFailed("dump: makeCommandBuffer failed")
                    }
                    mpsCB = MPSCommandBuffer(commandBuffer: cb2)
                }

                let isLastStage = (i == scaleList.count - 1)
                if isLastStage { continue }

                guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                    throw IFNetError.commandBufferFailed("stage \(i) post-encoder failed")
                }

                enc.setComputePipelineState(conv.flowSplit)
                enc.setBuffer(accFlowBuf!, offset: 0, index: 0)
                enc.setTexture(p.stageFlowTex0, index: 0)
                enc.setTexture(p.stageFlowTex1, index: 1)
                dispatch2D(enc, conv.flowSplit, gridW: intW, gridH: intH)

                warp.encode(into: enc, source: p.srcTex0Int, flow: p.stageFlowTex0, output: p.stageWarpedTex0)
                warp.encode(into: enc, source: p.srcTex1Int, flow: p.stageFlowTex1, output: p.stageWarpedTex1)

                enc.setComputePipelineState(conv.rgbaTexToRGBBuf)
                enc.setTexture(p.stageWarpedTex0, index: 0)
                enc.setBuffer(p.stageWarped0Buf, offset: 0, index: 0)
                dispatch2D(enc, conv.rgbaTexToRGBBuf, gridW: intW, gridH: intH)
                enc.setTexture(p.stageWarpedTex1, index: 0)
                enc.setBuffer(p.stageWarped1Buf, offset: 0, index: 0)
                dispatch2D(enc, conv.rgbaTexToRGBBuf, gridW: intW, gridH: intH)

                warp.encode(into: enc, source: p.f0Tex, flow: p.stageFlowTex0, output: p.stageWf0Tex)
                warp.encode(into: enc, source: p.f1Tex, flow: p.stageFlowTex1, output: p.stageWf1Tex)

                enc.setComputePipelineState(conv.rgbaTexTo4ChBuf)
                enc.setTexture(p.stageWf0Tex, index: 0)
                enc.setBuffer(p.stageWf0Buf, offset: 0, index: 0)
                dispatch2D(enc, conv.rgbaTexTo4ChBuf, gridW: intW, gridH: intH)
                enc.setTexture(p.stageWf1Tex, index: 0)
                enc.setBuffer(p.stageWf1Buf, offset: 0, index: 0)
                dispatch2D(enc, conv.rgbaTexTo4ChBuf, gridW: intW, gridH: intH)

                enc.endEncoding()

                warped0IntTD = stageWarped0TD
                warped1IntTD = stageWarped1TD
            }

            // 5. Final flow/mask + full-res warp + blend + writeout for THIS timestep.
            guard let _ = accFlowTD,
                  let _ = accMaskTD,
                  let finalFlowIntBuf = accFlowBuf else {
                // No stages ran — fallback. Convert prev img to this output.
                let output = outputs[outIdx]
                let outTex = try PixelBufferConvert.makeTexture(from: output, textureCache: textureCache)
                guard let enc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                    throw IFNetError.commandBufferFailed("fallback convert: makeComputeCommandEncoder failed")
                }
                enc.setComputePipelineState(conv.rgbToBGRA)
                enc.setBuffer(prevImgFullBuf, offset: 0, index: 0)
                enc.setTexture(outTex, index: 0)
                dispatch2D(enc, conv.rgbToBGRA, gridW: width, gridH: height)
                enc.endEncoding()
                mpsCB.commit()
                mpsCB.waitUntilCompleted()
                continue
            }

            guard let resolvedMaskBuf = accMaskBuf else { throw IFNetError.dimensionsMismatch }

            // 6. Full-res warp + blend + BGRA writeout.
            let output = outputs[outIdx]
            let outTex = try PixelBufferConvert.makeTexture(from: output, textureCache: textureCache)

            guard let finalEnc = mpsCB.commandBuffer.makeComputeCommandEncoder() else {
                throw IFNetError.commandBufferFailed("final warp encoder failed")
            }
            if useInternalScale {
                finalEnc.setComputePipelineState(conv.flowUpsampleSplit)
                finalEnc.setBuffer(finalFlowIntBuf, offset: 0, index: 0)
                var intDim: SIMD2<UInt32> = SIMD2(UInt32(intW), UInt32(intH))
                var valueScale: Float = Float(1.0 / internalScale)
                finalEnc.setBytes(&intDim, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
                finalEnc.setBytes(&valueScale, length: MemoryLayout<Float>.size, index: 2)
                finalEnc.setTexture(p.finalFlowTex0, index: 0)
                finalEnc.setTexture(p.finalFlowTex1, index: 1)
                dispatch2D(finalEnc, conv.flowUpsampleSplit, gridW: width, gridH: height)
            } else {
                finalEnc.setComputePipelineState(conv.flowSplit)
                finalEnc.setBuffer(finalFlowIntBuf, offset: 0, index: 0)
                finalEnc.setTexture(p.finalFlowTex0, index: 0)
                finalEnc.setTexture(p.finalFlowTex1, index: 1)
                dispatch2D(finalEnc, conv.flowSplit, gridW: width, gridH: height)
            }

            warp.encodeToBuffer(into: finalEnc,
                                 source: p.srcTex0Full,
                                 flow: p.finalFlowTex0,
                                 output: p.finalWarped0Buf,
                                 width: width, height: height)
            warp.encodeToBuffer(into: finalEnc,
                                 source: p.srcTex1Full,
                                 flow: p.finalFlowTex1,
                                 output: p.finalWarped1Buf,
                                 width: width, height: height)

            let cropped = outTex.width != width || outTex.height != height
            let blendMask = cropped ? conv.blendUpsampleMaskAndPackCropped : conv.blendUpsampleMaskAndPack
            if cropped {
                var sourceDim = SIMD2<UInt32>(UInt32(width), UInt32(height))
                finalEnc.setBytes(&sourceDim, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 4)
            }
            if useInternalScale {
                finalEnc.setComputePipelineState(blendMask)
                finalEnc.setBuffer(p.finalWarped0Buf, offset: 0, index: 0)
                finalEnc.setBuffer(p.finalWarped1Buf, offset: 0, index: 1)
                finalEnc.setBuffer(resolvedMaskBuf, offset: 0, index: 2)
                var intDimM: SIMD2<UInt32> = SIMD2(UInt32(intW), UInt32(intH))
                finalEnc.setBytes(&intDimM, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
                finalEnc.setTexture(outTex, index: 0)
                dispatch2D(finalEnc, blendMask, gridW: outTex.width, gridH: outTex.height)
            } else {
                finalEnc.setComputePipelineState(conv.blendAndPack)
                finalEnc.setBuffer(p.finalWarped0Buf, offset: 0, index: 0)
                finalEnc.setBuffer(p.finalWarped1Buf, offset: 0, index: 1)
                finalEnc.setBuffer(resolvedMaskBuf, offset: 0, index: 2)
                finalEnc.setTexture(outTex, index: 0)
                dispatch2D(finalEnc, conv.blendAndPack, gridW: width, gridH: height)
            }
            finalEnc.endEncoding()

            if gpuTimingEnabled && outIdx == timesteps.count - 1 {
                let cbRef = mpsCB.commandBuffer
                cbRef.addCompletedHandler { cb in
                    let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000
                    let kernelMs = (cb.kernelEndTime - cb.kernelStartTime) * 1000
                    let line = String(format: "[bench] stream gpu=%.2f ms kernel=%.2f ms\n", gpuMs, kernelMs)
                    FileHandle.standardError.write(Data(line.utf8))
                }
            }

            mpsCB.commit()
            mpsCB.waitUntilCompleted()
        }  // end per-timestep loop
    }

    // MARK: - Per-stage debug dump (gated by RIFE_DUMP_DIR env var)

    private static func tensorDataToFloat32(_ td: MPSGraphTensorData) -> [Float] {
        let shape = td.shape
        let elemCount = shape.reduce(1) { $0 * $1.intValue }
        var fp16 = [UInt16](repeating: 0, count: elemCount)
        td.mpsndarray().readBytes(&fp16, strideBytes: nil as UnsafeMutablePointer<Int>?)
        return fp16.map { HalfPrecision.bitsToFloat($0) }
    }

    private static func dumpStageActivations(stageIndex: Int,
                                              flow: MPSGraphTensorData,
                                              mask: MPSGraphTensorData,
                                              dumpDir: String) {
        let dir = URL(fileURLWithPath: dumpDir)
        let flowValues = tensorDataToFloat32(flow)
        let maskValues = tensorDataToFloat32(mask)

        let flowShape = flow.shape.map { $0.intValue }
        let maskShape = mask.shape.map { $0.intValue }

        let flowURL = dir.appendingPathComponent("stage\(stageIndex)_flow.npy")
        let maskURL = dir.appendingPathComponent("stage\(stageIndex)_mask.npy")

        try? writeNumpyFloat32(flowValues, shape: flowShape, to: flowURL)
        try? writeNumpyFloat32(maskValues, shape: maskShape, to: maskURL)
    }

    private static func writeNumpyFloat32(_ values: [Float], shape: [Int], to url: URL) throws {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(shape.map(String.init).joined(separator: ", ")))}"
        var headerStr = header
        let baseLen = 10 + headerStr.utf8.count + 1
        let padding = (64 - (baseLen % 64)) % 64
        headerStr.append(String(repeating: " ", count: padding))
        headerStr.append("\n")

        var data = Data()
        data.append(0x93)
        data.append(contentsOf: "NUMPY".utf8)
        data.append(0x01)
        data.append(0x00)
        let headerLenLE = UInt16(headerStr.utf8.count).littleEndian
        withUnsafeBytes(of: headerLenLE) { data.append(contentsOf: $0) }
        data.append(contentsOf: headerStr.utf8)
        values.withUnsafeBytes { rawBuf in
            data.append(contentsOf: rawBuf)
        }
        try data.write(to: url)
    }
}
