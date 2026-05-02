import Foundation
import Metal

public enum BackwardWarpError: Error {
    case shaderNotFound
    case pipelineCompilationFailed(String)
}

/// Compiles and dispatches the rifeBackwardWarp Metal kernel.
public final class BackwardWarpKernel {
    public let pipeline: MTLComputePipelineState
    /// Same warp but writes a packed fp16 RGB MTLBuffer instead of an RGBA texture.
    /// Used for the FINAL full-res warps so we can feed rifeBlendAndPack directly.
    public let bufferPipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            throw BackwardWarpError.pipelineCompilationFailed("library load failed: \(error)")
        }
        guard let function = library.makeFunction(name: "rifeBackwardWarp") else {
            throw BackwardWarpError.shaderNotFound
        }
        guard let bufFunction = library.makeFunction(name: "rifeBackwardWarpToBuffer") else {
            throw BackwardWarpError.shaderNotFound
        }
        do {
            self.pipeline = try device.makeComputePipelineState(function: function)
            self.bufferPipeline = try device.makeComputePipelineState(function: bufFunction)
        } catch {
            throw BackwardWarpError.pipelineCompilationFailed(String(describing: error))
        }
    }

    /// Encodes the warp into the given encoder. Caller is responsible for setting up textures and dispatch.
    public func encode(into encoder: MTLComputeCommandEncoder,
                       source: MTLTexture,
                       flow: MTLTexture,
                       output: MTLTexture) {
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(flow, index: 1)
        encoder.setTexture(output, index: 2)

        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let tg = MTLSize(width: w, height: h, depth: 1)
        let groups = MTLSize(
            width: (output.width + w - 1) / w,
            height: (output.height + h - 1) / h,
            depth: 1
        )
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
    }

    /// Encodes the buffer-output warp variant. The output buffer is an fp16 NHWC RGB
    /// buffer of shape [1, H, W, 3]. Caller passes (W, H) since it's not derivable
    /// from a buffer.
    public func encodeToBuffer(into encoder: MTLComputeCommandEncoder,
                                source: MTLTexture,
                                flow: MTLTexture,
                                output: MTLBuffer,
                                width: Int,
                                height: Int) {
        encoder.setComputePipelineState(bufferPipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(flow, index: 1)
        encoder.setBuffer(output, offset: 0, index: 0)
        var dim: SIMD2<UInt32> = SIMD2(UInt32(width), UInt32(height))
        encoder.setBytes(&dim, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)

        let w = bufferPipeline.threadExecutionWidth
        let h = max(1, bufferPipeline.maxTotalThreadsPerThreadgroup / w)
        let tg = MTLSize(width: w, height: h, depth: 1)
        let groups = MTLSize(
            width: (width + w - 1) / w,
            height: (height + h - 1) / h,
            depth: 1
        )
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
    }
}

/// Compiles and exposes the BGRA↔RGB conversion kernels and supporting helpers.
public final class ConversionKernels {
    public let bgraToRGB: MTLComputePipelineState
    public let rgbToBGRA: MTLComputePipelineState
    /// Split a [1,H,W,4] fp16 flow buffer into two RG16Float textures.
    public let flowSplit: MTLComputePipelineState
    /// Expand a packed fp16 RGB MTLBuffer into an RGBA16Float texture (alpha=1).
    public let rgbBufToRGBATex: MTLComputePipelineState
    /// Pack an RGBA16Float texture into a fp16 RGB MTLBuffer (drops alpha).
    public let rgbaTexToRGBBuf: MTLComputePipelineState
    /// Fused BGRA8 texture → fp16 RGB buffer at downsampled resolution (balanced tier only).
    public let bgraDownsampleToRGB: MTLComputePipelineState
    /// Fused blend + sigmoid + RGB→BGRA pack (full res, replaces blendExecutable+rgbToBGRA pair).
    public let blendAndPack: MTLComputePipelineState
    /// Fused bilinear upsample + flow split (balanced tier final flow).
    public let flowUpsampleSplit: MTLComputePipelineState
    /// Fused mask-upsample + blend + RGB→BGRA pack (balanced tier final blend).
    public let blendUpsampleMaskAndPack: MTLComputePipelineState
    /// Expand a packed fp16 4-channel buffer → RGBA16Float texture (v4.26 encoder feature buf→tex).
    public let rgba4BufToRGBATex: MTLComputePipelineState
    /// Pack an RGBA16Float texture → fp16 4-channel buffer (v4.26 warped feature tex→buf).
    public let rgbaTexTo4ChBuf: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            throw BackwardWarpError.pipelineCompilationFailed("library load failed: \(error)")
        }
        guard let f1  = library.makeFunction(name: "rifeBGRAToRGB"),
              let f2  = library.makeFunction(name: "rifeRGBToBGRA"),
              let f3  = library.makeFunction(name: "rifeFlowSplit"),
              let f4  = library.makeFunction(name: "rifeRGBBufToRGBATex"),
              let f5  = library.makeFunction(name: "rifeRGBATexToRGBBuf"),
              let f6  = library.makeFunction(name: "rifeBGRADownsampleToRGB"),
              let f7  = library.makeFunction(name: "rifeBlendAndPack"),
              let f8  = library.makeFunction(name: "rifeFlowUpsampleSplit"),
              let f9  = library.makeFunction(name: "rifeBlendUpsampleMaskAndPack"),
              let f10 = library.makeFunction(name: "rife4ChBufToRGBATex"),
              let f11 = library.makeFunction(name: "rifeRGBATexTo4ChBuf") else {
            throw BackwardWarpError.shaderNotFound
        }
        self.bgraToRGB = try device.makeComputePipelineState(function: f1)
        self.rgbToBGRA = try device.makeComputePipelineState(function: f2)
        self.flowSplit = try device.makeComputePipelineState(function: f3)
        self.rgbBufToRGBATex = try device.makeComputePipelineState(function: f4)
        self.rgbaTexToRGBBuf = try device.makeComputePipelineState(function: f5)
        self.bgraDownsampleToRGB = try device.makeComputePipelineState(function: f6)
        self.blendAndPack = try device.makeComputePipelineState(function: f7)
        self.flowUpsampleSplit = try device.makeComputePipelineState(function: f8)
        self.blendUpsampleMaskAndPack = try device.makeComputePipelineState(function: f9)
        self.rgba4BufToRGBATex = try device.makeComputePipelineState(function: f10)
        self.rgbaTexTo4ChBuf = try device.makeComputePipelineState(function: f11)
    }
}
