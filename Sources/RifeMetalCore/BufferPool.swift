import Foundation
import Metal

// MARK: - BufferPool

/// Pre-allocates all intermediate MTLBuffers and MTLTextures needed by IFNetGraph.run() at
/// IFNetGraph construction time, then exposes them as named properties so run() can reuse them
/// without any per-call heap traffic.
///
/// Layout (all fp16):
///   fullRGBByteCount  = W × H × 3 × 2
///   intRGBByteCount   = intW × intH × 3 × 2
///   intFlowByteCount  = intW × intH × 4 × 2
///   intMaskByteCount  = intW × intH × 1 × 2
///
/// Buffers that the CPU writes into (input RGB, t constant) use .storageModeShared.
/// Pure GPU-side intermediates (warped scratch, flow/mask accumulators, output) use .private
/// when allocated from a heap, falling back to .shared when not on a heap (for debugging
/// or when heap allocation fails).
///
/// RIFE_DUMP_DIR: The debug dump path reads back MPSGraphTensorData via MPSNDArray, not
/// directly via MTLBuffer.contents(), so private-storage buffers work fine for dump too.
final class BufferPool {

    // MARK: Stored heap (optional)

    /// Owns all private-mode allocations when heap creation succeeds.
    let heap: MTLHeap?

    // MARK: Input buffers (CPU-writable → .shared)

    /// Full-res packed fp16 RGB for input image 0:  [1, H,    W,    3]
    let img0FullBuf: MTLBuffer
    /// Full-res packed fp16 RGB for input image 1:  [1, H,    W,    3]
    let img1FullBuf: MTLBuffer
    /// Internal-res packed fp16 RGB for IFNet stage 0 image 0:  [1, intH, intW, 3]
    /// When internalScale == 1.0 this is the same object as img0FullBuf.
    let img0IntBuf: MTLBuffer
    /// Internal-res packed fp16 RGB for IFNet stage 0 image 1.
    /// When internalScale == 1.0 this is the same object as img1FullBuf.
    let img1IntBuf: MTLBuffer
    /// Timestep tensor at internal res:  [1, intH, intW, 1]. Filled with a uniform scalar via
    /// `setTimestep(_:)`. Initialized to 0.5 at construction so callers that never call
    /// setTimestep see the historical t=0.5 midframe behavior.
    let tBuf: MTLBuffer

    /// Rewrites every element of `tBuf` with the given uniform timestep (cast to Float16).
    /// Safe to call between runs — the previous run's GPU work has completed by the time the
    /// caller (`IFNetGraph.run`) reaches this on the next call (run is synchronous via
    /// `waitUntilCompleted`).
    func setTimestep(_ t: Float) {
        let count = tBuf.length / MemoryLayout<Float16>.stride
        let ptr = tBuf.contents().bindMemory(to: Float16.self, capacity: count)
        let h = Float16(t)
        for i in 0..<count { ptr[i] = h }
    }

    // MARK: Source RGBA textures (GPU-write from bgraToRGB + rgbBufToRGBATex → GPU-read in warp)

    /// Internal-res RGBA16Float source texture for warp (from img0Int).
    let srcTex0Int: MTLTexture
    /// Internal-res RGBA16Float source texture for warp (from img1Int).
    let srcTex1Int: MTLTexture
    /// Full-res RGBA16Float source texture for warp (from img0Full).
    let srcTex0Full: MTLTexture
    /// Full-res RGBA16Float source texture for warp (from img1Full).
    let srcTex1Full: MTLTexture

    // MARK: Per-stage flow/mask ping-pong (GPU-only)
    //
    // We run at most one stage at a time and commit all work before the next call, so we
    // can reuse a single pair of "raw" (stage output) buffers and a single pair of
    // "accumulator" buffers for every stage, plus a second accumulator pair for the addFlow/
    // addMask result so we never alias inputs with outputs.

    /// Raw flow output from current stage:    [1, intH, intW, 4]
    let flowRawBuf: MTLBuffer
    /// Raw mask output from current stage:    [1, intH, intW, 1]
    let maskRawBuf: MTLBuffer
    /// Accumulated flow (even stages / first):  [1, intH, intW, 4]
    let flowAccBufA: MTLBuffer
    /// Accumulated mask (even stages / first):  [1, intH, intW, 1]
    let maskAccBufA: MTLBuffer
    /// Accumulated flow (odd stages):           [1, intH, intW, 4]
    let flowAccBufB: MTLBuffer
    /// Accumulated mask (odd stages):           [1, intH, intW, 1]
    let maskAccBufB: MTLBuffer

    // MARK: Per-stage warp scratch (GPU-only, reused each non-last stage)

    /// Flow split RG16Float for direction 0 at internal res.
    let stageFlowTex0: MTLTexture
    /// Flow split RG16Float for direction 1 at internal res.
    let stageFlowTex1: MTLTexture
    /// Warped RGBA16Float for source 0 at internal res.
    let stageWarpedTex0: MTLTexture
    /// Warped RGBA16Float for source 1 at internal res.
    let stageWarpedTex1: MTLTexture
    /// Packed warped RGB output for source 0 at internal res:  [1, intH, intW, 3]
    let stageWarped0Buf: MTLBuffer
    /// Packed warped RGB output for source 1 at internal res:  [1, intH, intW, 3]
    let stageWarped1Buf: MTLBuffer

    // MARK: v4.26 encoder + feat propagation buffers (GPU-only, NHWC fp16)

    /// Encoder feature for img0 at internal res:  [1, intH, intW, 4]
    let f0Buf: MTLBuffer
    /// Encoder feature for img1 at internal res:  [1, intH, intW, 4]
    let f1Buf: MTLBuffer

    /// Per-stage feat output, ping-pong A:  [1, intH, intW, 8]
    let stageFeatBufA: MTLBuffer
    /// Per-stage feat output, ping-pong B:  [1, intH, intW, 8]
    let stageFeatBufB: MTLBuffer

    /// Per-stage warped f0 (4-ch) at internal res:  [1, intH, intW, 4]
    let stageWf0Buf: MTLBuffer
    /// Per-stage warped f1 (4-ch) at internal res:  [1, intH, intW, 4]
    let stageWf1Buf: MTLBuffer

    /// RGBA16Float texture views for warp source (read-only after encoder Head fills f0Buf/f1Buf).
    let f0Tex: MTLTexture
    let f1Tex: MTLTexture

    /// RGBA16Float texture views for warp output (filled per stage).
    let stageWf0Tex: MTLTexture
    let stageWf1Tex: MTLTexture

    // MARK: Final full-res warp + blend (GPU-only)

    /// Flow split RG16Float for direction 0 at full res.
    let finalFlowTex0: MTLTexture
    /// Flow split RG16Float for direction 1 at full res.
    let finalFlowTex1: MTLTexture
    /// Warped RGBA16Float for source 0 at full res.
    let finalWarpedTex0: MTLTexture
    /// Warped RGBA16Float for source 1 at full res.
    let finalWarpedTex1: MTLTexture
    /// Packed warped RGB for source 0 at full res:  [1, H, W, 3]
    let finalWarped0Buf: MTLBuffer
    /// Packed warped RGB for source 1 at full res:  [1, H, W, 3]
    let finalWarped1Buf: MTLBuffer

    // MARK: Initialisation

    init(device: MTLDevice,
         paddedW: Int,
         paddedH: Int,
         internalWidth intW: Int,
         internalHeight intH: Int,
         internalScale: Double) throws {

        let fullRGBBytes  = paddedW * paddedH * 3 * 2
        let intRGBBytes   = intW   * intH   * 3 * 2
        let intFlowBytes  = intW   * intH   * 4 * 2
        let intMaskBytes  = intW   * intH   * 1 * 2
        let intFeat4Bytes = intW   * intH   * 4 * 2  // 4-ch fp16 encoder/warped feat
        let intFeatBytes  = intW   * intH   * 8 * 2  // 8-ch fp16 per-stage feat ping-pong

        let useInternalScale = internalScale != 1.0

        // ------------------------------------------------------------------ //
        // Attempt MTLHeap for all private-mode allocations.                   //
        // Private buffers: flowRawBuf, maskRawBuf, flowAccBuf{A,B},          //
        //   maskAccBuf{A,B}, stageWarped{0,1}Buf,                            //
        //   finalWarped{0,1}Buf.                                             //
        // Private textures: srcTex{0,1}Int, srcTex{0,1}Full,                 //
        //   stageFlowTex{0,1}, stageWarpedTex{0,1},                         //
        //   finalFlowTex{0,1}, finalWarpedTex{0,1}.                         //
        // ------------------------------------------------------------------ //

        // Helper: size (bytes) an MTLTextureDescriptor would occupy on this device.
        func heapTexBytes(_ desc: MTLTextureDescriptor) -> Int {
            return device.heapTextureSizeAndAlign(descriptor: desc).size
        }
        func heapBufBytes(_ length: Int) -> Int {
            return device.heapBufferSizeAndAlign(length: length, options: .storageModePrivate).size
        }

        // Build descriptors for all private textures upfront so we can sum their heap sizes.
        func rgbaDesc(w: Int, h: Int) -> MTLTextureDescriptor {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            return d
        }
        func rgDesc(w: Int, h: Int) -> MTLTextureDescriptor {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rg16Float, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            return d
        }

        let srcTex0IntDesc   = rgbaDesc(w: intW,    h: intH)
        let srcTex1IntDesc   = rgbaDesc(w: intW,    h: intH)
        let srcTex0FullDesc  = rgbaDesc(w: paddedW, h: paddedH)
        let srcTex1FullDesc  = rgbaDesc(w: paddedW, h: paddedH)
        let stageFlowTex0Desc  = rgDesc(w: intW, h: intH)
        let stageFlowTex1Desc  = rgDesc(w: intW, h: intH)
        let stageWarpedTex0Desc = rgbaDesc(w: intW, h: intH)
        let stageWarpedTex1Desc = rgbaDesc(w: intW, h: intH)
        let finalFlowTex0Desc  = rgDesc(w: paddedW, h: paddedH)
        let finalFlowTex1Desc  = rgDesc(w: paddedW, h: paddedH)
        let finalWarpedTex0Desc = rgbaDesc(w: paddedW, h: paddedH)
        let finalWarpedTex1Desc = rgbaDesc(w: paddedW, h: paddedH)
        // v4.26: 4-ch feature texture descriptor (shared shape for f0/f1 and stageWf0/stageWf1).
        let feat4Desc           = rgbaDesc(w: intW,    h: intH)

        // Compute total heap bytes needed (buffers + textures, all private).
        var heapBytes = 0
        // Buffers
        heapBytes += heapBufBytes(intFlowBytes)   // flowRawBuf
        heapBytes += heapBufBytes(intMaskBytes)   // maskRawBuf
        heapBytes += heapBufBytes(intFlowBytes)   // flowAccBufA
        heapBytes += heapBufBytes(intMaskBytes)   // maskAccBufA
        heapBytes += heapBufBytes(intFlowBytes)   // flowAccBufB
        heapBytes += heapBufBytes(intMaskBytes)   // maskAccBufB
        heapBytes += heapBufBytes(intRGBBytes)    // stageWarped0Buf
        heapBytes += heapBufBytes(intRGBBytes)    // stageWarped1Buf
        heapBytes += heapBufBytes(fullRGBBytes)   // finalWarped0Buf
        heapBytes += heapBufBytes(fullRGBBytes)   // finalWarped1Buf
        // Textures
        heapBytes += heapTexBytes(srcTex0IntDesc)
        heapBytes += heapTexBytes(srcTex1IntDesc)
        heapBytes += heapTexBytes(srcTex0FullDesc)
        heapBytes += heapTexBytes(srcTex1FullDesc)
        heapBytes += heapTexBytes(stageFlowTex0Desc)
        heapBytes += heapTexBytes(stageFlowTex1Desc)
        heapBytes += heapTexBytes(stageWarpedTex0Desc)
        heapBytes += heapTexBytes(stageWarpedTex1Desc)
        heapBytes += heapTexBytes(finalFlowTex0Desc)
        heapBytes += heapTexBytes(finalFlowTex1Desc)
        heapBytes += heapTexBytes(finalWarpedTex0Desc)
        heapBytes += heapTexBytes(finalWarpedTex1Desc)
        // v4.26 encoder + feat propagation buffers.
        heapBytes += heapBufBytes(intFeat4Bytes) * 2     // f0Buf, f1Buf
        heapBytes += heapBufBytes(intFeatBytes)  * 2     // stageFeatBufA, stageFeatBufB
        heapBytes += heapBufBytes(intFeat4Bytes) * 2     // stageWf0Buf, stageWf1Buf
        heapBytes += heapTexBytes(feat4Desc)     * 4     // f0Tex, f1Tex, stageWf0Tex, stageWf1Tex

        // Add 25% slack so Metal's internal alignment bookkeeping doesn't push us over.
        let heapSize = (heapBytes * 5) / 4

        // Try to create the heap; fall back gracefully if creation or any sub-alloc fails.
        var attemptedHeap: MTLHeap? = nil
        do {
            let heapDesc = MTLHeapDescriptor()
            heapDesc.size = heapSize
            heapDesc.storageMode = .private
            heapDesc.hazardTrackingMode = .tracked
            attemptedHeap = device.makeHeap(descriptor: heapDesc)
        }
        // (If makeHeap returns nil we just leave attemptedHeap = nil)

        // Helper: allocate a private buffer from heap, fall back to device if heap is nil or full.
        func makePrivateBuf(_ length: Int) -> MTLBuffer? {
            if let h = attemptedHeap,
               let b = h.makeBuffer(length: length, options: .storageModePrivate) {
                return b
            }
            return device.makeBuffer(length: length, options: .storageModePrivate)
        }

        // Helper: allocate a private texture from heap, fall back to device.
        func makePrivateTex(_ desc: MTLTextureDescriptor) -> MTLTexture? {
            if let h = attemptedHeap,
               let t = h.makeTexture(descriptor: desc) {
                return t
            }
            // Fallback: allocate directly from device (descriptor storage mode may need to be adjusted)
            let fallbackDesc = desc.copy() as! MTLTextureDescriptor
            fallbackDesc.storageMode = .private
            return device.makeTexture(descriptor: fallbackDesc)
        }

        // ------------------------------------------------------------------ //
        // CPU-visible (shared) buffers — NOT on the private heap.             //
        // ------------------------------------------------------------------ //
        guard let _img0FullBuf = device.makeBuffer(length: fullRGBBytes, options: .storageModeShared),
              let _img1FullBuf = device.makeBuffer(length: fullRGBBytes, options: .storageModeShared),
              let _tBuf        = device.makeBuffer(length: intMaskBytes, options: .storageModeShared)
        else { throw IFNetError.textureAllocationFailed }
        self.img0FullBuf = _img0FullBuf
        self.img1FullBuf = _img1FullBuf
        self.tBuf = _tBuf

        // Initialize tBuf with t=0.5 so a default IFNetGraph.run() with the timestep arg omitted
        // yields the historical midframe; setTimestep(_:) overwrites this on every subsequent call.
        let tPtr = _tBuf.contents().bindMemory(to: Float16.self, capacity: intW * intH)
        for i in 0..<(intW * intH) { tPtr[i] = 0.5 }

        if useInternalScale {
            guard let b0 = device.makeBuffer(length: intRGBBytes, options: .storageModeShared),
                  let b1 = device.makeBuffer(length: intRGBBytes, options: .storageModeShared) else {
                throw IFNetError.textureAllocationFailed
            }
            self.img0IntBuf = b0
            self.img1IntBuf = b1
        } else {
            self.img0IntBuf = _img0FullBuf
            self.img1IntBuf = _img1FullBuf
        }

        // ------------------------------------------------------------------ //
        // Private allocations (heap-backed or device-direct on fallback).     //
        // ------------------------------------------------------------------ //
        guard let _srcTex0Int  = makePrivateTex(srcTex0IntDesc),
              let _srcTex1Int  = makePrivateTex(srcTex1IntDesc),
              let _srcTex0Full = makePrivateTex(srcTex0FullDesc),
              let _srcTex1Full = makePrivateTex(srcTex1FullDesc) else {
            throw IFNetError.textureAllocationFailed
        }
        self.srcTex0Int  = _srcTex0Int
        self.srcTex1Int  = _srcTex1Int
        self.srcTex0Full = _srcTex0Full
        self.srcTex1Full = _srcTex1Full

        guard let _flowRawBuf  = makePrivateBuf(intFlowBytes),
              let _maskRawBuf  = makePrivateBuf(intMaskBytes),
              let _flowAccBufA = makePrivateBuf(intFlowBytes),
              let _maskAccBufA = makePrivateBuf(intMaskBytes),
              let _flowAccBufB = makePrivateBuf(intFlowBytes),
              let _maskAccBufB = makePrivateBuf(intMaskBytes) else {
            throw IFNetError.textureAllocationFailed
        }
        self.flowRawBuf  = _flowRawBuf
        self.maskRawBuf  = _maskRawBuf
        self.flowAccBufA = _flowAccBufA
        self.maskAccBufA = _maskAccBufA
        self.flowAccBufB = _flowAccBufB
        self.maskAccBufB = _maskAccBufB

        guard let _stageFlowTex0    = makePrivateTex(stageFlowTex0Desc),
              let _stageFlowTex1    = makePrivateTex(stageFlowTex1Desc),
              let _stageWarpedTex0  = makePrivateTex(stageWarpedTex0Desc),
              let _stageWarpedTex1  = makePrivateTex(stageWarpedTex1Desc),
              let _stageWarped0Buf  = makePrivateBuf(intRGBBytes),
              let _stageWarped1Buf  = makePrivateBuf(intRGBBytes) else {
            throw IFNetError.textureAllocationFailed
        }
        self.stageFlowTex0   = _stageFlowTex0
        self.stageFlowTex1   = _stageFlowTex1
        self.stageWarpedTex0 = _stageWarpedTex0
        self.stageWarpedTex1 = _stageWarpedTex1
        self.stageWarped0Buf = _stageWarped0Buf
        self.stageWarped1Buf = _stageWarped1Buf

        // v4.26 encoder + feat propagation buffers — sized at internal resolution.
        guard let _f0Buf         = makePrivateBuf(intFeat4Bytes),
              let _f1Buf         = makePrivateBuf(intFeat4Bytes),
              let _stageFeatBufA = makePrivateBuf(intFeatBytes),
              let _stageFeatBufB = makePrivateBuf(intFeatBytes),
              let _stageWf0Buf   = makePrivateBuf(intFeat4Bytes),
              let _stageWf1Buf   = makePrivateBuf(intFeat4Bytes)
        else { throw IFNetError.textureAllocationFailed }
        self.f0Buf         = _f0Buf
        self.f1Buf         = _f1Buf
        self.stageFeatBufA = _stageFeatBufA
        self.stageFeatBufB = _stageFeatBufB
        self.stageWf0Buf   = _stageWf0Buf
        self.stageWf1Buf   = _stageWf1Buf

        // RGBA16Float texture views over the 4-ch feature buffers.
        guard let _f0Tex       = makePrivateTex(feat4Desc),
              let _f1Tex       = makePrivateTex(feat4Desc),
              let _stageWf0Tex = makePrivateTex(feat4Desc),
              let _stageWf1Tex = makePrivateTex(feat4Desc)
        else { throw IFNetError.textureAllocationFailed }
        self.f0Tex       = _f0Tex
        self.f1Tex       = _f1Tex
        self.stageWf0Tex = _stageWf0Tex
        self.stageWf1Tex = _stageWf1Tex

        guard let _finalFlowTex0   = makePrivateTex(finalFlowTex0Desc),
              let _finalFlowTex1   = makePrivateTex(finalFlowTex1Desc),
              let _finalWarpedTex0 = makePrivateTex(finalWarpedTex0Desc),
              let _finalWarpedTex1 = makePrivateTex(finalWarpedTex1Desc),
              let _finalWarped0Buf = makePrivateBuf(fullRGBBytes),
              let _finalWarped1Buf = makePrivateBuf(fullRGBBytes) else {
            throw IFNetError.textureAllocationFailed
        }
        self.finalFlowTex0   = _finalFlowTex0
        self.finalFlowTex1   = _finalFlowTex1
        self.finalWarpedTex0 = _finalWarpedTex0
        self.finalWarpedTex1 = _finalWarpedTex1
        self.finalWarped0Buf = _finalWarped0Buf
        self.finalWarped1Buf = _finalWarped1Buf

        // Commit the heap only if every sub-allocation succeeded.
        // (If any guard above threw, heap is deallocated by ARC.)
        self.heap = attemptedHeap
    }
}
