import Foundation
import CoreVideo
import Metal
import RifeMetalCore

/// A stateful inference session for video streams. Caches the previous
/// frame's encoder Head output across consecutive push() calls.
///
/// Created via `RifeInterpolator.makeStream(width:height:)`. Not
/// thread-safe — caller must serialize push/reset on a single instance.
public final class RifeStream: @unchecked Sendable {
    private var paddedInputBuffer: CVPixelBuffer?

    public let width: Int
    public let height: Int

    // Padded internal-resolution dims (computed at init).
    internal let paddedW: Int
    internal let paddedH: Int
    internal let paddedIntW: Int
    internal let paddedIntH: Int

    // Parent's serial queue; all GPU work routes through this.
    internal let queue: DispatchQueue
    internal let graph: IFNetGraph
    internal let interpolator: RifeInterpolator   // strong; outlives stream

    // Cache buffers (ping-pong slots). Allocated lazily on first push to
    // avoid up-front allocation cost on streams that get created but never used.
    internal var featBuf:    [MTLBuffer] = []  // size 0 until first push
    internal var imgIntBuf:  [MTLBuffer] = []
    internal var imgFullBuf: [MTLBuffer] = []

    internal var prevSlot: Int = 0
    internal var hasCachedPrev: Bool = false

    internal init(interpolator: RifeInterpolator,
                  graph: IFNetGraph,
                  width: Int, height: Int,
                  paddedW: Int, paddedH: Int,
                  paddedIntW: Int, paddedIntH: Int,
                  queue: DispatchQueue) {
        self.interpolator = interpolator
        self.graph = graph
        self.width = width
        self.height = height
        self.paddedW = paddedW
        self.paddedH = paddedH
        self.paddedIntW = paddedIntW
        self.paddedIntH = paddedIntH
        self.queue = queue
    }

    /// Push a frame onto the stream.
    ///
    /// - First call after init or reset(): returns []. The frame is encoded
    ///   and cached as the prev frame for the next call. `timesteps` is
    ///   ignored.
    /// - Subsequent calls with non-empty `timesteps`: returns one
    ///   CVPixelBuffer per requested timestep, interpolated between the
    ///   previously-pushed frame and this one. Then this frame replaces
    ///   the cached prev for the next call.
    /// - Subsequent calls with empty `timesteps`: returns []. The frame is
    ///   encoded and replaces the cached prev (cache rebase) without
    ///   producing any output. Useful for skipping a frame in the stream
    ///   while keeping the cache fresh.
    public func push(_ frame: CVPixelBuffer,
                     timesteps: [Float]) throws -> [CVPixelBuffer] {
        // Validate input dims and format.
        let frameW = CVPixelBufferGetWidth(frame)
        let frameH = CVPixelBufferGetHeight(frame)
        guard frameW == width, frameH == height else {
            throw RifeError.dimensionMismatch
        }
        let fmt = CVPixelBufferGetPixelFormatType(frame)
        let allowed: Set<OSType> = [kCVPixelFormatType_32BGRA, kCVPixelFormatType_32RGBA]
        guard allowed.contains(fmt) else {
            throw RifeError.unsupportedPixelFormat(fmt)
        }

        return try queue.sync { () -> [CVPixelBuffer] in
            do {
            try ensureCacheAllocated()

            // Pad input if needed.
            let needsPad = (paddedW != width) || (paddedH != height)
            let paddedFrame: CVPixelBuffer
            if needsPad {
                paddedFrame = try paddedInput(frame)
            } else {
                paddedFrame = frame
            }

            if !hasCachedPrev {
                // First push: encode and cache; no output produced.
                try encodeAndRotate(paddedFrame)
                hasCachedPrev = true
                return []
            }

            // Subsequent push with empty timesteps: cache-rebase only, no inference.
            guard !timesteps.isEmpty else {
                try encodeAndRotate(paddedFrame)
                return []
            }
            for t in timesteps {
                guard t > 0 && t < 1 else {
                    throw RifeError.invalidTimesteps("timestep \(t) is out of (0, 1)")
                }
            }

            // Allocate padded output buffers.
            var outBuffers: [CVPixelBuffer] = []
            for _ in timesteps {
                let outBuf = try PixelBufferConvert.makeOutputPixelBuffer(
                    width: paddedW, height: paddedH)
                outBuffers.append(outBuf)
            }

            try runStreamAndRotate(curr: paddedFrame,
                                   timesteps: timesteps,
                                   outputs: outBuffers)

            // Crop outputs back to caller-visible dims if we padded.
            if needsPad {
                var cropped: [CVPixelBuffer] = []
                for paddedOut in outBuffers {
                    cropped.append(try interpolator.cropPixelBufferForStream(
                        paddedOut, width: width, height: height))
                }
                return cropped
            }
            return outBuffers
            } catch {
                // 失败时不能假定所有已提交的 GPU 工作都已结束。
                paddedInputBuffer = nil
                throw error
            }
        }
    }

    /// Convenience: t = 0.5 only. First call returns nil.
    public func push(_ frame: CVPixelBuffer) throws -> CVPixelBuffer? {
        let outs = try push(frame, timesteps: [0.5])
        return outs.first
    }

    /// Push a frame, writing the t=0.5 midframe into `output`.
    ///
    /// - First call after init or reset(): returns `false`; `output` is left
    ///   untouched. The frame is encoded and cached as the prev for the next call.
    /// - Subsequent calls: writes the midframe into `output` and returns `true`.
    ///   This frame replaces the cached prev for the next call.
    ///
    /// `frame` and `output` must have dims equal to the stream's locked
    /// `(width, height)` and BGRA/RGBA pixel format. Otherwise throws
    /// `RifeError.dimensionMismatch` or `RifeError.unsupportedPixelFormat`.
    public func push(_ frame: CVPixelBuffer,
                     into output: CVPixelBuffer) throws -> Bool {
        // Validate input dims and format.
        let frameW = CVPixelBufferGetWidth(frame)
        let frameH = CVPixelBufferGetHeight(frame)
        let outW   = CVPixelBufferGetWidth(output)
        let outH   = CVPixelBufferGetHeight(output)
        guard frameW == width, frameH == height,
              outW == width, outH == height else {
            throw RifeError.dimensionMismatch
        }
        let allowed: Set<OSType> = [kCVPixelFormatType_32BGRA, kCVPixelFormatType_32RGBA]
        let fmt = CVPixelBufferGetPixelFormatType(frame)
        let outFmt = CVPixelBufferGetPixelFormatType(output)
        guard allowed.contains(fmt) else {
            throw RifeError.unsupportedPixelFormat(fmt)
        }
        guard allowed.contains(outFmt) else {
            throw RifeError.unsupportedPixelFormat(outFmt)
        }

        return try queue.sync { () -> Bool in
            do {
            try ensureCacheAllocated()

            let needsPad = (paddedW != width) || (paddedH != height)
            let paddedFrame: CVPixelBuffer
            if needsPad {
                paddedFrame = try paddedInput(frame)
            } else {
                paddedFrame = frame
            }

            if !hasCachedPrev {
                // First push: encode + cache; output left untouched.
                try encodeAndRotate(paddedFrame)
                hasCachedPrev = true
                return false
            }

            if needsPad && (!graph.supportsCroppedOutput || outFmt != kCVPixelFormatType_32BGRA || CVPixelBufferGetIOSurface(output) == nil) {
                // Allocate padded interim output, run inference into it, crop
                // into caller's output. (Same shape as stateless interpolate's
                // padded path.)
                let paddedOut = try PixelBufferConvert.makeOutputPixelBuffer(
                    width: paddedW, height: paddedH)
                try runStreamAndRotate(curr: paddedFrame,
                                       timesteps: [0.5],
                                       outputs: [paddedOut])
                try interpolator.cropPixelBufferForStream(paddedOut, into: output)
            } else {
                // Fast path: GPU writes directly into caller's output.
                // Balanced 的可绑定 BGRA 输出直接裁剪写入，其他档位和缓冲保留原路径。
                try runStreamAndRotate(curr: paddedFrame,
                                       timesteps: [0.5],
                                       outputs: [output])
            }
            return true
            } catch {
                paddedInputBuffer = nil
                throw error
            }
        }
    }

    /// Drop cached previous-frame state. Next push goes through the
    /// first-frame path. Cache buffers are retained for reuse.
    public func reset() {
        hasCachedPrev = false
        paddedInputBuffer = nil
    }

    // push 在串行队列内等待 GPU 完成，下一帧才可以覆盖这份输入。
    private func paddedInput(_ frame: CVPixelBuffer) throws -> CVPixelBuffer {
        if paddedInputBuffer == nil {
            paddedInputBuffer = try PixelBufferConvert.makeOutputPixelBuffer(width: paddedW, height: paddedH)
        }
        return try interpolator.padPixelBufferForStream(frame, paddedW: paddedW, paddedH: paddedH,
                                                        reusing: paddedInputBuffer)
    }

    /// Encodes `paddedFrame` into the curr slot's buffers and rotates `prevSlot`
    /// so the next call sees this slot as prev. Used for the first push and for
    /// empty-timesteps subsequent pushes (cache rebase without inference).
    private func encodeAndRotate(_ paddedFrame: CVPixelBuffer) throws {
        let writeSlot = 1 - prevSlot
        try graph.encodeFrame(
            frame: paddedFrame,
            featBufOut:    featBuf[writeSlot],
            imgIntBufOut:  imgIntBuf[writeSlot],
            imgFullBufOut: imgFullBuf[writeSlot])
        prevSlot = writeSlot
    }

    /// Runs the streaming inference graph using `prevSlot` for prev bindings and
    /// `1 - prevSlot` for curr (encoder writes), then rotates `prevSlot` so the
    /// freshly-encoded slot becomes prev for the next call.
    private func runStreamAndRotate(curr: CVPixelBuffer,
                                    timesteps: [Float],
                                    outputs: [CVPixelBuffer]) throws {
        let writeSlot = 1 - prevSlot
        try graph.runStream(
            curr: curr,
            prevFeatBuf:    featBuf[prevSlot],    currFeatBuf:    featBuf[writeSlot],
            prevImgIntBuf:  imgIntBuf[prevSlot],  currImgIntBuf:  imgIntBuf[writeSlot],
            prevImgFullBuf: imgFullBuf[prevSlot], currImgFullBuf: imgFullBuf[writeSlot],
            timesteps: timesteps,
            outputs:   outputs)
        prevSlot = writeSlot
    }

    private func ensureCacheAllocated() throws {
        if !featBuf.isEmpty { return }
        let device = graph.context.device

        let intRGBBytes  = paddedIntW * paddedIntH * 3 * 2     // fp16 packed
        let intFeatBytes = paddedIntW * paddedIntH * 4 * 2
        let fullRGBBytes = paddedW    * paddedH    * 3 * 2

        var f: [MTLBuffer] = []
        var i: [MTLBuffer] = []
        var fl: [MTLBuffer] = []
        let intScaleIs1 = (paddedW == paddedIntW && paddedH == paddedIntH)

        for _ in 0..<2 {
            guard let fb = device.makeBuffer(length: intFeatBytes,
                                             options: .storageModePrivate),
                  let fl_buf = device.makeBuffer(length: fullRGBBytes,
                                                  options: .storageModePrivate)
            else {
                throw RifeError.inferenceFailed("RifeStream: MTLBuffer alloc failed")
            }
            f.append(fb)
            fl.append(fl_buf)
            if intScaleIs1 {
                // imgInt aliases imgFull when internalScale == 1.
                i.append(fl_buf)
            } else {
                guard let ib = device.makeBuffer(length: intRGBBytes,
                                                 options: .storageModePrivate)
                else {
                    throw RifeError.inferenceFailed("RifeStream: imgInt alloc failed")
                }
                i.append(ib)
            }
        }
        self.featBuf    = f
        self.imgIntBuf  = i
        self.imgFullBuf = fl
    }
}
