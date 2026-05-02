import Foundation
import CoreVideo

enum SyntheticFrames {

    /// Generates a (W × H) BGRA frame containing vertical white bars at the
    /// columns listed in `barXs`. All other pixels are black (alpha 255).
    /// Used for synthetic motion fixtures where the number/position of bars
    /// can differ between frames to produce directionally-asymmetric scenes.
    static func barFrame(width: Int, height: Int, barXs: [Int],
                         barWidthPx: Int = 16) throws -> CVPixelBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary, &pb)
        guard let buffer = pb else {
            throw NSError(domain: "synthetic", code: 1)
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw NSError(domain: "synthetic", code: 2)
        }
        memset(base, 0, height * stride)
        for y in 0..<height {
            let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
            for x in 0..<width {
                row[x*4 + 3] = 255
                let inAnyBar = barXs.contains { barX in
                    x >= barX && x < min(width, barX + barWidthPx)
                }
                if inAnyBar {
                    row[x*4 + 0] = 255
                    row[x*4 + 1] = 255
                    row[x*4 + 2] = 255
                }
            }
        }
        return buffer
    }
}
