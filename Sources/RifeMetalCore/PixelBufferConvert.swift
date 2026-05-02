import Foundation
import CoreVideo
import Metal

/// Helpers for binding CVPixelBuffer / MTLTexture to MPSGraphTensorData.
public enum PixelBufferConvert {

    /// Creates an MTLTexture view of a CVPixelBuffer (BGRA8). Caller retains the buffer.
    public static func makeTexture(from pixelBuffer: CVPixelBuffer,
                                    textureCache: CVMetalTextureCache) throws -> MTLTexture {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var cvTex: CVMetalTexture?
        let err = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            .bgra8Unorm,
            width, height, 0,
            &cvTex
        )
        guard err == kCVReturnSuccess, let cvTex,
              let tex = CVMetalTextureGetTexture(cvTex) else {
            throw NSError(domain: "PixelBufferConvert", code: Int(err),
                          userInfo: [NSLocalizedDescriptionKey: "CVMetalTextureCacheCreateTextureFromImage failed"])
        }
        return tex
    }

    /// Allocates an output CVPixelBuffer (BGRA8) of the given size, Metal-compatible.
    public static func makeOutputPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        var buffer: CVPixelBuffer?
        let err = CVPixelBufferCreate(kCFAllocatorDefault,
                                      width, height,
                                      kCVPixelFormatType_32BGRA,
                                      attrs as CFDictionary,
                                      &buffer)
        guard err == kCVReturnSuccess, let buffer else {
            throw NSError(domain: "PixelBufferConvert", code: Int(err),
                          userInfo: [NSLocalizedDescriptionKey: "CVPixelBufferCreate failed"])
        }
        return buffer
    }
}
