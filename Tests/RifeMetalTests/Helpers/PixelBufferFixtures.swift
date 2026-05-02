import Foundation
import CoreVideo
import CoreGraphics
import ImageIO

func loadFixturePixelBuffer(_ url: URL) throws -> CVPixelBuffer {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        throw NSError(domain: "fixtures", code: 1)
    }
    let w = img.width, h = img.height
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
        kCVPixelBufferWidthKey as String: w,
        kCVPixelBufferHeightKey as String: h,
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
    ]
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                        kCVPixelFormatType_32BGRA,
                        attrs as CFDictionary, &pb)
    guard let buffer = pb else {
        throw NSError(domain: "fixtures", code: 2)
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let info: UInt32 = CGImageAlphaInfo.premultipliedFirst.rawValue
                     | CGBitmapInfo.byteOrder32Little.rawValue
    guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                              width: w, height: h,
                              bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                              space: cs,
                              bitmapInfo: info) else {
        throw NSError(domain: "fixtures", code: 3)
    }
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return buffer
}

/// Allocates a fresh BGRA pixel buffer for tests that need a caller-supplied
/// output target. Optionally fills with a sentinel byte for first-push tests.
func makeBGRABuffer(width: Int, height: Int, fillByte: UInt8? = nil) throws -> CVPixelBuffer {
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
        throw NSError(domain: "fixtures", code: 10)
    }
    if let byte = fillByte {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            memset(base, Int32(byte), h * stride)
        }
    }
    return buffer
}
