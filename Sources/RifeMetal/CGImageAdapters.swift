import Foundation
import CoreGraphics
import CoreVideo
import ImageIO

public extension RifeInterpolator {

    func interpolate(previous: CGImage, current: CGImage) throws -> CGImage {
        let prevPB = try cgImageToPixelBuffer(previous)
        let currPB = try cgImageToPixelBuffer(current)
        let outPB = try interpolate(previous: prevPB, current: currPB)
        return try pixelBufferToCGImage(outPB)
    }

    func interpolate(previous: CGImage, current: CGImage,
                     timesteps: [Float]) throws -> [CGImage] {
        let prevPB = try cgImageToPixelBuffer(previous)
        let currPB = try cgImageToPixelBuffer(current)
        let outs = try interpolate(previous: prevPB, current: currPB, timesteps: timesteps)
        return try outs.map { try pixelBufferToCGImage($0) }
    }
}

private func cgImageToPixelBuffer(_ image: CGImage) throws -> CVPixelBuffer {
    let width = image.width
    let height = image.height
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
        throw RifeError.inferenceFailed("CVPixelBufferCreate failed: \(err)")
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else {
        throw RifeError.inferenceFailed("CVPixelBufferGetBaseAddress failed")
    }
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let bitmapInfo: UInt32 = CGImageAlphaInfo.premultipliedFirst.rawValue
                            | CGBitmapInfo.byteOrder32Little.rawValue
    guard let ctx = CGContext(data: base,
                              width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                              space: cs,
                              bitmapInfo: bitmapInfo) else {
        throw RifeError.inferenceFailed("CGContext init failed")
    }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return buffer
}

private func pixelBufferToCGImage(_ buffer: CVPixelBuffer) throws -> CGImage {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    guard let base = CVPixelBufferGetBaseAddress(buffer) else {
        throw RifeError.inferenceFailed("CVPixelBufferGetBaseAddress failed")
    }
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let bitmapInfo: UInt32 = CGImageAlphaInfo.premultipliedFirst.rawValue
                            | CGBitmapInfo.byteOrder32Little.rawValue
    guard let ctx = CGContext(data: base,
                              width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bytesPerRow: bytesPerRow,
                              space: cs,
                              bitmapInfo: bitmapInfo) else {
        throw RifeError.inferenceFailed("CGContext init failed")
    }
    guard let cg = ctx.makeImage() else {
        throw RifeError.inferenceFailed("CGContext.makeImage failed")
    }
    return cg
}
