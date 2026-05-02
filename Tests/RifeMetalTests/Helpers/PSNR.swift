import Foundation
import CoreGraphics
import CoreVideo
import ImageIO

enum PSNRHelper {

    /// Loads two images of identical size from URLs, returns PSNR in dB.
    static func compare(_ a: URL, _ b: URL) throws -> Double {
        let imgA = try loadImage(at: a)
        let imgB = try loadImage(at: b)

        let bytesA = try rgbBytes(of: imgA)
        let bytesB = try rgbBytes(of: imgB)
        guard bytesA.count == bytesB.count else { return 0.0 }

        var mse = 0.0
        for i in 0..<bytesA.count {
            let d = Double(bytesA[i]) - Double(bytesB[i])
            mse += d * d
        }
        mse /= Double(bytesA.count)
        if mse == 0 { return .infinity }
        return 20.0 * log10(255.0 / sqrt(mse))
    }

    /// Compares two CVPixelBuffers (BGRA or RGBA) directly without going through PNG.
    /// Returns PSNR in dB (∞ if identical).
    static func compareInMemory(_ a: CVPixelBuffer, _ b: CVPixelBuffer) -> Double {
        let aw = CVPixelBufferGetWidth(a),  ah = CVPixelBufferGetHeight(a)
        let bw = CVPixelBufferGetWidth(b),  bh = CVPixelBufferGetHeight(b)
        guard aw == bw, ah == bh else { return 0 }

        CVPixelBufferLockBaseAddress(a, .readOnly)
        CVPixelBufferLockBaseAddress(b, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(a, .readOnly)
            CVPixelBufferUnlockBaseAddress(b, .readOnly)
        }

        let aStride = CVPixelBufferGetBytesPerRow(a)
        let bStride = CVPixelBufferGetBytesPerRow(b)
        guard let aBase = CVPixelBufferGetBaseAddress(a),
              let bBase = CVPixelBufferGetBaseAddress(b) else { return 0 }

        var mse = 0.0
        var samples = 0
        for y in 0..<ah {
            let aRow = aBase.advanced(by: y * aStride).assumingMemoryBound(to: UInt8.self)
            let bRow = bBase.advanced(by: y * bStride).assumingMemoryBound(to: UInt8.self)
            for x in 0..<aw {
                // Compare RGB channels only; skip alpha. BGRA layout: B,G,R,A.
                for c in 0..<3 {
                    let ai = Double(aRow[x*4 + c])
                    let bi = Double(bRow[x*4 + c])
                    let d = ai - bi
                    mse += d * d
                    samples += 1
                }
            }
        }
        if samples == 0 { return 0 }
        mse /= Double(samples)
        if mse == 0 { return .infinity }
        return 20.0 * log10(255.0 / sqrt(mse))
    }

    private static func loadImage(at url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw NSError(domain: "PSNRHelper", code: 1)
        }
        return img
    }

    private static func rgbBytes(of image: CGImage) throws -> [UInt8] {
        let width = image.width, height = image.height
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let info: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(data: &bytes,
                                  width: width, height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: width * 4,
                                  space: cs,
                                  bitmapInfo: info) else {
            throw NSError(domain: "PSNRHelper", code: 2)
        }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        var rgb = [UInt8](); rgb.reserveCapacity(width * height * 3)
        for i in stride(from: 0, to: bytes.count, by: 4) {
            rgb.append(bytes[i])
            rgb.append(bytes[i + 1])
            rgb.append(bytes[i + 2])
        }
        return rgb
    }
}
