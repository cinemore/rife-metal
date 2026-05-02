import XCTest
import CoreVideo
import CoreGraphics
import ImageIO
@testable import RifeMetal

final class CallerSuppliedOutputTests: XCTestCase {

    func testCallerSuppliedOutputMatchesAllocatingPath() throws {
        let fixtures = fixturesURL()
        let frameA = fixtures.appendingPathComponent("frame_a.png")
        let frameB = fixtures.appendingPathComponent("frame_b.png")
        let modelURL = RifeConfiguration.bundledModelURL
        for url in [frameA, frameB] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path),
                              "fixture missing: \(url.lastPathComponent)")
        }

        let prevPB = try makeBGRABuffer(from: frameA)
        let currPB = try makeBGRABuffer(from: frameB)

        for tier in RifeQualityTier.allCases {
            let interp = try RifeInterpolator(
                configuration: .init(modelURL: modelURL, qualityTier: tier))

            // Allocating path
            let allocated = try interp.interpolate(previous: prevPB, current: currPB)

            // Caller-supplied path
            let outPB = try makeBGRAOutput(width: CVPixelBufferGetWidth(prevPB),
                                            height: CVPixelBufferGetHeight(prevPB))
            try interp.interpolate(previous: prevPB, current: currPB, output: outPB)

            // Pixel-exact equality between the two paths (both run identical graph)
            try XCTAssertPixelsEqual(allocated, outPB,
                                     "tier \(tier): caller-supplied output diverges from allocating path")
        }
    }

    func testCallerSuppliedOutputMatchesAllocatingPathOnNoPadDimensions() throws {
        // Pick dims that are already multiples of 256, so all three tiers (hq=64,
        // balanced=128, fast=256) skip padding and exercise the direct-write branch
        // where the graph writes straight into the caller's output buffer.
        let width = 768
        let height = 512
        let modelURL = RifeConfiguration.bundledModelURL

        let prevPB = try makeSyntheticBGRA(width: width, height: height,
                                            seed: 0x1234)
        let currPB = try makeSyntheticBGRA(width: width, height: height,
                                            seed: 0x5678)

        for tier in RifeQualityTier.allCases {
            let interp = try RifeInterpolator(
                configuration: .init(modelURL: modelURL, qualityTier: tier))

            let allocated = try interp.interpolate(previous: prevPB, current: currPB)
            let outPB = try makeBGRAOutput(width: width, height: height)
            try interp.interpolate(previous: prevPB, current: currPB, output: outPB)

            try XCTAssertPixelsEqual(allocated, outPB,
                                     "tier \(tier): caller-supplied output diverges on no-pad path")
        }
    }

    private func fixturesURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("tests/fixtures")
    }

    private func makeBGRABuffer(from url: URL) throws -> CVPixelBuffer {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw NSError(domain: "test", code: 1)
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
        let err = CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                                       kCVPixelFormatType_32BGRA,
                                       attrs as CFDictionary, &pb)
        guard err == kCVReturnSuccess, let pb else {
            throw NSError(domain: "test", code: 2)
        }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb),
                            width: w, height: h, bitsPerComponent: 8,
                            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue)!
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return pb
    }

    private func makeSyntheticBGRA(width: Int, height: Int, seed: UInt32) throws -> CVPixelBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        var pb: CVPixelBuffer?
        let err = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                       kCVPixelFormatType_32BGRA,
                                       attrs as CFDictionary, &pb)
        guard err == kCVReturnSuccess, let pb else {
            throw NSError(domain: "test", code: 4)
        }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else {
            throw NSError(domain: "test", code: 5)
        }
        let stride = CVPixelBufferGetBytesPerRow(pb)
        // Smooth diagonal gradient + seed offset, gives motion-like differences
        // between two seeds without solid-color degeneracies.
        for y in 0..<height {
            let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
            for x in 0..<width {
                let i = x * 4
                let v = UInt8(((x + y) &+ Int(seed)) & 0xFF)
                row[i + 0] = v               // B
                row[i + 1] = UInt8((v &+ 0x40) & 0xFF)  // G
                row[i + 2] = UInt8((v &+ 0x80) & 0xFF)  // R
                row[i + 3] = 0xFF            // A
            }
        }
        return pb
    }

    private func makeBGRAOutput(width: Int, height: Int) throws -> CVPixelBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        var pb: CVPixelBuffer?
        let err = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                       kCVPixelFormatType_32BGRA,
                                       attrs as CFDictionary, &pb)
        guard err == kCVReturnSuccess, let pb else {
            throw NSError(domain: "test", code: 3)
        }
        return pb
    }

    private func XCTAssertPixelsEqual(_ a: CVPixelBuffer, _ b: CVPixelBuffer,
                                       _ msg: String,
                                       file: StaticString = #file,
                                       line: UInt = #line) throws {
        XCTAssertEqual(CVPixelBufferGetWidth(a), CVPixelBufferGetWidth(b), msg, file: file, line: line)
        XCTAssertEqual(CVPixelBufferGetHeight(a), CVPixelBufferGetHeight(b), msg, file: file, line: line)
        CVPixelBufferLockBaseAddress(a, .readOnly)
        CVPixelBufferLockBaseAddress(b, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(a, .readOnly)
            CVPixelBufferUnlockBaseAddress(b, .readOnly)
        }
        let h = CVPixelBufferGetHeight(a)
        let rowBytes = CVPixelBufferGetWidth(a) * 4
        let aRowStride = CVPixelBufferGetBytesPerRow(a)
        let bRowStride = CVPixelBufferGetBytesPerRow(b)
        guard let aBase = CVPixelBufferGetBaseAddress(a),
              let bBase = CVPixelBufferGetBaseAddress(b) else {
            XCTFail("\(msg): base address nil", file: file, line: line)
            return
        }
        for y in 0..<h {
            let aRow = aBase.advanced(by: y * aRowStride)
            let bRow = bBase.advanced(by: y * bRowStride)
            XCTAssertEqual(memcmp(aRow, bRow, rowBytes), 0,
                           "\(msg): row \(y) differs", file: file, line: line)
        }
    }
}
