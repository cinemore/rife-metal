import XCTest
import CoreVideo
import CoreGraphics
import ImageIO
@testable import RifeMetal
@testable import RifeMetalCore

final class MultiTimestepTests: XCTestCase {

    // MARK: - Single-element array equivalence (CVPixelBuffer)

    func testSingleElementArrayEquivalentTo2Arg() throws {
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

        let interp = try RifeInterpolator(
            configuration: .init(modelURL: modelURL, qualityTier: .hq))

        let viaScalar = try interp.interpolate(previous: prevPB, current: currPB)
        let viaArray  = try interp.interpolate(previous: prevPB, current: currPB,
                                                timesteps: [0.5])

        XCTAssertEqual(viaArray.count, 1)
        try XCTAssertPixelsEqual(viaScalar, viaArray[0],
                                 "single-element array entry must match 2-arg path at t=0.5")
    }

    // MARK: - CGImage variant

    func testCGImageArrayVariantReturnsExpectedCount() throws {
        let fixtures = fixturesURL()
        let frameA = fixtures.appendingPathComponent("frame_a.png")
        let frameB = fixtures.appendingPathComponent("frame_b.png")
        let modelURL = RifeConfiguration.bundledModelURL
        for url in [frameA, frameB] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path),
                              "fixture missing: \(url.lastPathComponent)")
        }

        guard let srcA = CGImageSourceCreateWithURL(frameA as CFURL, nil),
              let imgA = CGImageSourceCreateImageAtIndex(srcA, 0, nil),
              let srcB = CGImageSourceCreateWithURL(frameB as CFURL, nil),
              let imgB = CGImageSourceCreateImageAtIndex(srcB, 0, nil) else {
            XCTFail("could not load CGImage fixtures")
            return
        }

        let interp = try RifeInterpolator(
            configuration: .init(modelURL: modelURL, qualityTier: .hq))

        let outs = try interp.interpolate(previous: imgA, current: imgB,
                                           timesteps: [0.33, 0.67])
        XCTAssertEqual(outs.count, 2)
        for img in outs {
            XCTAssertEqual(img.width, imgA.width)
            XCTAssertEqual(img.height, imgA.height)
        }

        // Non-degenerate: at least one output must contain non-zero pixel data.
        guard let data = outs[0].dataProvider?.data,
              let bytes = CFDataGetBytePtr(data) else {
            return XCTFail("could not read bytes from outs[0]")
        }
        let len = CFDataGetLength(data)
        let allZero = (0..<len).allSatisfy { bytes[$0] == 0 }
        XCTAssertFalse(allZero, "interpolated CGImage at t=0.33 must not be all-zeros")
    }

    // MARK: - Input validation

    func testTimestepsValidation() throws {
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
        let interp = try RifeInterpolator(
            configuration: .init(modelURL: modelURL, qualityTier: .hq))

        let invalidCases: [[Float]] = [
            [],            // empty
            [0.0],         // boundary low
            [1.0],         // boundary high
            [-0.1],        // below range
            [1.5],         // above range
            [0.5, 0.0, 0.7],  // mixed valid + invalid
        ]

        for ts in invalidCases {
            XCTAssertThrowsError(
                try interp.interpolate(previous: prevPB, current: currPB, timesteps: ts),
                "expected throw for timesteps=\(ts)"
            ) { error in
                guard case RifeError.invalidTimesteps = error else {
                    XCTFail("expected RifeError.invalidTimesteps for \(ts), got \(error)")
                    return
                }
            }
        }
    }

    // MARK: - Multi-element self-consistency and order preservation

    func testMultiElementSelfConsistencyAndOrderAcrossTiers() throws {
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

            // Caller-supplied order: not sorted. The implementation must preserve it.
            let unsortedTs: [Float] = [0.7, 0.3]
            let multi = try interp.interpolate(previous: prevPB, current: currPB,
                                                timesteps: unsortedTs)
            XCTAssertEqual(multi.count, unsortedTs.count, "tier \(tier): output count")

            // Each multi[i] must equal a separate single-element call at the same t.
            for (i, t) in unsortedTs.enumerated() {
                let single = try interp.interpolate(previous: prevPB, current: currPB,
                                                     timesteps: [t])
                XCTAssertEqual(single.count, 1)
                try XCTAssertPixelsEqual(
                    multi[i], single[0],
                    "tier \(tier): multi[\(i)] (t=\(t)) diverges from single-call at same t"
                )
            }
        }
    }

    // MARK: - v4.26 fixture shape sanity

    func testV4_26FixtureHasExpectedShape() throws {
        let store = try WeightStore(url: RifeConfiguration.bundledModelURL)
        XCTAssertEqual(store.header.ifblockChannels, [192, 128, 96, 64, 32])
        XCTAssertEqual(store.header.scaleList, [16, 8, 4, 2, 1])
        XCTAssertEqual(store.header.model, "rife-v4.26")
    }

    // MARK: - Helpers (private — same pattern as CallerSuppliedOutputTests)

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
        let cvErr = CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                                        kCVPixelFormatType_32BGRA,
                                        attrs as CFDictionary, &pb)
        guard cvErr == kCVReturnSuccess, let pb else { throw NSError(domain: "test", code: 2) }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let bitmapInfo: UInt32 = CGImageAlphaInfo.premultipliedFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb),
                                  width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                                  space: cs, bitmapInfo: bitmapInfo) else {
            throw NSError(domain: "test", code: 3)
        }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return pb
    }

    private func XCTAssertPixelsEqual(_ a: CVPixelBuffer, _ b: CVPixelBuffer,
                                       _ message: String,
                                       file: StaticString = #filePath,
                                       line: UInt = #line) throws {
        let aw = CVPixelBufferGetWidth(a), ah = CVPixelBufferGetHeight(a)
        let bw = CVPixelBufferGetWidth(b), bh = CVPixelBufferGetHeight(b)
        XCTAssertEqual(aw, bw, "width mismatch: \(message)", file: file, line: line)
        XCTAssertEqual(ah, bh, "height mismatch: \(message)", file: file, line: line)
        CVPixelBufferLockBaseAddress(a, .readOnly)
        CVPixelBufferLockBaseAddress(b, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(a, .readOnly)
            CVPixelBufferUnlockBaseAddress(b, .readOnly)
        }
        let aStride = CVPixelBufferGetBytesPerRow(a)
        let bStride = CVPixelBufferGetBytesPerRow(b)
        guard let aBase = CVPixelBufferGetBaseAddress(a),
              let bBase = CVPixelBufferGetBaseAddress(b) else {
            throw NSError(domain: "test", code: 4)
        }
        let rowBytes = aw * 4
        for y in 0..<ah {
            let aRow = aBase.advanced(by: y * aStride)
            let bRow = bBase.advanced(by: y * bStride)
            if memcmp(aRow, bRow, rowBytes) != 0 {
                XCTFail("pixel mismatch at row \(y): \(message)", file: file, line: line)
                return
            }
        }
    }
}
