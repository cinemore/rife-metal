import XCTest
import CoreGraphics
import ImageIO
@testable import RifeMetal

final class EndToEndTests: XCTestCase {

    func testRifePsnrOver25() throws {
        try runPsnrTest(tier: .hq, psnrThreshold: 25.0)
    }

    /// Balanced tier (UHD mode) runs IFNet at half-res internally. PSNR is expected ~1 dB worse
    /// than hq at small frames; threshold of 25 dB is the same architectural bar (severe drift
    /// indicates a real bug). We do NOT enforce that balanced is faster than hq on the test
    /// fixture — at 640x360 the inference time is dominated by command-buffer overhead, not the
    /// IFNet stages, so a meaningful speedup only shows up at HD/4K. Wall-clock is sanity-
    /// checked elsewhere via the CLI bench harness.
    func testRifeBalancedPsnrOver25() throws {
        try runPsnrTest(tier: .balanced, psnrThreshold: 25.0)
    }

    /// Fast tier (1/4 internal scale) trades quality for ~3x speedup vs balanced at 4K.
    /// Fast tier (1/4 internal scale) trades quality for ~3x speedup vs balanced at 4K.
    /// On the 640×360 test fixture, v4.26's scale=16 coarsest stage operates on only 10×8
    /// pixels at internal resolution, producing ~20 dB — substantially less than v4.6's 4-stage
    /// network on the same fixture. This quality loss is geometry-driven (v4.26 adds a scale=16
    /// stage that degrades at this tiny size; at HD/4K the extra stage improves quality). The
    /// threshold of 19.0 dB is the architectural correctness bar for this tier/fixture combo;
    /// below that would indicate a numerical bug (e.g., all-zero output).
    func testRifeFastPsnrOver19() throws {
        try runPsnrTest(tier: .fast, psnrThreshold: 19.0)
    }

    /// Asserts all tiers run on the same input without crashing and produce
    /// outputs of the original (unpadded) dimensions.
    func testTierProducesCorrectOutputDimensions() throws {
        let fixtures = fixturesURL()
        let frameA = fixtures.appendingPathComponent("frame_a.png")
        let frameB = fixtures.appendingPathComponent("frame_b.png")
        let modelURL = RifeConfiguration.bundledModelURL

        for url in [frameA, frameB] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path),
                              "fixture missing: \(url.lastPathComponent)")
        }

        let imgA = try loadCG(frameA)
        let imgB = try loadCG(frameB)

        for tier in RifeQualityTier.allCases {
            let interpolator = try RifeInterpolator(
                configuration: .init(modelURL: modelURL, qualityTier: tier))
            let result = try interpolator.interpolate(previous: imgA, current: imgB)
            XCTAssertEqual(result.width, imgA.width, "tier \(tier): output width mismatch")
            XCTAssertEqual(result.height, imgA.height, "tier \(tier): output height mismatch")
        }
    }

    private func runPsnrTest(tier: RifeQualityTier, psnrThreshold: Double) throws {
        let fixtures = fixturesURL()
        let frameA = fixtures.appendingPathComponent("frame_a.png")
        let frameB = fixtures.appendingPathComponent("frame_b.png")
        let modelURL = RifeConfiguration.bundledModelURL
        let referenceURL = fixtures.appendingPathComponent("reference_mid.png")

        for url in [frameA, frameB, referenceURL] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path),
                              "fixture missing: \(url.lastPathComponent)")
        }

        let imgA = try loadCG(frameA)
        let imgB = try loadCG(frameB)

        let interpolator = try RifeInterpolator(
            configuration: .init(modelURL: modelURL, qualityTier: tier))
        let result = try interpolator.interpolate(previous: imgA, current: imgB)

        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rife-test-\(tier.rawValue)-\(UUID().uuidString).png")
        try writeCG(result, to: outURL)
        defer { try? FileManager.default.removeItem(at: outURL) }

        let psnr = try PSNRHelper.compare(outURL, referenceURL)
        XCTAssertGreaterThan(psnr, psnrThreshold,
                             "tier \(tier): PSNR \(psnr) below acceptance threshold \(psnrThreshold) dB")
    }

    private func fixturesURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()      // Tests/RifeMetalTests
            .deletingLastPathComponent()      // Tests
            .deletingLastPathComponent()      // <repo root>
            .appendingPathComponent("tests/fixtures")
    }

    private func loadCG(_ url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw NSError(domain: "EndToEndTests", code: 1)
        }
        return img
    }

    private func writeCG(_ image: CGImage, to url: URL) throws {
        let type = "public.png" as CFString
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, type, 1, nil) else {
            throw NSError(domain: "EndToEndTests", code: 2)
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw NSError(domain: "EndToEndTests", code: 3)
        }
    }
}
