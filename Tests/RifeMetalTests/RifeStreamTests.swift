import XCTest
import CoreVideo
@testable import RifeMetal
@testable import RifeMetalCore

final class RifeStreamTests: XCTestCase {

    private func makeInterpolator(tier: RifeQualityTier = .hq) throws -> RifeInterpolator {
        let cfg = RifeConfiguration.bundled(qualityTier: tier)
        return try RifeInterpolator(configuration: cfg)
    }

    func testMakeStream_returnsStreamWithLockedDims() throws {
        let interp = try makeInterpolator()
        let stream = try interp.makeStream(width: 640, height: 480)
        XCTAssertEqual(stream.width, 640)
        XCTAssertEqual(stream.height, 480)
    }

    func testFirstPush_returnsEmptyArray() throws {
        let interp = try makeInterpolator()
        let fixturesDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let frameA = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_a.png"))
        let stream = try interp.makeStream(
            width: CVPixelBufferGetWidth(frameA),
            height: CVPixelBufferGetHeight(frameA))
        let result = try stream.push(frameA, timesteps: [0.5])
        XCTAssertEqual(result.count, 0,
                       "first push must return empty array; got \(result.count) frames")
    }

    func testFirstPushConvenience_returnsNil() throws {
        let interp = try makeInterpolator()
        let fixturesDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let frameA = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_a.png"))
        let stream = try interp.makeStream(
            width: CVPixelBufferGetWidth(frameA),
            height: CVPixelBufferGetHeight(frameA))
        let result = try stream.push(frameA)
        XCTAssertNil(result, "first push convenience must return nil")
    }

    func testStreamMatchesStateless_HQ() throws {
        try runStreamParityTest(tier: .hq, threshold: 50.0)
    }

    func testStreamMatchesStateless_balanced() throws {
        try runStreamParityTest(tier: .balanced, threshold: 50.0)
    }

    func testMultiFrameStream_eachMidframeMatchesStateless() throws {
        let interp = try makeInterpolator()
        let fixturesDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let frameA = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_a.png"))
        let frameB = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_b.png"))
        let frames = [frameA, frameB, frameA, frameB]
        let w = CVPixelBufferGetWidth(frameA)
        let h = CVPixelBufferGetHeight(frameA)

        let stream = try interp.makeStream(width: w, height: h)
        var streamMids: [CVPixelBuffer] = []
        for (i, f) in frames.enumerated() {
            let mids = try stream.push(f, timesteps: [0.5])
            if i == 0 {
                XCTAssertEqual(mids.count, 0, "first push must be empty")
            } else {
                XCTAssertEqual(mids.count, 1, "subsequent pushes must emit one frame at t=0.5")
                streamMids.append(mids[0])
            }
        }

        XCTAssertEqual(streamMids.count, 3, "expect 3 midframes from 4 pushes")

        for i in 0..<3 {
            let prev = frames[i]
            let curr = frames[i + 1]
            let ref  = try interp.interpolate(previous: prev, current: curr)
            let psnr = PSNRHelper.compareInMemory(streamMids[i], ref)
            XCTAssertGreaterThan(psnr, 50.0,
                "frame pair (\(i), \(i+1)): stream vs stateless PSNR \(psnr) below 50")
        }
    }

    func testReset_isIndistinguishableFromFreshStream() throws {
        let interp = try makeInterpolator()
        let fixturesDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let frameA = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_a.png"))
        let frameB = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_b.png"))
        let w = CVPixelBufferGetWidth(frameA)
        let h = CVPixelBufferGetHeight(frameA)

        // Stream A: warm with one pair, then reset, then push A→B.
        let streamA = try interp.makeStream(width: w, height: h)
        _ = try streamA.push(frameA, timesteps: [])
        _ = try streamA.push(frameB, timesteps: [0.5])
        streamA.reset()
        let afterResetEmpty = try streamA.push(frameA, timesteps: [0.5])
        XCTAssertEqual(afterResetEmpty.count, 0,
                       "first push after reset must return empty")
        let midA = try streamA.push(frameB, timesteps: [0.5])
        XCTAssertEqual(midA.count, 1)

        // Stream B: fresh, push A→B.
        let streamB = try interp.makeStream(width: w, height: h)
        _ = try streamB.push(frameA, timesteps: [])
        let midB = try streamB.push(frameB, timesteps: [0.5])
        XCTAssertEqual(midB.count, 1)

        let psnr = PSNRHelper.compareInMemory(midA[0], midB[0])
        XCTAssertGreaterThan(psnr, 60.0,
            "reset stream vs fresh stream PSNR \(psnr) below 60 — reset is leaking state")
    }

    func testDirectionalAsymmetry_streamUsesPrevNotCurr() throws {
        let interp = try makeInterpolator()
        let w = 256, h = 256
        // Frame A: one bar at x=64. Frame B: two bars at x=192 and x=232.
        // 1-bar → 2-bar forward differs from 2-bar → 1-bar backward, so
        // interpolate(A,B) and interpolate(B,A) produce structurally
        // different midframes.
        let frameA = try SyntheticFrames.barFrame(width: w, height: h, barXs: [64])
        let frameB = try SyntheticFrames.barFrame(width: w, height: h, barXs: [192, 232])

        let midForward  = try interp.interpolate(previous: frameA, current: frameB)
        let midBackward = try interp.interpolate(previous: frameB, current: frameA)

        let asymmetryPsnr = PSNRHelper.compareInMemory(midForward, midBackward)
        XCTAssertLessThan(asymmetryPsnr, 35.0,
            "fixture must be directionally asymmetric (PSNR(fwd, bwd) = \(asymmetryPsnr))")

        let stream = try interp.makeStream(width: w, height: h)
        _ = try stream.push(frameA, timesteps: [])
        let mids = try stream.push(frameB, timesteps: [0.5])
        XCTAssertEqual(mids.count, 1)

        let psnrForward  = PSNRHelper.compareInMemory(mids[0], midForward)
        let psnrBackward = PSNRHelper.compareInMemory(mids[0], midBackward)
        XCTAssertGreaterThan(psnrForward, 50.0,
            "stream output should match forward direction; PSNR = \(psnrForward)")
        XCTAssertLessThan(psnrBackward, 40.0,
            "stream output must NOT match backward direction; PSNR = \(psnrBackward) — likely prev/curr swap bug")
    }

    func testMultiStream_independentStateThroughInterpolatorQueue() throws {
        let interp = try makeInterpolator()
        let fixturesDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let frameA = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_a.png"))
        let frameB = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_b.png"))
        let w = CVPixelBufferGetWidth(frameA)
        let h = CVPixelBufferGetHeight(frameA)

        let streamX = try interp.makeStream(width: w, height: h)
        let synthA = try SyntheticFrames.barFrame(width: w, height: h, barXs: [w/4])
        let synthB = try SyntheticFrames.barFrame(width: w, height: h, barXs: [3*w/4])
        let streamY = try interp.makeStream(width: w, height: h)

        _ = try streamX.push(frameA, timesteps: [])
        _ = try streamY.push(synthA,  timesteps: [])
        let midX = try streamX.push(frameB, timesteps: [0.5])
        let midY = try streamY.push(synthB,  timesteps: [0.5])
        XCTAssertEqual(midX.count, 1)
        XCTAssertEqual(midY.count, 1)

        let refX = try interp.interpolate(previous: frameA, current: frameB)
        let psnrX = PSNRHelper.compareInMemory(midX[0], refX)
        XCTAssertGreaterThan(psnrX, 50.0, "streamX output corrupted; PSNR \(psnrX)")

        let refY = try interp.interpolate(previous: synthA, current: synthB)
        let psnrY = PSNRHelper.compareInMemory(midY[0], refY)
        XCTAssertGreaterThan(psnrY, 50.0, "streamY output corrupted; PSNR \(psnrY)")
    }

    func testMultiTimestep_perPushMatchesStateless() throws {
        let interp = try makeInterpolator()
        let fixturesDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let frameA = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_a.png"))
        let frameB = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_b.png"))
        let w = CVPixelBufferGetWidth(frameA)
        let h = CVPixelBufferGetHeight(frameA)

        let timesteps: [Float] = [0.33, 0.67]

        let refMids = try interp.interpolate(previous: frameA, current: frameB,
                                              timesteps: timesteps)
        XCTAssertEqual(refMids.count, 2)

        let stream = try interp.makeStream(width: w, height: h)
        _ = try stream.push(frameA, timesteps: [])
        let mids = try stream.push(frameB, timesteps: timesteps)
        XCTAssertEqual(mids.count, 2, "expected 2 midframes for 2 timesteps")

        for i in 0..<2 {
            let psnr = PSNRHelper.compareInMemory(mids[i], refMids[i])
            XCTAssertGreaterThan(psnr, 50.0,
                "timestep \(timesteps[i]): stream vs stateless PSNR \(psnr)")
        }
    }

    func testStreamCallerOutput_matchesStateless() throws {
        let interp = try makeInterpolator()
        let fixturesDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let frameA = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_a.png"))
        let frameB = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_b.png"))
        let w = CVPixelBufferGetWidth(frameA)
        let h = CVPixelBufferGetHeight(frameA)

        // Stateless reference: caller-supplied output.
        let refOut = try makeBGRABuffer(width: w, height: h)
        try interp.interpolate(previous: frameA, current: frameB, output: refOut)

        // Stream output: caller-supplied output via push(_:into:).
        let streamOut = try makeBGRABuffer(width: w, height: h)
        let stream = try interp.makeStream(width: w, height: h)
        let firstResult = try stream.push(frameA, into: streamOut)
        XCTAssertFalse(firstResult, "first push must return false (no midframe yet)")
        let secondResult = try stream.push(frameB, into: streamOut)
        XCTAssertTrue(secondResult, "subsequent push must return true (midframe written)")

        let psnr = PSNRHelper.compareInMemory(streamOut, refOut)
        XCTAssertGreaterThan(psnr, 50.0,
            "stream caller-output vs stateless caller-output PSNR \(psnr) below 50")
    }

    func testStreamCallerOutput_firstPushLeavesOutputUntouched() throws {
        let interp = try makeInterpolator()
        let fixturesDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let frameA = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_a.png"))
        let w = CVPixelBufferGetWidth(frameA)
        let h = CVPixelBufferGetHeight(frameA)

        // Pre-fill output with a sentinel byte (0xAB) and verify first push
        // leaves it untouched.
        let sentinel: UInt8 = 0xAB
        let output = try makeBGRABuffer(width: w, height: h, fillByte: sentinel)
        let stream = try interp.makeStream(width: w, height: h)
        let result = try stream.push(frameA, into: output)
        XCTAssertFalse(result, "first push must return false")

        // Verify every byte is still the sentinel.
        CVPixelBufferLockBaseAddress(output, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(output, .readOnly) }
        let stride = CVPixelBufferGetBytesPerRow(output)
        guard let base = CVPixelBufferGetBaseAddress(output) else {
            XCTFail("output base address nil")
            return
        }
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<h {
            for x in 0..<(w*4) {
                XCTAssertEqual(ptr[y*stride + x], sentinel,
                    "byte at (\(x), \(y)) was modified — first push should leave output untouched")
                if ptr[y*stride + x] != sentinel { return }  // bail on first failure
            }
        }
    }

    private func runStreamParityTest(tier: RifeQualityTier,
                                      threshold: Double) throws {
        let interp = try makeInterpolator(tier: tier)
        let fixturesDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let frameA = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_a.png"))
        let frameB = try loadFixturePixelBuffer(fixturesDir.appendingPathComponent("frame_b.png"))
        let w = CVPixelBufferGetWidth(frameA)
        let h = CVPixelBufferGetHeight(frameA)

        // Stateless reference.
        let midStateless = try interp.interpolate(previous: frameA, current: frameB)

        // Stream output.
        let stream = try interp.makeStream(width: w, height: h)
        _ = try stream.push(frameA, timesteps: [])
        let mids = try stream.push(frameB, timesteps: [0.5])
        XCTAssertEqual(mids.count, 1, "stream must emit one frame on second push with one timestep")

        let psnr = PSNRHelper.compareInMemory(mids[0], midStateless)
        XCTAssertGreaterThan(psnr, threshold,
                             "tier \(tier): stream vs stateless PSNR \(psnr) below \(threshold)")
    }
}
