import XCTest
import Metal
@testable import RifeMetalCore

final class BackwardWarpKernelTests: XCTestCase {

    var device: MTLDevice!
    var queue: MTLCommandQueue!
    var kernel: BackwardWarpKernel!

    override func setUpWithError() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available")
        }
        guard let queue = device.makeCommandQueue() else {
            throw XCTSkip("Cannot make command queue")
        }
        self.device = device
        self.queue = queue
        self.kernel = try BackwardWarpKernel(device: device)
    }

    func testIdentityFlowReproducesSource() throws {
        let W = 4, H = 4
        // Source: 16 unique values, gradient.
        var source = [Float](repeating: 0, count: W * H * 4)
        for y in 0..<H {
            for x in 0..<W {
                let i = (y * W + x) * 4
                let v = Float(y * W + x) / 15.0
                source[i + 0] = v
                source[i + 1] = v
                source[i + 2] = v
                source[i + 3] = 1.0
            }
        }
        // Flow: zero everywhere → output should equal source.
        let flow = [Float](repeating: 0, count: W * H * 2)

        let result = try runKernel(source: source, flow: flow, width: W, height: H)
        for i in 0..<source.count {
            XCTAssertEqual(result[i], source[i], accuracy: 0.01,
                           "mismatch at index \(i)")
        }
    }

    func testRightShiftFlow() throws {
        let W = 4, H = 4
        var source = [Float](repeating: 0, count: W * H * 4)
        for y in 0..<H {
            for x in 0..<W {
                let i = (y * W + x) * 4
                source[i + 0] = Float(x)  // R = x
                source[i + 3] = 1.0
            }
        }
        // Flow (1, 0) → output[x, y] should be source[clamp(x+1), y].
        var flow = [Float](repeating: 0, count: W * H * 2)
        for y in 0..<H {
            for x in 0..<W {
                flow[(y * W + x) * 2 + 0] = 1.0
                flow[(y * W + x) * 2 + 1] = 0.0
            }
        }

        let result = try runKernel(source: source, flow: flow, width: W, height: H)
        for y in 0..<H {
            for x in 0..<W {
                let expectedX = Float(min(x + 1, W - 1))
                let i = (y * W + x) * 4
                XCTAssertEqual(result[i], expectedX, accuracy: 0.05,
                               "x=\(x) y=\(y)")
            }
        }
    }

    func testHalfPixelFlowProducesBilinearAverage() throws {
        let W = 4, H = 1
        // Source row: R channel = 0, 1, 2, 3.
        var source = [Float](repeating: 0, count: W * 4)
        for x in 0..<W {
            source[x * 4] = Float(x)
            source[x * 4 + 3] = 1.0
        }
        // Flow (0.5, 0) → output[x] should be 0.5*(source[x] + source[x+1]).
        var flow = [Float](repeating: 0, count: W * 2)
        for x in 0..<W {
            flow[x * 2] = 0.5
        }

        let result = try runKernel(source: source, flow: flow, width: W, height: H)
        for x in 0..<W {
            let nextX = min(x + 1, W - 1)
            let expected = 0.5 * Float(x) + 0.5 * Float(nextX)
            XCTAssertEqual(result[x * 4], expected, accuracy: 0.05, "x=\(x)")
        }
    }

    func testNegativeFlowClampsToLeftEdge() throws {
        let W = 4, H = 1
        var source = [Float](repeating: 0, count: W * 4)
        for x in 0..<W {
            source[x * 4] = Float(x)
            source[x * 4 + 3] = 1.0
        }
        // Flow (-1, 0) → output[x] should be source[max(x-1, 0)] thanks to clamp_to_edge.
        var flow = [Float](repeating: 0, count: W * 2)
        for x in 0..<W {
            flow[x * 2] = -1.0
        }

        let result = try runKernel(source: source, flow: flow, width: W, height: H)
        for x in 0..<W {
            let prevX = max(x - 1, 0)
            let expected = Float(prevX)
            XCTAssertEqual(result[x * 4], expected, accuracy: 0.05, "x=\(x)")
        }
    }

    // MARK: - Kernel runner

    private func runKernel(source: [Float], flow: [Float], width: Int, height: Int) throws -> [Float] {
        let srcTex = makeRGBATexture(values: source, width: width, height: height)
        let flowTex = makeRGTexture(values: flow, width: width, height: height)

        let outDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: width, height: height,
            mipmapped: false
        )
        outDesc.usage = [.shaderRead, .shaderWrite]
        outDesc.storageMode = .shared
        let outTex = device.makeTexture(descriptor: outDesc)!

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        kernel.encode(into: enc, source: srcTex, flow: flowTex, output: outTex)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        var bytes = [UInt16](repeating: 0, count: width * height * 4)
        outTex.getBytes(
            &bytes,
            bytesPerRow: width * 4 * 2,
            from: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0
        )
        return bytes.map { Float(Float16(bitPattern: $0)) }
    }

    private func makeRGBATexture(values: [Float], width: Int, height: Int) -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: width, height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        let tex = device.makeTexture(descriptor: desc)!
        let half = values.map { Float16($0).bitPattern }
        half.withUnsafeBufferPointer { buf in
            tex.replace(region: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0,
                        withBytes: buf.baseAddress!,
                        bytesPerRow: width * 4 * 2)
        }
        return tex
    }

    private func makeRGTexture(values: [Float], width: Int, height: Int) -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float,
            width: width, height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        let tex = device.makeTexture(descriptor: desc)!
        let half = values.map { Float16($0).bitPattern }
        half.withUnsafeBufferPointer { buf in
            tex.replace(region: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0,
                        withBytes: buf.baseAddress!,
                        bytesPerRow: width * 2 * 2)
        }
        return tex
    }
}
