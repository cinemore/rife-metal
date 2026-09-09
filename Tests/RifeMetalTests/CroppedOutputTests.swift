import Metal
import XCTest
@testable import RifeMetalCore

final class CroppedOutputTests: XCTestCase {
    func testCroppedBlendMatchesFullOutputBits() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let kernels = try ConversionKernels(device: device)
        let w = 96, h = 64
        func buffer(_ count: Int, _ seed: Int) throws -> MTLBuffer {
            let values = (0..<count).map { Float16(Float(($0 * seed) % 107) / 53 - 0.5).bitPattern }
            return try XCTUnwrap(values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        }
        let rgb0 = try buffer(w*h*3, 7), rgb1 = try buffer(w*h*3, 11)
        func texture(_ width: Int, _ height: Int) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            d.storageMode = .shared
            d.usage = [.shaderRead, .shaderWrite]
            return try XCTUnwrap(device.makeTexture(descriptor: d))
        }
        do {
            let mask = try buffer(17*11, 13)
            for (cw, ch) in [(1,1), (65,33), (96,64)] {
                let full = try texture(w,h), cropped = try texture(cw,ch)
                let command = try XCTUnwrap(queue.makeCommandBuffer())
                for (target, crop) in [(full,false), (cropped,true)] {
                    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                    let pipeline = crop ? kernels.blendUpsampleMaskAndPackCropped : kernels.blendUpsampleMaskAndPack
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(rgb0, offset: 0, index: 0)
                    encoder.setBuffer(rgb1, offset: 0, index: 1)
                    encoder.setBuffer(mask, offset: 0, index: 2)
                    var internalDim = SIMD2<UInt32>(17,11)
                    var sourceDim = SIMD2<UInt32>(UInt32(w),UInt32(h))
                    encoder.setBytes(&internalDim, length: 8, index: 3)
                    if crop { encoder.setBytes(&sourceDim, length: 8, index: 4) }
                    encoder.setTexture(target, index: 0)
                    encoder.dispatchThreadgroups(MTLSize(width: (target.width+7)/8, height: (target.height+7)/8, depth: 1), threadsPerThreadgroup: MTLSize(width: 8,height: 8,depth: 1))
                    encoder.endEncoding()
                }
                command.commit(); command.waitUntilCompleted()
                XCTAssertEqual(command.status, .completed)
                var a = [UInt8](repeating: 0, count: w*h*4)
                var b = [UInt8](repeating: 0, count: cw*ch*4)
                full.getBytes(&a, bytesPerRow: w*4, from: MTLRegionMake2D(0,0,w,h), mipmapLevel: 0)
                cropped.getBytes(&b, bytesPerRow: cw*4, from: MTLRegionMake2D(0,0,cw,ch), mipmapLevel: 0)
                for y in 0..<ch {
                    XCTAssertEqual(Array(a[y*w*4..<y*w*4+cw*4]), Array(b[y*cw*4..<(y+1)*cw*4]))
                }
            }
        }
    }
}
