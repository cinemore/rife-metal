import Metal
import XCTest
@testable import RifeMetalCore

final class AccumulationRoundingTests: XCTestCase {
    func testGPUAdditionMatchesBothRoundingModes() throws {
        let context = try InferenceContext(), device = context.device
        let pixels = 128*128, count = pixels*4
        let buffers = (0..<6).map { _ in device.makeBuffer(length:count*2,options:.storageModeShared)! }
        let lhs = buffers[0].contents().bindMemory(to:UInt16.self,capacity:count)
        let rhs = buffers[1].contents().bindMemory(to:UInt16.self,capacity:count)
        for i in 0..<count {
            lhs[i] = UInt16(i%0x7c00) | (i.isMultiple(of:2) ? 0x8000 : 0)
            rhs[i] = UInt16((i*193+79)%0x7c00) | (i.isMultiple(of:3) ? 0x8000 : 0)
        }
        // 精确中点、正负抵消、次正规数和溢出都参与对照。
        let edges: [(UInt16,UInt16)] = [(0x3c00,0x1000),(0xbc00,0x9000),(0x3c00,0xbc00),(1,1),(0x7bff,0x7bff),(0xfbff,0xfbff),(0x8000,0x8000)]
        for (i,pair) in edges.enumerated() { lhs[i]=pair.0;rhs[i]=pair.1 }
        memcpy(buffers[3].contents(),buffers[0].contents(),pixels*2)
        memcpy(buffers[4].contents(),buffers[1].contents(),pixels*2)
        for mode: UInt32 in [0,1] {
            let command = context.commandQueue.makeCommandBuffer()!
            let encoder = command.makeComputeCommandEncoder()!
            let pipeline = context.conversionKernels.accumulateFlowMask
            encoder.setComputePipelineState(pipeline)
            for i in 0..<6 { encoder.setBuffer(buffers[i],offset:0,index:i) }
            var dimensions=SIMD2<UInt32>(128,128),rounding=SIMD2<UInt32>(mode,mode)
            encoder.setBytes(&dimensions,length:8,index:6);encoder.setBytes(&rounding,length:8,index:7)
            encoder.dispatchThreads(MTLSize(width:128,height:128,depth:1),threadsPerThreadgroup:MTLSize(width:8,height:8,depth:1))
            encoder.endEncoding();command.commit();command.waitUntilCompleted()
            XCTAssertEqual(command.status,.completed)
            for (outputIndex,n) in [(2,count),(5,pixels)] {
                let output=buffers[outputIndex].contents().bindMemory(to:UInt16.self,capacity:n)
                var mismatches=0
                for i in 0..<n {
                    let sum=Float(Float16(bitPattern:lhs[i]))+Float(Float16(bitPattern:rhs[i]))
                    var expected=Float16(sum).bitPattern
                    let magnitude=abs(Float(Float16(bitPattern:expected)))
                    if mode == 1, magnitude < abs(sum), expected & 0x7fff < 0x7bff {
                        let next=Float(Float16(bitPattern:(expected & 0x7fff)+1))
                        if abs(sum)==(magnitude+next)/2 { expected += 1 }
                    }
                    if output[i] != expected { mismatches += 1 }
                }
                XCTAssertEqual(mismatches,0,"mode=\(mode) buffer=\(outputIndex)")
            }
        }
    }
}
