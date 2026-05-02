import Foundation
import Metal
import MetalPerformanceShadersGraph

public enum InferenceContextError: Error {
    case noMetalDevice
    case commandQueueCreationFailed
}

public final class InferenceContext {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let graph: MPSGraph
    public let warpKernel: BackwardWarpKernel
    public let conversionKernels: ConversionKernels
    public let mpsGraphDevice: MPSGraphDevice

    public init(preferredDevice: MTLDevice? = nil) throws {
        let device: MTLDevice
        if let preferred = preferredDevice {
            device = preferred
        } else if let system = MTLCreateSystemDefaultDevice() {
            device = system
        } else {
            throw InferenceContextError.noMetalDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw InferenceContextError.commandQueueCreationFailed
        }
        self.device = device
        self.commandQueue = queue
        self.graph = MPSGraph()
        self.warpKernel = try BackwardWarpKernel(device: device)
        self.conversionKernels = try ConversionKernels(device: device)
        self.mpsGraphDevice = MPSGraphDevice(mtlDevice: device)
    }
}
