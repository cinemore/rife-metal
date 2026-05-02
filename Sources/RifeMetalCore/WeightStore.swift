import Foundation
import Metal
import MetalPerformanceShadersGraph

public enum WeightStoreError: Error, Equatable {
    case invalidMagic
    case unsupportedVersion(UInt32)
    case headerParseFailed(String)
    case tensorNotFound(String)
    case fileTooSmall
}

public final class WeightStore {
    public let header: RifeWeightHeader
    private let mappedData: Data
    private let blobOffset: Int

    public init(url: URL) throws {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= 12 else { throw WeightStoreError.fileTooSmall }

        let magic = data.subdata(in: 0..<4)
        guard magic == Data([0x52, 0x4D, 0x57, 0x31]) else {
            throw WeightStoreError.invalidMagic
        }

        let version = data.readUInt32LE(at: 4)
        guard version == 1 else { throw WeightStoreError.unsupportedVersion(version) }

        let headerLen = Int(data.readUInt32LE(at: 8))
        guard data.count >= 12 + headerLen else { throw WeightStoreError.fileTooSmall }
        let headerData = data.subdata(in: 12..<(12 + headerLen))

        let decoder = JSONDecoder()
        do {
            self.header = try decoder.decode(RifeWeightHeader.self, from: headerData)
        } catch {
            throw WeightStoreError.headerParseFailed(String(describing: error))
        }

        let unpadded = 12 + headerLen
        let padding = (16 - (unpadded % 16)) % 16
        self.blobOffset = unpadded + padding
        self.mappedData = data
    }

    /// Returns the raw fp16 bytes for the named tensor.
    public func tensorBytes(named name: String) throws -> Data {
        guard let entry = header.tensors.first(where: { $0.name == name }) else {
            throw WeightStoreError.tensorNotFound(name)
        }
        guard entry.offset >= 0, entry.size >= 0 else {
            throw WeightStoreError.fileTooSmall
        }
        let start = blobOffset + entry.offset
        let end = start + entry.size
        guard end <= mappedData.count else { throw WeightStoreError.fileTooSmall }
        return mappedData.subdata(in: start..<end)
    }

    /// Wraps the named tensor's bytes as MPSGraphTensorData (fp16, NHWC).
    public func tensorData(named name: String,
                           on device: MTLDevice) throws -> MPSGraphTensorData {
        guard let entry = header.tensors.first(where: { $0.name == name }) else {
            throw WeightStoreError.tensorNotFound(name)
        }
        guard entry.offset >= 0, entry.size >= 0 else {
            throw WeightStoreError.fileTooSmall
        }
        let start = blobOffset + entry.offset
        let end = start + entry.size
        guard end <= mappedData.count else { throw WeightStoreError.fileTooSmall }
        let bytes = mappedData.subdata(in: start ..< end)
        return MPSGraphTensorData(
            device: MPSGraphDevice(mtlDevice: device),
            data: bytes,
            shape: entry.shape.map(NSNumber.init),
            dataType: .float16
        )
    }
}

// MARK: - Helpers

private extension Data {
    func readUInt32LE(at offset: Int) -> UInt32 {
        return self.withUnsafeBytes { buf -> UInt32 in
            buf.load(fromByteOffset: offset, as: UInt32.self).littleEndian
        }
    }
}
