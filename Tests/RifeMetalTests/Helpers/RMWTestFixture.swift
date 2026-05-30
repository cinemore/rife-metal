import Foundation
@testable import RifeMetalCore

/// Builds a minimal in-memory .rmw file for unit tests.
/// Each tensor is a flat array of fp16 values; shape is honored only for size computation.
struct RMWTestFixture {
    struct Tensor {
        let name: String
        let shape: [Int]
        let values: [Float]   // fp32 in, encoded to fp16 in the file
    }

    static func write(tensors: [Tensor], to url: URL,
                      model: String = "test-model",
                      ifblockChannels: [Int] = [4, 2],
                      scaleList: [Int] = [2, 1]) throws {
        var blob = Data()
        var entries: [[String: Any]] = []

        for t in tensors {
            let bytes = encodeFloat16(t.values)
            let entry: [String: Any] = [
                "name": t.name,
                "dtype": "f16",
                "shape": t.shape,
                "offset": blob.count,
                "size": bytes.count,
            ]
            entries.append(entry)
            blob.append(bytes)
        }

        let header: [String: Any] = [
            "model": model,
            "ifblock_channels": ifblockChannels,
            "scale_list": scaleList,
            "input_layout": "NHWC",
            "tensors": entries,
        ]
        let headerJSON = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])

        var out = Data()
        out.append(contentsOf: [0x52, 0x4D, 0x57, 0x31])  // "RMW1"
        out.append(uint32LE(1))                             // version
        out.append(uint32LE(UInt32(headerJSON.count)))      // headerLen
        out.append(headerJSON)
        let pad = (16 - (out.count % 16)) % 16
        out.append(Data(repeating: 0, count: pad))
        out.append(blob)

        try out.write(to: url)
    }

    private static func encodeFloat16(_ values: [Float]) -> Data {
        var data = Data(capacity: values.count * 2)
        for v in values {
            let f16 = HalfPrecision.floatToBits(v).littleEndian
            withUnsafeBytes(of: f16) { data.append(contentsOf: $0) }
        }
        return data
    }

    private static func uint32LE(_ x: UInt32) -> Data {
        var le = x.littleEndian
        return Data(bytes: &le, count: 4)
    }
}
