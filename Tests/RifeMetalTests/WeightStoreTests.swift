import XCTest
@testable import RifeMetal
@testable import RifeMetalCore

final class WeightStoreTests: XCTestCase {

    var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("RifeMetalTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    func testLoadValidFile() throws {
        let url = tempDir.appendingPathComponent("test.rmw")
        try RMWTestFixture.write(
            tensors: [
                .init(name: "block0.weight", shape: [2, 1, 1, 2], values: [1, 2, 3, 4]),
                .init(name: "block0.bias",   shape: [2],           values: [0.5, -0.5]),
            ],
            to: url
        )

        let store = try WeightStore(url: url)
        XCTAssertEqual(store.header.model, "test-model")
        XCTAssertEqual(store.header.tensors.count, 2)
        XCTAssertEqual(store.header.tensors[0].name, "block0.weight")
        XCTAssertEqual(store.header.tensors[0].shape, [2, 1, 1, 2])
        XCTAssertEqual(store.header.tensors[0].size, 8)  // 4 fp16 values
    }

    func testTensorBytesCorrectness() throws {
        let url = tempDir.appendingPathComponent("test.rmw")
        try RMWTestFixture.write(
            tensors: [.init(name: "x", shape: [4], values: [1, 2, 3, 4])],
            to: url
        )

        let store = try WeightStore(url: url)
        let bytes = try store.tensorBytes(named: "x")
        XCTAssertEqual(bytes.count, 8)

        // Decode fp16 back to fp32 and compare.
        let decoded = bytes.withUnsafeBytes { buf -> [Float] in
            let ptr = buf.bindMemory(to: UInt16.self)
            return ptr.map { HalfPrecision.bitsToFloat(UInt16(littleEndian: $0)) }
        }
        XCTAssertEqual(decoded, [1, 2, 3, 4])
    }

    func testMissingTensorThrows() throws {
        let url = tempDir.appendingPathComponent("test.rmw")
        try RMWTestFixture.write(
            tensors: [.init(name: "x", shape: [1], values: [1])],
            to: url
        )

        let store = try WeightStore(url: url)
        XCTAssertThrowsError(try store.tensorBytes(named: "nope"))
    }

    func testInvalidMagicThrows() throws {
        let url = tempDir.appendingPathComponent("bad.rmw")
        try Data(repeating: 0xFF, count: 32).write(to: url)
        XCTAssertThrowsError(try WeightStore(url: url))
    }

    func testBundledModelHasConvertedTensorLayout() throws {
        let store = try WeightStore(url: RifeConfiguration.bundledModelURL)
        XCTAssertEqual(store.header.model, "rife-v4.25")
        XCTAssertTrue(store.header.tensors.contains(where: { $0.name == "block0.conv0.0.0.weight" }))
        let bytes = try store.tensorBytes(named: "block0.conv0.0.0.weight")
        // PyTorch shape: [96, 15, 3, 3]. After HWIO permute: [3, 3, 15, 96].
        // Total elements: 96 * 15 * 3 * 3 = 12960. fp16 = 2 bytes each → 25920 bytes.
        XCTAssertEqual(bytes.count, 96 * 15 * 3 * 3 * 2)
    }
}
