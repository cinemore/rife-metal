import XCTest
@testable import RifeMetalCore

final class HalfPrecisionTests: XCTestCase {
    func testEncodesKnownBinary16Patterns() {
        XCTAssertEqual(HalfPrecision.floatToBits(0), 0x0000)
        XCTAssertEqual(HalfPrecision.floatToBits(-0.0), 0x8000)
        XCTAssertEqual(HalfPrecision.floatToBits(0.5), 0x3800)
        XCTAssertEqual(HalfPrecision.floatToBits(1), 0x3C00)
        XCTAssertEqual(HalfPrecision.floatToBits(-2), 0xC000)
    }

    func testRoundTripsRepresentativeValues() {
        for value in [Float]([-4, -1, -0.25, 0, 0.125, 0.5, 1, 3.5, 1024]) {
            let decoded = HalfPrecision.bitsToFloat(HalfPrecision.floatToBits(value))
            XCTAssertEqual(decoded, value, accuracy: 0.001)
        }
    }
}
