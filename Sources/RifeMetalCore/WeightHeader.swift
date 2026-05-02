import Foundation

/// Decoded representation of the JSON header inside a .rmw file.
public struct RifeWeightHeader: Codable, Sendable {
    public struct TensorEntry: Codable, Sendable {
        public let name: String
        public let dtype: String
        public let shape: [Int]
        public let offset: Int
        public let size: Int
    }

    public let model: String
    public let ifblockChannels: [Int]
    public let scaleList: [Int]
    public let inputLayout: String
    public let tensors: [TensorEntry]

    enum CodingKeys: String, CodingKey {
        case model
        case ifblockChannels = "ifblock_channels"
        case scaleList = "scale_list"
        case inputLayout = "input_layout"
        case tensors
    }
}
