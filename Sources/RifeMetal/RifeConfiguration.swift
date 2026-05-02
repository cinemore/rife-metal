import Foundation
import Metal

/// Quality / performance trade-off knob.
///
/// `hq` runs IFNet at the full padded resolution (the historical behavior).
/// `balanced` runs IFNet at half-resolution internally (UHD mode): motion is
/// estimated on a 4x-cheaper grid, then the resulting flow/mask are upsampled
/// to full resolution for the final warp+blend so output sharpness is preserved.
/// `fast` runs IFNet at quarter-resolution internally (1/16 the pixel count vs hq),
/// targeting 4K 30fps real-time throughput.
public enum RifeQualityTier: String, Sendable, CaseIterable {
    case hq         // internal scale 1.0 (full-res inference)
    case balanced   // internal scale 0.5 (UHD mode)
    case fast       // internal scale 0.25 (1/4 res; ~16x fewer IFNet pixels vs hq)

    /// Ratio of internal IFNet resolution to padded full resolution.
    public var internalScale: Double {
        switch self {
        case .hq:       return 1.0
        case .balanced: return 0.5
        case .fast:     return 0.25
        }
    }

    /// Padded dims must be a multiple of this so that `padded * internalScale` is itself
    /// a multiple of 64 (required by v4.26's deepest stage at scale=16 combined with
    /// IFBlock's internal 4× spatial downsample).
    public var paddingMultiple: Int {
        switch self {
        case .hq:       return 64
        case .balanced: return 128
        case .fast:     return 256
        }
    }
}

public struct RifeConfiguration: Sendable {
    public var modelURL: URL
    public var qualityTier: RifeQualityTier
    public var preferredDevice: MTLDevice?

    public init(modelURL: URL,
                qualityTier: RifeQualityTier = .hq,
                preferredDevice: MTLDevice? = nil) {
        self.modelURL = modelURL
        self.qualityTier = qualityTier
        self.preferredDevice = preferredDevice
    }
}

public extension RifeConfiguration {

    /// File URL of the rife-v4.26 model weights bundled inside this Swift package.
    /// Resolved via SwiftPM's per-target `Bundle.module`. Force-unwrap is safe — if
    /// the resource is missing, the package itself is malformed.
    static let bundledModelURL: URL = Bundle.module.url(forResource: "rife-v4.26",
                                                          withExtension: "rmw")!

    /// Configuration that uses the package's bundled rife-v4.26 weights. Most callers
    /// want this — no need to source weights separately.
    static func bundled(qualityTier: RifeQualityTier = .hq,
                        preferredDevice: MTLDevice? = nil) -> RifeConfiguration {
        RifeConfiguration(modelURL: bundledModelURL,
                          qualityTier: qualityTier,
                          preferredDevice: preferredDevice)
    }
}
