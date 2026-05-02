import Foundation

public enum RifeError: Error, Sendable {
    case modelLoadFailed(String)
    case unsupportedPixelFormat(OSType)
    case dimensionMismatch
    case metalUnavailable
    case shaderCompilationFailed(String)
    case inferenceFailed(String)
    case invalidTimesteps(String)
}
