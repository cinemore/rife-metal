import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

enum ImageIOAdapter {

    static func readImage(at url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw NSError(domain: "ImageIOAdapter", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "cannot create image source: \(url.path)"])
        }
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NSError(domain: "ImageIOAdapter", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "cannot decode image at: \(url.path)"])
        }
        return image
    }

    static func writeImage(_ image: CGImage, to url: URL) throws {
        let type = imageType(forExtension: url.pathExtension) ?? UTType.png.identifier as CFString
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, type, 1, nil) else {
            throw NSError(domain: "ImageIOAdapter", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "cannot create destination: \(url.path)"])
        }
        CGImageDestinationAddImage(dest, image, nil)
        if !CGImageDestinationFinalize(dest) {
            throw NSError(domain: "ImageIOAdapter", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "finalize failed: \(url.path)"])
        }
    }

    private static func imageType(forExtension ext: String) -> CFString? {
        switch ext.lowercased() {
        case "png": return UTType.png.identifier as CFString
        case "jpg", "jpeg": return UTType.jpeg.identifier as CFString
        case "heic": return UTType.heic.identifier as CFString
        case "tiff": return UTType.tiff.identifier as CFString
        default: return nil
        }
    }
}
