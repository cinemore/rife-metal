import Foundation
import CoreGraphics
import CoreVideo
import ArgumentParser
import RifeMetal

@main
struct RifeMetalCLI: ParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: "rife-metal",
        abstract: "Native Apple Silicon RIFE frame interpolation."
    )

    @Option(name: [.customShort("0")], help: "First input image (png/jpeg/heic).")
    var firstFrame: String

    @Option(name: [.customShort("1")], help: "Second input image (png/jpeg/heic).")
    var secondFrame: String

    @Option(name: .shortAndLong, help: "Output image path. With --timesteps producing multiple frames, this becomes a template — see --timesteps help for the naming rule.")
    var output: String

    @Option(name: .shortAndLong, help: "Path to .rmw weight file.")
    var model: String

    @Option(name: .long, help: "Benchmark mode: run inference N times and print mean/min/max ms (per-frame when --timesteps has multiple values).")
    var bench: Int = 0

    @Option(name: .long,
            help: "Stream-bench mode: run N stateless calls + N stream pushes (after a warm-up push), print mean/delta.")
    var benchStream: Int = 0

    @Option(name: .long,
            help: "Quality tier: hq (default; full-res inference) | balanced (UHD mode, half-res inference, ~4x faster on large frames) | fast (quarter-res inference, targets 4K 30fps real-time).")
    var tier: String = "hq"

    @Option(name: .long,
            help: "Comma-separated target timesteps in (0, 1). Single value writes to -o exactly. Multiple values write to filenames templated from -o (e.g. out.png + 0.33,0.67 → out_t0.33.png, out_t0.67.png).")
    var timesteps: String = "0.5"

    func run() throws {
        let firstURL = URL(fileURLWithPath: firstFrame)
        let secondURL = URL(fileURLWithPath: secondFrame)
        let modelURL = URL(fileURLWithPath: model)

        if benchStream > 0 {
            try runBenchStream(firstURL: firstURL, secondURL: secondURL, modelURL: modelURL)
            return
        }

        // Parse and validate --timesteps.
        let parsedTimesteps: [Float]
        do {
            parsedTimesteps = try parseTimesteps(timesteps)
        } catch {
            FileHandle.standardError.write(
                Data("rife-metal: \(error)\n".utf8))
            throw ExitCode(2)
        }

        guard let qualityTier = RifeQualityTier(rawValue: tier) else {
            let validTiers = RifeQualityTier.allCases.map(\.rawValue).joined(separator: ", ")
            FileHandle.standardError.write(
                Data("rife-metal: unknown --tier '\(tier)'. Valid: \(validTiers)\n".utf8))
            throw ExitCode(2)
        }

        let config = RifeConfiguration(modelURL: modelURL, qualityTier: qualityTier)
        let interpolator: RifeInterpolator
        do {
            interpolator = try RifeInterpolator(configuration: config)
        } catch {
            FileHandle.standardError.write(
                Data("rife-metal: model load failed: \(error)\n".utf8))
            throw ExitCode(3)
        }

        let firstImage: CGImage
        let secondImage: CGImage
        do {
            firstImage = try ImageIOAdapter.readImage(at: firstURL)
            secondImage = try ImageIOAdapter.readImage(at: secondURL)
        } catch {
            FileHandle.standardError.write(
                Data("rife-metal: input read failed: \(error)\n".utf8))
            throw ExitCode(2)
        }

        let results: [CGImage]
        do {
            results = try interpolator.interpolate(previous: firstImage,
                                                    current: secondImage,
                                                    timesteps: parsedTimesteps)
        } catch {
            FileHandle.standardError.write(
                Data("rife-metal: inference failed: \(error)\n".utf8))
            throw ExitCode(4)
        }

        if bench > 0 {
            // Warm graph already built by the call above. Run N more times to measure steady-state.
            var samples: [Double] = []
            samples.reserveCapacity(bench)
            for _ in 0..<bench {
                let t0 = DispatchTime.now()
                _ = try interpolator.interpolate(previous: firstImage,
                                                  current: secondImage,
                                                  timesteps: parsedTimesteps)
                let t1 = DispatchTime.now()
                let ms = Double(t1.uptimeNanoseconds &- t0.uptimeNanoseconds) / 1_000_000
                samples.append(ms)
            }
            let n = parsedTimesteps.count
            let mean = (samples.reduce(0, +) / Double(samples.count)) / Double(n)
            let minV = (samples.min() ?? 0) / Double(n)
            let maxV = (samples.max() ?? 0) / Double(n)
            let line = String(
                format: "[bench] tier=%@ N=%d timesteps=%d mean=%.2f ms/frame min=%.2f ms/frame max=%.2f ms/frame\n",
                tier, bench, n, mean, minV, maxV
            )
            FileHandle.standardError.write(Data(line.utf8))
        }

        // Write outputs. Single timestep → exactly -o. Multiple → template from -o.
        let outputPaths = resolveOutputPaths(template: output, timesteps: parsedTimesteps)
        for (img, path) in zip(results, outputPaths) {
            do {
                try ImageIOAdapter.writeImage(img, to: URL(fileURLWithPath: path))
            } catch {
                FileHandle.standardError.write(
                    Data("rife-metal: output write failed for \(path): \(error)\n".utf8))
                throw ExitCode(2)
            }
        }
    }

    private func runBenchStream(firstURL: URL, secondURL: URL, modelURL: URL) throws {
        let iters = benchStream

        guard let qualityTier = RifeQualityTier(rawValue: tier) else {
            let validTiers = RifeQualityTier.allCases.map(\.rawValue).joined(separator: ", ")
            FileHandle.standardError.write(
                Data("rife-metal: unknown --tier '\(tier)'. Valid: \(validTiers)\n".utf8))
            throw ExitCode(2)
        }

        let config = RifeConfiguration(modelURL: modelURL, qualityTier: qualityTier)
        let interpolator: RifeInterpolator
        do {
            interpolator = try RifeInterpolator(configuration: config)
        } catch {
            FileHandle.standardError.write(
                Data("rife-metal: model load failed: \(error)\n".utf8))
            throw ExitCode(3)
        }

        let firstImage: CGImage
        let secondImage: CGImage
        do {
            firstImage = try ImageIOAdapter.readImage(at: firstURL)
            secondImage = try ImageIOAdapter.readImage(at: secondURL)
        } catch {
            FileHandle.standardError.write(
                Data("rife-metal: input read failed: \(error)\n".utf8))
            throw ExitCode(2)
        }

        let frameA = try cgImageToBGRAPixelBuffer(firstImage)
        let frameB = try cgImageToBGRAPixelBuffer(secondImage)

        // Stateless baseline: 1 warm-up + iters timed.
        _ = try interpolator.interpolate(previous: frameA, current: frameB)
        var statelessMs: [Double] = []
        statelessMs.reserveCapacity(iters)
        for _ in 0..<iters {
            let t0 = DispatchTime.now()
            _ = try interpolator.interpolate(previous: frameA, current: frameB)
            let t1 = DispatchTime.now()
            statelessMs.append(Double(t1.uptimeNanoseconds &- t0.uptimeNanoseconds) / 1_000_000)
        }

        // Stream steady-state: makeStream + push(A) warm-up + iters timed pushes alternating B/A.
        let w = CVPixelBufferGetWidth(frameA)
        let h = CVPixelBufferGetHeight(frameA)
        let stream = try interpolator.makeStream(width: w, height: h)
        _ = try stream.push(frameA, timesteps: [])
        var streamMs: [Double] = []
        streamMs.reserveCapacity(iters)
        var nextIsB = true
        for _ in 0..<iters {
            let pushFrame = nextIsB ? frameB : frameA
            nextIsB.toggle()
            let t0 = DispatchTime.now()
            _ = try stream.push(pushFrame, timesteps: [0.5])
            let t1 = DispatchTime.now()
            streamMs.append(Double(t1.uptimeNanoseconds &- t0.uptimeNanoseconds) / 1_000_000)
        }

        let statelessMean = statelessMs.reduce(0, +) / Double(statelessMs.count)
        let streamMean    = streamMs.reduce(0, +)    / Double(streamMs.count)
        let delta = statelessMean - streamMean

        let lines = [
            String(format: "[bench-stream] tier=%@ stateless mean=%.2f ms (%d iters)\n",
                   tier, statelessMean, iters),
            String(format: "[bench-stream] tier=%@ stream    mean=%.2f ms (%d iters, steady state)\n",
                   tier, streamMean, iters),
            String(format: "[bench-stream] tier=%@ delta=%.2f ms\n", tier, delta),
        ]
        for line in lines {
            FileHandle.standardError.write(Data(line.utf8))
        }
    }

    private func cgImageToBGRAPixelBuffer(_ image: CGImage) throws -> CVPixelBuffer {
        let w = image.width, h = image.height
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary, &pb)
        guard let buffer = pb else {
            throw CLIError.parseError("cgImageToBGRAPixelBuffer: CVPixelBufferCreate failed")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let info: UInt32 = CGImageAlphaInfo.premultipliedFirst.rawValue
                         | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                  width: w, height: h,
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: cs,
                                  bitmapInfo: info) else {
            throw CLIError.parseError("cgImageToBGRAPixelBuffer: CGContext failed")
        }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Parses `"0.5"` or `"0.33,0.67"` into `[Float]`. Validates non-empty and `(0, 1)` range.
    private func parseTimesteps(_ s: String) throws -> [Float] {
        let parts = s.split(separator: ",", omittingEmptySubsequences: true)
        guard !parts.isEmpty else {
            throw CLIError.parseError("--timesteps must not be empty")
        }
        var values: [Float] = []
        for p in parts {
            let trimmed = p.trimmingCharacters(in: .whitespaces)
            guard let d = Double(trimmed) else {
                throw CLIError.parseError("--timesteps: cannot parse '\(trimmed)' as a number")
            }
            guard d > 0.0, d < 1.0 else {
                throw CLIError.parseError("--timesteps: value \(d) must be in (0, 1)")
            }
            values.append(Float(d))
        }
        return values
    }

    /// For 1 timestep returns `[template]`. For N>=2 builds `{stem}_t{value}{ext}` paths.
    /// Value formatting: up to 4 fraction digits, trailing zeros trimmed.
    private func resolveOutputPaths(template: String, timesteps: [Float]) -> [String] {
        if timesteps.count == 1 { return [template] }
        let url = URL(fileURLWithPath: template)
        let dir = url.deletingLastPathComponent()
        let base = url.lastPathComponent
        let dotIdx = base.lastIndex(of: ".")
        let stem: String
        let ext: String
        if let dotIdx, dotIdx != base.startIndex {
            stem = String(base[..<dotIdx])
            ext  = String(base[dotIdx...])  // includes the dot
        } else {
            stem = base
            ext  = ""
        }
        return timesteps.map { t in
            let value = formatTimestep(t)
            let filename = "\(stem)_t\(value)\(ext)"
            return dir.appendingPathComponent(filename).path
        }
    }

    /// Formats e.g. 0.5 → "0.5", 0.333 → "0.333", 0.66666 → "0.6667".
    private func formatTimestep(_ t: Float) -> String {
        let s = String(format: "%.4f", t)
        // Trim trailing zeros after the decimal point. If everything after the dot is zero,
        // leave one trailing zero so the value is still recognisable as a float.
        if s.contains(".") {
            var trimmed = s
            while trimmed.hasSuffix("0") { trimmed.removeLast() }
            if trimmed.hasSuffix(".") { trimmed.append("0") }
            return trimmed
        }
        return s
    }

    private enum CLIError: Error, CustomStringConvertible {
        case parseError(String)
        var description: String {
            switch self {
            case .parseError(let msg): return msg
            }
        }
    }
}
