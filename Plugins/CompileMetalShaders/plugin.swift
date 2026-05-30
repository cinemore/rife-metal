import PackagePlugin
import Foundation

@main
struct CompileMetalShaders: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
        guard let target = target as? SourceModuleTarget else { return [] }

        // Discover .metal files under the target's Shaders/ directory directly.
        // We can't use target.sourceFiles because Shaders/ is excluded from the
        // SwiftPM source list (so Xcode's auto-MetalLink phase doesn't pick them
        // up and collide with our plugin).
        let shadersDir = target.directory.appending("Shaders")
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: shadersDir.string)) ?? []
        let metalFiles = entries
            .filter { $0.hasSuffix(".metal") }
            .map { shadersDir.appending($0) }
        guard !metalFiles.isEmpty else { return [] }

        let outputDir = context.pluginWorkDirectory.appending("MetalLib")
        let moduleCacheDir = context.pluginWorkDirectory.appending("MetalModuleCache")
        let airFiles = metalFiles.map { metal -> Path in
            outputDir.appending(metal.stem + ".air")
        }
        let metallibPath = outputDir.appending("default.metallib")

        // Locate metal and metallib via xcrun
        let metalPath = try xcrunFind("metal")
        let metallibToolPath = try xcrunFind("metallib")

        var commands: [Command] = []

        // Compile each .metal -> .air
        for (metal, air) in zip(metalFiles, airFiles) {
            commands.append(.buildCommand(
                displayName: "Compile \(metal.lastComponent)",
                executable: Path(metalPath),
                arguments: [
                    "-c",
                    "-fmodules-cache-path=\(moduleCacheDir.string)",
                    metal.string,
                    "-o", air.string,
                ],
                inputFiles: [metal],
                outputFiles: [air]
            ))
        }

        // Link .air files -> default.metallib
        commands.append(.buildCommand(
            displayName: "Link default.metallib",
            executable: Path(metallibToolPath),
            arguments: airFiles.map { $0.string } + ["-o", metallibPath.string],
            inputFiles: airFiles,
            outputFiles: [metallibPath]
        ))

        return commands
    }

    private func xcrunFind(_ tool: String) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = ["--find", tool]
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let path = String(data: data, encoding: .utf8)!
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else {
            throw PluginError.toolNotFound(tool)
        }
        return path
    }

    enum PluginError: Error {
        case toolNotFound(String)
    }
}
