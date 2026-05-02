// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "RifeMetal",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "RifeMetal", targets: ["RifeMetal", "RifeMetalCore"]),
        .executable(name: "rife-metal", targets: ["rife-metal-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "RifeMetal",
            dependencies: ["RifeMetalCore"],
            resources: [.copy("Resources/rife-v4.26.rmw")]
        ),
        .target(
            name: "RifeMetalCore",
            exclude: ["Shaders"],
            plugins: [.plugin(name: "CompileMetalShaders")]
        ),
        .plugin(
            name: "CompileMetalShaders",
            capability: .buildTool()
        ),
        .executableTarget(
            name: "rife-metal-cli",
            dependencies: [
                "RifeMetal",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "RifeMetalTests",
            dependencies: ["RifeMetal", "RifeMetalCore"]
        ),
    ]
)
