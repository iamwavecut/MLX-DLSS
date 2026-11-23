// swift-tools-version: 6.2
import PackageDescription

let package = Package(
  name: "MLXDLSS",
  platforms: [
    .macOS(.v14)
  ],
  products: [
    .library(name: "DLSSCore", targets: ["DLSSCore"]),
    .library(name: "DLSSCoreML", targets: ["DLSSCoreML"]),
    .library(name: "DLSSMLX", targets: ["DLSSMLX"]),
    .library(name: "DLSSMedia", targets: ["DLSSMedia"]),
    .executable(name: "mlxdlss", targets: ["mlxdlss"]),
    .executable(name: "MLXDLSSApp", targets: ["MLXDLSSApp"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/ml-explore/mlx-swift",
      exact: "0.33.0"
    )
  ],
  targets: [
    .target(
      name: "DLSSCore",
      linkerSettings: [
        .linkedFramework("Accelerate"),
        .linkedFramework("Metal"),
      ]
    ),
    .target(
      name: "DLSSCoreML",
      dependencies: ["DLSSCore"],
      linkerSettings: [.linkedFramework("CoreML")]
    ),
    .target(
      name: "DLSSMLX",
      dependencies: [
        "DLSSCore",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
      ],
      linkerSettings: [.linkedFramework("CoreVideo"), .linkedFramework("Metal"), .linkedFramework("IOSurface")]
    ),
    .target(
      name: "DLSSMedia",
      dependencies: ["DLSSCore", "DLSSMLX"],
      linkerSettings: [
        .linkedFramework("AVFoundation"), .linkedFramework("VideoToolbox"),
        .linkedFramework("CoreImage"), .linkedFramework("ImageIO"),
        .linkedFramework("Vision"),
      ]
    ),
    .executableTarget(
      name: "mlxdlss",
      dependencies: [
        "DLSSCore",
        "DLSSCoreML",
        "DLSSMLX",
        "DLSSMedia",
      ],
      linkerSettings: [
        .linkedFramework("CoreGraphics"),
        .linkedFramework("ImageIO"),
      ]
    ),
    .testTarget(
      name: "DLSSCoreTests",
      dependencies: ["DLSSCore"]
    ),
    .executableTarget(
      name: "MLXDLSSApp",
      dependencies: ["DLSSCore", "DLSSMLX", "DLSSMedia"],
      linkerSettings: [.linkedFramework("SwiftUI"), .linkedFramework("AVKit")]
    ),
    .testTarget(
      name: "DLSSCoreMLTests",
      dependencies: [
        "DLSSCore",
        "DLSSCoreML",
      ]
    ),
    .testTarget(
      name: "DLSSMLXTests",
      dependencies: [
        "DLSSCore",
        "DLSSMLX",
        .product(name: "MLX", package: "mlx-swift"),
      ]
    ),
    .testTarget(
      name: "DLSSMediaTests",
      dependencies: ["DLSSMedia", "DLSSMLX"]
    ),
    .testTarget(
      name: "MLXDLSSCLITests",
      dependencies: ["mlxdlss", "DLSSMLX"]
    ),
  ]
)
