import XCTest

@testable import DLSSMLX

final class MLXGPUProfileTests: XCTestCase {
  private func profile(_ architecture: String, device: String? = nil, os: Bool = true) -> MLXGPUProfile {
    MLXGPUProfile(
      deviceName: "test", architecture: architecture, deviceArchitecture: device ?? architecture,
      operatingSystemSupportsNeuralAccelerators: os)
  }

  func testGenerationAndSizeClassFollowMLXParsing() {
    XCTAssertEqual(profile("applegpu_g14s").generation, 14)
    XCTAssertEqual(profile("applegpu_g14s").sizeClass, "s")
    XCTAssertEqual(profile("applegpu_g17g").generation, 17)
    XCTAssertEqual(profile("applegpu_g18p").sizeClass, "p")
    XCTAssertNil(profile("unknown").generation)
    XCTAssertNil(profile("").sizeClass)
  }

  func testNeuralAcceleratorsStartAtM5AndA20() {
    for name in ["applegpu_g13g", "applegpu_g14s", "applegpu_g15c", "applegpu_g16d", "applegpu_g17p"] {
      XCTAssertFalse(profile(name).hasNeuralAccelerators, name)
      XCTAssertFalse(profile(name).mlxUsesNeuralAccelerators, name)
    }
    for name in ["applegpu_g17g", "applegpu_g17s", "applegpu_g17d", "applegpu_g18g", "applegpu_g18p"] {
      XCTAssertTrue(profile(name).hasNeuralAccelerators, name)
      XCTAssertTrue(profile(name).mlxUsesNeuralAccelerators, name)
    }
  }

  func testOperatingSystemAndOverrideGateMLXUse() {
    // Before macOS 26.2 the hardware is present but MLX does not use it.
    XCTAssertTrue(profile("applegpu_g17s", os: false).hasNeuralAccelerators)
    XCTAssertFalse(profile("applegpu_g17s", os: false).mlxUsesNeuralAccelerators)
    // MLX_METAL_GPU_ARCH set to an older generation turns the path off on an M5...
    let off = profile("applegpu_g16s", device: "applegpu_g17s")
    XCTAssertTrue(off.hasNeuralAccelerators)
    XCTAssertFalse(off.mlxUsesNeuralAccelerators)
    XCTAssertEqual(off.summary["deviceArchitecture"] as? String, "applegpu_g17s")
    // ...and set to a newer one forces it on older hardware (diagnostics only).
    XCTAssertTrue(profile("applegpu_g17s", device: "applegpu_g14s").mlxUsesNeuralAccelerators)
  }

  func testCurrentProfileDescribesThisMachine() throws {
    let current = try XCTUnwrap(MLXGPUProfile.current)
    XCTAssertFalse(current.deviceName.isEmpty)
    XCTAssertNotNil(current.deviceGeneration)
  }
}
