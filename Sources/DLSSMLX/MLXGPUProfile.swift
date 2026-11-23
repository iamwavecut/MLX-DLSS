import Foundation
import Metal

/// The GPU MLX runs on and whether MLX routes its matrix kernels to the GPU
/// Neural Accelerators.
///
/// M5, M6 and later (GPU generation 17 and up; phone GPUs from generation 18)
/// have a Neural Accelerator in every GPU core. MLX uses them through Metal 4
/// tensor operations ("NAX" kernels) for float16, bfloat16 and, with TF32,
/// float32 matrix multiplications on macOS 26.2 and later; mlx-swift compiles
/// those kernels at runtime. The rule here mirrors `mlx::core::metal::is_nax_available`
/// in MLX 0.32, including the `MLX_METAL_GPU_ARCH` override MLX reads in place of
/// the device's architecture name. Setting `MLX_METAL_GPU_ARCH` to an older
/// generation (for example `applegpu_g16s` on an M5 Max) turns the Neural
/// Accelerator path off, which `scripts/bench-neural-accelerators.sh` uses to
/// compare both paths on the same machine.
public struct MLXGPUProfile: Equatable, Sendable {
  /// Metal device name, for example "Apple M2 Max".
  public let deviceName: String
  /// The architecture name MLX uses: `MLX_METAL_GPU_ARCH` if set, else the
  /// device's, for example "applegpu_g14s".
  public let architecture: String
  /// The device's own architecture name, which `MLX_METAL_GPU_ARCH` does not change.
  public let deviceArchitecture: String
  /// True when the operating system has the tensor APIs MLX's NAX kernels need (26.2).
  public let operatingSystemSupportsNeuralAccelerators: Bool

  public init(deviceName: String, architecture: String, deviceArchitecture: String, operatingSystemSupportsNeuralAccelerators: Bool) {
    self.deviceName = deviceName
    self.architecture = architecture
    self.deviceArchitecture = deviceArchitecture
    self.operatingSystemSupportsNeuralAccelerators = operatingSystemSupportsNeuralAccelerators
  }

  /// GPU generation parsed like MLX: the two digits before the size-class letter
  /// (13 M1, 14 M2, 15 M3, 16 M4, 17 M5, 18 M6). Nil when the name has no such digits.
  public var generation: Int? { Self.generation(of: architecture) }
  /// Size-class letter: "g" base, "s" Pro/Max, "c" and "d" larger parts, "p" phone.
  public var sizeClass: Character? { architecture.last.flatMap { $0.isLetter ? $0 : nil } }
  /// Generation of the physical device, independent of `MLX_METAL_GPU_ARCH`.
  public var deviceGeneration: Int? { Self.generation(of: deviceArchitecture) }

  /// The hardware has GPU Neural Accelerators (by the device's own architecture).
  public var hasNeuralAccelerators: Bool { Self.supportsNeuralAccelerators(deviceArchitecture) }
  /// MLX routes matrix multiplications to the Neural Accelerators in this process.
  public var mlxUsesNeuralAccelerators: Bool {
    operatingSystemSupportsNeuralAccelerators && Self.supportsNeuralAccelerators(architecture)
  }

  static func generation(of architecture: String) -> Int? {
    let characters = Array(architecture)
    guard characters.count >= 3,
      let tens = characters[characters.count - 3].wholeNumberValue,
      let ones = characters[characters.count - 2].wholeNumberValue
    else { return nil }
    return tens * 10 + ones
  }

  static func supportsNeuralAccelerators(_ architecture: String) -> Bool {
    guard let generation = generation(of: architecture) else { return false }
    return generation >= (architecture.last == "p" ? 18 : 17)
  }

  /// The profile of the default Metal device in this process; nil without Metal.
  public static let current: MLXGPUProfile? = {
    guard let device = MTLCreateSystemDefaultDevice() else { return nil }
    let deviceArchitecture = device.architecture.name
    let override = ProcessInfo.processInfo.environment["MLX_METAL_GPU_ARCH"].flatMap { $0.isEmpty ? nil : $0 }
    let osSupport: Bool
    if #available(macOS 26.2, iOS 26.2, *) { osSupport = true } else { osSupport = false }
    return MLXGPUProfile(
      deviceName: device.name,
      architecture: override ?? deviceArchitecture,
      deviceArchitecture: deviceArchitecture,
      operatingSystemSupportsNeuralAccelerators: osSupport
    )
  }()

  /// JSON-ready summary for the CLI and benchmark logs.
  public var summary: [String: Any] {
    var result: [String: Any] = [
      "device": deviceName,
      "architecture": architecture,
      "neuralAccelerators": hasNeuralAccelerators,
      "mlxUsesNeuralAccelerators": mlxUsesNeuralAccelerators,
      "osSupportsNeuralAccelerators": operatingSystemSupportsNeuralAccelerators,
    ]
    if architecture != deviceArchitecture { result["deviceArchitecture"] = deviceArchitecture }
    if let generation { result["gpuGeneration"] = generation }
    if let sizeClass { result["sizeClass"] = String(sizeClass) }
    return result
  }
}
