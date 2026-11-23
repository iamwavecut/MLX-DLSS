import Foundation
import MLX
import DLSSCore

/// Actor-transfer wrapper for immutable MLX graphs.
///
/// The sending actor does not mutate the array while the receiving actor builds
/// its lazy output graph, and the receiving call completes before either array is
/// used again.
struct MLXArrayTransfer: @unchecked Sendable {
  let array: MLXArray
}

public enum MLXExecutionMode: String, CaseIterable, Equatable, Sendable {
  case eager
  case blockCompiled = "block-compiled"
  case int8Fast = "int8-fast"
  case metalFused = "metal-fused"
  case compiled
}

public enum MLXComputePrecision: String, CaseIterable, Equatable, Sendable {
  case float32
  case float16

  var mlxDataType: DType {
    switch self {
    case .float32: .float32
    case .float16: .float16
    }
  }

  public var tensorDataType: TensorDataType {
    switch self {
    case .float32: .float32
    case .float16: .float16
    }
  }
}

public actor MLXNeuralRenderer: NeuralRenderBackend {
  public nonisolated let temporalCadence: NeuralRenderTemporalCadence
  private let package: LoadedModelPackage
  private let model: LoadedMLXModel
  private let compiledModel: (@Sendable (MLXArray) -> MLXArray)?
  private let computePrecision: MLXComputePrecision
  /// Block layout of a transformer package, inferred from its tensors.
  nonisolated let topology: NeuralRenderingTopology?
  /// The temporal blend cap the package carries (`blockN.layer0.blend_scale`
  /// of the post block): 0.7397 for the 71-block network, 0.767 for the
  /// 31-block small network; `nil` for packages without one.
  public nonisolated let blendScale: Float?
  /// `"ht (4,4,6,8,8, 71 blocks)"`, `nil` for non-transformer packages.
  public nonisolated var networkDescription: String? { topology?.description }

  /// The transformer packages this backend runs: the current identifier and the
  /// one written before the project was renamed (packages built earlier still load).
  public static let transformerArchitectures: Set<String> = [
    "mlxdlss.neural-rendering-transformer.v1",
    "nrk.neural-rendering-transformer.v1",
  ]

  public init(
    packageURL: URL,
    executionMode: MLXExecutionMode = .eager,
    computePrecision: MLXComputePrecision = .float32
  ) throws {
    let package = try ModelPackageLoader.load(url: packageURL)
    if executionMode == .blockCompiled || executionMode == .int8Fast
      || executionMode == .metalFused,
      !Self.transformerArchitectures.contains(package.manifest.architecture)
    {
      throw MLXBackendError.compiledExecutionUnsupported(
        architecture: package.manifest.architecture
      )
    }
    let model: LoadedMLXModel
    var topology: NeuralRenderingTopology?
    var blendScale: Float?
    switch package.manifest.architecture {
    case "mlxdlss.pixel-affine.v1":
      try Self.requireStateKind(.stateless, package: package)
      let weights = try SafeTensorValidation.loadAndValidate(package: package)
        .cast(to: computePrecision)
      model = .pixelAffine(
        PixelAffineModel(
          scale: try weights.required("scale"),
          bias: try weights.required("bias")
        )
      )
    case "mlxdlss.neural-rendering-transformer.v1", "nrk.neural-rendering-transformer.v1":
      try Self.requireStateKind(.stateless, package: package)
      guard executionMode != .compiled else {
        throw MLXBackendError.compiledExecutionUnsupported(
          architecture: package.manifest.architecture
        )
      }
      let weights = try SafeTensorValidation.loadAndValidate(package: package)
        .cast(to: computePrecision)
      let networkTopology = try NeuralRenderingTopology.infer(weights: weights)
      topology = networkTopology
      blendScale = weights["block\(networkTopology.post).layer0.blend_scale"].map {
        $0.asType(.float32).reshaped([-1])[0].item(Float.self)
      }
      model = .neuralRenderingTransformer(
        try NeuralRenderingTransformerModel(
          weights: weights,
          // Fused NR blocks publish half values and require float16 inputs.
          // Keep the reference MLX graph for float32 instead of trapping or reducing precision.
          compileBlocks: executionMode == .metalFused && computePrecision == .float16,
          quantizeGlobalFFN: executionMode == .int8Fast,
          topology: networkTopology
        )
      )
    default:
      throw MLXBackendError.unsupportedArchitecture(package.manifest.architecture)
    }

    self.temporalCadence = package.manifest.state.cadence
    self.topology = topology
    self.blendScale = blendScale
    self.package = package
    self.model = model
    self.computePrecision = computePrecision
    switch executionMode {
    case .eager, .blockCompiled, .int8Fast, .metalFused:
      self.compiledModel = nil
    case .compiled:
      // The compiled graph captures the model, which is not Sendable; the
      // renderer serializes calls into it like the uncompiled paths.
      nonisolated(unsafe) let compiled = compile(inputs: []) { input in
        model.statelessOutput(input)
      }
      self.compiledModel = { compiled($0) }
    }
  }

  public func render(_ request: NeuralRenderRequest) async throws -> NeuralRenderResult {
    let bindings = try package.manifest.resolveInputs(request.inputs)
    guard let color = request.input(named: "color") else {
      throw ManifestError.missingInput("color")
    }
    guard color.descriptor.dataType == .float32 else {
      throw MLXBackendError.unsupportedTensorDataType(color.descriptor.dataType)
    }
    try model.validateInputShape(color.descriptor.shape)

    let input = MLXArray(
      color.bytes,
      color.descriptor.shape,
      dtype: .float32
    )
    .asType(computePrecision.mlxDataType)
    let clock = ContinuousClock()
    let start = clock.now
    let modelOutput = try statelessOutputArray(input)
    let output = modelOutput.asType(.float32)
    eval(output)
    let executionNanoseconds = nanoseconds(in: start.duration(to: clock.now))

    let descriptors = try package.manifest.resolveOutputDescriptors(bindings: bindings)
    guard let outputDescriptor = descriptors.first(where: { $0.name == "color" }) else {
      throw MLXBackendError.missingOutput("color")
    }
    let outputTensor = try HostTensor(
      descriptor: outputDescriptor,
      bytes: contiguous(output).asData(access: .copy).data   // one host copy
    )

    return try NeuralRenderResult(
      outputs: [outputTensor],
      timing: NeuralRenderTiming(executionNanoseconds: executionNanoseconds)
    )
  }

  public func reset(sequenceID: UInt64?) async {}

  public func reset(_ request: NeuralRenderResetRequest) async {}

  func statelessOutputArray(_ input: MLXArray) throws -> MLXArray {
    try model.validateInputShape(input.shape)
    return Device.withDefaultDevice(.gpu) {
      executeModel(input)
    }
  }

  func statelessOutputTransfer(_ input: MLXArrayTransfer) throws -> MLXArrayTransfer {
    MLXArrayTransfer(array: try statelessOutputArray(input.array))
  }

  private func executeModel(_ input: MLXArray) -> MLXArray {
    if let compiledModel {
      compiledModel(input)
    } else {
      model.statelessOutput(input)
    }
  }

  private func nanoseconds(in duration: Duration) -> UInt64 {
    let components = duration.components
    let seconds = max(0, components.seconds)
    let attoseconds = max(0, components.attoseconds)
    return UInt64(seconds) * 1_000_000_000 + UInt64(attoseconds / 1_000_000_000)
  }

  private static func requireStateKind(
    _ expected: ModelStateKind,
    package: LoadedModelPackage
  ) throws {
    let actual = package.manifest.state.kind
    guard actual == expected else {
      throw MLXBackendError.modelStateKindMismatch(
        architecture: package.manifest.architecture,
        expected: expected,
        actual: actual
      )
    }
  }

}

private enum LoadedMLXModel {
  case pixelAffine(PixelAffineModel)
  case neuralRenderingTransformer(NeuralRenderingTransformerModel)

  func statelessOutput(_ input: MLXArray) -> MLXArray {
    switch self {
    case .pixelAffine(let model):
      model(input)
    case .neuralRenderingTransformer(let model):
      model(input)
    }
  }

  func validateInputShape(_ shape: [Int]) throws {
    guard case .neuralRenderingTransformer = self else { return }
    let minimum = NeuralRenderingGraphContract.minimumInputExtent
    let multiple = NeuralRenderingGraphContract.inputExtentMultiple
    let height = shape[1]
    let width = shape[2]
    guard height >= minimum,
      width >= minimum,
      height.isMultiple(of: multiple),
      width.isMultiple(of: multiple)
    else {
      throw MLXBackendError.unsupportedSpatialShape(
        architecture: "mlxdlss.neural-rendering-transformer.v1",
        height: height,
        width: width,
        minimum: minimum,
        multiple: multiple
      )
    }
  }
}
