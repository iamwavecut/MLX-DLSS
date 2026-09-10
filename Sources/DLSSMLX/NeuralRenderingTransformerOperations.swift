import Foundation
import MLX

enum NeuralRenderingBlockFamily: Equatable, Sendable {
  case pre
  case window
  case downsample
  case splitWindow
  case splitDownsample
  case globalAttention
  case bridge
  case upsample
  case post
}

struct NeuralRenderingBlockSpec: Equatable, Sendable {
  let index: Int
  let family: NeuralRenderingBlockFamily
  let inputChannels: Int
  let outputChannels: Int
  let hiddenChannels: Int?
  let headCount: Int?
}

struct NeuralRenderingStageSpec: Equatable, Sendable {
  let blockRange: ClosedRange<Int>
  let channels: Int
}

struct NeuralRenderingWindowOrigin: Equatable, Sendable {
  let y: Int
  let x: Int

  static let zero = NeuralRenderingWindowOrigin(y: 0, x: 0)
}

enum NeuralRenderingGraphContract {
  static let windowSize = 8
  static let minimumInputExtent = 128
  static let inputExtentMultiple = 64

  static func windowOrigin(for blockIndex: Int) -> NeuralRenderingWindowOrigin {
    let phase: Int
    if blockIndex == 0 {
      phase = 0
    } else if (1...4).contains(blockIndex) {
      phase = blockIndex - 1
    } else if (5...8).contains(blockIndex) {
      phase = blockIndex - 5
    } else if (9...14).contains(blockIndex) {
      phase = blockIndex - 9
    } else if (15...22).contains(blockIndex) {
      phase = blockIndex - 15
    } else if (23...30).contains(blockIndex) {
      phase = blockIndex - 23
    } else if (40...47).contains(blockIndex) {
      phase = blockIndex - 40
    } else if (48...55).contains(blockIndex) {
      phase = blockIndex - 48
    } else if (56...61).contains(blockIndex) {
      phase = blockIndex - 54
    } else if (62...65).contains(blockIndex) {
      phase = blockIndex - 62
    } else if (66...69).contains(blockIndex) {
      phase = blockIndex - 66
    } else if blockIndex == 70 {
      phase = 1
    } else {
      return .zero
    }
    return [
      .zero,
      NeuralRenderingWindowOrigin(y: -4, x: -4),
      NeuralRenderingWindowOrigin(y: 0, x: -4),
      NeuralRenderingWindowOrigin(y: -4, x: 0),
    ][phase % 4]
  }

  static let encoderStages = [
    NeuralRenderingStageSpec(blockRange: 0...4, channels: 32),
    NeuralRenderingStageSpec(blockRange: 5...8, channels: 64),
    NeuralRenderingStageSpec(blockRange: 9...14, channels: 128),
    NeuralRenderingStageSpec(blockRange: 15...22, channels: 256),
  ]

  static let decoderStages = [
    NeuralRenderingStageSpec(blockRange: 48...55, channels: 256),
    NeuralRenderingStageSpec(blockRange: 56...61, channels: 128),
    NeuralRenderingStageSpec(blockRange: 62...65, channels: 64),
    NeuralRenderingStageSpec(blockRange: 66...69, channels: 32),
  ]

  static let blocks: [NeuralRenderingBlockSpec] = {
    var result: [NeuralRenderingBlockSpec] = []

    func append(
      _ range: ClosedRange<Int>,
      family: NeuralRenderingBlockFamily,
      inputChannels: Int,
      outputChannels: Int? = nil,
      hiddenChannels: Int? = nil,
      headCount: Int? = nil
    ) {
      for index in range {
        result.append(
          NeuralRenderingBlockSpec(
            index: index,
            family: family,
            inputChannels: inputChannels,
            outputChannels: outputChannels ?? inputChannels,
            hiddenChannels: hiddenChannels,
            headCount: headCount
          )
        )
      }
    }

    append(
      0...0, family: .pre, inputChannels: 16, outputChannels: 32,
      hiddenChannels: 128, headCount: 1)
    append(
      1...3, family: .window, inputChannels: 32,
      hiddenChannels: 128, headCount: 1)
    append(
      4...4, family: .downsample, inputChannels: 32, outputChannels: 64,
      hiddenChannels: 128, headCount: 1)
    append(
      5...7, family: .window, inputChannels: 64,
      hiddenChannels: 224, headCount: 2)
    append(
      8...8, family: .downsample, inputChannels: 64, outputChannels: 128,
      hiddenChannels: 224, headCount: 2)
    append(
      9...13, family: .window, inputChannels: 128,
      hiddenChannels: 384, headCount: 4)
    append(
      14...14, family: .downsample, inputChannels: 128, outputChannels: 256,
      hiddenChannels: 384, headCount: 4)
    append(
      15...21, family: .window, inputChannels: 256,
      hiddenChannels: 704, headCount: 8)
    append(
      22...22, family: .downsample, inputChannels: 256, outputChannels: 512,
      hiddenChannels: 704, headCount: 8)
    append(
      23...29, family: .splitWindow, inputChannels: 512,
      hiddenChannels: 512, headCount: 16)
    append(
      30...30, family: .splitDownsample, inputChannels: 512,
      outputChannels: 1024, hiddenChannels: 512, headCount: 16)
    append(
      31...38, family: .globalAttention, inputChannels: 1024,
      hiddenChannels: 4096, headCount: 32)
    append(
      39...39, family: .bridge, inputChannels: 1024,
      outputChannels: 512)
    append(
      40...47, family: .splitWindow, inputChannels: 512,
      hiddenChannels: 512, headCount: 16)
    append(
      48...48, family: .upsample, inputChannels: 512,
      outputChannels: 256, hiddenChannels: 704, headCount: 8)
    append(
      49...55, family: .window, inputChannels: 256,
      hiddenChannels: 704, headCount: 8)
    append(
      56...56, family: .upsample, inputChannels: 256,
      outputChannels: 128, hiddenChannels: 384, headCount: 4)
    append(
      57...61, family: .window, inputChannels: 128,
      hiddenChannels: 384, headCount: 4)
    append(
      62...62, family: .upsample, inputChannels: 128,
      outputChannels: 64, hiddenChannels: 224, headCount: 2)
    append(
      63...65, family: .window, inputChannels: 64,
      hiddenChannels: 224, headCount: 2)
    append(
      66...66, family: .upsample, inputChannels: 64,
      outputChannels: 32, hiddenChannels: 128, headCount: 1)
    append(
      67...69, family: .window, inputChannels: 32,
      hiddenChannels: 128, headCount: 1)
    append(
      70...70, family: .post, inputChannels: 32, outputChannels: 4,
      hiddenChannels: 128, headCount: 1)

    precondition(result.map(\.index) == Array(0...70))
    return result
  }()
}

struct NeuralRenderingWindowBlock {
  private let body: (MLXArray, Bool?) -> MLXArray
  /// Multi-head blocks fold the final residual and publication into one kernel.
  private let fusesOutputResidual: Bool
  /// Single-kernel path (32 channels, one head) with the publication folded in.
  private let fusedBody: ((MLXArray, Bool) -> MLXArray)?
  private let publishesOutput: Bool

  init(
    weights: ValidatedWeights,
    blockIndex: Int,
    channels: Int,
    hiddenChannels: Int,
    headCount: Int,
    compileBlock: Bool = false
  ) throws {
    let prefix = "block\(blockIndex).layer0"
    let windowOrigin = NeuralRenderingGraphContract.windowOrigin(for: blockIndex)
    let feedForwardCosine = try Self.require(
      weights, name: "\(prefix).ffn_cos_skip", shape: [channels]
    )
    let qkvWeight = try Self.require(
      weights, name: "\(prefix).qkv_weight", shape: [channels, channels * 3]
    )
    let attentionScale = try Self.require(
      weights, name: "\(prefix).attn_scale", shape: [headCount]
    )
    let storedAttentionBias = try Self.require(
      weights,
      name: "\(prefix).attn_bias",
      shape: [headCount, 64, 64]
    )
    let attentionBias =
      NeuralRenderingAttentionBiasLayout.usesFragmentSwizzle(
        blockIndex: blockIndex,
        headCount: headCount
      )
      ? NeuralRenderingAttentionBiasLayout.recoverFragmentSwizzle(storedAttentionBias)
      : storedAttentionBias
    let attentionProjectionWeight = try Self.require(
      weights, name: "\(prefix).projection_weight", shape: [channels, channels]
    )
    let attentionCosine = try Self.require(
      weights, name: "\(prefix).attn_cos_skip", shape: [channels]
    )
    let operation: (MLXArray, Bool?) -> MLXArray
    var fusedWindow: ((MLXArray, Bool) -> MLXArray)?
    if channels >= 64 {
      let expansionWeight = try Self.require(
        weights, name: "\(prefix).ffn_expand_weight",
        shape: [headCount, 4, headCount, 32, 32]
      )
      let branchProjectionWeight = try Self.require(
        weights, name: "\(prefix).ffn_branch_projection_weight",
        shape: [headCount, 4, 32, 32]
      )
      let outputProjectionWeight = try Self.require(
        weights, name: "\(prefix).ffn_output_projection_weight",
        shape: [channels, channels]
      )
      let denseExpansion = compileBlock
        ? NeuralRenderingTransformerOperations.denseExpansionWeight(expansionWeight) : nil
      let groupedProjection = compileBlock
        ? NeuralRenderingTransformerOperations.groupedProjectionWeight(branchProjectionWeight) : nil
      operation = { input, publishedOutput in
        NeuralRenderingTransformerOperations.branchedWindowBlock(
          input,
          expansionWeight: expansionWeight,
          branchProjectionWeight: branchProjectionWeight,
          outputProjectionWeight: outputProjectionWeight,
          feedForwardCosine: feedForwardCosine,
          qkvWeight: qkvWeight,
          attentionScale: attentionScale,
          attentionBias: attentionBias,
          attentionProjectionWeight: attentionProjectionWeight,
          attentionCosine: attentionCosine,
          headCount: headCount,
          windowSize: NeuralRenderingGraphContract.windowSize,
          windowOrigin: windowOrigin,
          preciseSoftmax: true,
          fusedFeedForward: compileBlock,
          fusedAttention: compileBlock,
          denseExpansionWeight: denseExpansion,
          groupedProjectionWeight: groupedProjection,
          publishedOutput: publishedOutput
        )
      }
    } else {
      let expansionWeight = try Self.require(
        weights, name: "\(prefix).weight1", shape: [channels, hiddenChannels]
      )
      let feedForwardProjectionWeight = try Self.require(
        weights, name: "\(prefix).weight2", shape: [hiddenChannels, channels]
      )
      operation = { input, _ in
        NeuralRenderingTransformerOperations.windowBlock(
          input,
          expansionWeight: expansionWeight,
          feedForwardProjectionWeight: feedForwardProjectionWeight,
          feedForwardCosine: feedForwardCosine,
          qkvWeight: qkvWeight,
          attentionScale: attentionScale,
          attentionBias: attentionBias,
          attentionProjectionWeight: attentionProjectionWeight,
          attentionCosine: attentionCosine,
          headCount: headCount,
          windowSize: NeuralRenderingGraphContract.windowSize,
          windowOrigin: windowOrigin,
          preciseSoftmax: true,
          fusedFeedForward: compileBlock
        )
      }
      if compileBlock, channels == NeuralRenderingFusedWindowBlock.channels,
        hiddenChannels == NeuralRenderingFusedWindowBlock.hiddenChannels, headCount == 1
      {
        fusedWindow = { input, publish in
          NeuralRenderingFusedWindowBlock.apply(
            input,
            expansionWeight: expansionWeight,
            feedForwardProjectionWeight: feedForwardProjectionWeight,
            feedForwardCosine: feedForwardCosine,
            qkvWeight: qkvWeight,
            attentionScale: attentionScale,
            attentionBias: attentionBias,
            attentionProjectionWeight: attentionProjectionWeight,
            attentionCosine: attentionCosine,
            windowOrigin: windowOrigin,
            publish: publish
          )
        }
      }
    }
    // MLX compile currently changes custom vendor-kernel semantics.
    self.body = operation
    self.fusedBody = fusedWindow
    self.fusesOutputResidual =
      compileBlock && channels >= 64 && NeuralRenderingTransformerOperations.fusedOutputResidualEnabled
    // Block 0 is average-pooled before its single E4M3 publication and block 70
    // feeds the output head directly.
    self.publishesOutput = blockIndex != 0 && blockIndex != 70
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    if let fusedBody, input.dtype == .float16, input.shape[0] == 1 {
      return fusedBody(input, publishesOutput)
    }
    if fusesOutputResidual, input.dtype == .float16 {
      return body(input, publishesOutput)
    }
    let output = body(input, nil)
    return publishesOutput
      ? NeuralRenderingTransformerOperations.e4m3RoundTrip(output)
      : output
  }

  /// The block output before its E4M3 publication, for the fused downsample
  /// kernels that pool the half-precision result (vendor transition captures:
  /// 0.010-0.021 versus 0.026-0.031 when pooling the published tensor).
  func unpublished(_ input: MLXArray) -> MLXArray {
    if let fusedBody, input.dtype == .float16, input.shape[0] == 1 {
      return fusedBody(input, false)
    }
    if fusesOutputResidual, input.dtype == .float16 {
      return body(input, false)
    }
    return body(input, nil)
  }

  private static func require(
    _ weights: ValidatedWeights,
    name: String,
    shape: [Int]
  ) throws -> MLXArray {
    let value = try weights.required(name)
    guard value.shape == shape else {
      throw MLXBackendError.weightShapeMismatch(
        name: name,
        expected: shape,
        actual: value.shape
      )
    }
    return value
  }
}

struct NeuralRenderingWindowSequence {
  private let body: (MLXArray) -> MLXArray

  init(
    weights: ValidatedWeights,
    blockIndices: ClosedRange<Int>,
    channels: Int,
    hiddenChannels: Int,
    headCount: Int,
    compileSequence: Bool = false
  ) throws {
    let blocks = try blockIndices.map { index in
      try NeuralRenderingWindowBlock(
        weights: weights,
        blockIndex: index,
        channels: channels,
        hiddenChannels: hiddenChannels,
        headCount: headCount,
        compileBlock: compileSequence
      )
    }
    let operation = { input in
      blocks.reduce(input) { value, block in block(value) }
    }
    self.body = operation
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    body(input)
  }
}

struct NeuralRenderingSplitWindowBlock {
  private let windowOrigin: NeuralRenderingWindowOrigin
  private let firstProjectionWeight: MLXArray
  private let groupExpandWeight: MLXArray
  private let groupProjectWeight: MLXArray
  private let feedForwardProjectionWeight: MLXArray
  private let feedForwardCosine: MLXArray
  private let qkvWeight: MLXArray
  private let attentionScale: MLXArray
  private let attentionBias: MLXArray
  private let attentionProjectionWeight: MLXArray
  private let attentionCosine: MLXArray
  private let preciseAttention: Bool
  private let fusedFeedForward: Bool

  init(
    weights: ValidatedWeights,
    blockIndex: Int,
    preciseAttention: Bool = true,
    fusedFeedForward: Bool = false
  ) throws {
    let blockPrefix = "block\(blockIndex)"
    self.windowOrigin = NeuralRenderingGraphContract.windowOrigin(for: blockIndex)
    self.firstProjectionWeight = try Self.require(
      weights,
      name: "\(blockPrefix).layer0.first_projection_weight",
      shape: [512, 512]
    )
    self.groupExpandWeight = try Self.require(
      weights,
      name: "\(blockPrefix).layer0.group_expand_weight",
      shape: [8, 64, 256]
    )
    self.groupProjectWeight = try Self.require(
      weights,
      name: "\(blockPrefix).layer0.group_project_weight",
      shape: [8, 256, 64]
    )
    self.feedForwardProjectionWeight = try Self.require(
      weights,
      name: "\(blockPrefix).layer1.weight3",
      shape: [512, 512]
    )
    let feedForwardCosine = try Self.require(
      weights,
      name: "\(blockPrefix).layer1.ffn_cos_skip",
      shape: [512]
    )
    self.feedForwardCosine = feedForwardCosine
    self.qkvWeight = try Self.require(
      weights,
      name: "\(blockPrefix).layer2.qkv_weight",
      shape: [512, 1536]
    )
    self.attentionScale = try Self.require(
      weights,
      name: "\(blockPrefix).layer2.attn_scale",
      shape: [16]
    )
    let storedAttentionBias = try Self.require(
      weights,
      name: "\(blockPrefix).layer2.attn_bias",
      shape: [16, 64, 64]
    )
    self.attentionBias =
      NeuralRenderingAttentionBiasLayout.usesFragmentSwizzle(
        blockIndex: blockIndex,
        headCount: 16
      )
      ? NeuralRenderingAttentionBiasLayout.recoverFragmentSwizzle(storedAttentionBias)
      : storedAttentionBias
    self.attentionProjectionWeight = try Self.require(
      weights,
      name: "\(blockPrefix).layer3.projection_weight",
      shape: [512, 512]
    )
    let attentionCosine = try Self.require(
      weights,
      name: "\(blockPrefix).layer3.attn_cos_skip",
      shape: [512]
    )
    self.attentionCosine = attentionCosine
    self.preciseAttention = preciseAttention
    self.fusedFeedForward = fusedFeedForward
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    let fusesOutput = fusedFeedForward && input.dtype == .float16
      && NeuralRenderingTransformerOperations.fusedOutputResidualEnabled
    let output = NeuralRenderingTransformerOperations.splitWindowBlock(
      input,
      firstProjectionWeight: firstProjectionWeight,
      expandWeight: groupExpandWeight,
      projectWeight: groupProjectWeight,
      feedForwardProjectionWeight: feedForwardProjectionWeight,
      feedForwardCosine: feedForwardCosine,
      qkvWeight: qkvWeight,
      attentionScale: attentionScale,
      attentionBias: attentionBias,
      attentionProjectionWeight: attentionProjectionWeight,
      attentionCosine: attentionCosine,
      headCount: 16,
      windowSize: NeuralRenderingGraphContract.windowSize,
      windowOrigin: windowOrigin,
      preciseSoftmax: preciseAttention,
      fusedFeedForward: fusedFeedForward,
      publishedOutput: fusesOutput ? true : nil
    )
    return fusesOutput ? output : NeuralRenderingTransformerOperations.e4m3RoundTrip(output)
  }

  private static func require(
    _ weights: ValidatedWeights,
    name: String,
    shape: [Int]
  ) throws -> MLXArray {
    let value = try weights.required(name)
    guard value.shape == shape else {
      throw MLXBackendError.weightShapeMismatch(
        name: name,
        expected: shape,
        actual: value.shape
      )
    }
    return value
  }
}

struct NeuralRenderingSplitEncoderOutput {
  let skip: MLXArray
  let latent: MLXArray
}

struct NeuralRenderingSplitEncoderStage {
  private let blocks: [NeuralRenderingSplitWindowBlock]
  private let transition: NeuralRenderingSplitWindowBlock
  private let transitionWeight: MLXArray

  init(
    weights: ValidatedWeights,
    preciseAttention: Bool = true,
    fusedFeedForward: Bool = false
  ) throws {
    self.blocks = try (23...29).map { index in
      try NeuralRenderingSplitWindowBlock(
        weights: weights,
        blockIndex: index,
        preciseAttention: preciseAttention,
        fusedFeedForward: fusedFeedForward
      )
    }
    self.transition = try NeuralRenderingSplitWindowBlock(
      weights: weights,
      blockIndex: 30,
      preciseAttention: preciseAttention,
      fusedFeedForward: fusedFeedForward
    )
    let name = "block30.layer4.weight"
    let value = try weights.required(name)
    let expectedShape = [512, 1024]
    guard value.shape == expectedShape else {
      throw MLXBackendError.weightShapeMismatch(
        name: name,
        expected: expectedShape,
        actual: value.shape
      )
    }
    self.transitionWeight = value
  }

  func callAsFunction(_ input: MLXArray) -> NeuralRenderingSplitEncoderOutput {
    let transformed = blocks.reduce(input) { value, block in block(value) }
    let skip = transition(transformed)
    let transitionInput = NeuralRenderingTransformerOperations.padSpatialEnd(
      skip,
      multiple: NeuralRenderingGraphContract.windowSize
    )
    return NeuralRenderingSplitEncoderOutput(
      skip: skip,
      latent: NeuralRenderingTransformerOperations.e4m3RoundTrip(
        NeuralRenderingTransformerOperations.downsample(
          transitionInput,
          weight: transitionWeight
        )
      )
    )
  }
}

struct NeuralRenderingLinearWeight {
  private let weight: MLXArray
  private let scales: MLXArray?
  private let biases: MLXArray?

  init(_ weight: MLXArray, quantize: Bool = false) {
    self.weight = weight
    self.scales = nil
    self.biases = nil
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    if let scales {
      quantizedMM(
        input,
        weight,
        scales: scales,
        biases: biases,
        transpose: false,
        groupSize: 32,
        bits: 8
      )
    } else {
      matmul(input, weight)
    }
  }
}

struct NeuralRenderingGlobalBlock {
  private let body: (MLXArray) -> MLXArray
  private let compiledBody: ((MLXArray) -> MLXArray)?

  init(
    weights: ValidatedWeights,
    blockIndex: Int,
    quantizeFFN: Bool = false,
    preciseAttention: Bool = true,
    fusedOperations: Bool = false,
    compileGraph: Bool = false
  ) throws {
    let prefix = "block\(blockIndex)"
    let expansionWeight = NeuralRenderingLinearWeight(
      try Self.require(
        weights, name: "\(prefix).layer0.weight", shape: [1024, 4096]
      ),
      quantize: quantizeFFN
    )
    let feedForwardProjectionWeight = NeuralRenderingLinearWeight(
      try Self.require(
        weights, name: "\(prefix).layer1.weight", shape: [4096, 1024]
      )
    )
    let feedForwardCosine = try Self.require(
      weights, name: "\(prefix).layer1.ffn_cos_skip", shape: [1024]
    )
    let qkvWeight = try Self.require(
      weights, name: "\(prefix).layer2.qkv_weight", shape: [1024, 3072]
    )
    let attentionScale = try Self.require(
      weights, name: "\(prefix).layer2.attn_scale", shape: [32]
    )
    let attentionProjectionWeight = try Self.require(
      weights,
      name: "\(prefix).layer4.projection_weight",
      shape: [1024, 1024]
    )
    let attentionCosine = try Self.require(
      weights, name: "\(prefix).layer4.attn_cos_skip", shape: [1024]
    )
    let body: (MLXArray) -> MLXArray = { input in
      NeuralRenderingTransformerOperations.e4m3RoundTrip(
        NeuralRenderingTransformerOperations.globalBlock(
          input,
          expansionWeight: expansionWeight,
          feedForwardProjectionWeight: feedForwardProjectionWeight,
          feedForwardCosine: feedForwardCosine,
          qkvWeight: qkvWeight,
          attentionScale: attentionScale,
          attentionBias: nil,
          attentionProjectionWeight: attentionProjectionWeight,
          attentionCosine: attentionCosine,
          headCount: 32,
          logitCap: Self.experimentalLogitCap,
          preciseSoftmax: preciseAttention,
          fusedOperations: fusedOperations
        )
      )
    }
    self.body = body
    if compileGraph && fusedOperations {
      self.compiledBody = compile { (input: MLXArray) -> MLXArray in body(input) }
    } else {
      self.compiledBody = nil
    }
  }

  /// Global attention logit cap; `MLXDLSS_EXPERIMENTAL_GLOBAL_LOGIT_CAP` overrides
  /// the verified default (`none` disables it, for experiments).
  static let experimentalLogitCap: Float? = {
    guard let text = ProcessInfo.processInfo.environment["MLXDLSS_EXPERIMENTAL_GLOBAL_LOGIT_CAP"] else {
      return NeuralRenderingTransformerOperations.globalAttentionLogitCap
    }
    if text.lowercased() == "none" { return nil }
    guard let value = Float(text), value.isFinite else {
      return NeuralRenderingTransformerOperations.globalAttentionLogitCap
    }
    return value
  }()

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    if let compiledBody, input.shape[1] * input.shape[2] <= 512 {
      return compiledBody(input)
    }
    return body(input)
  }

  private static func require(
    _ weights: ValidatedWeights,
    name: String,
    shape: [Int]
  ) throws -> MLXArray {
    let value = try weights.required(name)
    guard value.shape == shape else {
      throw MLXBackendError.weightShapeMismatch(
        name: name,
        expected: shape,
        actual: value.shape
      )
    }
    return value
  }
}

struct NeuralRenderingGlobalStage {
  private let blocks: [NeuralRenderingGlobalBlock]

  init(
    weights: ValidatedWeights,
    quantizeFFN: Bool = false,
    preciseAttention: Bool = true,
    fusedOperations: Bool = false,
    compileGraph: Bool = false
  ) throws {
    self.blocks = try (31...38).map { index in
      try NeuralRenderingGlobalBlock(
        weights: weights,
        blockIndex: index,
        quantizeFFN: quantizeFFN,
        preciseAttention: preciseAttention,
        fusedOperations: fusedOperations,
        compileGraph: compileGraph
      )
    }
  }

  /// Perf-spike diagnostic: `MLXDLSS_GLOBAL_EVAL=0` skips the per-block materialization.
  nonisolated(unsafe) static var perBlockEvalEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_GLOBAL_EVAL"] != "0"

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    let tokenCount = input.shape[1] * input.shape[2]
    // Streamed attention keeps no quadratic intermediates, so the per-block
    // materialization (and the cache clear that slows every later allocation)
    // is only kept for the materialized path above the resident token limit.
    let streamed = NeuralRenderingStreamedGlobalAttention.isEnabled(tokens: tokenCount)
      && tokenCount <= NeuralRenderingStreamedGlobalAttention.residentMaxTokens
    guard tokenCount > 64, Self.perBlockEvalEnabled || !streamed else {
      return blocks.reduce(input) { value, block in block(value) }
    }
    var value = input
    for block in blocks {
      value = block(value)
      // Materialize each block to bound live attention intermediates, while
      // allowing subsequent blocks to reuse their same-sized scratch buffers.
      eval(value)
    }
    Memory.clearCache()
    return value
  }
}

struct NeuralRenderingTrunkOutput {
  let fullResolutionSkip: MLXArray
  let skips: [MLXArray]
  let splitSkip: MLXArray
  let latent: MLXArray
}

struct NeuralRenderingTrunk {
  private let encoder: NeuralRenderingEncoder
  private let splitEncoder: NeuralRenderingSplitEncoderStage
  private let globalStage: NeuralRenderingGlobalStage

  init(
    weights: ValidatedWeights,
    compileBlocks: Bool = false,
    quantizeGlobalFFN: Bool = false
  ) throws {
    self.encoder = try NeuralRenderingEncoder(
      weights: weights,
      compileBlocks: compileBlocks
    )
    self.splitEncoder = try NeuralRenderingSplitEncoderStage(
      weights: weights,
      preciseAttention: true,
      fusedFeedForward: compileBlocks
    )
    self.globalStage = try NeuralRenderingGlobalStage(
      weights: weights,
      quantizeFFN: quantizeGlobalFFN,
      preciseAttention: true,
      fusedOperations: compileBlocks,
      compileGraph: compileBlocks && ProcessInfo.processInfo.environment["MLXDLSS_COMPILE_GLOBAL"] != "0"
    )
  }

  func callAsFunction(_ input: MLXArray) -> NeuralRenderingTrunkOutput {
    let encoded = encoder(input)
    let split = splitEncoder(encoded.latent)
    return NeuralRenderingTrunkOutput(
      fullResolutionSkip: encoded.fullResolutionSkip,
      skips: encoded.skips,
      splitSkip: split.skip,
      latent: globalStage(split.latent)
    )
  }
}

struct NeuralRenderingDecoderInput {
  private let projectionWeight: MLXArray
  private let skipSine: MLXArray

  init(weights: ValidatedWeights) throws {
    let weightName = "block39.layer0.conv_weight"
    let weight = try weights.required(weightName)
    guard weight.shape == [1024, 512] else {
      throw MLXBackendError.weightShapeMismatch(
        name: weightName,
        expected: [1024, 512],
        actual: weight.shape
      )
    }
    let skipSineName = "block39.layer0.inp_upsample_sin"
    let skipSine = try weights.required(skipSineName)
    guard skipSine.shape == [512] else {
      throw MLXBackendError.weightShapeMismatch(
        name: skipSineName,
        expected: [512],
        actual: skipSine.shape
      )
    }
    self.projectionWeight = weight
    self.skipSine = skipSine
  }

  func callAsFunction(_ input: MLXArray, skip: MLXArray) -> MLXArray {
    let projected = matmul(input, projectionWeight)
    if NeuralRenderingFusedGlue.canMerge(low: projected, skip: skip) {
      return NeuralRenderingFusedGlue.upsampleMerge(
        low: projected, skip: skip, lowScale: nil, skipScale: skipSine, publish: true)
    }
    return NeuralRenderingTransformerOperations.e4m3RoundTrip(
      NeuralRenderingTransformerOperations.decoderInputMerge(
        projected,
        skip: skip,
        skipSine: skipSine
      )
    )
  }
}

struct NeuralRenderingUpsampleBlock {
  private let projectionWeight: MLXArray
  private let skipSine: MLXArray
  private let windowBlock: NeuralRenderingWindowBlock

  init(
    weights: ValidatedWeights,
    blockIndex: Int,
    channels: Int,
    hiddenChannels: Int,
    headCount: Int,
    compileBlocks: Bool = false
  ) throws {
    let prefix = "block\(blockIndex).layer0"
    let weightName = "\(prefix).weight0"
    let weight = try weights.required(weightName)
    guard weight.shape == [channels * 2, channels] else {
      throw MLXBackendError.weightShapeMismatch(
        name: weightName,
        expected: [channels * 2, channels],
        actual: weight.shape
      )
    }
    let skipSineName = "\(prefix).sin"
    let skipSine = try weights.required(skipSineName)
    guard skipSine.shape == [channels] else {
      throw MLXBackendError.weightShapeMismatch(
        name: skipSineName,
        expected: [channels],
        actual: skipSine.shape
      )
    }
    self.projectionWeight = weight
    self.skipSine = skipSine
    self.windowBlock = try NeuralRenderingWindowBlock(
      weights: weights,
      blockIndex: blockIndex,
      channels: channels,
      hiddenChannels: hiddenChannels,
      headCount: headCount,
      compileBlock: compileBlocks
    )
  }

  func callAsFunction(_ input: MLXArray, skip: MLXArray) -> MLXArray {
    let projected = matmul(input, projectionWeight)
    // The fused upsample kernels read the merged tensor as E4M3 (vendor
    // decoder captures: blocks 48/56/62 move from 0.037-0.041 to 0.019-0.022).
    if NeuralRenderingFusedGlue.canMerge(low: projected, skip: skip) {
      return windowBlock(
        NeuralRenderingFusedGlue.upsampleMerge(
          low: projected, skip: skip, lowScale: nil, skipScale: skipSine, publish: true))
    }
    let skipPath = skip * skipSine
    let upsampled = NeuralRenderingTransformerOperations.nearestUpsample2Crop(
      projected,
      height: skip.shape[1],
      width: skip.shape[2]
    )
    return windowBlock(
      NeuralRenderingTransformerOperations.e4m3RoundTrip(upsampled + skipPath)
    )
  }
}

struct NeuralRenderingFirstDecoderStage {
  private let splitBlocks: [NeuralRenderingSplitWindowBlock]
  private let upsample: NeuralRenderingUpsampleBlock
  private let blocks: NeuralRenderingWindowSequence

  init(weights: ValidatedWeights, compileBlocks: Bool = false) throws {
    self.splitBlocks = try (40...47).map { index in
      try NeuralRenderingSplitWindowBlock(
        weights: weights,
        blockIndex: index,
        preciseAttention: true,
        fusedFeedForward: compileBlocks
      )
    }
    self.upsample = try NeuralRenderingUpsampleBlock(
      weights: weights,
      blockIndex: 48,
      channels: 256,
      hiddenChannels: 704,
      headCount: 8,
      compileBlocks: compileBlocks
    )
    self.blocks = try NeuralRenderingWindowSequence(
      weights: weights,
      blockIndices: 49...55,
      channels: 256,
      hiddenChannels: 704,
      headCount: 8,
      compileSequence: compileBlocks
    )
  }

  func callAsFunction(_ input: MLXArray, skip: MLXArray) -> MLXArray {
    let split = splitBlocks.reduce(input) { value, block in block(value) }
    return blocks(upsample(split, skip: skip))
  }
}

struct NeuralRenderingDecoderStage {
  private let upsample: NeuralRenderingUpsampleBlock
  private let blocks: NeuralRenderingWindowSequence

  init(
    weights: ValidatedWeights,
    upsampleBlock: Int,
    regularBlocks: ClosedRange<Int>,
    channels: Int,
    hiddenChannels: Int,
    headCount: Int,
    compileBlocks: Bool = false
  ) throws {
    self.upsample = try NeuralRenderingUpsampleBlock(
      weights: weights,
      blockIndex: upsampleBlock,
      channels: channels,
      hiddenChannels: hiddenChannels,
      headCount: headCount,
      compileBlocks: compileBlocks
    )
    self.blocks = try NeuralRenderingWindowSequence(
      weights: weights,
      blockIndices: regularBlocks,
      channels: channels,
      hiddenChannels: hiddenChannels,
      headCount: headCount,
      compileSequence: compileBlocks
    )
  }

  func callAsFunction(_ input: MLXArray, skip: MLXArray) -> MLXArray {
    blocks(upsample(input, skip: skip))
  }
}

struct NeuralRenderingDecoder {
  private let first: NeuralRenderingFirstDecoderStage
  private let second: NeuralRenderingDecoderStage
  private let third: NeuralRenderingDecoderStage
  private let fourth: NeuralRenderingDecoderStage

  init(weights: ValidatedWeights, compileBlocks: Bool = false) throws {
    self.first = try NeuralRenderingFirstDecoderStage(
      weights: weights,
      compileBlocks: compileBlocks
    )
    self.second = try NeuralRenderingDecoderStage(
      weights: weights,
      upsampleBlock: 56,
      regularBlocks: 57...61,
      channels: 128,
      hiddenChannels: 384,
      headCount: 4,
      compileBlocks: compileBlocks
    )
    self.third = try NeuralRenderingDecoderStage(
      weights: weights,
      upsampleBlock: 62,
      regularBlocks: 63...65,
      channels: 64,
      hiddenChannels: 224,
      headCount: 2,
      compileBlocks: compileBlocks
    )
    self.fourth = try NeuralRenderingDecoderStage(
      weights: weights,
      upsampleBlock: 66,
      regularBlocks: 67...69,
      channels: 32,
      hiddenChannels: 128,
      headCount: 1,
      compileBlocks: compileBlocks
    )
  }

  func callAsFunction(_ input: MLXArray, skips: [MLXArray]) -> MLXArray {
    precondition(skips.count == 4)
    let firstOutput = first(input, skip: skips[3])
    let secondOutput = second(firstOutput, skip: skips[2])
    let thirdOutput = third(secondOutput, skip: skips[1])
    return fourth(thirdOutput, skip: skips[0])
  }
}

struct NeuralRenderingPostBlock {
  /// Perf spike: `MLXDLSS_FUSED_HEAD=1` runs the output head as one `[32, 8]`
  /// GEMM whose column halves equal the two 16-wide GEMMs (zero taps add
  /// nothing to the float accumulators), avoiding two full-resolution slice copies.
  nonisolated(unsafe) static var fusedHeadEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_HEAD"] == "1"
  private let sine: MLXArray
  private let cosine: MLXArray
  private let windowBlock: NeuralRenderingWindowBlock
  let outputGain: MLXArray
  let outputConvolution: MLXArray
  let headWeight: MLXArray

  init(weights: ValidatedWeights, compileBlocks: Bool = false) throws {
    let prefix = "block70.layer0"
    self.sine = try Self.require(
      weights, name: "\(prefix).inp_merge_sin", shape: [32]
    )
    self.cosine = try Self.require(
      weights, name: "\(prefix).inp_merge_cos", shape: [32]
    )
    self.windowBlock = try NeuralRenderingWindowBlock(
      weights: weights,
      blockIndex: 70,
      channels: 32,
      hiddenChannels: 128,
      headCount: 1,
      compileBlock: compileBlocks
    )
    let outputGain = try Self.require(
      weights, name: "\(prefix).out_gain", shape: [16, 4]
    )
    let outputConvolution = try Self.require(
      weights, name: "\(prefix).out_conv_weight", shape: [16, 4]
    )
    self.outputGain = outputGain
    self.outputConvolution = outputConvolution
    let zeros = MLXArray.zeros([16, 4], dtype: outputGain.dtype)
    let headWeight = concatenated([
      concatenated([outputGain, zeros], axis: 0),
      concatenated([zeros, outputConvolution], axis: 0),
    ], axis: 1)
    eval(headWeight)
    self.headWeight = headWeight
  }

  func callAsFunction(_ input: MLXArray, skip: MLXArray) -> MLXArray {
    let merged: MLXArray
    if NeuralRenderingFusedGlue.canMerge(low: input, skip: skip) {
      merged = NeuralRenderingFusedGlue.upsampleMerge(
        low: input, skip: skip, lowScale: sine, skipScale: cosine, publish: false)
    } else {
      let upsampled = NeuralRenderingTransformerOperations.nearestUpsample2Crop(
        input,
        height: skip.shape[1],
        width: skip.shape[2]
      )
      merged = NeuralRenderingTransformerOperations.postMerge(
        upsampled,
        skip: skip,
        sine: sine,
        cosine: cosine
      )
    }
    let features = windowBlock(merged)
    return Self.head(features, gain: outputGain, convolution: outputConvolution,
      headWeight: headWeight, fused: Self.fusedHeadEnabled)
  }

  static func head(
    _ features: MLXArray, gain: MLXArray, convolution: MLXArray, headWeight: MLXArray, fused: Bool
  ) -> MLXArray {
    if fused {
      let projected = matmul(features, headWeight)
      return projected[0..., 0..., 0..., 0..<4] + projected[0..., 0..., 0..., 4..<8]
    }
    let first = features[0..., 0..., 0..., 0..<16]
    let second = features[0..., 0..., 0..., 16..<32]
    return matmul(first, gain) + matmul(second, convolution)
  }

  private static func require(
    _ weights: ValidatedWeights,
    name: String,
    shape: [Int]
  ) throws -> MLXArray {
    let value = try weights.required(name)
    guard value.shape == shape else {
      throw MLXBackendError.weightShapeMismatch(
        name: name,
        expected: shape,
        actual: value.shape
      )
    }
    return value
  }
}

struct NeuralRenderingTransformerModel {
  private let trunk: NeuralRenderingTrunk
  private let decoderInput: NeuralRenderingDecoderInput
  private let decoder: NeuralRenderingDecoder
  private let post: NeuralRenderingPostBlock

  init(
    weights: ValidatedWeights,
    compileBlocks: Bool = false,
    quantizeGlobalFFN: Bool = false
  ) throws {
    self.trunk = try NeuralRenderingTrunk(
      weights: weights,
      compileBlocks: compileBlocks,
      quantizeGlobalFFN: quantizeGlobalFFN
    )
    self.decoderInput = try NeuralRenderingDecoderInput(weights: weights)
    self.decoder = try NeuralRenderingDecoder(
      weights: weights,
      compileBlocks: compileBlocks
    )
    self.post = try NeuralRenderingPostBlock(
      weights: weights,
      compileBlocks: compileBlocks
    )
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    let encoded = trunk(input)
    let decoderStart = decoderInput(encoded.latent, skip: encoded.splitSkip)
    let decoded = decoder(decoderStart, skips: encoded.skips)
    return post(decoded, skip: encoded.fullResolutionSkip)
  }
}

struct NeuralRenderingPreBlock {
  private let inputAdapterWeight: MLXArray
  private let windowBlock: NeuralRenderingWindowBlock

  init(
    weights: ValidatedWeights,
    inputChannels: Int = 16,
    channels: Int = 32,
    hiddenChannels: Int = 128,
    headCount: Int = 1,
    compileBlocks: Bool = false
  ) throws {
    let name = "block0.layer0.input_adapter_weight"
    let adapter = try weights.required(name)
    let expectedShape = [inputChannels, channels]
    guard adapter.shape == expectedShape else {
      throw MLXBackendError.weightShapeMismatch(
        name: name,
        expected: expectedShape,
        actual: adapter.shape
      )
    }
    self.inputAdapterWeight = adapter
    self.windowBlock = try NeuralRenderingWindowBlock(
      weights: weights,
      blockIndex: 0,
      channels: channels,
      hiddenChannels: hiddenChannels,
      headCount: headCount,
      compileBlock: compileBlocks
    )
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    precondition(input.ndim == 4)
    precondition(input.shape.last == inputAdapterWeight.shape[0])
    return transform(project(input))
  }

  func project(_ input: MLXArray) -> MLXArray {
    precondition(input.ndim == 4)
    precondition(input.shape.last == inputAdapterWeight.shape[0])
    return matmul(input, inputAdapterWeight)
  }

  func transform(_ input: MLXArray) -> MLXArray {
    windowBlock(input)
  }
}

struct NeuralRenderingEncoderStemOutput {
  let fullResolutionSkip: MLXArray
  let skip: MLXArray
}

struct NeuralRenderingEncoderStem {
  private let pre: NeuralRenderingPreBlock
  private let blocks: NeuralRenderingWindowSequence

  init(weights: ValidatedWeights, compileBlocks: Bool = false) throws {
    self.pre = try NeuralRenderingPreBlock(
      weights: weights,
      compileBlocks: compileBlocks
    )
    self.blocks = try NeuralRenderingWindowSequence(
      weights: weights,
      blockIndices: 1...3,
      channels: 32,
      hiddenChannels: 128,
      headCount: 1,
      compileSequence: compileBlocks
    )
  }

  func callAsFunction(_ input: MLXArray) -> NeuralRenderingEncoderStemOutput {
    // The fused pre kernel runs the block-0 window attention on the full
    // adapter output, then 2x2-average-pools and publishes once; vendor
    // block-0 captures move from 0.0254 to 0.0075 MAE with this order.
    // The full-resolution skip the post block merges is that block-0 output
    // published as E4M3, not the raw adapter output: the adapter output
    // carries the noise channels straight into the head and shows up as
    // pixel-level grain (four DLL goldens at 256: MAE 0.0118 -> 0.0055,
    // high-pass correlation 0.52 -> 0.94; five native crops: 0.0073-0.0113
    // -> 0.0037-0.0046).
    let block0Output = pre.transform(pre.project(input))
    if NeuralRenderingFusedGlue.canPool(block0Output) {
      let published = NeuralRenderingFusedGlue.publishAndPool(block0Output, emitFull: true)
      return NeuralRenderingEncoderStemOutput(
        fullResolutionSkip: published.full,
        skip: blocks(published.pooled)
      )
    }
    let fullResolutionSkip = NeuralRenderingTransformerOperations.e4m3RoundTrip(
      block0Output
    )
    let pooled = NeuralRenderingTransformerOperations.e4m3RoundTrip(
      NeuralRenderingTransformerOperations.averagePool2(block0Output)
    )
    return NeuralRenderingEncoderStemOutput(
      fullResolutionSkip: fullResolutionSkip,
      skip: blocks(pooled)
    )
  }
}

struct NeuralRenderingDownsampleBlock {
  private let windowBlock: NeuralRenderingWindowBlock
  private let weight: MLXArray
  private let padsForSplitTransition: Bool

  init(
    weights: ValidatedWeights,
    blockIndex: Int,
    channels: Int,
    hiddenChannels: Int,
    headCount: Int,
    compileBlocks: Bool = false
  ) throws {
    self.windowBlock = try NeuralRenderingWindowBlock(
      weights: weights,
      blockIndex: blockIndex,
      channels: channels,
      hiddenChannels: hiddenChannels,
      headCount: headCount,
      compileBlock: compileBlocks
    )
    self.padsForSplitTransition = blockIndex == 22
    let name = "block\(blockIndex).layer0.weight0"
    let value = try weights.required(name)
    let expectedShape = [channels, channels * 2]
    guard value.shape == expectedShape else {
      throw MLXBackendError.weightShapeMismatch(
        name: name,
        expected: expectedShape,
        actual: value.shape
      )
    }
    self.weight = value
  }

  func callAsFunction(_ input: MLXArray) -> MLXArray {
    // The fused ``ds`` kernels pool the block's unpublished half-precision
    // output, publish the pooled tensor as E4M3, project it with QMMA and
    // publish again (vendor transition captures: 0.010-0.021 versus
    // 0.026-0.031 for pool-then-project on the published tensor).
    let transformed = windowBlock.unpublished(input)
    let downsampleInput =
      padsForSplitTransition
      ? NeuralRenderingTransformerOperations.padSpatialEnd(
        transformed,
        multiple: NeuralRenderingGraphContract.windowSize
      )
      : transformed
    let pooled =
      NeuralRenderingFusedGlue.canPool(downsampleInput)
      ? NeuralRenderingFusedGlue.publishAndPool(downsampleInput, emitFull: false).pooled
      : NeuralRenderingTransformerOperations.e4m3RoundTrip(
        NeuralRenderingTransformerOperations.averagePool2(downsampleInput)
      )
    return NeuralRenderingTransformerOperations.e4m3RoundTrip(matmul(pooled, weight))
  }
}

struct NeuralRenderingStageOutput {
  let skip: MLXArray
  let downsampled: MLXArray
}

struct NeuralRenderingFirstStageOutput {
  let fullResolutionSkip: MLXArray
  let skip: MLXArray
  let downsampled: MLXArray
}

struct NeuralRenderingFirstEncoderStage {
  private let stem: NeuralRenderingEncoderStem
  private let transition: NeuralRenderingDownsampleBlock

  init(weights: ValidatedWeights, compileBlocks: Bool = false) throws {
    self.stem = try NeuralRenderingEncoderStem(
      weights: weights,
      compileBlocks: compileBlocks
    )
    self.transition = try NeuralRenderingDownsampleBlock(
      weights: weights,
      blockIndex: 4,
      channels: 32,
      hiddenChannels: 128,
      headCount: 1,
      compileBlocks: compileBlocks
    )
  }

  func callAsFunction(_ input: MLXArray) -> NeuralRenderingFirstStageOutput {
    let stemOutput = stem(input)
    return NeuralRenderingFirstStageOutput(
      fullResolutionSkip: stemOutput.fullResolutionSkip,
      skip: stemOutput.skip,
      downsampled: transition(stemOutput.skip)
    )
  }
}

struct NeuralRenderingEncoderStage {
  private let blocks: NeuralRenderingWindowSequence
  private let transition: NeuralRenderingDownsampleBlock

  init(
    weights: ValidatedWeights,
    regularBlocks: ClosedRange<Int>,
    transitionBlock: Int,
    channels: Int,
    hiddenChannels: Int,
    headCount: Int,
    compileBlocks: Bool = false
  ) throws {
    self.blocks = try NeuralRenderingWindowSequence(
      weights: weights,
      blockIndices: regularBlocks,
      channels: channels,
      hiddenChannels: hiddenChannels,
      headCount: headCount,
      compileSequence: compileBlocks
    )
    self.transition = try NeuralRenderingDownsampleBlock(
      weights: weights,
      blockIndex: transitionBlock,
      channels: channels,
      hiddenChannels: hiddenChannels,
      headCount: headCount,
      compileBlocks: compileBlocks
    )
  }

  func callAsFunction(_ input: MLXArray) -> NeuralRenderingStageOutput {
    let skip = blocks(input)
    return NeuralRenderingStageOutput(
      skip: skip,
      downsampled: transition(skip)
    )
  }
}

struct NeuralRenderingEncoderOutput {
  let fullResolutionSkip: MLXArray
  let skips: [MLXArray]
  let latent: MLXArray
}

struct NeuralRenderingEncoder {
  private let first: NeuralRenderingFirstEncoderStage
  private let second: NeuralRenderingEncoderStage
  private let third: NeuralRenderingEncoderStage
  private let fourth: NeuralRenderingEncoderStage

  init(weights: ValidatedWeights, compileBlocks: Bool = false) throws {
    self.first = try NeuralRenderingFirstEncoderStage(
      weights: weights,
      compileBlocks: compileBlocks
    )
    self.second = try NeuralRenderingEncoderStage(
      weights: weights,
      regularBlocks: 5...7,
      transitionBlock: 8,
      channels: 64,
      hiddenChannels: 224,
      headCount: 2,
      compileBlocks: compileBlocks
    )
    self.third = try NeuralRenderingEncoderStage(
      weights: weights,
      regularBlocks: 9...13,
      transitionBlock: 14,
      channels: 128,
      hiddenChannels: 384,
      headCount: 4,
      compileBlocks: compileBlocks
    )
    self.fourth = try NeuralRenderingEncoderStage(
      weights: weights,
      regularBlocks: 15...21,
      transitionBlock: 22,
      channels: 256,
      hiddenChannels: 704,
      headCount: 8,
      compileBlocks: compileBlocks
    )
  }

  func callAsFunction(_ input: MLXArray) -> NeuralRenderingEncoderOutput {
    let firstOutput = first(input)
    let secondOutput = second(firstOutput.downsampled)
    let thirdOutput = third(secondOutput.downsampled)
    let fourthOutput = fourth(thirdOutput.downsampled)
    return NeuralRenderingEncoderOutput(
      fullResolutionSkip: firstOutput.fullResolutionSkip,
      skips: [
        firstOutput.skip,
        secondOutput.skip,
        thirdOutput.skip,
        fourthOutput.skip,
      ],
      latent: fourthOutput.downsampled
    )
  }
}

enum NeuralRenderingTransformerOperations {
  private static let cosineNormFloor: Float = 0.000_061_988_830_566_406_25
  /// The global (`vit_1d`) attention kernels saturate their logits at +3.
  /// Verified teacher-forced on every global block with the 2026-09-02
  /// per-launch captures (blocks 31–38: rel 0.046–0.144 → 0.032–0.093, optimum
  /// 3.0 on each block); window-block kernels show no cap.
  static let globalAttentionLogitCap: Float = 3
  static let experimentalGlobalAttentionLogitCap: Float = globalAttentionLogitCap
  /// Perf spike: `MLXDLSS_FAST_E4M3=1` rounds half inputs with 16-bit integer
  /// arithmetic (exhaustively verified bit-exact against the float path).
  nonisolated(unsafe) static var fastE4M3Enabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_FAST_E4M3"] == "1"
  static var e4m3MetalHeaderText: String { fastE4M3Enabled ? fastE4M3MetalHeader : e4m3MetalHeader }
  static var e4m3ReferenceMetalHeaderText: String { e4m3MetalHeader }
  static var e4m3FastMetalHeaderText: String { fastE4M3MetalHeader }

  private static let e4m3MetalHeader = #"""
    struct mlxdlss_e4m3 {
      template <typename T>
      mlxdlss_e4m3(T value) {
        uint32_t fp8Max = 543 << 21;
        uint32_t denormMask = 141 << 23;
        uint32_t valueBits = as_type<uint32_t>(static_cast<float>(value));
        uint32_t sign = valueBits & 0x80000000;
        valueBits ^= sign;
        if (valueBits >= fp8Max) {
          bits = 0x7E;
        } else if (valueBits < (121 << 23)) {
          valueBits = as_type<uint32_t>(
            as_type<float>(valueBits) + as_type<float>(denormMask)
          );
          bits = static_cast<uint8_t>(valueBits - denormMask);
        } else {
          uint8_t mantissaOdd = (valueBits >> 20) & 1;
          valueBits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
          valueBits += mantissaOdd;
          bits = static_cast<uint8_t>(valueBits >> 20);
        }
        bits |= static_cast<uint8_t>(sign >> 24);
      }

      operator float() {
        uint16_t value = (bits & 127) << 7;
        half converted = as_type<half>(value);
        converted *= 256.0;
        return float((bits & 128) ? -converted : converted);
      }

      uint8_t bits;
    };
    """#
  private static let fastE4M3MetalHeader = #"""
    struct mlxdlss_e4m3 {
      // Half inputs: saturate at 448, round-to-nearest-even at three mantissa bits,
      // and quantise the E4M3 subnormal range (< 2^-6) to multiples of 2^-9 with
      // the half magic-number add. Same value as the float path for every half.
      mlxdlss_e4m3(half value) {
        ushort b = as_type<ushort>(value);
        ushort m = b & 0x7FFF;
        // NaN inputs saturate to +448 like the float path (their sign is dropped by the conversion).
        ushort s = m > 0x7C00 ? ushort(0) : ushort(b & 0x8000);
        ushort normal = ushort((m + 0x3F + ((m >> 7) & 1)) & 0xFF80);
        half small = (as_type<half>(m) + 2.0h) - 2.0h;
        ushort r = m >= 0x5F00 ? ushort(0x5F00) : (m < 0x2400 ? as_type<ushort>(small) : normal);
        rounded = as_type<half>(ushort(r | s));
      }
      mlxdlss_e4m3(float value) {
        uint32_t fp8Max = 543 << 21;
        uint32_t denormMask = 141 << 23;
        uint32_t valueBits = as_type<uint32_t>(value);
        uint32_t sign = valueBits & 0x80000000;
        valueBits ^= sign;
        uint8_t bits;
        if (valueBits >= fp8Max) {
          bits = 0x7E;
        } else if (valueBits < (121 << 23)) {
          valueBits = as_type<uint32_t>(
            as_type<float>(valueBits) + as_type<float>(denormMask)
          );
          bits = static_cast<uint8_t>(valueBits - denormMask);
        } else {
          uint8_t mantissaOdd = (valueBits >> 20) & 1;
          valueBits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
          valueBits += mantissaOdd;
          bits = static_cast<uint8_t>(valueBits >> 20);
        }
        uint16_t halfBits = (bits & 127) << 7;
        half converted = as_type<half>(halfBits);
        converted *= 256.0;
        rounded = sign ? -converted : converted;
      }

      operator float() {
        return float(rounded);
      }

      half rounded;
    };
    """#
  private static let e4m3RoundTripKernel = MLXFast.metalKernel(
    name: "mlxdlss_e4m3_round_trip",
    inputNames: ["input"],
    outputNames: ["output"],
    source: #"""
      uint element = thread_position_in_grid.x;
      if (element >= uint(elementCount)) {
        return;
      }
      output[element] = float(mlxdlss_e4m3(input[element]));
      """#,
    header: e4m3MetalHeader
  )
  private static let vendorSoftmaxKernel = MLXFast.metalKernel(
    name: "mlxdlss_vendor_softmax",
    inputNames: ["input"],
    outputNames: ["output"],
    source: #"""
      uint row = thread_position_in_grid.x;
      if (row >= uint(rowCount)) {
        return;
      }
      uint base = row * uint(tokenCount);
      half total = 0.0h;
      for (uint token = 0; token < uint(tokenCount); token += 2) {
        half2 score = half2(input[base + token], input[base + token + 1]);
        half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
        affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
        ushort2 bits = as_type<ushort2>(affine);
        uint packed = uint(bits.x) | (uint(bits.y) << 16);
        uint transformed = (packed << 5) + 0x7FF88000;
        half2 weight = as_type<half2>(
          ushort2(ushort(transformed), ushort(transformed >> 16))
        );
        total += weight.x;
        total += weight.y;
      }
      half reciprocal = half(1.0f / float(total));
      for (uint token = 0; token < uint(tokenCount); token += 2) {
        half2 score = half2(input[base + token], input[base + token + 1]);
        half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
        affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
        ushort2 bits = as_type<ushort2>(affine);
        uint packed = uint(bits.x) | (uint(bits.y) << 16);
        uint transformed = (packed << 5) + 0x7FF88000;
        half2 weight = as_type<half2>(
          ushort2(ushort(transformed), ushort(transformed >> 16))
        );
        output[base + token] = float(mlxdlss_e4m3(weight.x * reciprocal));
        output[base + token + 1] = float(mlxdlss_e4m3(weight.y * reciprocal));
      }
      """#,
    header: e4m3MetalHeader
  )
  private static let vendorCosineNormalize32Kernel = MLXFast.metalKernel(
    name: "mlxdlss_vendor_cosine_normalize_32",
    inputNames: ["input", "scale"],
    outputNames: ["output"],
    source: #"""
      uint vector = thread_position_in_grid.x;
      if (vector >= uint(vectorCount)) {
        return;
      }
      uint base = vector * 32;
      half value[32];
      for (uint channel = 0; channel < 32; ++channel) {
        value[channel] = half(input[base + channel]);
      }
      half partial[4][2];
      for (uint lane = 0; lane < 4; ++lane) {
        for (uint parity = 0; parity < 2; ++parity) {
          uint channel = lane * 2 + parity;
          half first = fma(
            value[channel + 8],
            value[channel + 8],
            value[channel] * value[channel]
          );
          half second = fma(
            value[channel + 24],
            value[channel + 24],
            value[channel + 16] * value[channel + 16]
          );
          partial[lane][parity] = first + second;
        }
      }
      half xorTwo[4][2];
      for (uint lane = 0; lane < 4; ++lane) {
        for (uint parity = 0; parity < 2; ++parity) {
          xorTwo[lane][parity] = partial[lane][parity]
            + partial[lane ^ 2][parity];
        }
      }
      half xorOne[4][2];
      for (uint lane = 0; lane < 4; ++lane) {
        for (uint parity = 0; parity < 2; ++parity) {
          xorOne[lane][parity] = xorTwo[lane][parity]
            + xorTwo[lane ^ 1][parity];
        }
      }
      half norm = max(
        xorOne[0][0] + xorOne[0][1],
        half(0.00006198883056640625)
      );
      half reciprocal = half(metal::fast::rsqrt(float(norm)));
      for (uint channel = 0; channel < 32; ++channel) {
        half normalized = value[channel] * reciprocal;
        if (hasScale) {
          uint head = (vector / uint(tokenCount)) % uint(headCount);
          normalized *= half(scale[head]);
        }
        output[base + channel] = publish
          ? float(mlxdlss_e4m3(normalized))
          : normalized;
      }
      """#,
    header: e4m3MetalHeader
  )
  private static let quadraticGateKernel = MLXFast.metalKernel(
    name: "mlxdlss_quadratic_gate",
    inputNames: ["input"],
    outputNames: ["output"],
    source: #"""
      uint element = thread_position_in_grid.x;
      if (element >= uint(elementCount)) {
        return;
      }
      half value = half(input[element]);
      half clamped = clamp(value, half(-4.0), half(4.0));
      half linear = fma(
        abs(clamped),
        half(-0.055908203125),
        half(0.447265625)
      );
      half gate = fma(clamped, linear, half(0.89453125));
      output[element] = float(activate == 0 ? gate : value * gate);
      """#
  )
  private static let branchedExpansionKernel = MLXFast.metalKernel(
    name: "mlxdlss_branched_expansion",
    inputNames: ["input", "weight"],
    outputNames: ["output"],
    source: #"""
      uint tile = (threadgroup_position_in_grid.z * uint(groupRows)
        + threadgroup_position_in_grid.y) * 8
        + simdgroup_index_in_threadgroup;
      if (tile >= uint(tileCount)) {
        return;
      }
      uint outputTileCount = uint(groupCount * 16);
      uint rowBase = (tile / outputTileCount) * 8;
      uint outputTile = tile % outputTileCount;
      uint outputHead = outputTile / 16;
      uint branch = (outputTile / 4) % 4;
      uint columnBase = (outputTile % 4) * 8;
      simdgroup_matrix<T, 8, 8> accumulator;
      accumulator.thread_elements()[0] = 0.0f;
      accumulator.thread_elements()[1] = 0.0f;
      for (uint inputHead = 0; inputHead < uint(groupCount); ++inputHead) {
        for (uint kTile = 0; kTile < 4; ++kTile) {
          simdgroup_matrix<T, 8, 8> left;
          simdgroup_matrix<T, 8, 8> right;
          uint inputBase = rowBase * uint(channelCount)
            + inputHead * 32 + kTile * 8;
          uint weightBase = (((outputHead * 4 + branch) * uint(groupCount)
            + inputHead) * 32 + kTile * 8) * 32 + columnBase;
          simdgroup_load(left, input + inputBase, uint(channelCount), ulong2(0), false);
          simdgroup_load(right, weight + weightBase, 32, ulong2(0), false);
          simdgroup_multiply_accumulate(
            accumulator, left, right, accumulator
          );
        }
      }
      uint outputStride = uint(groupCount * 4 * 32);
      uint outputBase = rowBase * outputStride
        + (outputHead * 4 + branch) * 32 + columnBase;
      simdgroup_store(
        accumulator, output + outputBase, outputStride, ulong2(0), false
      );
      """#,
    header: "#include <metal_simdgroup_matrix>\n"
  )
  private static let branchedProjectionKernel = MLXFast.metalKernel(
    name: "mlxdlss_branched_projection",
    inputNames: ["input", "weight"],
    outputNames: ["output"],
    source: #"""
      uint tile = (threadgroup_position_in_grid.z * uint(groupRows)
        + threadgroup_position_in_grid.y) * 8
        + simdgroup_index_in_threadgroup;
      if (tile >= uint(tileCount)) {
        return;
      }
      uint outputTileCount = uint(groupCount * 4);
      uint rowBase = (tile / outputTileCount) * 8;
      uint outputTile = tile % outputTileCount;
      uint outputHead = outputTile / 4;
      uint columnBase = (outputTile % 4) * 8;
      simdgroup_matrix<T, 8, 8> branches[4];
      for (uint branch = 0; branch < 4; ++branch) {
        branches[branch].thread_elements()[0] = 0.0f;
        branches[branch].thread_elements()[1] = 0.0f;
        for (uint kTile = 0; kTile < 4; ++kTile) {
          simdgroup_matrix<T, 8, 8> left;
          simdgroup_matrix<T, 8, 8> right;
          uint inputStride = uint(groupCount * 4 * 32);
          uint inputBase = rowBase * inputStride
            + (outputHead * 4 + branch) * 32 + kTile * 8;
          uint weightBase = ((outputHead * 4 + branch) * 32
            + kTile * 8) * 32 + columnBase;
          simdgroup_load(left, input + inputBase, inputStride, ulong2(0), false);
          simdgroup_load(right, weight + weightBase, 32, ulong2(0), false);
          simdgroup_multiply_accumulate(
            branches[branch], left, right, branches[branch]
          );
        }
      }
      simdgroup_matrix<T, 8, 8> merged;
      merged.thread_elements()[0] = branches[0].thread_elements()[0]
        + branches[1].thread_elements()[0]
        + branches[2].thread_elements()[0]
        + branches[3].thread_elements()[0];
      merged.thread_elements()[1] = branches[0].thread_elements()[1]
        + branches[1].thread_elements()[1]
        + branches[2].thread_elements()[1]
        + branches[3].thread_elements()[1];
      uint outputStride = uint(groupCount * 32);
      uint outputBase = rowBase * outputStride
        + outputHead * 32 + columnBase;
      simdgroup_store(
        merged, output + outputBase, outputStride, ulong2(0), false
      );
      """#,
    header: "#include <metal_simdgroup_matrix>\n"
  )
  private static let feedForwardExpansionGateKernel = MLXFast.metalKernel(
    name: "mlxdlss_feed_forward_expansion_gate",
    inputNames: ["input", "weight"],
    outputNames: ["output"],
    source: #"""
      uint tile = (threadgroup_position_in_grid.z * uint(groupRows)
        + threadgroup_position_in_grid.y) * 8
        + simdgroup_index_in_threadgroup;
      if (tile >= uint(tileCount)) {
        return;
      }
      uint columnTileCount = uint(hiddenChannels / 8);
      uint rowBase = (tile / columnTileCount) * 8;
      uint columnBase = (tile % columnTileCount) * 8;
      simdgroup_matrix<T, 8, 8> result;
      result.thread_elements()[0] = 0.0f;
      result.thread_elements()[1] = 0.0f;
      for (uint kTile = 0; kTile < uint(inputChannels / 8); ++kTile) {
        simdgroup_matrix<T, 8, 8> left;
        simdgroup_matrix<T, 8, 8> right;
        simdgroup_load(
          left,
          input + rowBase * uint(inputChannels) + kTile * 8,
          uint(inputChannels),
          ulong2(0),
          false
        );
        simdgroup_load(
          right,
          weight + kTile * 8 * uint(hiddenChannels) + columnBase,
          uint(hiddenChannels),
          ulong2(0),
          false
        );
        simdgroup_multiply_accumulate(result, left, right, result);
      }
      for (uint element = 0; element < 2; ++element) {
        half value = half(result.thread_elements()[element]);
        half clamped = clamp(value, half(-4.0), half(4.0));
        half linear = fma(
          abs(clamped), half(-0.055908203125), half(0.447265625)
        );
        half gate = fma(clamped, linear, half(0.89453125));
        result.thread_elements()[element] = float(value * gate);
      }
      simdgroup_store(
        result,
        output + rowBase * uint(hiddenChannels) + columnBase,
        uint(hiddenChannels),
        ulong2(0),
        false
      );
      """#,
    header: "#include <metal_simdgroup_matrix>\n"
  )
  private static let feedForwardProjectionKernel = MLXFast.metalKernel(
    name: "mlxdlss_feed_forward_projection",
    inputNames: ["input", "weight"],
    outputNames: ["output"],
    source: #"""
      uint tile = (threadgroup_position_in_grid.z * uint(groupRows)
        + threadgroup_position_in_grid.y) * 8
        + simdgroup_index_in_threadgroup;
      if (tile >= uint(tileCount)) {
        return;
      }
      uint columnTileCount = uint(outputChannels / 8);
      uint rowBase = (tile / columnTileCount) * 8;
      uint columnBase = (tile % columnTileCount) * 8;
      simdgroup_matrix<T, 8, 8> result;
      result.thread_elements()[0] = 0.0f;
      result.thread_elements()[1] = 0.0f;
      for (uint kTile = 0; kTile < uint(hiddenChannels / 8); ++kTile) {
        simdgroup_matrix<T, 8, 8> left;
        simdgroup_matrix<T, 8, 8> right;
        simdgroup_load(
          left,
          input + rowBase * uint(hiddenChannels) + kTile * 8,
          uint(hiddenChannels),
          ulong2(0),
          false
        );
        simdgroup_load(
          right,
          weight + kTile * 8 * uint(outputChannels) + columnBase,
          uint(outputChannels),
          ulong2(0),
          false
        );
        simdgroup_multiply_accumulate(result, left, right, result);
      }
      simdgroup_store(
        result,
        output + rowBase * uint(outputChannels) + columnBase,
        uint(outputChannels),
        ulong2(0),
        false
      );
      """#,
    header: "#include <metal_simdgroup_matrix>\n"
  )

  static func quadraticGate(_ input: MLXArray) -> MLXArray {
    quadraticGateKernel(
      [input],
      template: [
        ("elementCount", input.size),
        ("activate", 0),
      ],
      grid: (input.size, 1, 1),
      threadGroup: (min(input.size, 256), 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [input.dtype]
    )[0]
  }

  static func quadraticGateActivation(_ input: MLXArray) -> MLXArray {
    quadraticGateKernel(
      [input],
      template: [
        ("elementCount", input.size),
        ("activate", 1),
      ],
      grid: (input.size, 1, 1),
      threadGroup: (min(input.size, 256), 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [input.dtype]
    )[0]
  }

  /// Perf spike: `MLXDLSS_E4M3_VEC=1` publishes half tensors four elements per thread.
  nonisolated(unsafe) static var vectorizedE4M3Enabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_E4M3_VEC"] == "1"
  private static let e4m3RoundTripHalf4Kernel = MLXFast.metalKernel(
    name: "mlxdlss_e4m3_round_trip_half4",
    inputNames: ["input", "params"],
    outputNames: ["output"],
    source: #"""
      const uint index = thread_position_in_grid.x;
      if (index >= params[0]) { return; }
      const device half4* input4 = (const device half4*)input;
      device half4* output4 = (device half4*)output;
      half4 value = input4[index];
      output4[index] = half4(
        half(float(mlxdlss_e4m3(value.x))), half(float(mlxdlss_e4m3(value.y))),
        half(float(mlxdlss_e4m3(value.z))), half(float(mlxdlss_e4m3(value.w))));
      """#,
    header: e4m3MetalHeader
  )

  static func e4m3RoundTrip(_ input: MLXArray) -> MLXArray {
    precondition(input.dtype == .float16 || input.dtype == .float32)
    if vectorizedE4M3Enabled, input.dtype == .float16, input.size.isMultiple(of: 4), input.size > 0 {
      let count = input.size / 4
      return e4m3RoundTripHalf4Kernel(
        [contiguous(input), NeuralRenderingKernelParameters.array([UInt32(count)])],
        grid: (count, 1, 1),
        threadGroup: (min(count, 256), 1, 1),
        outputShapes: [input.shape],
        outputDTypes: [.float16]
      )[0]
    }
    return e4m3RoundTripKernel(
      [input],
      template: [("elementCount", input.size)],
      grid: (input.size, 1, 1),
      threadGroup: (min(input.size, 256), 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [input.dtype]
    )[0]
  }

  static func vendorApproximateSoftmax(_ input: MLXArray) -> MLXArray {
    precondition(input.ndim > 0)
    let tokenCount = input.shape.last!
    precondition(tokenCount > 0 && tokenCount.isMultiple(of: 2))
    let rowCount = input.size / tokenCount
    return vendorSoftmaxKernel(
      [input],
      template: [
        ("rowCount", rowCount),
        ("tokenCount", tokenCount),
      ],
      grid: (rowCount, 1, 1),
      threadGroup: (min(rowCount, 256), 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [input.dtype]
    )[0]
  }

  static func vendorCosineNormalize(_ input: MLXArray) -> MLXArray {
    precondition(input.ndim > 0)
    let half = input.asType(.float16)
    guard input.shape.last == 32 else {
      let squaredNorm = sum(square(half), axis: -1, keepDims: true)
      return (half * rsqrt(maximum(squaredNorm, cosineNormFloor))).asType(input.dtype)
    }
    let vectorCount = input.size / 32
    return vendorCosineNormalize32Kernel(
      [half, MLXArray.ones([1], dtype: .float16)],
      template: [
        ("vectorCount", vectorCount),
        ("publish", false),
        ("hasScale", false),
        ("headCount", 1),
        ("tokenCount", 1),
      ],
      grid: (vectorCount, 1, 1),
      threadGroup: (min(vectorCount, 256), 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [.float16]
    )[0].asType(input.dtype)
  }

  static func vendorCosinePublish(
    _ input: MLXArray,
    scale: MLXArray? = nil
  ) -> MLXArray {
    precondition(input.ndim == 4)
    let headCount = input.shape[1]
    let tokenCount = input.shape[2]
    if let scale {
      precondition(scale.shape == [headCount])
    }
    guard input.shape.last == 32 else {
      let normalized = vendorCosineNormalize(input)
      return e4m3RoundTrip(
        scale.map { normalized * $0.reshaped([1, headCount, 1, 1]) }
          ?? normalized
      )
    }
    let vectorCount = input.size / 32
    return vendorCosineNormalize32Kernel(
      [input.asType(.float16), scale ?? MLXArray.ones([1], dtype: .float16)],
      template: [
        ("vectorCount", vectorCount),
        ("publish", true),
        ("hasScale", scale != nil),
        ("headCount", headCount),
        ("tokenCount", tokenCount),
      ],
      grid: (vectorCount, 1, 1),
      threadGroup: (min(vectorCount, 256), 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [input.dtype]
    )[0]
  }

  static func cosineResidual(
    skip: MLXArray,
    branch: MLXArray,
    cosine: MLXArray
  ) -> MLXArray {
    precondition(skip.shape == branch.shape)
    precondition(cosine.ndim == 1)
    precondition(skip.shape.last == cosine.shape[0])
    return branch + skip * cosine
  }

  static func cosineAttention(
    _ input: MLXArray,
    qkvWeight: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray?,
    projectionWeight: MLXArray,
    headCount: Int,
    logitCap: Float? = nil,
    symmetricLogitCap: Bool = false,
    preciseSoftmax: Bool = true,
    streamedGlobalAttention: Bool = false
  ) -> MLXArray {
    precondition(input.ndim == 3)
    precondition(headCount > 0)
    let batchCount = input.shape[0]
    let tokenCount = input.shape[1]
    let channels = input.shape[2]
    precondition(channels.isMultiple(of: headCount))
    precondition(qkvWeight.shape == [channels, channels * 3])
    precondition(attentionScale.shape == [headCount])
    if let attentionBias {
      precondition(attentionBias.shape == [headCount, tokenCount, tokenCount])
    }
    precondition(projectionWeight.shape == [channels, channels])

    let projected = matmul(input, qkvWeight)
    let qkv = split(projected, parts: 3, axis: -1)
    let headChannels = channels / headCount
    let headShape = [batchCount, tokenCount, headCount, headChannels]
    let query = qkv[0].reshaped(headShape).transposed(0, 2, 1, 3)
    let key = qkv[1].reshaped(headShape).transposed(0, 2, 1, 3)
    let value = qkv[2].reshaped(headShape).transposed(0, 2, 1, 3)
    let normalizedQuery =
      preciseSoftmax
      ? vendorCosinePublish(query, scale: attentionScale)
      : e4m3RoundTrip(
        frameworkCosineNormalize(query)
          * attentionScale.reshaped([1, headCount, 1, 1])
      )
    let normalizedKey =
      preciseSoftmax
      ? vendorCosinePublish(key)
      : e4m3RoundTrip(frameworkCosineNormalize(key))
    let publishedValue = e4m3RoundTrip(value)
    if streamedGlobalAttention, preciseSoftmax, attentionBias == nil,
      symmetricLogitCap, logitCap == 3, headChannels == 32,
      normalizedQuery.dtype == .float16, tokenCount.isMultiple(of: 2),
      NeuralRenderingStreamedGlobalAttention.isEnabled(tokens: tokenCount)
    {
      let attended = NeuralRenderingStreamedGlobalAttention.apply(
        query: normalizedQuery, key: normalizedKey, value: publishedValue)
        .transposed(0, 2, 1, 3).reshaped([batchCount, tokenCount, channels])
      return matmul(attended, projectionWeight)
    }
    let rawScores = matmul(
      normalizedQuery,
      normalizedKey.transposed(0, 1, 3, 2)
    )
    let biasedScores = attentionBias.map {
      rawScores + $0.reshaped([1, headCount, tokenCount, tokenCount])
    } ?? rawScores
    // The global vit_1d kernels clamp logits to [-cap, cap] (symmetric); the
    // window families are never clamped.
    let scores = logitCap.map { cap -> MLXArray in
      let upper = MLXArray(cap).asType(biasedScores.dtype)
      let clipped = minimum(biasedScores, upper)
      return symmetricLogitCap
        ? maximum(clipped, MLXArray(-cap).asType(biasedScores.dtype))
        : clipped
    } ?? biasedScores
    let probabilities =
      preciseSoftmax
      ? vendorApproximateSoftmax(scores)
      : softmax(scores, axis: -1, precise: false)
    let attended = e4m3RoundTrip(
      matmul(probabilities, publishedValue)
        .transposed(0, 2, 1, 3)
        .reshaped([batchCount, tokenCount, channels])
    )
    return matmul(attended, projectionWeight)
  }

  static func windowAttention(
    _ input: MLXArray,
    qkvWeight: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray,
    projectionWeight: MLXArray,
    headCount: Int,
    windowSize: Int,
    windowOrigin: NeuralRenderingWindowOrigin,
    preciseSoftmax: Bool,
    fusedAttention: Bool = false
  ) -> MLXArray {
    precondition(input.ndim == 4)
    precondition(windowOrigin.y <= 0 && windowOrigin.y > -windowSize)
    precondition(windowOrigin.x <= 0 && windowOrigin.x > -windowSize)
    if fusedAttention, fusedAttentionEnabled, fusedAttentionHeadCounts?.contains(headCount) ?? true,
      preciseSoftmax, windowSize == NeuralRenderingFusedWindowAttention.windowSize,
      input.shape[0] == 1, input.dtype == .float16, input.shape[3] == headCount * 32
    {
      // Same numerics as the padded per-operation path: the projection GEMM on
      // the image rows, the window core in one kernel, the output GEMM.
      let projected = matmul(input, qkvWeight)
      let attended = NeuralRenderingFusedWindowAttention.apply(
        qkv: projected,
        attentionScale: attentionScale,
        attentionBias: attentionBias,
        headCount: headCount,
        windowOrigin: windowOrigin
      )
      return matmul(attended, projectionWeight)
    }
    let shape = input.shape
    let padTop = -windowOrigin.y
    let padLeft = -windowOrigin.x
    let padBottom = (windowSize - (shape[1] + padTop) % windowSize) % windowSize
    let padRight = (windowSize - (shape[2] + padLeft) % windowSize) % windowSize
    let windowInput: MLXArray
    if padTop == 0, padBottom == 0, padLeft == 0, padRight == 0 {
      windowInput = input
    } else {
      windowInput = padded(
        input,
        widths: [
          IntOrPair((0, 0)),
          IntOrPair((padTop, padBottom)),
          IntOrPair((padLeft, padRight)),
          IntOrPair((0, 0)),
        ]
      )
    }
    let windows = NeuralRenderingWindowLayout.partitionWindows(
      windowInput,
      windowSize: windowSize
    )
    let attentionWindows = cosineAttention(
      windows,
      qkvWeight: qkvWeight,
      attentionScale: attentionScale,
      attentionBias: attentionBias,
      projectionWeight: projectionWeight,
      headCount: headCount,
      preciseSoftmax: preciseSoftmax
    )
    let attention = NeuralRenderingWindowLayout.reverseWindows(
      attentionWindows,
      batchCount: shape[0],
      height: windowInput.shape[1],
      width: windowInput.shape[2],
      windowSize: windowSize
    )
    return attention[
      0...,
      padTop..<(padTop + shape[1]),
      padLeft..<(padLeft + shape[2]),
      0...
    ]
  }

  static func windowBlock(
    _ input: MLXArray,
    expansionWeight: MLXArray,
    feedForwardProjectionWeight: MLXArray,
    feedForwardCosine: MLXArray,
    qkvWeight: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray,
    attentionProjectionWeight: MLXArray,
    attentionCosine: MLXArray,
    headCount: Int,
    windowSize: Int,
    windowOrigin: NeuralRenderingWindowOrigin = .zero,
    preciseSoftmax: Bool = true,
    fusedFeedForward: Bool = false
  ) -> MLXArray {
    precondition(input.ndim == 4)
    let shape = input.shape
    let channels = shape[3]
    precondition(expansionWeight.shape[0] == channels)
    precondition(feedForwardProjectionWeight.shape[1] == channels)
    precondition(expansionWeight.shape[1] == feedForwardProjectionWeight.shape[0])

    let feedForwardBranch =
      fusedFeedForward
      ? fusedSimpleFeedForward(
        input,
        expansionWeight: expansionWeight,
        projectionWeight: feedForwardProjectionWeight
      )
      : matmul(
        e4m3RoundTrip(
          quadraticGateActivation(matmul(input, expansionWeight))
        ),
        feedForwardProjectionWeight
      )
    let feedForwardOutput = cosineResidual(
      skip: input,
      branch: feedForwardBranch,
      cosine: feedForwardCosine
    )

    let attentionBranch = windowAttention(
      feedForwardOutput,
      qkvWeight: qkvWeight,
      attentionScale: attentionScale,
      attentionBias: attentionBias,
      projectionWeight: attentionProjectionWeight,
      headCount: headCount,
      windowSize: windowSize,
      windowOrigin: windowOrigin,
      preciseSoftmax: preciseSoftmax
    )
    return cosineResidual(
      skip: feedForwardOutput,
      branch: attentionBranch,
      cosine: attentionCosine
    )
  }

  static func fusedSimpleFeedForward(
    _ input: MLXArray,
    expansionWeight: MLXArray,
    projectionWeight: MLXArray,
    maximumIntermediateBytes: Int = 512 * 1024 * 1024
  ) -> MLXArray {
    let inputChannels = input.shape[input.ndim - 1]
    let hiddenChannels = expansionWeight.shape[1]
    let outputChannels = projectionWeight.shape[1]
    let rowCount = input.size / inputChannels
    precondition(rowCount.isMultiple(of: 8))
    precondition(inputChannels.isMultiple(of: 8))
    precondition(hiddenChannels.isMultiple(of: 8))
    precondition(outputChannels.isMultiple(of: 8))
    precondition(expansionWeight.shape == [inputChannels, hiddenChannels])
    precondition(projectionWeight.shape == [hiddenChannels, outputChannels])
    let dataType = expansionWeight.dtype
    precondition(maximumIntermediateBytes > 0)
    let rowsPerChunk = max(
      8,
      maximumIntermediateBytes / (hiddenChannels * dataType.size) / 8 * 8
    )
    let flattened = input.reshaped([rowCount, inputChannels])
    guard rowsPerChunk < rowCount else {
      return fusedSimpleFeedForwardChunk(
        flattened,
        expansionWeight: expansionWeight,
        projectionWeight: projectionWeight
      ).reshaped(input.shape)
    }

    var chunks: [MLXArray] = []
    for start in stride(from: 0, to: rowCount, by: rowsPerChunk) {
      let end = min(rowCount, start + rowsPerChunk)
      let output = fusedSimpleFeedForwardChunk(
        flattened[start..<end, 0...],
        expansionWeight: expansionWeight,
        projectionWeight: projectionWeight
      )
      eval(output)
      chunks.append(output)
    }
    // Keep the per-chunk evaluation bound, but reuse scratch within this FFN.
    Memory.clearCache()
    return concatenated(chunks, axis: 0).reshaped(input.shape)
  }

  private static func fusedSimpleFeedForwardChunk(
    _ input: MLXArray,
    expansionWeight: MLXArray,
    projectionWeight: MLXArray
  ) -> MLXArray {
    let inputChannels = input.shape[input.ndim - 1]
    let hiddenChannels = expansionWeight.shape[1]
    let outputChannels = projectionWeight.shape[1]
    let rowCount = input.size / inputChannels
    let dataType = expansionWeight.dtype
    let expansionTileCount = rowCount / 8 * hiddenChannels / 8
    let expansionDispatch = fusedDispatchGrid(tileCount: expansionTileCount)
    let activated = feedForwardExpansionGateKernel(
      [input.asType(dataType), expansionWeight],
      template: [
        ("T", dataType),
        ("tileCount", expansionTileCount),
        ("groupRows", expansionDispatch.groupRows),
        ("inputChannels", inputChannels),
        ("hiddenChannels", hiddenChannels),
      ],
      grid: expansionDispatch.grid,
      threadGroup: (256, 1, 1),
      outputShapes: [[rowCount, hiddenChannels]],
      outputDTypes: [dataType]
    )[0]
    let projectionTileCount = rowCount / 8 * outputChannels / 8
    let projectionDispatch = fusedDispatchGrid(tileCount: projectionTileCount)
    let projected = feedForwardProjectionKernel(
      [e4m3RoundTrip(activated), projectionWeight],
      template: [
        ("T", dataType),
        ("tileCount", projectionTileCount),
        ("groupRows", projectionDispatch.groupRows),
        ("hiddenChannels", hiddenChannels),
        ("outputChannels", outputChannels),
      ],
      grid: projectionDispatch.grid,
      threadGroup: (256, 1, 1),
      outputShapes: [[rowCount, outputChannels]],
      outputDTypes: [dataType]
    )[0]
    return projected.reshaped(input.shape)
  }

  static func branchedFeedForward(
    _ input: MLXArray,
    expansionWeight: MLXArray,
    branchProjectionWeight: MLXArray,
    outputProjectionWeight: MLXArray
  ) -> MLXArray {
    let channels = input.shape[input.ndim - 1]
    precondition(channels >= 64 && channels.isMultiple(of: 32))
    let channelGroups = channels / 32
    precondition(
      expansionWeight.shape == [channelGroups, 4, channelGroups, 32, 32]
    )
    precondition(branchProjectionWeight.shape == [channelGroups, 4, 32, 32])
    precondition(outputProjectionWeight.shape == [channels, channels])

    let inputHeads = (0..<channelGroups).map { inputHead in
      input[.ellipsis, inputHead * 32..<(inputHead + 1) * 32]
    }
    let outputHeads = (0..<channelGroups).map { outputHead in
      let branches = (0..<4).map { branch in
        let partials = (0..<channelGroups).map { inputHead in
          matmul(
            inputHeads[inputHead],
            expansionWeight[outputHead, branch, inputHead]
          )
        }
        let expanded = partials.dropFirst().reduce(partials[0], +)
        // The fused multi-head kernels publish the gated expansion and the
        // per-head branch sum as E4M3 (vendor block-5 FFN captures: residual
        // 0.0386 -> 0.0077 MAE).
        return matmul(
          e4m3RoundTrip(quadraticGateActivation(expanded)),
          branchProjectionWeight[outputHead, branch]
        )
      }
      return e4m3RoundTrip(branches.dropFirst().reduce(branches[0], +))
    }
    return matmul(
      concatenated(outputHeads, axis: -1),
      outputProjectionWeight
    )
  }

  static func fusedBranchedFeedForward(
    _ input: MLXArray,
    expansionWeight: MLXArray,
    branchProjectionWeight: MLXArray,
    outputProjectionWeight: MLXArray
  ) -> MLXArray {
    let channels = input.shape[input.ndim - 1]
    precondition(channels >= 64 && channels.isMultiple(of: 32))
    let groupCount = channels / 32
    let rowCount = input.size / channels
    precondition(rowCount.isMultiple(of: 8))
    precondition(
      expansionWeight.shape == [groupCount, 4, groupCount, 32, 32]
    )
    precondition(branchProjectionWeight.shape == [groupCount, 4, 32, 32])
    precondition(outputProjectionWeight.shape == [channels, channels])
    let dataType = expansionWeight.dtype
    let expandedShape = [rowCount, groupCount, 4, 32]
    let expansionTileCount = rowCount / 8 * groupCount * 16
    let expansionDispatch = fusedDispatchGrid(tileCount: expansionTileCount)
    let expanded = branchedExpansionKernel(
      [input.asType(dataType), expansionWeight],
      template: [
        ("T", dataType),
        ("tileCount", expansionTileCount),
        ("groupRows", expansionDispatch.groupRows),
        ("groupCount", groupCount),
        ("channelCount", channels),
      ],
      grid: expansionDispatch.grid,
      threadGroup: (256, 1, 1),
      outputShapes: [expandedShape],
      outputDTypes: [dataType]
    )[0]
    let projectionTileCount = rowCount / 8 * groupCount * 4
    let projectionDispatch = fusedDispatchGrid(tileCount: projectionTileCount)
    let projected = branchedProjectionKernel(
      [e4m3RoundTrip(quadraticGateActivation(expanded)), branchProjectionWeight],
      template: [
        ("T", dataType),
        ("tileCount", projectionTileCount),
        ("groupRows", projectionDispatch.groupRows),
        ("groupCount", groupCount),
      ],
      grid: projectionDispatch.grid,
      threadGroup: (256, 1, 1),
      outputShapes: [[rowCount, channels]],
      outputDTypes: [dataType]
    )[0]
    return matmul(
      e4m3RoundTrip(projected).reshaped(input.shape),
      outputProjectionWeight
    )
  }

  /// Experiment toggle (`MLXDLSS_DENSE_FFN=0` disables): the branched feed-forward
  /// as dense GEMMs with float accumulation instead of the half-accumulating
  /// tile kernels.
  nonisolated(unsafe) static var denseFeedForwardEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_DENSE_FFN"] != "0"
  /// Diagnostics toggles (`MLXDLSS_FUSED_ATTENTION=0`, `MLXDLSS_FUSED_RESIDUAL=0`).
  nonisolated(unsafe) static var fusedAttentionEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_ATTENTION"] != "0"
  nonisolated(unsafe) static var fusedResidualEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_RESIDUAL"] != "0"
  /// Perf spike: `MLXDLSS_FUSED_OUTPUT_RESIDUAL=1` folds the block's final
  /// residual and its E4M3 publication into one kernel (multi-head and split families).
  nonisolated(unsafe) static var fusedOutputResidualEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_OUTPUT_RESIDUAL"] == "1"
  /// Diagnostics: head counts allowed to use the fused attention core (`MLXDLSS_FUSED_ATTENTION_HEADS=2,4`).
  nonisolated(unsafe) static var fusedAttentionHeadCounts: Set<Int>? =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_ATTENTION_HEADS"].map { Set($0.split(separator: ",").compactMap { Int($0) }) }

  private static let quadraticGatePublishKernel = MLXFast.metalKernel(
    name: "mlxdlss_quadratic_gate_publish",
    inputNames: ["input", "params"],
    outputNames: ["output"],
    source: #"""
      const uint index = thread_position_in_grid.x;
      if (index >= params[0]) { return; }
      const device half4* input4 = (const device half4*)input;
      device half4* output4 = (device half4*)output;
      half4 value = input4[index];
      half4 clamped = clamp(value, half4(-4.0h), half4(4.0h));
      half4 linear = fma(abs(clamped), half4(-0.055908203125h), half4(0.447265625h));
      half4 gate = fma(clamped, linear, half4(0.89453125h));
      half4 activated = value * gate;
      output4[index] = half4(
        half(float(mlxdlss_e4m3(activated.x))), half(float(mlxdlss_e4m3(activated.y))),
        half(float(mlxdlss_e4m3(activated.z))), half(float(mlxdlss_e4m3(activated.w))));
      """#,
    header: e4m3MetalHeader
  )

  /// `e4m3RoundTrip(quadraticGateActivation(input))` in one pass (half4 lanes).
  static func quadraticGatePublish(_ input: MLXArray) -> MLXArray {
    precondition(input.dtype == .float16 && input.size.isMultiple(of: 4))
    let count = input.size / 4
    return quadraticGatePublishKernel(
      [input, NeuralRenderingKernelParameters.array([UInt32(count)])],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [.float16]
    )[0]
  }

  /// Dense arrangement of the branched expansion weight `[G, 4, G, 32, 32]`
  /// (output group, branch, input group, k, c) as `[C, 4C]` with columns
  /// ordered (output group, branch, c), matching the fused kernels' layout.
  static func denseExpansionWeight(_ expansionWeight: MLXArray) -> MLXArray {
    let groupCount = expansionWeight.shape[0]
    let channels = groupCount * 32
    let dense = expansionWeight.transposed(2, 3, 0, 1, 4).reshaped([channels, channels * 4])
    eval(dense)
    return dense
  }

  /// Grouped arrangement of the branch projection `[G, 4, 32, 32]` as `[G, 128, 32]`.
  static func groupedProjectionWeight(_ branchProjectionWeight: MLXArray) -> MLXArray {
    let grouped = branchProjectionWeight.reshaped([branchProjectionWeight.shape[0], 128, 32])
    eval(grouped)
    return grouped
  }

  /// Perf spike: `MLXDLSS_FUSED_GROUPED_PROJECTION=1` runs gate + E4M3 +
  /// grouped projection + E4M3 as one kernel over the expansion output.
  nonisolated(unsafe) static var fusedGroupedProjectionEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_GROUPED_PROJECTION"] == "1"
  /// Perf spike: `MLXDLSS_GROUPED_EXPANSION=1` runs the expansion as a broadcast
  /// batched GEMM into the grouped layout, removing the hidden-tensor transposes.
  nonisolated(unsafe) static var groupedExpansionEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_GROUPED_EXPANSION"] == "1"

  private static let gateGroupedProjectionKernel = MLXFast.metalKernel(
    name: "mlxdlss_gate_grouped_projection",
    inputNames: ["expanded", "weight", "params"],
    outputNames: ["output"],
    source: #"""
      // params = [rows, groups]. expanded: [rows, groups * 128] half (columns
      // ordered (group, branch, c)); weight: [groups, 128, 32] half; output:
      // [rows, groups * 32] half = E4M3(E4M3(gate(expanded_g)) · W_g), float
      // accumulation over the 16 k-tiles in order, like the batched GEMM.
      // One simdgroup: 32 rows × one group; a threadgroup covers up to four groups.
      const uint rows = params[0];
      const uint groups = params[1];
      const uint simd = simdgroup_index_in_threadgroup;
      const uint simdgroups = threads_per_threadgroup.x / 32;
      const uint groupBlocks = (groups + simdgroups - 1) / simdgroups;
      const uint rowBlock = threadgroup_position_in_grid.x / groupBlocks;
      const uint group = (threadgroup_position_in_grid.x % groupBlocks) * simdgroups + simd;
      if (group >= groups) { return; }
      const uint row0 = rowBlock * 32;
      const uint expandedStride = groups * 128;
      const uint outputStride = groups * 32;
      const device half* a = expanded + group * 128;
      const device half* w = weight + group * 128 * 32;
      simdgroup_matrix<float, 8, 8> acc[4][4];
      for (uint t = 0; t < 4; ++t) {
        for (uint o = 0; o < 4; ++o) {
          acc[t][o].thread_elements()[0] = 0.0f;
          acc[t][o].thread_elements()[1] = 0.0f;
        }
      }
      const uint tiles = min(4u, (rows - row0) / 8);
      for (uint k = 0; k < 16; ++k) {
        simdgroup_matrix<float, 8, 8> right[4];
        for (uint o = 0; o < 4; ++o) {
          simdgroup_matrix<half, 8, 8> rh;
          simdgroup_load(rh, w + k * 8 * 32 + o * 8, 32, ulong2(0), false);
          right[o].thread_elements()[0] = float(rh.thread_elements()[0]);
          right[o].thread_elements()[1] = float(rh.thread_elements()[1]);
        }
        for (uint t = 0; t < tiles; ++t) {
          simdgroup_matrix<half, 8, 8> lh;
          simdgroup_load(lh, a + (row0 + t * 8) * expandedStride + k * 8, expandedStride, ulong2(0), false);
          simdgroup_matrix<float, 8, 8> left;
          for (uint e = 0; e < 2; ++e) {
            half value = lh.thread_elements()[e];
            half clamped = clamp(value, half(-4.0), half(4.0));
            half linear = fma(abs(clamped), half(-0.055908203125), half(0.447265625));
            half gate = fma(clamped, linear, half(0.89453125));
            left.thread_elements()[e] = float(mlxdlss_e4m3(value * gate));
          }
          for (uint o = 0; o < 4; ++o) {
            simdgroup_multiply_accumulate(acc[t][o], left, right[o], acc[t][o]);
          }
        }
      }
      for (uint t = 0; t < tiles; ++t) {
        for (uint o = 0; o < 4; ++o) {
          simdgroup_matrix<half, 8, 8> result;
          result.thread_elements()[0] = half(float(mlxdlss_e4m3(half(acc[t][o].thread_elements()[0]))));
          result.thread_elements()[1] = half(float(mlxdlss_e4m3(half(acc[t][o].thread_elements()[1]))));
          simdgroup_store(result, output + (row0 + t * 8) * outputStride + group * 32 + o * 8, outputStride, ulong2(0), false);
        }
      }
      """#,
    header: "#include <metal_simdgroup_matrix>\n" + e4m3MetalHeader
  )

  /// `e4m3(grouped projection of e4m3(gate(expanded)))` in one kernel; `expanded`
  /// is `[rows, groups * 128]` half, `groupedWeight` `[groups, 128, 32]` half.
  static func gateGroupedProjection(_ expanded: MLXArray, groupedWeight: MLXArray) -> MLXArray {
    precondition(expanded.ndim == 2 && expanded.dtype == .float16 && groupedWeight.dtype == .float16)
    let rows = expanded.shape[0]
    let groups = groupedWeight.shape[0]
    precondition(expanded.shape[1] == groups * 128 && groupedWeight.shape == [groups, 128, 32])
    precondition(rows.isMultiple(of: 8))
    let simdgroups = min(groups, 4)
    let groupBlocks = (groups + simdgroups - 1) / simdgroups
    let threadgroups = (rows + 31) / 32 * groupBlocks
    let params = NeuralRenderingKernelParameters.array([UInt32(rows), UInt32(groups)])
    return gateGroupedProjectionKernel(
      [expanded, groupedWeight, params],
      grid: (threadgroups * 32 * simdgroups, 1, 1),
      threadGroup: (32 * simdgroups, 1, 1),
      outputShapes: [[rows, groups * 32]],
      outputDTypes: [.float16]
    )[0]
  }

  /// Branched feed-forward as dense GEMMs: expansion `[rows, C] × [C, 4C]`,
  /// gate + E4M3, grouped projection `[G, rows, 128] × [G, 128, 32]` (the
  /// branch sum folded into K), E4M3, output projection. Same publication
  /// points as the tile kernels; the GEMMs accumulate in float.
  static func denseBranchedFeedForward(
    _ input: MLXArray,
    denseExpansionWeight: MLXArray,
    groupedProjectionWeight: MLXArray,
    outputProjectionWeight: MLXArray
  ) -> MLXArray {
    let channels = input.shape[input.ndim - 1]
    let groupCount = channels / 32
    let rowCount = input.size / channels
    let flat = input.reshaped([rowCount, channels])
    if groupedExpansionEnabled {
      // Per-group expansion straight into the grouped layout: [1, rows, C] × [G, C, 128]
      // (a broadcast batch, no copy) gives [G, rows, 128]; every output column is
      // the same K-sequential GEMM as its dense counterpart, so the values match.
      let groupedExpansion = denseExpansionWeight.reshaped([channels, groupCount, 128]).transposed(1, 0, 2)
      let expandedGrouped = matmul(flat.expandedDimensions(axis: 0), groupedExpansion)
      let gated = quadraticGatePublish(expandedGrouped)
      let projected = matmul(gated, groupedProjectionWeight).transposed(1, 0, 2).reshaped([rowCount, channels])
      return matmul(e4m3RoundTrip(projected), outputProjectionWeight).reshaped(input.shape)
    }
    let expanded = matmul(flat, denseExpansionWeight)
    if fusedGroupedProjectionEnabled, expanded.dtype == .float16, groupedProjectionWeight.dtype == .float16 {
      let projected = gateGroupedProjection(expanded, groupedWeight: groupedProjectionWeight)
      return matmul(projected, outputProjectionWeight).reshaped(input.shape)
    }
    let gated = quadraticGatePublish(expanded)
    let grouped = gated.reshaped([rowCount, groupCount, 128]).transposed(1, 0, 2)
    let projected = matmul(grouped, groupedProjectionWeight).transposed(1, 0, 2).reshaped([rowCount, channels])
    return matmul(e4m3RoundTrip(projected), outputProjectionWeight).reshaped(input.shape)
  }

  static func branchedWindowBlock(
    _ input: MLXArray,
    expansionWeight: MLXArray,
    branchProjectionWeight: MLXArray,
    outputProjectionWeight: MLXArray,
    feedForwardCosine: MLXArray,
    qkvWeight: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray,
    attentionProjectionWeight: MLXArray,
    attentionCosine: MLXArray,
    headCount: Int,
    windowSize: Int,
    windowOrigin: NeuralRenderingWindowOrigin = .zero,
    preciseSoftmax: Bool = true,
    fusedFeedForward: Bool = false,
    fusedAttention: Bool = false,
    denseExpansionWeight: MLXArray? = nil,
    groupedProjectionWeight: MLXArray? = nil,
    publishedOutput: Bool? = nil
  ) -> MLXArray {
    precondition(input.ndim == 4)
    let feedForwardBranch: MLXArray
    if fusedFeedForward, denseFeedForwardEnabled, let denseExpansionWeight, let groupedProjectionWeight {
      feedForwardBranch = denseBranchedFeedForward(
        input,
        denseExpansionWeight: denseExpansionWeight,
        groupedProjectionWeight: groupedProjectionWeight,
        outputProjectionWeight: outputProjectionWeight
      )
    } else {
      feedForwardBranch =
      fusedFeedForward
      ? fusedBranchedFeedForward(
        input,
        expansionWeight: expansionWeight,
        branchProjectionWeight: branchProjectionWeight,
        outputProjectionWeight: outputProjectionWeight
      )
      : branchedFeedForward(
        input,
        expansionWeight: expansionWeight,
        branchProjectionWeight: branchProjectionWeight,
        outputProjectionWeight: outputProjectionWeight
      )
    }
    // Published as E4M3 before the attention reads it (multi-head kernels only;
    // the single-head window block keeps its residual in fp16 registers).
    let feedForwardOutput =
      fusedAttention && fusedResidualEnabled && input.dtype == .float16 && feedForwardBranch.dtype == .float16
      ? NeuralRenderingFusedWindowAttention.cosineResidual(
        skip: input, branch: feedForwardBranch, cosine: feedForwardCosine, publish: true)
      : e4m3RoundTrip(
        cosineResidual(
          skip: input,
          branch: feedForwardBranch,
          cosine: feedForwardCosine
        )
      )
    let attentionBranch = windowAttention(
      feedForwardOutput,
      qkvWeight: qkvWeight,
      attentionScale: attentionScale,
      attentionBias: attentionBias,
      projectionWeight: attentionProjectionWeight,
      headCount: headCount,
      windowSize: windowSize,
      windowOrigin: windowOrigin,
      preciseSoftmax: preciseSoftmax,
      fusedAttention: fusedAttention
    )
    return outputResidual(
      skip: feedForwardOutput, branch: attentionBranch, cosine: attentionCosine,
      publishedOutput: publishedOutput)
  }

  /// The block's final residual. With `publishedOutput`, the caller delegates
  /// the E4M3 publication: one fused kernel when the toggle allows, otherwise
  /// the per-operation chain.
  static func outputResidual(
    skip: MLXArray, branch: MLXArray, cosine: MLXArray, publishedOutput: Bool?
  ) -> MLXArray {
    guard let publishedOutput else {
      return cosineResidual(skip: skip, branch: branch, cosine: cosine)
    }
    if fusedOutputResidualEnabled, fusedResidualEnabled, skip.dtype == .float16, branch.dtype == .float16 {
      return NeuralRenderingFusedWindowAttention.cosineResidual(
        skip: skip, branch: branch, cosine: cosine, publish: publishedOutput)
    }
    let output = cosineResidual(skip: skip, branch: branch, cosine: cosine)
    return publishedOutput ? e4m3RoundTrip(output) : output
  }

  static func downsample(
    _ input: MLXArray,
    weight: MLXArray
  ) -> MLXArray {
    precondition(input.ndim == 4)
    let shape = input.shape
    precondition(shape[1].isMultiple(of: 2))
    precondition(shape[2].isMultiple(of: 2))
    precondition(weight.ndim == 2)
    precondition(weight.shape[0] == shape[3])
    return matmul(averagePool2(input), weight)
  }

  static func averagePool2(_ input: MLXArray) -> MLXArray {
    precondition(input.ndim == 4)
    let shape = input.shape
    precondition(shape[1].isMultiple(of: 2))
    precondition(shape[2].isMultiple(of: 2))
    return mean(
      input.reshaped([
        shape[0], shape[1] / 2, 2, shape[2] / 2, 2, shape[3],
      ]),
      axes: [2, 4]
    )
  }

  static func padSpatialEnd(_ input: MLXArray, multiple: Int) -> MLXArray {
    precondition(input.ndim == 4)
    precondition(multiple > 0)
    let padBottom = (multiple - input.shape[1] % multiple) % multiple
    let padRight = (multiple - input.shape[2] % multiple) % multiple
    guard padBottom != 0 || padRight != 0 else { return input }
    return padded(
      input,
      widths: [
        IntOrPair((0, 0)),
        IntOrPair((0, padBottom)),
        IntOrPair((0, padRight)),
        IntOrPair((0, 0)),
      ]
    )
  }

  /// Split-family feed-forward core (blocks 23-30 and 40-47): the first
  /// projection is published as E4M3, then every 64-channel group runs a
  /// `64 -> 256 -> 64` MLP with the quadratic-gate activation, and the
  /// concatenated group outputs are published as E4M3 before `weight3`.
  /// This mirrors the vendor `ffwd_512_chained` kernel output byte-for-byte
  /// on captured DLL activations (2% mean error on every split block).
  static func splitGroupFeedForward(
    _ input: MLXArray,
    firstProjectionWeight: MLXArray,
    expandWeight: MLXArray,
    projectWeight: MLXArray
  ) -> MLXArray {
    let channels = input.shape[input.ndim - 1]
    precondition(channels.isMultiple(of: 64))
    let groups = channels / 64
    precondition(firstProjectionWeight.shape == [channels, channels])
    precondition(expandWeight.shape == [groups, 64, 256])
    precondition(projectWeight.shape == [groups, 256, 64])

    let hidden = e4m3RoundTrip(matmul(input, firstProjectionWeight))
    let rows = hidden.size / channels
    let grouped = hidden.reshaped([rows, groups, 64]).transposed(1, 0, 2)
    let expanded = quadraticGateActivation(matmul(grouped, expandWeight))
    let projected = matmul(expanded, projectWeight)
    return e4m3RoundTrip(projected.transposed(1, 0, 2).reshaped(input.shape))
  }

  private static func fusedDispatchGrid(
    tileCount: Int
  ) -> (grid: (Int, Int, Int), groupRows: Int) {
    let totalGroupRows = (tileCount + 7) / 8
    let groupRows = min(totalGroupRows, 65_535)
    let groupLayers = (totalGroupRows + groupRows - 1) / groupRows
    return ((256, groupRows, groupLayers), groupRows)
  }

  static func splitWindowBlock(
    _ input: MLXArray,
    firstProjectionWeight: MLXArray,
    expandWeight: MLXArray,
    projectWeight: MLXArray,
    feedForwardProjectionWeight: MLXArray,
    feedForwardCosine: MLXArray,
    qkvWeight: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray,
    attentionProjectionWeight: MLXArray,
    attentionCosine: MLXArray,
    headCount: Int,
    windowSize: Int,
    windowOrigin: NeuralRenderingWindowOrigin = .zero,
    preciseSoftmax: Bool = true,
    fusedFeedForward: Bool = false,
    publishedOutput: Bool? = nil
  ) -> MLXArray {
    precondition(input.ndim == 4)
    let shape = input.shape
    let channels = shape[3]
    precondition(feedForwardProjectionWeight.shape == [channels, channels])

    // The group MLP runs as batched matmuls in both execution modes; the
    // `fusedFeedForward` flag selects the fused window attention core.
    let feedForwardInput = splitGroupFeedForward(
      input,
      firstProjectionWeight: firstProjectionWeight,
      expandWeight: expandWeight,
      projectWeight: projectWeight
    )
    let feedForwardBranch = matmul(
      feedForwardInput,
      feedForwardProjectionWeight
    )
    let feedForwardOutput = cosineResidual(
      skip: input,
      branch: feedForwardBranch,
      cosine: feedForwardCosine
    )
    let attentionBranch = windowAttention(
      feedForwardOutput,
      qkvWeight: qkvWeight,
      attentionScale: attentionScale,
      attentionBias: attentionBias,
      projectionWeight: attentionProjectionWeight,
      headCount: headCount,
      windowSize: windowSize,
      windowOrigin: windowOrigin,
      preciseSoftmax: preciseSoftmax,
      fusedAttention: fusedFeedForward
    )
    return outputResidual(
      skip: feedForwardOutput, branch: attentionBranch, cosine: attentionCosine,
      publishedOutput: publishedOutput)
  }

  static func globalBlock(
    _ input: MLXArray,
    expansionWeight: MLXArray,
    feedForwardProjectionWeight: MLXArray,
    feedForwardCosine: MLXArray,
    qkvWeight: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray? = nil,
    attentionProjectionWeight: MLXArray,
    attentionCosine: MLXArray,
    headCount: Int,
    logitCap: Float? = globalAttentionLogitCap,
    preciseSoftmax: Bool = true
  ) -> MLXArray {
    globalBlock(
      input,
      expansionWeight: NeuralRenderingLinearWeight(expansionWeight),
      feedForwardProjectionWeight: NeuralRenderingLinearWeight(
        feedForwardProjectionWeight
      ),
      feedForwardCosine: feedForwardCosine,
      qkvWeight: qkvWeight,
      attentionScale: attentionScale,
      attentionBias: attentionBias,
      attentionProjectionWeight: attentionProjectionWeight,
      attentionCosine: attentionCosine,
      headCount: headCount,
      logitCap: logitCap,
      preciseSoftmax: preciseSoftmax
    )
  }

  static func globalBlock(
    _ input: MLXArray,
    expansionWeight: NeuralRenderingLinearWeight,
    feedForwardProjectionWeight: NeuralRenderingLinearWeight,
    feedForwardCosine: MLXArray,
    qkvWeight: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray? = nil,
    attentionProjectionWeight: MLXArray,
    attentionCosine: MLXArray,
    headCount: Int,
    logitCap: Float? = globalAttentionLogitCap,
    preciseSoftmax: Bool = true,
    fusedOperations: Bool = false
  ) -> MLXArray {
    precondition(input.ndim == 4)
    let shape = input.shape
    let channels = shape[3]
    let tokens = input.reshaped([shape[0], shape[1] * shape[2], channels])
    // Launch 58 publishes the gated expansion as E4M3 (98.3% exact bytes).
    let expanded = expansionWeight(tokens)
    let published = fusedOperations && expanded.dtype == .float16
      ? quadraticGatePublish(expanded) : e4m3RoundTrip(quadraticGateActivation(expanded))
    let feedForwardBranch = feedForwardProjectionWeight(published)
    let feedForwardOutput = cosineResidual(
      skip: tokens,
      branch: feedForwardBranch,
      cosine: feedForwardCosine
    )
    let globalAttentionScale =
      attentionScale * Float(channels / headCount).squareRoot()
    let attentionBranch = cosineAttention(
      feedForwardOutput,
      qkvWeight: qkvWeight,
      attentionScale: globalAttentionScale,
      attentionBias: attentionBias,
      projectionWeight: attentionProjectionWeight,
      headCount: headCount,
      logitCap: logitCap,
      symmetricLogitCap: true,
      preciseSoftmax: preciseSoftmax,
      streamedGlobalAttention: fusedOperations
    )
    return cosineResidual(
      skip: feedForwardOutput,
      branch: attentionBranch,
      cosine: attentionCosine
    ).reshaped(shape)
  }

  static func learnedUpsample2(
    _ input: MLXArray,
    interpolation: MLXArray
  ) -> MLXArray {
    precondition(input.ndim == 4)
    let shape = input.shape
    precondition(interpolation.shape == [shape[3]])

    let right = concatenated(
      [
        input[0..., 0..., 1..., 0...],
        input[0..., 0..., (shape[2] - 1)..<shape[2], 0...],
      ], axis: 2)
    let horizontalMidpoint = input * (1 - interpolation) + right * interpolation
    let horizontal = stacked([input, horizontalMidpoint], axis: 3)
      .reshaped([shape[0], shape[1], shape[2] * 2, shape[3]])

    let below = concatenated(
      [
        horizontal[0..., 1..., 0..., 0...],
        horizontal[0..., (shape[1] - 1)..<shape[1], 0..., 0...],
      ], axis: 1)
    let verticalMidpoint = horizontal * (1 - interpolation) + below * interpolation
    return stacked([horizontal, verticalMidpoint], axis: 2)
      .reshaped([shape[0], shape[1] * 2, shape[2] * 2, shape[3]])
  }

  static func decoderInputMerge(
    _ input: MLXArray,
    skip: MLXArray,
    skipSine: MLXArray
  ) -> MLXArray {
    precondition(input.ndim == 4 && skip.ndim == 4)
    precondition(input.shape[0] == skip.shape[0])
    precondition(input.shape[3] == skip.shape[3])
    precondition(skipSine.shape == [input.shape[3]])
    let cropped = nearestUpsample2Crop(
      input,
      height: skip.shape[1],
      width: skip.shape[2]
    )
    return cropped + skip * skipSine
  }

  static func nearestUpsample2Crop(
    _ input: MLXArray,
    height: Int,
    width: Int
  ) -> MLXArray {
    precondition(input.ndim == 4)
    precondition(height > 0 && width > 0)
    let shape = input.shape
    let horizontal = stacked([input, input], axis: 3)
      .reshaped([shape[0], shape[1], shape[2] * 2, shape[3]])
    let doubled = stacked([horizontal, horizontal], axis: 2)
      .reshaped([shape[0], shape[1] * 2, shape[2] * 2, shape[3]])
    precondition(height <= doubled.shape[1])
    precondition(width <= doubled.shape[2])
    return doubled[
      0...,
      0..<height,
      0..<width,
      0...
    ]
  }

  static func postMerge(
    _ input: MLXArray,
    skip: MLXArray,
    sine: MLXArray,
    cosine: MLXArray
  ) -> MLXArray {
    precondition(input.shape == skip.shape)
    precondition(sine.shape == [input.shape.last!])
    precondition(cosine.shape == sine.shape)
    return input * sine + skip * cosine
  }

  private static func frameworkCosineNormalize(_ input: MLXArray) -> MLXArray {
    let half = input.asType(.float16)
    let squaredNorm = sum(square(half), axis: -1, keepDims: true)
    return (half * rsqrt(maximum(squaredNorm, cosineNormFloor))).asType(input.dtype)
  }

}

/// NHWC window partition and reverse used by the window blocks.
enum NeuralRenderingWindowLayout {
  static func partitionWindows(
      _ input: MLXArray,
      windowSize: Int
    ) -> MLXArray {
      precondition(input.ndim == 4, "window partition expects NHWC rank 4")
      precondition(windowSize > 0, "window size must be positive")
      let shape = input.shape
      precondition(
        shape[1].isMultiple(of: windowSize)
          && shape[2].isMultiple(of: windowSize),
        "spatial dimensions must be divisible by window size"
      )
      return
        input
        .reshaped([
          shape[0], shape[1] / windowSize, windowSize,
          shape[2] / windowSize, windowSize, shape[3],
        ])
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped([
          shape[0] * shape[1] * shape[2] / (windowSize * windowSize),
          windowSize * windowSize,
          shape[3],
        ])
    }

  static func reverseWindows(
      _ windows: MLXArray,
      batchCount: Int,
      height: Int,
      width: Int,
      windowSize: Int
    ) -> MLXArray {
      precondition(windows.ndim == 3, "window reverse expects rank 3 windows")
      precondition(batchCount > 0, "batch count must be positive")
      precondition(windowSize > 0, "window size must be positive")
      precondition(
        height.isMultiple(of: windowSize) && width.isMultiple(of: windowSize),
        "spatial dimensions must be divisible by window size"
      )
      let channels = windows.shape[2]
      let verticalWindows = height / windowSize
      let horizontalWindows = width / windowSize
      precondition(
        windows.shape[0] == batchCount * verticalWindows * horizontalWindows,
        "window count does not match the requested output shape"
      )
      precondition(
        windows.shape[1] == windowSize * windowSize,
        "window token count does not match window size"
      )
      return
        windows
        .reshaped([
          batchCount, verticalWindows, horizontalWindows,
          windowSize, windowSize, channels,
        ])
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped([batchCount, height, width, channels])
    }
}
