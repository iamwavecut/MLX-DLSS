import AVFoundation
import DLSSCore
import DLSSMLX
import Foundation

public struct MediaPreviewRequest: Equatable, Sendable {
  public let input: URL
  public let isVideo: Bool
  public let time: Double
  public let options: MediaProcessingOptions

  public init(input: URL, isVideo: Bool, time: Double = 0, options: MediaProcessingOptions) {
    self.input = input
    self.isVideo = isVideo
    self.time = time
    self.options = options
  }
}

public struct MediaPreviewResult: Sendable {
  public let original: CGImage
  public let processed: CGImage
  public let time: Double
  public let duration: Double
  public let frameInterval: Double
  public let historyFrames: Int
  public let elapsedSeconds: Double
}

/// Retains the selected source frames and loaded weights across control changes.
/// Callers submit one request at a time and discard superseded results. Preview
/// history is a bounded local window, independent of export's full sequence.
@available(macOS 26.0, *)
public actor NativeMediaPreview {
  private let io: NativeImageIO
  private var sourceKey: SourceKey?
  private var frames: [NativeDecodedFrame] = []
  private var original: CGImage?
  private var duration: Double = 0
  private var frameInterval: Double = 1 / 30
  private var renderer: MLXNeuralRenderingDeviceTemporalBackend?
  private var modelURL: URL?
  private var precision: MLXComputePrecision?
  private var upscaler: MLXNativeSuperResolver?
  private var superResolutionURL: URL?
  private var dlss: MLXNativeDLSSSuperResolver?
  private var dlssURL: URL?
  private var motionKey: MotionKey?
  private var motions: [MLXVideoMotion?] = []
  /// The last neural-rendering result and the controls it was rendered with.
  /// Composition-only controls are re-applied to it, and the intensity too when
  /// no temporal history is involved; everything else re-renders.
  private var lastRendering: LastRendering?
  private var busy = false

  private struct LastRendering {
    let source: SourceKey
    let options: MediaProcessingOptions   // live controls neutralised
    let temporal: Bool
    var intensity: Float
    var composition: [Float]
    var result: MLXVideoFrame
    let historyFrames: Int
  }

  /// Everything that changes the head input or history, with the live controls
  /// (intensity, detail, colour, radius) and post-NR upscaling set to fixed values.
  private static func renderingKey(_ options: MediaProcessingOptions) -> MediaProcessingOptions {
    var key = options
    key.intensity = 1
    key.detailStrength = 1
    key.colourStrength = 1
    key.detailRadius = 4
    key.superResolutionWeights = nil
    return key
  }

  private struct SourceKey: Equatable {
    let url: URL
    let isVideo: Bool
    let time: Double
  }
  private struct MotionKey: Equatable {
    let mode: MediaMotion
    let threshold: Float
  }

  public init() throws { io = try NativeImageIO() }

  public func render(_ request: MediaPreviewRequest) async throws -> MediaPreviewResult {
    guard !busy else { throw MLXMediaError("Submit preview requests sequentially") }
    guard request.time.isFinite, request.time >= 0 else { throw MLXMediaError("Invalid preview time") }
    if request.options.renderingModel != nil || request.options.superResolutionWeights != nil || request.options.dlssSuperResolutionModel != nil {
      try request.options.validate()
    }
    guard request.isVideo || request.options.dlssSuperResolutionModel == nil else {
      throw MLXMediaError("Use RTX VSR for images; DLSS SR requires video")
    }
    busy = true
    defer { busy = false }
    let started = ContinuousClock.now
    let key = SourceKey(url: request.input, isVideo: request.isVideo, time: request.isVideo ? request.time : 0)
    if sourceKey != key {
      try await loadSource(key)
      sourceKey = key
      original = nil
      motionKey = nil
      motions = []
      lastRendering = nil
    }
    guard let selected = frames.last else { throw MLXMediaError("No frame at the selected time") }
    if original == nil { original = try await io.displayImage(selected.rgb) }
    try Task.checkCancellation()
    let options = request.options
    var result = selected.rgb
    var historyFrames = 0
    if let url = options.dlssSuperResolutionModel {
      if dlss == nil || dlssURL != url {
        dlss = nil
        dlss = try MLXNativeDLSSSuperResolver(packageURL: url)
        dlssURL = url
      }
      await dlss!.reset()
    } else { dlss = nil; dlssURL = nil }
    if let url = options.renderingModel {
      if renderer == nil || modelURL != url || precision != options.precision {
        renderer = nil
        renderer = try MLXNeuralRenderingDeviceTemporalBackend(packageURL: url,
          executionMode: .metalFused, computePrecision: options.precision)
        modelURL = url
        precision = options.precision
        lastRendering = nil
      }
    } else {
      lastRendering = nil
    }
    let reusable = try await reuseLastRendering(key: key, options: options, temporal: request.isVideo && options.temporal,
      selected: selected)
    if let reusable {
      result = reusable.result
      historyFrames = reusable.historyFrames
    } else if options.renderingModel != nil || dlss != nil {
      if let renderer { await renderer.reset(sequenceID: 1) }
      let temporal = request.isVideo && options.temporal
      if temporal {
        let key = MotionKey(mode: options.motion, threshold: options.sceneCutThreshold)
        if motionKey != key {
          let flow = try NativeOpticalFlow(width: selected.rgb.width, height: selected.rgb.height, mode: options.motion)
          var prepared: [MLXVideoMotion?] = []
          for (index, frame) in frames.enumerated() {
            try Task.checkCancellation()
            prepared.append(try await flow.prepare(frame.rgb, index: index, sceneCutThreshold: options.sceneCutThreshold))
          }
          motions = prepared
          motionKey = key
        }
      }
      let outputOptions = try MLXVideoOutputOptions(width: selected.rgb.width, height: selected.rgb.height,
        detailStrength: options.detailStrength, colourStrength: options.colourStrength, radius: options.detailRadius)
      let indices = temporal ? Array(frames.indices) : [frames.count - 1]
      for (ordinal, index) in indices.enumerated() {
        try Task.checkCancellation()
        result = frames[index].rgb
        if options.renderingModel != nil, let renderer {
          result = try await renderer.renderVideoFrame(result, motion: temporal ? motions[index] : nil,
            context: .init(streamID: 1, frameIndex: UInt64(ordinal)), processingScale: options.processingScale,
            temporal: temporal, outputOptions: outputOptions, featureControls: options.profile.featureControls,
            intensity: options.intensity)
        }
        if let dlss {
          result = try await dlss.upscale(result, motion: temporal ? motions[index] : nil, temporal: temporal)
        }
      }
      try Task.checkCancellation()
      historyFrames = indices.count - 1
      lastRendering = options.renderingModel != nil && dlss == nil
        ? LastRendering(source: key, options: Self.renderingKey(options), temporal: temporal, intensity: options.intensity,
          composition: [options.detailStrength, options.colourStrength, options.detailRadius], result: result,
          historyFrames: historyFrames)
        : nil
    }
    if let url = options.superResolutionWeights {
      if upscaler == nil || superResolutionURL != url {
        upscaler = nil
        upscaler = try MLXNativeSuperResolver(weightsURL: url)
        superResolutionURL = url
      }
      result = try await upscaler!.upscale(result)
    } else {
      upscaler = nil
      superResolutionURL = nil
    }
    try Task.checkCancellation()
    let processed = options.renderingModel != nil || options.superResolutionWeights != nil || dlss != nil
      ? try await io.displayImage(result) : original!
    let elapsed = started.duration(to: .now).components
    return MediaPreviewResult(original: original!, processed: processed, time: selected.time.seconds,
      duration: duration, frameInterval: frameInterval, historyFrames: historyFrames,
      elapsedSeconds: Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18)
  }

  /// Reuses the last neural-rendering result when only live controls changed:
  /// detail/colour/radius are recomposed, and the intensity is re-applied to the
  /// retained head output when the frame has no temporal history (a still image
  /// or a non-temporal preview). Returns nil when the head must run again.
  private func reuseLastRendering(key: SourceKey, options: MediaProcessingOptions, temporal: Bool,
                                  selected: NativeDecodedFrame) async throws -> (result: MLXVideoFrame, historyFrames: Int)? {
    guard options.renderingModel != nil, dlss == nil, let renderer, var last = lastRendering,
      last.source == key, last.temporal == temporal, last.options == Self.renderingKey(options)
    else { return nil }
    let composition = [options.detailStrength, options.colourStrength, options.detailRadius]
    if last.intensity == options.intensity, last.composition == composition {
      return (last.result, last.historyFrames)
    }
    let outputOptions = try MLXVideoOutputOptions(width: selected.rgb.width, height: selected.rgb.height,
      detailStrength: options.detailStrength, colourStrength: options.colourStrength, radius: options.detailRadius)
    let frame: MLXVideoFrame?
    if last.intensity == options.intensity {
      frame = try await renderer.recomposeLastFrame(outputOptions: outputOptions)
    } else if !temporal {
      frame = try await renderer.reintensifyLastFrame(intensity: options.intensity, outputOptions: outputOptions)
    } else {
      frame = nil
    }
    guard let frame else { return nil }
    last.intensity = options.intensity
    last.composition = composition
    last.result = frame
    lastRendering = last
    return (frame, last.historyFrames)
  }

  private func loadSource(_ key: SourceKey) async throws {
    if !key.isVideo {
      let image = try await io.read(key.url)
      frames = [NativeDecodedFrame(rgb: image, time: .zero, duration: .zero)]
      duration = 0
      return
    }
    let asset = AVURLAsset(url: key.url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw MLXMediaError("The file contains no video track")
    }
    let trackRange = try await track.load(.timeRange)
    let rate = try await track.load(.nominalFrameRate)
    let frameInterval = rate.isFinite && rate > 0 ? 1 / Double(rate) : 1 / 30
    guard trackRange.end.seconds.isFinite, trackRange.duration > .zero else {
      throw MLXMediaError("Video preview requires a finite duration")
    }
    let duration = trackRange.end.seconds
    let time = min(max(trackRange.start.seconds, key.time), max(trackRange.start.seconds, duration - frameInterval))
    let start = CMTime(seconds: max(trackRange.start.seconds, time - 3 * frameInterval), preferredTimescale: 60000)
    let reader = try await NativeVideoReader(url: key.url, options: MediaProcessingOptions(),
      timeRange: CMTimeRange(start: start, end: trackRange.end))
    do {
      var selected: [NativeDecodedFrame] = []
      while let frame = try await reader.next() {
        selected.append(frame)
        if selected.count > 4 { selected.removeFirst() }
        if frame.time.seconds >= time - 1e-6 { break }
      }
      await reader.cancel()
      guard !selected.isEmpty else { throw MLXMediaError("No frame at the selected time") }
      frames = selected
      self.duration = duration
      self.frameInterval = frameInterval
    } catch { await reader.cancel(); throw error }
  }
}
