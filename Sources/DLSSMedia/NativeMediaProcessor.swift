import CoreMedia
import DLSSCore
import DLSSMLX
import Foundation

/// Shared by the command line and the macOS app. One processor runs one job at a
/// time so that temporal state and MLX lazy graph construction cannot interleave.
public actor NativeMediaProcessor {
  private var busy = false
  /// The next frame's motion is estimated while the current frame renders
  /// (rendering-then-generation order); `MLXDLSS_OVERLAP_MOTION=0` serialises them.
  nonisolated(unsafe) static var overlapMotionEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_OVERLAP_MOTION"] != "0"
  public init() {}

  public func processImage(input: URL, output: URL, options: MediaProcessingOptions) async throws -> MediaProcessingResult {
    try options.validate()
    guard !busy else { throw MLXMediaError("This processor is already running a job") }
    guard options.frameGenerationWeights == nil else {
      throw MLXMediaError("Frame generation requires video")
    }
    guard options.dlssSuperResolutionModel == nil else { throw MLXMediaError("Use RTX VSR for images; DLSS SR requires video") }
    busy = true
    defer { busy = false }
    let start = ContinuousClock.now
    let io = try NativeImageIO()
    let source = try await io.read(input)
    let renderer = try makeRenderer(width: source.width, height: source.height, options: options)
    try Task.checkCancellation()
    var rendered = source
    if let renderer {
      rendered = try await renderer.renderVideoFrame(source, motion: nil,
        context: .init(streamID: 1, frameIndex: 0), processingScale: options.processingScale, temporal: false)
    }
    if let weights = options.superResolutionWeights {
      let upscaler = try MLXNativeSuperResolver(weightsURL: weights)
      rendered = try await upscaler.upscale(rendered)
    }
    try Task.checkCancellation()
    try await io.write(rendered, to: output)
    return MediaProcessingResult(output: output, inputFrames: 1, outputFrames: 1, sceneResets: 0,
      elapsedSeconds: seconds(since: start), motionBackend: "disabled")
  }

  @available(macOS 26.0, *)
  public func processVideo(input: URL, output: URL, options: MediaProcessingOptions,
    progress: @Sendable (MediaProgress) async -> Void = { _ in }) async throws -> MediaProcessingResult {
    try options.validate()
    guard !busy else { throw MLXMediaError("This processor is already running a job") }
    guard !FileManager.default.fileExists(atPath: output.path) else {
      throw MLXMediaError("Output already exists: \(output.lastPathComponent)")
    }
    guard ["mp4", "mov", "m4v"].contains(output.pathExtension.lowercased()) else {
      throw MLXMediaError("Video output must be MP4, M4V or MOV")
    }
    busy = true
    defer { busy = false }
    let started = ContinuousClock.now
    var timing = MediaStageTiming()
    let reader = try await NativeVideoReader(url: input, options: options)
    guard let first = try await reader.next() else { throw MLXMediaError("No frames in the selected video range") }
    timing.decodingSeconds = seconds(since: started)
    let setupStarted = ContinuousClock.now
    let width = first.rgb.width, height = first.rgb.height
    let renderer = try makeRenderer(width: width, height: height, options: options)
    let generator = try options.frameGenerationWeights.map { try MLXNativeFrameGenerator(weightsURL: $0,
      precision: options.precision == .float16 ? .float16 : .float32) }
    let upscaler = try options.superResolutionWeights.map { try MLXNativeSuperResolver(weightsURL: $0) }
    let dlss = try options.dlssSuperResolutionModel.map { try MLXNativeDLSSSuperResolver(packageURL: $0) }
    let srFlow = options.temporal && dlss != nil
      ? try NativeOpticalFlow(width: width, height: height, mode: options.motion) : nil
    let flow = options.temporal && renderer != nil
      ? try NativeOpticalFlow(width: width, height: height, mode: options.motion) : nil
    let factor = generator == nil ? 1 : options.frameGenerationFactor
    let timeScale = options.slowMotion ? factor : 1
    let audio = options.includeAudio
      ? try await NativeAudioReader.open(url: input, start: first.time, timeScale: timeScale) : nil
    // AVAssetWriter can leave safe-save sidecars after cancellation. Keep its
    // entire transaction in a private directory and publish only the final file.
    let staging = output.deletingLastPathComponent().appendingPathComponent(
      ".\(output.deletingPathExtension().lastPathComponent)-\(UUID().uuidString).tmp")
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false,
      attributes: [.posixPermissions: 0o700])
    defer { try? FileManager.default.removeItem(at: staging) }
    let temporary = staging.appendingPathComponent(output.lastPathComponent)
    let scale = upscaler == nil && dlss == nil ? 1 : 2
    let writer = try NativeVideoWriter(url: temporary, width: width * scale, height: height * scale,
      frameRate: Double(reader.nominalFrameRate) * Double(factor) / Double(timeScale), options: options, hasAudio: audio != nil)
    // Each receiver can suspend until the other track advances. Feeding audio
    // and video from the same task can deadlock the muxer's interleave queues.
    let audioTask = Task {
      let started = ContinuousClock.now
      do {
        if let audio {
          while let sample = try await audio.next(upTo: .positiveInfinity) {
            try await writer.appendAudio(sample)
          }
        }
        await writer.finishAudio()
        return audio == nil ? 0 : seconds(since: started)
      } catch {
        await writer.cancel()
        throw error
      }
    }
    timing.setupSeconds = seconds(since: setupStarted)
    var inputFrames = 0, outputFrames = 0, resets = 0, renderIndex = 0
    var lastDuration = first.duration
    var previousSource: NativeDecodedFrame?
    var previousDisplay: MLXVideoFrame?

    func render(_ source: MLXVideoFrame, prepared: MLXVideoMotion?? = nil, scheduled: Bool = false,
                isolation: isolated (any Actor)? = #isolation) async throws -> MLXVideoFrame {
      guard let renderer else { return source }
      let motion: MLXVideoMotion?
      if let prepared {
        motion = prepared
      } else {
        let motionStarted = ContinuousClock.now
        motion = try await flow?.prepare(source, index: renderIndex, sceneCutThreshold: options.sceneCutThreshold)
        if flow != nil { timing.motionSeconds += seconds(since: motionStarted) }
      }
      if motion?.reset == true { resets += 1 }
      let renderingStarted = ContinuousClock.now
      let result = try await renderer.renderVideoFrame(source, motion: motion,
        context: NeuralRenderFrameContext(streamID: 1, frameIndex: UInt64(renderIndex)),
        processingScale: options.processingScale, temporal: options.temporal, evaluated: !scheduled)
      timing.renderingSeconds += seconds(since: renderingStarted)
      renderIndex += 1
      return result
    }

    // Generated frames before a rendered frame, then the frame itself, then the
    // bookkeeping the sequential loop does per input frame.
    func emitRendered(_ frame: NativeDecodedFrame, _ display: MLXVideoFrame,
                      isolation: isolated (any Actor)? = #isolation) async throws {
      if let generator, let previousSource, let previousDisplay {
        let generationStarted = ContinuousClock.now
        let intermediates = try await generator.interpolate(previousDisplay, display, factor: factor)
        timing.generationSeconds += seconds(since: generationStarted)
        for (index, intermediate) in intermediates.enumerated() {
          let time = previousSource.time + CMTimeMultiplyByRatio(frame.time - previousSource.time,
            multiplier: Int32(index + 1), divisor: Int32(factor))
          try await emit(intermediate, sourceTime: time)
        }
      }
      try await emit(display, sourceTime: frame.time)
      previousDisplay = display
      inputFrames += 1
      lastDuration = frame.duration
      previousSource = frame
      await progress(MediaProgress(inputFrames: inputFrames, outputFrames: outputFrames,
        estimatedInputFrames: reader.estimatedFrames, sceneResets: resets, elapsedSeconds: seconds(since: started)))
    }

    func emit(_ frame: MLXVideoFrame, sourceTime: CMTime, isolation: isolated (any Actor)? = #isolation) async throws {
      try Task.checkCancellation()
      var frame = frame
      if let dlss {
        let motionStarted = ContinuousClock.now
        let motion = try await srFlow?.prepare(frame, index: outputFrames, sceneCutThreshold: options.sceneCutThreshold)
        if srFlow != nil { timing.motionSeconds += seconds(since: motionStarted) }
        if renderer == nil && motion?.reset == true { resets += 1 }
        let upscalingStarted = ContinuousClock.now
        frame = try await dlss.upscale(frame, motion: motion, temporal: options.temporal)
        timing.superResolutionSeconds = (timing.superResolutionSeconds ?? 0) + seconds(since: upscalingStarted)
      }
      if let upscaler {
        let upscalingStarted = ContinuousClock.now
        frame = try await upscaler.upscale(frame)
        timing.superResolutionSeconds = (timing.superResolutionSeconds ?? 0) + seconds(since: upscalingStarted)
      }
      let time = CMTimeMultiply(sourceTime - first.time, multiplier: Int32(timeScale))
      let encodingStarted = ContinuousClock.now
      try await writer.append(frame, at: time)
      timing.encodingSeconds += seconds(since: encodingStarted)
      outputFrames += 1
    }

    do {
      var current: NativeDecodedFrame? = first
      let overlap = Self.overlapMotionEnabled && options.order != .generationThenRendering && flow != nil && renderer != nil
      let threshold = options.sceneCutThreshold
      // Motion for `current`, started while the previous frame rendered.
      // The rendered frame waiting to be encoded while the next one renders.
      var pendingDisplay: (frame: NativeDecodedFrame, display: MLXVideoFrame)?
      var pendingMotion: Task<(MLXVideoMotion?, Double), any Error>? = overlap ? Task.detached { [flow, rgb = first.rgb] in
        let started = ContinuousClock.now
        let motion = try await flow!.prepare(rgb, index: 0, sceneCutThreshold: threshold)
        let elapsed = started.duration(to: .now).components
        return (motion, Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18)
      } : nil
      while let frame = current {
        try Task.checkCancellation()
        if let previousSource, frame.time <= previousSource.time {
          throw MLXMediaError("Input video timestamps are not increasing")
        }
        if overlap {
          // Wait for this frame's motion (in order), decode the next frame and
          // start its motion estimation, schedule this frame's rendering on the
          // GPU, and only then encode the previous frame, so the GPU keeps
          // working while the encoder and the decoder do their part.
          let (motion, motionSeconds) = try await pendingMotion!.value
          timing.motionSeconds += motionSeconds
          let decodingStarted = ContinuousClock.now
          let next = try await reader.next()
          timing.decodingSeconds += seconds(since: decodingStarted)
          if let next {
            let index = renderIndex + 1
            pendingMotion = Task.detached { [flow, rgb = next.rgb] in
              let started = ContinuousClock.now
              let motion = try await flow!.prepare(rgb, index: index, sceneCutThreshold: threshold)
              let elapsed = started.duration(to: .now).components
              return (motion, Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18)
            }
          } else {
            pendingMotion = nil
          }
          let display = try await render(frame.rgb, prepared: .some(motion), scheduled: true)
          if let pending = pendingDisplay {
            try await emitRendered(pending.frame, pending.display)
          }
          pendingDisplay = (frame, display)
          current = next
          continue
        }
        if options.order == .generationThenRendering {
          if let generator, let previousSource {
            let generationStarted = ContinuousClock.now
            let intermediates = try await generator.interpolate(previousSource.rgb, frame.rgb, factor: factor)
            timing.generationSeconds += seconds(since: generationStarted)
            for (index, intermediate) in intermediates.enumerated() {
              let time = previousSource.time + CMTimeMultiplyByRatio(frame.time - previousSource.time,
                multiplier: Int32(index + 1), divisor: Int32(factor))
              try await emit(render(intermediate), sourceTime: time)
            }
          }
          try await emit(render(frame.rgb), sourceTime: frame.time)
        } else {
          let display = try await render(frame.rgb)
          if let generator, let previousSource, let previousDisplay {
            let generationStarted = ContinuousClock.now
            let intermediates = try await generator.interpolate(previousDisplay, display, factor: factor)
            timing.generationSeconds += seconds(since: generationStarted)
            for (index, intermediate) in intermediates.enumerated() {
              let time = previousSource.time + CMTimeMultiplyByRatio(frame.time - previousSource.time,
                multiplier: Int32(index + 1), divisor: Int32(factor))
              try await emit(intermediate, sourceTime: time)
            }
          }
          try await emit(display, sourceTime: frame.time)
          previousDisplay = display
        }
        inputFrames += 1
        lastDuration = frame.duration
        previousSource = frame
        await progress(MediaProgress(inputFrames: inputFrames, outputFrames: outputFrames,
          estimatedInputFrames: reader.estimatedFrames, sceneResets: resets, elapsedSeconds: seconds(since: started)))
        let decodingStarted = ContinuousClock.now
        current = try await reader.next()
        timing.decodingSeconds += seconds(since: decodingStarted)
      }
      if let pending = pendingDisplay {
        try await emitRendered(pending.frame, pending.display)
      }
      // The final original frame lasts one output interval, matching (N-1)*F+1
      // frames. VFR intervals before it retain their original presentation times.
      let end = CMTimeMultiply(previousSource!.time - first.time +
        CMTimeMultiplyByRatio(lastDuration, multiplier: 1, divisor: Int32(factor)), multiplier: Int32(timeScale))
      let encodingStarted = ContinuousClock.now
      await audio?.limit(to: end)
      await writer.finishVideo(at: end)
      timing.audioSeconds = try await audioTask.value
      try await writer.finish(at: end)
      timing.encodingSeconds += seconds(since: encodingStarted)
      try Task.checkCancellation()
      try FileManager.default.moveItem(at: temporary, to: output)
      await reader.cancel()
      await audio?.cancel()
      return MediaProcessingResult(output: output, inputFrames: inputFrames, outputFrames: outputFrames,
        sceneResets: resets, elapsedSeconds: seconds(since: started), motionBackend: srFlow?.backend ?? flow?.backend ?? "disabled", timing: timing)
    } catch {
      audioTask.cancel()
      await writer.cancel()
      await reader.cancel()
      await audio?.cancel()
      _ = try? await audioTask.value
      throw error
    }
  }

  private func makeRenderer(width: Int, height: Int, options: MediaProcessingOptions) throws -> MLXNeuralRenderingDeviceTemporalBackend? {
    guard let model = options.renderingModel else { return nil }
    return try MLXNeuralRenderingDeviceTemporalBackend(packageURL: model, executionMode: .metalFused,
      computePrecision: options.precision, controlMaskIntensity: options.intensity,
      featureControls: options.profile.featureControls,
      videoOutput: try MLXVideoOutputOptions(width: width, height: height,
        detailStrength: options.detailStrength, colourStrength: options.colourStrength, radius: options.detailRadius))
  }

  private func seconds(since start: ContinuousClock.Instant) -> Double {
    let elapsed = start.duration(to: .now).components
    return Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
  }
}
