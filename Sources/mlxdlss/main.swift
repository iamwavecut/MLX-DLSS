import Foundation
import DLSSMLX

// macOS kills GPU command buffers that block display compositing
// ("Impacting Interactivity", IOGPU 0xe) while the display is active. MLX cannot
// catch that from the Metal completion thread, so relax the driver's context-store
// timeout before the first Metal device is created (the workaround recommended in
// ml-explore/mlx#3267); an explicit value in the environment is respected.
setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", 0)

enum CLIError: Error, Sendable {
    case usage(String)
    case missingOutput(String)
    case destinationExists(URL)
}

enum CLIOutput {
    static func writeJSON(_ object: [String: Any]) throws {
        var data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        data.append(0x0a)
        FileHandle.standardOutput.write(data)
    }

    static func writeEncodable<T: Encodable>(_ value: T) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(value)
        data.append(0x0a)
        FileHandle.standardOutput.write(data)
    }

    static func writeError(_ error: any Error) {
        let message: String
        switch error {
        case let CLIError.usage(detail):
            message = "usage error: \(detail)"
        case let CLIError.missingOutput(name):
            message = "missing output: \(name)"
        case let CLIError.destinationExists(url):
            message = "destination already exists: \(url.path)"
        case let DemoModelPackageError.destinationExists(url):
            message = "destination already exists: \(url.path)"
        case let DemoModelPackageError.parentIsNotDirectory(url):
            message = "parent is not a directory: \(url.path)"
        default:
            message = String(describing: error)
        }
        FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    }
}

func runCommand(_ arguments: [String]) async throws {
    guard let command = arguments.first else {
        throw CLIError.usage(
            "expected process-video, process-image, preview-stream, inspect, run, run-sequence, render-image, framegen, framegen-stream, or stream"
        )
    }
    let commandArguments = Array(arguments.dropFirst())
    switch command {
    case "preview-stream":
        try await PreviewStreamCommand.run(arguments: commandArguments)
    case "process-video":
        try await ProcessMediaCommand.run(arguments: commandArguments, video: true)
    case "process-image":
        try await ProcessMediaCommand.run(arguments: commandArguments, video: false)
    case "render-image":
        try await RenderImageCommand.run(arguments: commandArguments)
    case "inspect":
        try InspectCommand.run(arguments: commandArguments)
    case "run":
        try await RunCommand.run(arguments: commandArguments)
    case "run-sequence":
        try await RunSequenceCommand.run(arguments: commandArguments)
    case "stream":
        try await StreamCommand.run(arguments: commandArguments)
    case "framegen":
        try await FrameGenCommand.run(arguments: commandArguments)
    case "framegen-stream":
        try await FrameGenStreamCommand.run(arguments: commandArguments)
    default:
        throw CLIError.usage("unknown command '\(command)'")
    }
}

do {
    try await runCommand(Array(CommandLine.arguments.dropFirst()))
} catch {
    CLIOutput.writeError(error)
    exit(2)
}
