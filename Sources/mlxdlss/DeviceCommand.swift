import DLSSMLX
import Foundation

/// `mlxdlss device`: the GPU MLX runs on and whether MLX uses its Neural Accelerators.
enum DeviceCommand {
    static func run(arguments: [String]) throws {
        guard arguments.isEmpty else {
            throw CLIError.usage("device takes no arguments")
        }
        guard let profile = MLXGPUProfile.current else {
            throw CLIError.usage("no Metal device is available")
        }
        try CLIOutput.writeJSON(profile.summary)
    }
}
