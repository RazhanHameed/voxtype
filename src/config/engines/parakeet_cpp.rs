//! parakeet.cpp engine configuration.

use serde::{Deserialize, Serialize};

use super::super::default_on_demand_loading;

/// parakeet.cpp speech-to-text configuration (ggml/GGUF Parakeet models).
///
/// Runs through the system `libparakeet.so` inside an isolated worker
/// process, so no ONNX Runtime and no extra Cargo feature are needed.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ParakeetCppConfig {
    /// Model name from `voxtype info models`, or a path to a `.gguf` file
    pub model: String,

    /// Path to `libparakeet.so` (or `.dylib`). Default: search the usual
    /// library directories and `LD_LIBRARY_PATH`
    #[serde(default)]
    pub library_path: Option<String>,

    /// ggml device to run on: "cpu", or a name such as "Vulkan0" or "CUDA0".
    /// Default: parakeet.cpp's own choice (the first GPU it reports, which on
    /// a laptop is often the integrated GPU)
    #[serde(default)]
    pub device: Option<String>,

    /// Number of CPU threads (default: runtime default)
    #[serde(default)]
    pub threads: Option<usize>,

    /// Load model on-demand when recording starts (true) or keep loaded (false)
    #[serde(default = "default_on_demand_loading")]
    pub on_demand_loading: bool,
}

impl Default for ParakeetCppConfig {
    fn default() -> Self {
        Self {
            model: "fluid-speech-mini".to_string(),
            library_path: None,
            device: None,
            threads: None,
            on_demand_loading: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let c = ParakeetCppConfig::default();
        assert_eq!(c.model, "fluid-speech-mini");
        assert!(c.library_path.is_none());
        assert!(!c.on_demand_loading);
    }

    #[test]
    fn test_parse_minimal() {
        let c: ParakeetCppConfig = toml::from_str(r#"model = "/m/x.gguf""#).unwrap();
        assert_eq!(c.model, "/m/x.gguf");
        assert!(c.device.is_none());
    }
}
