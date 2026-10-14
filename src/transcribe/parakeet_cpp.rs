//! parakeet.cpp transcription (ggml/GGUF Parakeet models)
//!
//! parakeet.cpp ships a flat C API in `libparakeet.so`. Voxtype loads it with
//! `dlopen` inside a separate worker process (`voxtype parakeet-cpp-worker`)
//! rather than in the daemon: whisper-rs already links its own copy of ggml
//! into this binary, and two ggml builds must never share an address space.
//! The worker also keeps GPU memory out of the daemon.
//!
//! The library is found at runtime, so this engine needs no Cargo feature
//! and no ONNX Runtime.
//!
//! Protocol (worker stdio):
//! 1. Worker loads the library and model, then writes `READY\n` to stdout.
//!    On failure it writes one JSON error line and exits.
//! 2. Parent sends requests: `[u32 sample_count LE][f32 samples LE...]`.
//! 3. Worker answers each request with one JSON line (`WorkerResponse`).
//! 4. Worker exits when stdin closes.

use super::worker::{WorkerResponse, READY_SIGNAL};
use super::Transcriber;
use crate::config::ParakeetCppConfig;
use crate::error::TranscribeError;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

/// File name of the shared library on this platform
#[cfg(target_os = "macos")]
const LIBRARY_NAME: &str = "libparakeet.dylib";
#[cfg(not(target_os = "macos"))]
const LIBRARY_NAME: &str = "libparakeet.so";

/// Directories searched for the library besides `LD_LIBRARY_PATH`
const LIBRARY_DIRS: &[&str] = &[
    "/usr/local/lib",
    "/usr/lib",
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/opt/parakeet.cpp/lib",
];

/// Where users get the library, quoted in error messages
const INSTALL_HINT: &str = "Install parakeet.cpp's shared library (libparakeet.so) from \
https://github.com/mudler/parakeet.cpp/releases (the parakeet-*-lib-linux-* archive for your GPU), \
or set [parakeet_cpp] library_path in config.toml.";

/// Longest audio accepted by the worker (10 minutes at 16 kHz)
const MAX_SAMPLES: usize = 16000 * 60 * 10;

// ---------------------------------------------------------------------------
// Resolution helpers (parent side)
// ---------------------------------------------------------------------------

/// Find `libparakeet.so`: explicit path, then library directories.
///
/// Returns an absolute path when the file is found, otherwise the bare
/// library name so the worker's `dlopen` can still try the loader's own
/// search path (ld.so.cache, RPATH).
fn resolve_library(configured: Option<&str>) -> Result<PathBuf, TranscribeError> {
    if let Some(path) = configured {
        let path = PathBuf::from(path);
        let candidate = if path.is_dir() {
            path.join(LIBRARY_NAME)
        } else {
            path
        };
        if candidate.exists() {
            return Ok(candidate);
        }
        return Err(TranscribeError::InitFailed(format!(
            "parakeet.cpp library not found at {}\n  {}",
            candidate.display(),
            INSTALL_HINT
        )));
    }

    let mut dirs: Vec<PathBuf> = std::env::var("LD_LIBRARY_PATH")
        .unwrap_or_default()
        .split(':')
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .collect();
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".local/lib"));
    }
    dirs.extend(LIBRARY_DIRS.iter().map(PathBuf::from));

    Ok(dirs
        .into_iter()
        .map(|d| d.join(LIBRARY_NAME))
        .find(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(LIBRARY_NAME)))
}

/// Resolve a model name or path to a `.gguf` file.
///
/// Accepts an absolute or relative `.gguf` path, a file or directory under
/// the models directory, or `./models/<name>`.
pub(super) fn resolve_model_path(model: &str) -> Result<PathBuf, TranscribeError> {
    let direct = PathBuf::from(model);
    let models_dir = crate::config::Config::models_dir();
    let candidates = [
        direct.clone(),
        models_dir.join(model),
        PathBuf::from("models").join(model),
    ];

    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
        if candidate.is_dir() {
            let mut ggufs: Vec<PathBuf> = std::fs::read_dir(candidate)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
                .collect();
            ggufs.sort();
            if let Some(first) = ggufs.into_iter().next() {
                return Ok(first);
            }
        }
    }

    Err(TranscribeError::ModelNotFound(format!(
        "parakeet.cpp model '{}' not found. Looked in:\n  - {}\n  - {}\n  - ./models/{}\n\n  \
         Pass a path to a .gguf file, or place the model under the models directory.",
        model,
        direct.display(),
        models_dir.join(model).display(),
        model
    )))
}

// ---------------------------------------------------------------------------
// Parent side: manages the worker process
// ---------------------------------------------------------------------------

/// A running worker with the model loaded
struct Worker {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl Worker {
    /// Send one clip and read the JSON response line.
    fn request(&mut self, samples: &[f32]) -> Result<WorkerReply, io::Error> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "worker stdin closed"))?;

        let mut payload = Vec::with_capacity(4 + samples.len() * 4);
        payload.extend_from_slice(&(samples.len() as u32).to_le_bytes());
        for s in samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }
        stdin.write_all(&payload)?;
        stdin.flush()?;

        let mut line = String::new();
        if self.stdout.read_line(&mut line)? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "worker exited without a response",
            ));
        }
        serde_json::from_str::<WorkerReply>(&line)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Closing stdin tells the worker to exit; kill covers a stuck one.
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Debug, serde::Deserialize)]
struct WorkerReply {
    ok: bool,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

/// parakeet.cpp transcriber backed by an isolated worker process
pub struct ParakeetCppTranscriber {
    model_path: PathBuf,
    library: PathBuf,
    device: Option<String>,
    threads: Option<usize>,
    on_demand: bool,
    worker: Mutex<Option<Worker>>,
}

impl ParakeetCppTranscriber {
    pub fn new(config: &ParakeetCppConfig) -> Result<Self, TranscribeError> {
        let model_path = resolve_model_path(&config.model)?;
        let library = resolve_library(config.library_path.as_deref())?;

        let transcriber = Self {
            model_path,
            library,
            device: config.device.clone(),
            threads: config.threads,
            on_demand: config.on_demand_loading,
            worker: Mutex::new(None),
        };

        // Resident mode: load now so config errors surface at startup and
        // the first recording does not pay the model load.
        if !transcriber.on_demand {
            let worker = transcriber.spawn()?;
            *transcriber.worker.lock().unwrap() = Some(worker);
        }

        Ok(transcriber)
    }

    fn spawn(&self) -> Result<Worker, TranscribeError> {
        let exe = std::env::current_exe().map_err(|e| {
            TranscribeError::InitFailed(format!("Cannot find voxtype executable: {}", e))
        })?;

        let mut cmd = Command::new(exe);
        cmd.arg("parakeet-cpp-worker")
            .arg("--model")
            .arg(&self.model_path)
            .arg("--library")
            .arg(&self.library)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        // parakeet.cpp reads its device from the environment, and its CPU
        // kernels use OpenMP.
        if let Some(device) = &self.device {
            cmd.env("PARAKEET_DEVICE", device);
        }
        if let Some(threads) = self.threads {
            cmd.env("OMP_NUM_THREADS", threads.to_string());
        }

        let start = std::time::Instant::now();
        let mut child = cmd.spawn().map_err(|e| {
            TranscribeError::InitFailed(format!("Failed to spawn parakeet-cpp-worker: {}", e))
        })?;

        let stdin = child.stdin.take();
        let mut stdout = BufReader::new(child.stdout.take().ok_or_else(|| {
            TranscribeError::InitFailed("Worker stdout not available".to_string())
        })?);

        let mut line = String::new();
        let read = stdout.read_line(&mut line).map_err(|e| {
            TranscribeError::InitFailed(format!("Failed to read worker READY signal: {}", e))
        })?;

        if line.trim() != READY_SIGNAL {
            let _ = child.kill();
            let _ = child.wait();
            let detail = serde_json::from_str::<WorkerReply>(&line)
                .ok()
                .and_then(|r| r.error)
                .unwrap_or_else(|| {
                    if read == 0 {
                        "worker exited before loading the model (see log above)".to_string()
                    } else {
                        format!("unexpected worker output: {:?}", line.trim())
                    }
                });
            return Err(TranscribeError::InitFailed(detail));
        }

        tracing::info!(
            model = %self.model_path.display(),
            "parakeet.cpp worker ready in {:.2}s",
            start.elapsed().as_secs_f32()
        );

        Ok(Worker {
            child,
            stdin,
            stdout,
        })
    }
}

impl Transcriber for ParakeetCppTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Err(TranscribeError::AudioFormat(
                "Empty audio buffer".to_string(),
            ));
        }
        if samples.len() > MAX_SAMPLES {
            return Err(TranscribeError::AudioFormat(format!(
                "Audio too long for parakeet.cpp worker: {} samples (max {})",
                samples.len(),
                MAX_SAMPLES
            )));
        }

        let mut guard = self.worker.lock().unwrap();

        // A worker that died between recordings gets one fresh start.
        for attempt in 0..2 {
            if guard.is_none() {
                *guard = Some(self.spawn()?);
            }
            let worker = guard.as_mut().expect("worker was just spawned");

            match worker.request(samples) {
                Ok(reply) => {
                    if self.on_demand {
                        *guard = None;
                    }
                    return if reply.ok {
                        Ok(reply.text.unwrap_or_default().trim().to_string())
                    } else {
                        Err(TranscribeError::InferenceFailed(
                            reply.error.unwrap_or_else(|| "unknown worker error".into()),
                        ))
                    };
                }
                Err(e) => {
                    tracing::warn!("parakeet.cpp worker request failed: {}", e);
                    *guard = None;
                    if attempt == 1 {
                        return Err(TranscribeError::InferenceFailed(format!(
                            "parakeet.cpp worker failed: {}",
                            e
                        )));
                    }
                }
            }
        }
        unreachable!("loop returns on the second attempt")
    }

    /// On-demand mode starts loading when recording starts, so the model
    /// loads while the user is speaking.
    fn prepare(&self) {
        if !self.on_demand {
            return;
        }
        let mut guard = self.worker.lock().unwrap();
        if guard.is_none() {
            match self.spawn() {
                Ok(worker) => *guard = Some(worker),
                Err(e) => tracing::warn!("Failed to prepare parakeet.cpp worker: {}", e),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Worker side: dlopen wrapper and request loop
// ---------------------------------------------------------------------------

/// Minimal view of the parakeet.cpp C API (ABI 10, `parakeet_capi.h`)
struct Api {
    // Keeps the library mapped for as long as the function pointers live.
    _lib: libloading::Library,
    load: unsafe extern "C" fn(*const c_char) -> *mut c_void,
    load_error: Option<unsafe extern "C" fn() -> *const c_char>,
    free: unsafe extern "C" fn(*mut c_void),
    transcribe_pcm:
        unsafe extern "C" fn(*mut c_void, *const f32, c_int, c_int, c_int) -> *mut c_char,
    free_string: unsafe extern "C" fn(*mut c_char),
    last_error: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    abi_version: unsafe extern "C" fn() -> c_int,
}

impl Api {
    fn open(library: &Path) -> Result<Self, String> {
        // SAFETY: loading a shared library runs its initializers. The path is
        // user-configured or found in standard library directories, the same
        // trust level as any dynamically linked dependency.
        let lib = unsafe { libloading::Library::new(library) }.map_err(|e| {
            format!(
                "Failed to load {}: {}\n  {}",
                library.display(),
                e,
                INSTALL_HINT
            )
        })?;

        // SAFETY: each symbol is looked up by the exact name and signature in
        // parakeet_capi.h, and `_lib` outlives the copied pointers.
        unsafe {
            macro_rules! sym {
                ($name:literal) => {
                    *lib.get(concat!($name, "\0").as_bytes())
                        .map_err(|e| format!("{} is missing {}: {}", library.display(), $name, e))?
                };
            }
            let load_error = lib
                .get::<unsafe extern "C" fn() -> *const c_char>(b"parakeet_capi_load_error\0")
                .ok()
                .map(|s| *s);
            Ok(Self {
                load: sym!("parakeet_capi_load"),
                free: sym!("parakeet_capi_free"),
                transcribe_pcm: sym!("parakeet_capi_transcribe_pcm"),
                free_string: sym!("parakeet_capi_free_string"),
                last_error: sym!("parakeet_capi_last_error"),
                abi_version: sym!("parakeet_capi_abi_version"),
                load_error,
                _lib: lib,
            })
        }
    }
}

/// A loaded parakeet.cpp model
struct Model<'a> {
    api: &'a Api,
    ctx: *mut c_void,
}

impl<'a> Model<'a> {
    fn load(api: &'a Api, path: &Path) -> Result<Self, String> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| "model path contains a NUL byte".to_string())?;
        // SAFETY: c_path is a valid NUL-terminated string.
        let ctx = unsafe { (api.load)(c_path.as_ptr()) };
        if ctx.is_null() {
            let reason = api
                .load_error
                // SAFETY: the function takes no arguments and returns a
                // valid string that lives until the next load call.
                .map(|f| unsafe { cstr_to_string(f()) })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "unknown error".to_string());
            return Err(format!(
                "parakeet.cpp could not load {}: {}",
                path.display(),
                reason
            ));
        }
        Ok(Self { api, ctx })
    }

    fn transcribe(&self, samples: &[f32]) -> Result<String, String> {
        // SAFETY: ctx is non-null and owned by self; samples is a valid
        // buffer of `len` f32 values; decoder 0 selects the model default.
        let text = unsafe {
            (self.api.transcribe_pcm)(self.ctx, samples.as_ptr(), samples.len() as c_int, 16000, 0)
        };
        if text.is_null() {
            // SAFETY: last_error returns a valid string for this context.
            let reason = unsafe { cstr_to_string((self.api.last_error)(self.ctx)) };
            return Err(if reason.is_empty() {
                "transcription failed".to_string()
            } else {
                reason
            });
        }
        // SAFETY: text is a malloc'd NUL-terminated string we now own.
        let out = unsafe { cstr_to_string(text) };
        unsafe { (self.api.free_string)(text) };
        Ok(out)
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        // SAFETY: ctx came from parakeet_capi_load and is freed exactly once.
        unsafe { (self.api.free)(self.ctx) };
    }
}

/// Copy a C string into a `String` (empty for NULL)
///
/// # Safety
/// `ptr` must be NULL or point to a valid NUL-terminated string.
unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Entry point for `voxtype parakeet-cpp-worker`
pub fn run_worker(model_path: &Path, library: &Path) -> anyhow::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let fail = |out: &mut io::StdoutLock, msg: String| {
        eprintln!("[parakeet-cpp-worker] {}", msg);
        if let Ok(json) = serde_json::to_string(&WorkerResponse::error(msg)) {
            let _ = writeln!(out, "{}", json);
            let _ = out.flush();
        }
    };

    let api = match Api::open(library) {
        Ok(api) => api,
        Err(msg) => {
            fail(&mut out, msg);
            return Ok(());
        }
    };
    // SAFETY: abi_version takes no arguments.
    let abi = unsafe { (api.abi_version)() };
    eprintln!("[parakeet-cpp-worker] library ABI v{}", abi);

    let load_start = std::time::Instant::now();
    let model = match Model::load(&api, model_path) {
        Ok(m) => m,
        Err(msg) => {
            fail(&mut out, msg);
            return Ok(());
        }
    };
    eprintln!(
        "[parakeet-cpp-worker] model loaded in {:.2}s",
        load_start.elapsed().as_secs_f32()
    );

    // The first inference pays one-time GPU setup (shader compilation on
    // Vulkan) that does not depend on audio length. Pay it here, before
    // READY, so it is hidden behind startup or the user's recording instead
    // of the first transcription. The result is irrelevant.
    let warm_start = std::time::Instant::now();
    let _ = model.transcribe(&vec![0.0f32; 8000]);
    eprintln!(
        "[parakeet-cpp-worker] warm-up in {:.2}s",
        warm_start.elapsed().as_secs_f32()
    );

    writeln!(out, "{}", READY_SIGNAL)?;
    out.flush()?;

    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let mut count_buf = [0u8; 4];

    loop {
        match stdin.read_exact(&mut count_buf) {
            Ok(()) => {}
            // Parent closed stdin: normal shutdown
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        let count = u32::from_le_bytes(count_buf) as usize;
        if count == 0 || count > MAX_SAMPLES {
            fail(&mut out, format!("invalid sample count: {}", count));
            return Ok(());
        }

        let mut samples = Vec::with_capacity(count);
        let mut word = [0u8; 4];
        for _ in 0..count {
            stdin.read_exact(&mut word)?;
            samples.push(f32::from_le_bytes(word));
        }

        let start = std::time::Instant::now();
        let response = match model.transcribe(&samples) {
            Ok(text) => {
                eprintln!(
                    "[parakeet-cpp-worker] {:.2}s of audio in {:.2}s",
                    count as f32 / 16000.0,
                    start.elapsed().as_secs_f32()
                );
                WorkerResponse::success(text, None)
            }
            Err(msg) => WorkerResponse::error(msg),
        };
        writeln!(out, "{}", serde_json::to_string(&response)?)?;
        out.flush()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_resolve_model_path_absolute_file() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("m.gguf");
        std::fs::write(&file, b"x").unwrap();
        assert_eq!(resolve_model_path(file.to_str().unwrap()).unwrap(), file);
    }

    #[test]
    fn test_resolve_model_path_directory_picks_gguf() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("README.md"), b"x").unwrap();
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"x").unwrap();
        assert_eq!(
            resolve_model_path(dir.path().to_str().unwrap()).unwrap(),
            gguf
        );
    }

    #[test]
    fn test_resolve_model_path_missing() {
        let err = resolve_model_path("/nonexistent/none.gguf").unwrap_err();
        assert!(matches!(err, TranscribeError::ModelNotFound(_)));
    }

    #[test]
    fn test_resolve_library_explicit_missing() {
        let err = resolve_library(Some("/nonexistent/libparakeet.so")).unwrap_err();
        assert!(err.to_string().contains("releases"));
    }

    #[test]
    fn test_resolve_library_explicit_directory() {
        let dir = TempDir::new().unwrap();
        let lib = dir.path().join(LIBRARY_NAME);
        std::fs::write(&lib, b"x").unwrap();
        assert_eq!(resolve_library(dir.path().to_str()).unwrap(), lib);
    }

    #[test]
    fn test_worker_reply_parses_error() {
        let r: WorkerReply = serde_json::from_str(r#"{"ok":false,"error":"boom"}"#).unwrap();
        assert!(!r.ok);
        assert_eq!(r.error.as_deref(), Some("boom"));
    }

    #[test]
    fn test_worker_reply_parses_text() {
        let r: WorkerReply = serde_json::from_str(r#"{"ok":true,"text":"hi"}"#).unwrap();
        assert!(r.ok);
        assert_eq!(r.text.as_deref(), Some("hi"));
    }
}
