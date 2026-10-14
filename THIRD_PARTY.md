# Third-Party Licenses

This file lists third-party models and assets distributed with or downloaded by voxtype.

## Fluid Speech Mini

- **File:** `fluid-speech-mini-q2k-q4k.gguf` (downloaded by `voxtype setup --download --model fluid-speech-mini`)
- **Source:** https://huggingface.co/FluidVoice-Labs/fluid-speech-mini-gguf
- **License:** CC BY 4.0
- **Copyright:** Copyright (c) 2026 FluidVoice
- **Attribution:** Fluid Speech Mini by FluidVoice. Built on NVIDIA Parakeet TDT 0.6B v3 and Moondream Parakeet Ultra.

The model is built on NVIDIA Parakeet TDT 0.6B v3
(https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3, CC BY 4.0, NVIDIA
Corporation) and Moondream Parakeet Ultra
(https://huggingface.co/moondream/parakeet-ultra, CC BY 4.0, Moondream). The
decoder and joint network are Parakeet Ultra's. The GGUF file is a converted
and quantized copy of the original weights. Voxtype does not ship the file; it
is mirrored on the voxtype model CDN and downloaded on request.

## parakeet.cpp

- **Library:** `libparakeet.so`, loaded at runtime by the `parakeet-cpp` engine
- **Source:** https://github.com/mudler/parakeet.cpp
- **License:** MIT

Voxtype does not distribute parakeet.cpp. Users install the shared library
from the parakeet.cpp releases.

## GTCRN Speech Enhancement Model

- **File:** `gtcrn_simple.onnx` (auto-downloaded on first meeting start)
- **Source:** https://github.com/Xiaobin-Rong/gtcrn
- **License:** MIT
- **Copyright:** Copyright (c) 2024 Rong Xiaobin

Used for removing speaker bleed-through from microphone audio during meeting
transcription. The model is downloaded from the
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project's release assets.
