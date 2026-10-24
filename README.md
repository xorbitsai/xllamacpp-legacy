> [!IMPORTANT]
> **xllamacpp has moved to [xorbitsai/xllamacpp](https://github.com/xorbitsai/xllamacpp).**
> Please use the new repository for ongoing development, current documentation, issues, and pull requests.
> This legacy repository is retained for historical issues, pull requests, releases, and downloads.

<div align="center">
<img src="./assets/logo.png" width="400px" alt="xllamacpp Python bindings for llama.cpp" />

# xllamacpp — Python bindings for llama.cpp

[![PyPI Latest Release](https://img.shields.io/pypi/v/xllamacpp.svg?style=for-the-badge)](https://pypi.org/project/xllamacpp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/xllamacpp/)
[![License](https://img.shields.io/pypi/l/xllamacpp.svg?style=for-the-badge)](https://github.com/xorbitsai/xllamacpp/blob/main/LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)](#cuda-gpu-architecture-coverage)
[![ROCm](https://img.shields.io/badge/ROCm-ED1C24.svg?style=for-the-badge&logo=amd&logoColor=white)](#rocm-hip-gpu-architecture-coverage)
[![Vulkan](https://img.shields.io/badge/Vulkan-AC162C.svg?style=for-the-badge&logo=vulkan&logoColor=white)](#installation)
[![Apple Metal](https://img.shields.io/badge/Apple_Metal-000000.svg?style=for-the-badge&logo=apple&logoColor=white)](#installation)

</div>
<br />

**xllamacpp** is a fast, Cython-based Python binding for
[llama.cpp](https://github.com/ggml-org/llama.cpp). It runs GGUF large language
models locally and exposes llama.cpp's native server and parameter APIs directly
to Python, including OpenAI-compatible chat completions, continuous batching,
embeddings, reranking, multimodal input, and structured output.

Prebuilt Python wheels are available for CPU, Metal, CUDA, ROCm, and Vulkan, so
you can run local LLM inference on Linux, macOS, or Windows without compiling
llama.cpp yourself.

## Why xllamacpp?

- **Thin native bindings:** Cython extension types stay close to llama.cpp's C/C++ APIs.
- **Production server engine:** llama.cpp's native C++ server provides continuous batching, parallel decoding, prompt caching, and thread-safe inference.
- **OpenAI-compatible API:** use chat completions, text completions, embeddings, reranking, and the Responses API from Python or over HTTP.
- **Broad hardware support:** install prebuilt CPU, Metal, NVIDIA CUDA, AMD ROCm, or Vulkan wheels.
- **Modern model features:** GGUF models, multimodal inference, LoRA adapters, speculative decoding, reasoning models, and JSON Schema-constrained output with LLGuidance.

## Quick start: run llama.cpp from Python

Install the CPU or macOS wheel from PyPI:

```sh
pip install -U xllamacpp
```

Run a local GGUF model from Python:

```python
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
params.n_ctx = 2048

server = xlc.Server(params)
result = server.handle_chat_completions({
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "max_tokens": 128,
})

print(result["choices"][0]["message"]["content"])
```

Need a model? Download a [GGUF model from Hugging Face](#download-models).
For GPU-specific wheels, see [Installation](#installation). For the complete
Python API, continue to [Usage](#usage).

This project was originally forked from
[cyllama](https://github.com/shakfu/cyllama) and has since evolved into the
llama.cpp backend used by [Xinference](https://github.com/xorbitsai/inference).

## xllamacpp vs. llama-cpp-python

The following table provides an overview of the two Python bindings:

| implementations / features |      xllamacpp      |         llama-cpp-python         |
|:---------------------------|:-------------------:|:--------------------------------:|
| Wrapper type               |       Cython        |              ctypes              |
| API                        | Server & Params API |            Llama API             |
| Server implementation      |         C++         | Python through wrapped Llama API |
| Continuous batching        |         Yes         |                No                |
| Thread safe                |         Yes         |                No                |
| Release package            |    Prebuilt wheels  |    Build during installation     |
| Structured outputs         | C++ / [LLGuidance](https://github.com/guidance-ai/llguidance) (Rust) |  Python (`SchemaConverter`)    |

It goes without saying that any help / collaboration / contributions to accelerate the above would be welcome!

## llama.cpp Python FAQ

### Is xllamacpp a Python binding for llama.cpp?

Yes. xllamacpp packages llama.cpp as a Cython extension and exposes its native
server and parameter APIs to Python. You can embed local GGUF inference in a
Python application without launching a separate `llama-server` process.

### Can I install llama.cpp for Python with pip?

Yes. `pip install xllamacpp` installs a prebuilt wheel for CPU or macOS. Prebuilt
CUDA, ROCm, and Vulkan wheels are also available from the project package
indexes, while source builds remain available for custom hardware tuning.

### Does the Python API support OpenAI-compatible requests?

Yes. The embedded llama.cpp server supports OpenAI-compatible chat completions,
text completions, embeddings, reranking, and Responses API requests. You can
call it directly from Python or use its local HTTP endpoints with the OpenAI
Python client.

## Wrapping Guidelines

As the intent is to provide a very thin wrapping layer and play to the strengths of the original c++ library as well as python, the approach to wrapping intentionally adopts the following guidelines:

- In general, key structs are implemented as cython extension classses with related functions implemented as methods of said classes.

- Be as consistent as possible with llama.cpp's naming of its api elements, except when it makes sense to shorten functions names which are used as methods.

- Minimize non-wrapper python code.


## Installation

**Note on Performance and Compatibility**

> ⚠️ **The pre-built Linux/Windows CPU-only release wheels are NOT fully optimized.** They are compiled for **maximum compatibility** across a wide range of hardware, not for maximum performance. CPU-native optimizations (e.g., AVX-512, SVE, AMX) are **disabled** in these builds to ensure they run on as many machines as possible.
>
> ✅ **macOS wheels are fully optimized.** Since each macOS architecture (Apple Silicon / Intel) uses a consistent CPU, native optimizations are enabled in release builds.
>
> ✅ **GPU-accelerated wheels (CUDA / ROCm / Vulkan) are already optimized for GPU computation.** The heavy lifting is done on the GPU, so the lack of CPU-native optimizations has minimal impact on overall performance.
>
> **If you are on Linux/Windows using CPU-only inference and want the best performance, build from source.** When building locally, all native CPU optimizations are automatically enabled for your specific machine. See [Build from Source](#build-from-source) for instructions.

Specifically:
- **macOS** (Apple Silicon & Intel): wheels are built with native CPU optimizations enabled — **no action needed**.
- **CUDA / ROCm / Vulkan** wheels: GPU acceleration is fully optimized; CPU-native optimizations are not critical since inference runs primarily on the GPU.
- **Linux x86_64 CPU-only** wheels disable `-march=native`, so advanced instruction sets like AVX-512 or AMX are not used.
- **Linux aarch64 CPU-only** wheels target the baseline `armv8-a` architecture, so advanced features like SVE are not enabled.
- **Windows x86_64 CPU-only** wheels disable `-march=native`, similar to Linux x86_64.

If you are on Linux/Windows using **CPU-only** inference and your CPU supports these advanced features, building from source can provide **significantly better performance**.

- From pypi for `CPU` or `Mac`:

```sh
pip install -U xllamacpp
```

- From github pypi for `CUDA` (use `--force-reinstall` to replace the installed CPU version):

  - CUDA 13.2
    ```sh
    pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/cu132
    ```

  - CUDA 12.8
    ```sh
    pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128
    ```

- From github pypi for `HIP` AMD GPU (use `--force-reinstall` to replace the installed CPU version):

  - ROCm 7.2.4
    ```sh
    pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/rocm-7.2.4
    ```

  - ROCm 6.4.1
    ```sh
    pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/rocm-6.4.1
    ```

- From github pypi for `Vulkan` (use `--force-reinstall` to replace the installed CPU version):

  ```sh
  pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan
  ```

## Prerequisites for Prebuilt Wheels

Before pip installing `xllamacpp`, please ensure your system meets the following requirements based on your build type:

- **CPU (macOS)**:
  - Pre-built wheels are already optimized for native CPU — no additional steps needed

- **CPU (Linux aarch64)**:
  - Requires ARMv8-A or later architecture
  - For best performance, build from source if your CPU supports advanced instruction sets (e.g., SVE)

- **CUDA (Linux/Windows)**:
  - Requires glibc 2.35 or later (Linux)
  - Compatible NVIDIA GPU with appropriate drivers (CUDA 12.8 or 13.2)
  - See [CUDA GPU Architecture Coverage](#cuda-gpu-architecture-coverage) below for the list of supported GPUs per CUDA version

- **ROCm (Linux)**:
  - Requires glibc 2.35 or later
  - Requires gcc 10 or later (ROCm libraries have this dependency)
  - Compatible AMD GPU with ROCm support (ROCm 6.4.1 or 7.2.4)
  - See [ROCm (HIP) GPU Architecture Coverage](#rocm-hip-gpu-architecture-coverage) below for the list of supported GPUs per ROCm version

- **Vulkan (Linux/Windows, Intel/AMD/NVIDIA where supported)**:
  - Install the Vulkan SDK and GPU drivers with Vulkan support
  - Linux users may need distro packages and the LunarG SDK
  - macOS Intel is supported via Vulkan; Apple Silicon Vulkan is not supported in this project

## CUDA GPU Architecture Coverage

The prebuilt CUDA wheels are compiled for a curated set of NVIDIA GPU architectures
that mirrors the upstream llama.cpp *release* default (`ggml/src/ggml-cuda/CMakeLists.txt`,
`GGML_NATIVE=OFF` branch). The set differs between CUDA versions because newer toolkits
drop older architectures and add newer ones.

Each architecture is shipped in one of two forms:

- **SASS** (`-real`): native machine code (cubin) for exactly that GPU architecture.
  It loads and runs directly with full optimization, but only on that specific architecture.
- **PTX** (`-virtual`): a forward-compatible virtual ISA. The driver JIT-compiles it to
  SASS on first launch (then caches it). PTX built for `compute_X` can JIT onto any GPU
  with compute capability `>= X` (never older).

At runtime CUDA picks matching SASS if available; otherwise it JIT-compiles compatible PTX;
otherwise the load fails (CUDA error 209, "no kernel image"). This is why at least one PTX
(`-virtual`) entry is always shipped, so GPUs without native SASS still run via JIT.

| Arch | Form | GPUs | CUDA 12.8 wheel | CUDA 13.2 wheel |
|:-----|:-----|:-----|:---------------:|:---------------:|
| `50`   | PTX  | Maxwell (GTX 9xx)            | ✅ JIT  | — |
| `61`   | PTX  | Pascal (GTX 10xx, P40)       | ✅ JIT  | — |
| `70`   | PTX  | Volta (V100)                 | ✅ JIT  | — |
| `75`   | PTX (12.8) / SASS (13.2) | Turing (RTX 20xx, T4) | ✅ JIT  | ✅ native |
| `80`   | PTX  | Ampere DC (A100); also forward-covers Hopper (90) and Blackwell DC (100/103) via JIT | ✅ JIT  | ✅ JIT  |
| `86`   | SASS | Ampere (RTX 30xx, A10/A40)   | ✅ native | ✅ native |
| `89`   | SASS | Ada (RTX 40xx, L4/L40)       | ✅ native | ✅ native |
| `120a` | SASS | Blackwell (RTX 50xx)         | ✅ native | ✅ native |
| `121a` | SASS | Blackwell variant            | —       | ✅ native |

Notes:

- **CUDA 13 dropped Maxwell/Pascal/Volta**, so `50`/`61`/`70` are omitted there; the CUDA 13
  floor is `sm_75` (Turing) via `75-real` SASS.
- **JIT needs a driver at least as new as the wheel's toolkit.** The CUDA 13.2 wheel's PTX is
  ISA 9.2, so its JIT rows need a driver that supports CUDA 13.2. An older 13.x driver, e.g.
  the R580 branch (580.xx, which `nvidia-smi` reports as CUDA 13.0), fails with "the provided
  PTX was compiled with an unsupported toolchain".
  Native SASS has no such limit, which is why Turing ships as `75-real` in that wheel.
- **`121a` requires CUDA >= 12.9**, so it is only present in the CUDA 13.2 wheel.
- **Native SASS is shipped only for mainstream consumer cards.** Datacenter parts
  (A100/Hopper/Blackwell-DC) run via JIT from the `80-virtual` PTX, which works but incurs a
  one-time JIT compile on first launch.
- **Local source builds** (where `CUDA_ARCHITECTURES` is unset) instead use CMake's
  `native`, producing native SASS for the GPU architectures detected on the build machine —
  see `scripts/build.py`. This keeps local builds fast and optimized for your hardware, but
  the resulting binary is not portable to other GPU architectures. If you need native
  datacenter performance from a prebuilt wheel, add `90-real` / `100-real` to the
  architecture list.

The same architecture lists apply to both Linux (x86_64 and arm64) and Windows CUDA wheels,
since the `sm_XX` value is the GPU architecture and is independent of the host CPU.

## ROCm (HIP) GPU Architecture Coverage

The prebuilt ROCm (HIP) wheels are compiled for a curated set of AMD GPU
architectures. The set differs between ROCm versions because newer ROCm
releases add support for newer GPU architectures.

Each target is compiled as native GCN ISA (no PTX/JIT equivalent in the ROCm
world) — the compiled code runs directly on the specified `gfxXXX` architecture
and does not auto-upgrade to newer GPUs. This is why the ROCm 7.2.4 wheel ships
a wider list, covering both older and newer architectures.

| Target | Arch family | GPUs | ROCm 6.4.1 wheel | ROCm 7.2.4 wheel |
|:-------|:------------|:-----|:-----------------:|:-----------------:|
| `gfx1030` | RDNA2     | RX 6800 / 6900 XT              | ✅ | ✅ |
| `gfx1031` | RDNA2     | RX 6700 / 6750 XT              | ✅ | ✅ |
| `gfx1032` | RDNA2     | RX 6600 / 6650 XT              | ✅ | ✅ |
| `gfx1100` | RDNA3     | RX 7900 XTX / XT               | ✅ | ✅ |
| `gfx1101` | RDNA3     | RX 7800 / 7700 XT               | ✅ | ✅ |
| `gfx1102` | RDNA3     | RX 7600 XT                     | ✅ | ✅ |
| `gfx1150` | RDNA 3.5  | Strix Point (Ryzen AI 300)     | —  | ✅ |
| `gfx1151` | RDNA 3.5  | Strix Halo (Ryzen AI Max 300)  | —  | ✅ |
| `gfx1200` | RDNA4     | RX 9060 / 9070 XT              | —  | ✅ |
| `gfx1201` | RDNA4     | RX 9070 variants               | —  | ✅ |
| `gfx950`  | CDNA4     | Instinct MI350X / MI355X       | —  | ✅ |

Notes:

- **ROCm 6.4.1** covers RDNA2 + RDNA3 only. It is retained as the fallback for
  users who cannot upgrade to ROCm 7.x.
- **ROCm 7.2.4** adds RDNA 3.5 (Strix Halo / Strix Point APUs), RDNA4 (RX 9060 /
  9070), and CDNA4 (MI350X datacenter). It is a superset — all GPUs supported by
  the 6.4.1 wheel are also covered.
- **ABI compatibility:** a binary built with ROCm 7.2.4 runs on any ROCm 7.x
  runtime (including 7.13/7.14 from TheRock), since the major version is the
  ABI boundary. It does **not** run on ROCm 6.x.
- **rocWMMA flash attention** is enabled on all builds, providing faster
  attention on supported architectures.
- **Local source builds** (where `AMDGPU_TARGETS` is unset) default to RDNA2 +
  RDNA3 targets — see `scripts/build.py`. Set `AMDGPU_TARGETS` explicitly to
  target your specific GPU (e.g., `export AMDGPU_TARGETS=gfx1151`).

## Build from Source

### (Optional) Preparation

- CUDA

  This provides GPU acceleration using an NVIDIA GPU. Make sure to have the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed.
  
  #### Download directly from NVIDIA
  You may find the official downloads here: [NVIDIA developer site](https://developer.nvidia.com/cuda-downloads).
  
  
  #### Compile and run inside a Fedora Toolbox Container
  We also have a [guide](./backend/CUDA-FEDORA.md) for setting up CUDA toolkit in a Fedora [toolbox container](https://containertoolbx.org/).
  
  **Recommended for:**
  - ***Necessary*** for users of [Atomic Desktops for Fedora](https://fedoraproject.org/atomic-desktops/); such as: [Silverblue](https://fedoraproject.org/atomic-desktops/silverblue/) and [Kinoite](https://fedoraproject.org/atomic-desktops/kinoite/).
    - (there are no supported CUDA packages for these systems)
  - ***Necessary*** for users that have a host that is not a: [Supported Nvidia CUDA Release Platform](https://developer.nvidia.com/cuda-downloads).
    - (for example, you may have [Fedora 42 Beta](https://fedoramagazine.org/announcing-fedora-linux-42-beta/) as your your host operating system)
  - ***Convenient*** For those running [Fedora Workstation](https://fedoraproject.org/workstation/) or [Fedora KDE Plasma Desktop](https://fedoraproject.org/spins/kde), and want to keep their host system clean.
  - *Optionally* toolbox packages are available: [Arch Linux](https://archlinux.org/), [Red Hat Enterprise Linux >= 8.5](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux), or [Ubuntu](https://ubuntu.com/download)

- HIP

  This provides GPU acceleration on HIP-supported AMD GPUs.
  Make sure to have ROCm installed.
  You can download it from your Linux distro's package manager or from here: [ROCm Quick Start (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html#rocm-install-quick).

  
  Or you can try to build inside the [ROCm docker container](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html).

- Vulkan

  Install the Vulkan SDK and drivers for your platform.
  - Linux: use your distro packages and/or the [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home).
  - Windows: install [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home) and vendor GPU drivers.
  - macOS: Intel only; Apple Silicon is not supported for Vulkan in this project.

### Build `xllamacpp`

1. A recent version of `python3` (testing on python 3.12)

2. Install Rust toolchain (required for building):

 ```sh
 curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
 ```

 For more installation options, see the [rustup installation guide](https://rustup.rs/).

3. Git clone the latest version of `xllamacpp`:

 ```sh
 git clone git@github.com:xorbitsai/xllamacpp.git
 cd xllamacpp
 git submodule init
 git submodule update
 ```

4. Install dependencies of `cython`, `setuptools`, and `pytest` for testing:

 ```sh
 pip install -r requirements.txt
 ```

5. Install platform build tools:

   - Linux: install a C/C++ toolchain, CMake, and Python development headers through your distro package manager.
   - macOS: install Xcode Command Line Tools and CMake.
   - Windows: install CMake and a native Windows C/C++ toolchain. For the current CI-compatible x86_64 build, install [w64devkit](https://github.com/skeeto/w64devkit) and put its `bin` directory on `PATH`. WSL is not required.

6. Select backend via environment and build through `setup.py` or the PEP 517 build frontend. The package build configures and builds the vendored `llama.cpp` with CMake before compiling the Python extension. `scripts/build.py` is an internal helper invoked by the package build, not a standalone package build command. `make` is only a convenience wrapper for `python setup.py build_ext --inplace`; it is not required.

   Linux/macOS examples:

   - CPU (default):
     ```sh
     python setup.py build_ext --inplace
     ```

   - CUDA:
     ```sh
     export XLLAMACPP_BUILD_CUDA=1
     python setup.py build_ext --inplace
     ```

   - HIP (AMD):
     ```sh
     export XLLAMACPP_BUILD_HIP=1
     python setup.py build_ext --inplace
     ```

   - Vulkan:
     ```sh
     export XLLAMACPP_BUILD_VULKAN=1
     python setup.py build_ext --inplace
     ```

   - Enable BLAS (optional):
     ```sh
     export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
     python setup.py build_ext --inplace
     ```

   Windows PowerShell examples:

   - CPU (default):
     ```powershell
     python setup.py build_ext --inplace
     ```

   - CUDA:
     ```powershell
     $env:XLLAMACPP_BUILD_CUDA = "1"
     python setup.py build_ext --inplace
     ```

   - Vulkan:
     ```powershell
     $env:XLLAMACPP_BUILD_VULKAN = "1"
     python setup.py build_ext --inplace
     ```

   - Enable BLAS (optional):
     ```powershell
     $env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
     python setup.py build_ext --inplace
     ```

   To build a wheel directly on any platform:

   ```sh
   python -m build --wheel
   ```

## Updating llama.cpp

`xllamacpp` vendors `llama.cpp` as the Git submodule at `thirdparty/llama.cpp`.
The repository includes a Codex skill for refreshing that submodule and checking
the xllamacpp bindings against the updated upstream headers.

Ask Codex to use the skill from the repository root:

```text
Use the update-llamacpp skill to update thirdparty/llama.cpp to the latest b* release tag.
```

The skill definition is in:

```text
.codex/skills/update-llamacpp/SKILL.md
```

The skill performs the update workflow end to end:

- checks the parent repository and `thirdparty/llama.cpp` worktrees before
  changing branches
- updates the vendored `llama.cpp` checkout and its nested submodules
- refreshes upstream tags and pins `thirdparty/llama.cpp` to the latest `b*`
  release tag
- builds xllamacpp after the update
- checks changed llama.cpp headers against `src/xllamacpp/xllamacpp.pxd` and
  `src/xllamacpp/xllamacpp.pyx`
- runs the local test suite
- summarizes the updated llama.cpp tag, commit, compatibility fixes, and
  build/test results

Do not run `git submodule update thirdparty/llama.cpp` after the skill updates
the submodule, because it resets the checkout back to the commit currently
recorded by the parent repository.

## Usage

All examples below assume you have `xllamacpp` installed and a GGUF model file available. See [Installation](#installation) for installation instructions and [Testing](#testing) for how to download models.

### Configuring Parameters

`CommonParams` is the central configuration object. It controls model loading, inference, sampling, server behavior, and more. For a complete list of all parameters with types, defaults, and descriptions, see the [Parameters Reference](params_reference.md).

```python
import xllamacpp as xlc

params = xlc.CommonParams()

# Model path
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"

# Context and prediction
params.n_ctx = 512          # Context window size
params.n_predict = 128      # Max tokens to predict
params.n_batch = 2048       # Batch size for prompt processing
params.n_ubatch = 512       # Micro-batch size
params.n_parallel = 1       # Number of parallel sequences

# CPU threading
params.cpuparams.n_threads = 4
params.cpuparams_batch.n_threads = 4

# Sampling parameters
params.sampling.temp = 0.8
params.sampling.top_k = 40
params.sampling.top_p = 0.95
params.sampling.seed = 42

# Server options
params.warmup = True
params.endpoint_metrics = True
```

### Environment Variables

xllamacpp supports several environment variables to control low-level behavior:

- **`LLAMA_ATTN_ROT_DISABLE`**: Set to `1` to disable attention rotation in KV cache quantization, enabling classic KV attention. This is useful for troubleshooting when comparing behavior with older versions or when the rotation feature causes issues. Default: rotation is enabled.

```python
import os
import xllamacpp as xlc

# Disable attention rotation for classic KV attention
os.environ["LLAMA_ATTN_ROT_DISABLE"] = "1"

params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
server = xlc.Server(params)
```

### Text Completions

Generate text from a prompt using the completions API:

```python
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
params.n_predict = 128
params.n_ctx = 256
params.cpuparams.n_threads = 4

server = xlc.Server(params)

# Non-streaming completion (returns a dict)
result = server.handle_completions({
    "max_tokens": 128,
    "prompt": "Write the fibonacci function in C++.",
})
print(result["choices"][0]["text"])
```

### Chat Completions

Use the chat completions API with message-based conversations:

```python
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
params.n_predict = 128
params.n_ctx = 256
params.cpuparams.n_threads = 4

server = xlc.Server(params)

result = server.handle_chat_completions({
    "max_tokens": 128,
    "messages": [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write the fibonacci function in C++."},
    ],
})
print(result["choices"][0]["message"]["content"])
```

### Decision Models (System One)

Native decision models answer typed questions about a `state` through the
TypeSafe-compatible [`/v1/systemone` API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api).
This API requires a decision-model GGUF and an xllamacpp build containing
llama.cpp b11361 or later. For example, download [Laya-Q8_0.gguf](https://huggingface.co/ggml-org/Laya-GGUF):

```sh
mkdir -p models
curl -fL https://huggingface.co/ggml-org/Laya-GGUF/resolve/main/Laya-Q8_0.gguf -o models/Laya-Q8_0.gguf
```

Start the embedded server and call its Python method:

```python
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/Laya-Q8_0.gguf"
params.n_ctx = 1024
params.n_batch = 512
params.n_ubatch = 512
server = xlc.Server(params)

request = {
    "state": "I was charged twice for my order and need a refund today.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": None, "shipping": None, "technical": None},
        },
        "refund": {
            "type": "noul",
            "instructions": "Is a refund requested?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["can wait", "this week", "today", "right now"],
        },
    },
}
answers = server.handle_systemone(request)["answers"]
print(answers["route"]["choice"])  # Highest-probability option
print(answers["refund"]["noul"])  # Probability that the answer is true
print(answers["urgency"]["score"])  # Expected index into the criteria list
```

`handle_systemone()` accepts a `dict`, JSON `str`, or JSON `bytes` and returns
the corresponding Python or JSON type. The same request is available to HTTP
clients at `POST /v1/systemone` on `server.listening_address`.

The response also includes per-option probabilities for `choice` and `score`
questions. Decision models return scores rather than generated text; ordinary
chat GGUF models cannot serve this endpoint.

### Streaming

Enable streaming to receive tokens as they are generated. Provide a callback function:

```python
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
params.n_predict = 64
params.n_ctx = 256
params.cpuparams.n_threads = 4

server = xlc.Server(params)

# Streaming completions with a callback
server.handle_completions(
    {
        "max_tokens": 128,
        "prompt": "Tell me a story.",
        "stream": True,
    },
    lambda chunk: print(chunk),  # Called for each chunk
)

# Streaming chat completions
server.handle_chat_completions(
    {
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Tell me a joke."}],
        "stream": True,
    },
    lambda chunk: print(chunk),
)
```

You can **stop streaming early** by returning `True` from the callback:

```python
count = 0

def stop_after_one(chunk):
    global count
    count += 1
    print(chunk)
    return True  # Returning True stops the stream

server.handle_completions(
    {"prompt": "Write a long story.", "stream": True, "max_tokens": 128},
    stop_after_one,
)
# Only one chunk will be received
```

### Structured Output with JSON Grammar

Constrain model output to conform to a JSON schema or GBNF grammar. xllamacpp is built with [LLGuidance](https://github.com/guidance-ai/llguidance) enabled (`LLAMA_LLGUIDANCE=ON`), a high-performance Rust-based constrained decoding library. When LLGuidance is enabled, JSON Schema requests are automatically routed through LLGuidance instead of the default GBNF grammar engine, providing faster token mask computation (~50μs average) and more spec-compliant JSON Schema support. The entire structured output pipeline — schema conversion, grammar parsing, and token mask computation — runs in native C++/Rust with no Python overhead.

In contrast, `llama-cpp-python` does not enable LLGuidance and implements JSON Schema → GBNF conversion in Python via a `SchemaConverter` class. It relies solely on llama.cpp's built-in GBNF grammar engine for constrained sampling.

**Important:** For server endpoints, grammar constraints must be passed **in the request body**, not via `params.sampling.grammar` (which only affects the CLI path). See the comparison below.

All three methods below share the same setup:

```python
import json
import xllamacpp as xlc

schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "score": {"type": "number"},
    },
    "required": ["answer"],
}

params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
params.n_predict = 64
params.n_ctx = 256
params.sampling.temp = 0
params.sampling.top_k = 1

server = xlc.Server(params)

messages = [
    {"role": "system", "content": "Respond with a JSON object matching the provided schema."},
    {"role": "user", "content": "Provide an answer string and an optional numeric score."},
]
```

#### Method 1: `grammar` in Request Body

Convert the JSON schema to a grammar string using `xlc.json_schema_to_grammar()`, then pass it in the request. The output format depends on the build: LLGuidance Lark format when built with `LLAMA_LLGUIDANCE=ON` (default for xllamacpp), or GBNF otherwise.

```python
# Convert schema to grammar string (LLGuidance Lark format when built with LLAMA_LLGUIDANCE=ON)
grammar_str = xlc.json_schema_to_grammar(schema)

result = server.handle_chat_completions({
    "max_tokens": 64,
    "messages": messages,
    # Grammar MUST be passed in the request body for server endpoints
    "grammar": grammar_str,
})
```

#### Method 2: `json_schema` in Request Body

Pass the JSON schema directly in the request — llama.cpp converts it to a grammar constraint internally (LLGuidance Lark format when built with `LLAMA_LLGUIDANCE=ON`, or GBNF otherwise):

```python
result = server.handle_chat_completions({
    "max_tokens": 64,
    "messages": messages,
    "json_schema": schema,
})
```

#### Method 3: `response_format` (OpenAI-Compatible)

Use the OpenAI-compatible `response_format` field for chat completions:

```python
# Type "json_object" with optional schema
result = server.handle_chat_completions({
    "messages": messages,
    "response_format": {"type": "json_object", "schema": schema},
})

# Type "json_schema" (OpenAI-style nested format)
result = server.handle_chat_completions({
    "messages": messages,
    "response_format": {
        "type": "json_schema",
        "json_schema": {"schema": schema},
    },
})
```

For all methods above, parse the result the same way:

```python
content = result["choices"][0]["message"]["content"]
parsed = json.loads(content)
print(parsed)  # e.g. {"answer": "Hello", "score": 42}
```

> **Note:** `json_schema` and `grammar` cannot be used simultaneously in the same request.

#### Structured Outputs: xllamacpp vs llama-cpp-python

| Aspect | xllamacpp | llama-cpp-python |
|:---|:---|:---|
| [LLGuidance](https://github.com/guidance-ai/llguidance) | Enabled (`LLAMA_LLGUIDANCE=ON`) | Not enabled |
| JSON Schema → Grammar | C++ / Rust (LLGuidance Lark format, ~50μs/mask) | Python (`SchemaConverter` → GBNF) |
| Grammar enforcement | LLGuidance sampler (Rust) | GBNF sampler (C++, via ctypes) |
| Structured output API | Request body: `grammar`, `json_schema`, or `response_format` | Python args: `grammar=LlamaGrammar(...)` or `response_format={...}` |

### Embeddings

Generate embeddings for a list of texts. Requires an embedding model (e.g., [Qwen3-Embedding-0.6B-Q8_0.gguf](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf)):

```python
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/Qwen3-Embedding-0.6B-Q8_0.gguf"
params.embedding = True
params.n_ctx = 512
params.n_batch = 128
params.n_ubatch = 128
params.pooling_type = xlc.llama_pooling_type.LLAMA_POOLING_TYPE_LAST

server = xlc.Server(params)

result = server.handle_embeddings({
    "input": [
        "I believe the meaning of life is",
        "This is a test",
    ],
    "model": "My Qwen3 Model",
})

for item in result["data"]:
    print(f"Index {item['index']}: {len(item['embedding'])} dimensions")
```

### Reranking

Rerank documents by relevance to a query. Requires a reranker model (e.g., `bge-reranker-v2-m3-Q2_K.gguf`):

```python
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/bge-reranker-v2-m3-Q2_K.gguf"
params.embedding = True
params.n_ctx = 512
params.n_batch = 128
params.n_ubatch = 128
params.pooling_type = xlc.llama_pooling_type.LLAMA_POOLING_TYPE_RANK

server = xlc.Server(params)

result = server.handle_rerank({
    "query": "What is the capital of France?",
    "documents": [
        "Paris is the capital of France.",
        "The Eiffel Tower is in Paris.",
        "Germany is located in Europe.",
    ],
})

for doc in result["results"]:
    print(f"Document {doc['index']}: relevance_score={doc['relevance_score']:.4f}")
```

### Multimodal (Vision)

Process images alongside text using a vision model. Requires a multimodal model and its mmproj file:

```python
import base64
import xllamacpp as xlc

# Load and encode an image
with open("image.png", "rb") as f:
    img_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

params = xlc.CommonParams()
params.model.path = "models/tinygemma3-Q8_0.gguf"
params.mmproj.path = "models/mmproj-tinygemma3.gguf"
params.n_predict = 128
params.n_ctx = 1024
params.sampling.temp = 0
params.sampling.top_k = 1

server = xlc.Server(params)

result = server.handle_chat_completions({
    "max_tokens": 128,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this:"},
                {"type": "image_url", "image_url": {"url": img_b64}},
            ],
        },
    ],
})
print(result["choices"][0]["message"]["content"])
```

### LoRA Adapters

Load and apply LoRA adapters to customize model behavior:

```python
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/stories15M_MOE-F16.gguf"
params.n_predict = 64
params.n_ctx = 256
params.sampling.seed = 42
params.sampling.temp = 0.0
params.sampling.top_k = 1

# Attach a LoRA adapter with a scale factor
lora = xlc.CommonAdapterLoraInfo("models/moe_shakespeare15M.gguf", 1.0)
params.lora_adapters = [lora]

# You can adjust the scale dynamically
params.lora_adapters[0].scale = 0.5  # Half strength

server = xlc.Server(params)

result = server.handle_completions({
    "max_tokens": 64,
    "prompt": "Look in thy glass",
    "temperature": 0.0,
})
print(result["choices"][0]["text"])
```

### Reasoning Support

Control reasoning/thinking behavior for models that support it (e.g., DeepSeek, QwQ):

```python
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/my-reasoning-model.gguf"

# Reasoning format: controls how thought tags are parsed
# COMMON_REASONING_FORMAT_DEEPSEEK (default) - puts thoughts in reasoning_content
# COMMON_REASONING_FORMAT_NONE - leaves thoughts unparsed in content
params.reasoning_format = xlc.common_reasoning_format.COMMON_REASONING_FORMAT_DEEPSEEK

# Enable/disable reasoning: -1=auto, 0=off, 1=on
params.enable_reasoning = 1

# Token budget for thinking: -1=unrestricted, 0=immediate end, N>0=budget
params.reasoning_budget = 1024

# Sampling-level reasoning budget
params.sampling.reasoning_budget_tokens = 500
```

### Speculative Decoding

Speculative decoding pairs a small, fast "draft" model with a larger "target" model to speed up generation. The draft model proposes several tokens at once; the target model verifies them in a single forward pass and accepts the longest matching prefix. With a well-aligned tokenizer/vocabulary, the produced text is **bit-exact** to running the target model alone, but throughput improves because multiple tokens are validated per target-model decode step.

This is especially helpful for unified-memory APU/iGPU users (e.g., AMD Strix Halo, Apple Silicon) who want to run larger dense models: the draft model fits easily alongside the target and trades a small amount of memory for a meaningful tokens/sec speedup on memory-bandwidth-bound hardware.

#### Draft-Model Speculative Decoding

A real-world example pairing a large model with a smaller same-family checkpoint:

```python
import xllamacpp as xlc

params = xlc.CommonParams()

# Target (main) model — the large, high-quality model.
params.model.path = "models/Qwen3-4B-Instruct-Q8_0.gguf"

# Draft model — a smaller, faster model from the same family.
# Both models must share the same tokenizer/vocabulary.
params.speculative.mparams_dft.path = "models/Qwen3-0.6B-Instruct-Q4_0.gguf"

# Draft N-tokens window: propose between n_min and n_max tokens per step.
params.speculative.n_min = 4   # Min tokens to draft before verification
params.speculative.n_max = 8   # Max tokens to draft per step

params.n_predict = 128
params.n_ctx = 512
params.cpuparams.n_threads = 4

# Deterministic sampling (recommended for verifying equivalence with non-speculative)
params.sampling.seed = 42
params.sampling.temp = 0.0
params.sampling.top_k = 1

server = xlc.Server(params)

# Use exactly like a normal server — speculative decoding is transparent.
result = server.handle_completions({
    "max_tokens": 128,
    "prompt": "I believe the meaning of life is",
    "temperature": 0.0,
})
print(result["choices"][0]["text"])
```

> **Note:** Setting `speculative.type` is *not* required when a draft model path is provided — the draft-model path automatically triggers the draft-model speculative decoding path.

#### Per-Request Speculative Overrides

You can override the speculative decoding parameters on a per-request basis:

```python
# Tighten the draft window for a single request.
result = server.handle_completions({
    "max_tokens": 64,
    "prompt": "Once upon a time,",
    "temperature": 0.0,
    "speculative.n_min": 1,
    "speculative.n_max": 4,
    "speculative.p_min": 0.0,
})
print(result["choices"][0]["text"])
```

#### N-gram Based Speculative Decoding (No Draft Model)

For self-speculative decoding without a separate draft model, use n-gram lookup:

```python
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
params.n_predict = 128
params.n_ctx = 512

# N-gram speculative decoding — no draft model needed.
params.speculative.type = xlc.common_speculative_type.COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE
params.speculative.n_max = 16      # Max tokens to draft
params.speculative.n_min = 5       # Min tokens to draft
params.speculative.ngram_size_n = 12
params.speculative.ngram_size_m = 48
params.speculative.ngram_min_hits = 1

server = xlc.Server(params)

result = server.handle_completions({
    "max_tokens": 128,
    "prompt": "The quick brown fox",
})
print(result["choices"][0]["text"])
```

### Request-Level Options

Both `handle_completions()` and `handle_chat_completions()` accept all [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#api-endpoints) in the request dict. Common options include:

```python
result = server.handle_chat_completions({
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "seed": 42,
    "stream": False,
    "stop": ["\n\n", "User:"],          # Stop sequences
    "cache_prompt": True,                # Reuse KV cache from previous requests
    "n_probs": 5,                        # Return top-N token probabilities
    "repeat_penalty": 1.1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "grammar": "...",                    # BNF-like grammar constraint
    "json_schema": {"type": "object"},   # JSON schema constraint
    "response_format": {"type": "json_object"},  # OpenAI-compatible format
    "samplers": ["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
    "lora": [{"id": 0, "scale": 0.5}],  # Per-request LoRA scaling
})
```

### GPU Memory Estimation

Estimate how many layers can be offloaded to GPU(s) given available memory:

```python
from xllamacpp import estimate_gpu_layers

devices = [
    {"name": "cuda", "memory_free": 8 * 1024**3, "memory_min": 2048},
]

estimate = estimate_gpu_layers(
    devices,
    "models/my-model.gguf",
    [],                      # projector paths
    context_length=2048,
    batch_size=512,
    num_parallel=1,
    kv_cache_type="",
)

print(f"Layers to offload: {estimate.layers}")
print(f"VRAM usage: {estimate.vram_size / 1024**3:.2f} GB")
print(f"Total model size: {estimate.total_size / 1024**3:.2f} GB")
print(f"Tensor split: {estimate.tensor_split}")
```

### System Info & Device Info

Query the runtime environment, including CPU features and available compute devices:

```python
import xllamacpp as xlc

# Get system info string (CPU features, build options, etc.)
print(xlc.get_system_info())

# Get available compute devices (CPU, CUDA, Metal, etc.)
devices = xlc.get_device_info()
for dev in devices:
    print(dev["name"])
```

## OpenAI API Compatible HTTP Server

`xlc.Server` automatically starts an HTTP server that provides OpenAI API compatible endpoints. The server supports continuous batching, parallel decoding, prompt caching, and multimodal inputs. For the full specification, see the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#api-endpoints).

### Available HTTP Endpoints

The server exposes the following endpoints. For full details on request/response formats, see the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#api-endpoints).

**General:**

| Endpoint | Method | Description |
|:---------|:------:|:------------|
| `/health`, `/v1/health` | GET | Health check (public, no API key required) |
| `/models`, `/v1/models` | GET | List loaded models (OpenAI compatible) |
| `/props` | GET/POST | Server properties & default generation settings |
| `/metrics` | GET | Prometheus-format metrics (requires `endpoint_metrics = True`) |
| `/slots` | GET | View inference slot states (requires `endpoint_slots = True`) |

**Inference:**

| Endpoint | Method | Description |
|:---------|:------:|:------------|
| `/completion` | POST | Text completions (llama.cpp native format) |
| `/v1/completions`, `/completions` | POST | Text completions (OpenAI compatible) |
| `/v1/chat/completions`, `/chat/completions` | POST | Chat completions (OpenAI compatible) |
| `/v1/responses` | POST | Responses API (OpenAI compatible) |
| `/v1/messages` | POST | Messages API (Anthropic compatible) |
| `/v1/systemone` | POST | Typed questions with a native decision model (TypeSafe compatible) |
| `/infill` | POST | Code infill (FIM: fill-in-the-middle) |

**Embeddings & Reranking:**

| Endpoint | Method | Description |
|:---------|:------:|:------------|
| `/v1/embeddings` | POST | Generate embeddings (OpenAI compatible) |
| `/embedding` | POST | Generate embeddings (llama.cpp native, supports `pooling none`) |
| `/rerank`, `/v1/rerank`, `/v1/reranking` | POST | Rerank documents by relevance |

**Tokenization & Templates:**

| Endpoint | Method | Description |
|:---------|:------:|:------------|
| `/tokenize` | POST | Tokenize text to token IDs |
| `/detokenize` | POST | Convert token IDs back to text |
| `/apply-template` | POST | Apply chat template without inference |

**LoRA & Slot Management:**

| Endpoint | Method | Description |
|:---------|:------:|:------------|
| `/lora-adapters` | GET | List loaded LoRA adapters with scales |
| `/lora-adapters` | POST | Update global LoRA adapter scales |
| `/slots/{id}?action=save` | POST | Save slot KV cache to file |
| `/slots/{id}?action=restore` | POST | Restore slot KV cache from file |
| `/slots/{id}?action=erase` | POST | Erase slot KV cache |

### Using the OpenAI Python Client

```python
import xllamacpp as xlc
from openai import OpenAI

# Start server
params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
server = xlc.Server(params)

# Connect using OpenAI client
client = OpenAI(
    base_url=server.listening_address + "/v1",
    api_key="not-required"  # No API key needed for local server
)

# Chat completion
response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=10
)
print(response.choices[0].message.content)
```

### Using `requests` Directly

```python
import requests
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
params.n_ctx = 256
params.endpoint_metrics = True

server = xlc.Server(params)
base_url = server.listening_address

# Health check
resp = requests.get(f"{base_url}/health")
print(resp.json())  # {"status": "ok"}

# Chat completion
resp = requests.post(f"{base_url}/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32,
})
print(resp.json()["choices"][0]["message"]["content"])

# Streaming
resp = requests.post(f"{base_url}/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "max_tokens": 64,
    "stream": True,
}, stream=True)
for line in resp.iter_lines():
    if line:
        print(line.decode())

# Tokenize / Detokenize
resp = requests.post(f"{base_url}/tokenize", json={
    "content": "Hello world, how are you?",
    "add_special": False,
    "with_pieces": True,  # Return token pieces alongside IDs
})
print(resp.json()["tokens"])

resp = requests.post(f"{base_url}/detokenize", json={
    "tokens": [1, 2, 3, 4, 5]
})
print(resp.json()["content"])

# Apply chat template (format messages without inference)
resp = requests.post(f"{base_url}/apply-template", json={
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
})
print(resp.json()["prompt"])  # Formatted prompt string

# Metrics (Prometheus format)
resp = requests.get(f"{base_url}/metrics")
print(resp.text)
```

### Embedding via HTTP

```python
import requests
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/Qwen3-Embedding-0.6B-Q8_0.gguf"
params.embedding = True
params.n_ctx = 512
params.n_batch = 128
params.n_ubatch = 128
params.pooling_type = xlc.llama_pooling_type.LLAMA_POOLING_TYPE_LAST

server = xlc.Server(params)
base_url = server.listening_address

resp = requests.post(f"{base_url}/v1/embeddings", json={
    "input": ["I believe the meaning of life is", "This is a test"],
})
data = resp.json()
for item in data["data"]:
    print(f"Index {item['index']}: {len(item['embedding'])} dimensions")
```

### Reranking via HTTP

```python
import requests
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/bge-reranker-v2-m3-Q2_K.gguf"
params.embedding = True
params.reranking = True
params.n_ctx = 512
params.n_batch = 128
params.n_ubatch = 128
params.pooling_type = xlc.llama_pooling_type.LLAMA_POOLING_TYPE_RANK

server = xlc.Server(params)
base_url = server.listening_address

resp = requests.post(f"{base_url}/v1/rerank", json={
    "model": "bge-reranker",
    "query": "What is panda?",
    "top_n": 3,
    "documents": [
        "hi",
        "it is a bear",
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
    ],
})
for doc in resp.json()["results"]:
    print(f"Document {doc['index']}: relevance_score={doc['relevance_score']:.4f}")
```

### Code Infill via HTTP

The `/infill` endpoint supports fill-in-the-middle (FIM) for code completion:

```python
import requests

resp = requests.post(f"{base_url}/infill", json={
    "input_prefix": "def fibonacci(n):\n    if n <= 1:\n        return n\n",
    "input_suffix": "\n    return fibonacci(n-1) + fibonacci(n-2)",
    "max_tokens": 64,
})
print(resp.json()["content"])
```

### LoRA Adapters via HTTP

Manage LoRA adapters at runtime through HTTP endpoints:

```python
import requests

# List loaded LoRA adapters
resp = requests.get(f"{base_url}/lora-adapters")
print(resp.json())  # [{"id": 0, "path": "...", "scale": 1.0}, ...]

# Update global LoRA scales
requests.post(f"{base_url}/lora-adapters", json=[
    {"id": 0, "scale": 0.5},
])

# Per-request LoRA scaling (overrides global scale)
resp = requests.post(f"{base_url}/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32,
    "lora": [{"id": 0, "scale": 0.8}],
})
```

### Parallel Slots

The server supports multiple concurrent requests via parallel slots:

```python
params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
params.n_parallel = 4   # 4 concurrent request slots
params.n_ctx = 2048     # Context shared across slots
params.cont_batching = True  # Enable continuous batching (default: True)

server = xlc.Server(params)
```

### Prompt Caching

Prompt caching is enabled by default and reuses KV cache from previous requests when prompts share a common prefix:

```python
params = xlc.CommonParams()
params.cache_prompt = True     # Enabled by default
params.n_cache_reuse = 256     # Min chunk size for KV shifting reuse (0 = disabled)
```

You can also control caching per-request:

```python
result = server.handle_chat_completions({
    "messages": [{"role": "user", "content": "Hello!"}],
    "cache_prompt": True,  # Reuse KV cache from previous request
})
```

### Server Sleep & Wake

The server supports automatic sleep after idle time to free resources. When sleeping, the model and KV cache are unloaded from memory. Any new inference request automatically wakes the server.

```python
import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
params.sleep_idle_seconds = 60  # Sleep after 60 seconds of inactivity

server = xlc.Server(params)
# Server will automatically sleep/wake as needed
# Check status via: GET /props → {"is_sleeping": true/false}
```

Note: `/health`, `/props`, and `/models` endpoints remain responsive during sleep and do not trigger a wake-up.

### Web UI

llama.cpp ships a browser-based Web UI for chatting with a running server. This is analogous to step 2 ("Start llama-server") of the upstream [llama.cpp Web UI Getting Started guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/webui/README.md#getting-started), where running `./llama-server -m model.gguf` brings up the UI — with xllamacpp you simply start an `xlc.Server` instead.

> **Note:** xllamacpp's prebuilt wheels are compiled **without** the embedded Web UI (`LLAMA_BUILD_WEBUI` is not defined), so the UI assets are not baked into the binary. To serve the UI, download the official prebuilt UI package from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) (the `*-ui.tar.gz` asset, e.g. <https://github.com/ggml-org/llama.cpp/releases/download/b9536/llama-b9536-ui.tar.gz>), extract it, and point `params.public_path` at the extracted directory:
>
> ```sh
> curl -L -o llama-ui.tar.gz https://github.com/ggml-org/llama.cpp/releases/download/b9536/llama-b9536-ui.tar.gz
> mkdir -p webui && tar -xzf llama-ui.tar.gz -C webui
> ```

```python
import time
import webbrowser

import xllamacpp as xlc

params = xlc.CommonParams()
params.model.path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
params.hostnames = ["127.0.0.1"]
params.port = 8080

# The Web UI is enabled by default (params.ui = True; params.webui is a
# deprecated alias). Serve the prebuilt static assets via public_path.
# Point this at the directory where you extracted the official UI package.
params.public_path = "webui"

server = xlc.Server(params)

# The Web UI is served at the server root.
print(f"Web UI available at {server.listening_address}")
webbrowser.open(server.listening_address)

# The server runs in a background thread, so keep the process alive
# to leave the Web UI reachable.
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
```

Open the printed address (e.g. `http://127.0.0.1:8080`) in your browser to chat with the model.

> **Note:** This Web UI is handy for quick local testing, but if you are running xllamacpp through [Xinference](https://github.com/xorbitsai/inference), the Xinference web UI provides a more full-featured experience (model management, multiple backends, cluster deployment, etc.) and is the recommended choice for most users.

## Testing

The `tests` directory provides extensive examples and test coverage for xllamacpp.

### Download Models

As a first step, download a small GGUF model from [HuggingFace](https://huggingface.co/models?search=gguf). The default test model is [Llama-3.2-1B-Instruct-Q8_0.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf). Models should be placed in the `models/` directory:

```sh
make download
```

This basically just does:

```sh
cd xllamacpp
mkdir models && cd models
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf 
```

### Run Tests

You can verify your installation with `llama-cli`:

```sh
bin/llama-cli -c 512 -n 32 -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
 -p "Is mathematics discovered or invented?"
```

Run the full test suite:

```sh
make test
```

To run the optional decision-model integration test with the GGUF downloaded
above (requires llama.cpp b11361 or later):

```sh
XLLAMACPP_DECISION_MODEL=models/Laya-Q8_0.gguf pytest -q tests/test_decision.py
```
