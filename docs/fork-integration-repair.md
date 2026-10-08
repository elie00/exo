# Fork integration repair — 2026-10-08

This work repairs `elie00/exo` after merge `894a687e`. The upstream EXO engine
remains the work of `exo-explore/exo`. These changes are contributions to a fork,
not evidence of creating the distributed engine or reproducing upstream results.

## What changed

- **NVIDIA telemetry:** optional NVML loading, serialized initialization/polling/
  shutdown, supported-driver version decoding, and unknown sensors represented
  as `null`. [Telemetry contract and device protocol](nvidia-telemetry.md).
- **Placement:** ported the GPU-memory policy to `MemoryUsage`, `Cycle` and
  `ModelCard`. `prefer_gpu` now governs filtering, ranking and layer allocation.
  All-GPU cycles use VRAM; mixed CPU/GPU cycles use RAM consistently. A known GPU
  with missing free-VRAM telemetry contributes zero, and measured zero survives
  serialization. This is a capacity heuristic, not a runtime peak-memory proof.
- **GGUF/Ollama boundaries:** removed imports of deleted upstream models; parse
  GGUF scalar types and Ollama manifests with validation; reject missing model
  dimensions instead of inventing a 32-layer/4096-hidden model. Optional llama.cpp
  responses pass through validated completion/stream contracts and explicit close.
  Fixed parallel Ollama calls passing a token limit as the model name.
- **Network prototype:** replace network `pickle.loads` with version-1 JSON
  envelopes and base64 numeric arrays. Check frame size before reading the body,
  reject object/structured dtypes, and verify shape against byte length. The
  transport remains unauthenticated TCP and is not a production deployment.
- **Honest unsupported states:** reject master GGUF placement without a model path
  and ports, unwired RPC workers, and unimplemented partial GGUF loading. Optional
  image generation rejects missing mflux, unsupported models and invented URL
  responses. The `placeholder` model is an explicitly selected UI demo.

## Reproduce the local gates

Python 3.13 is required. The existing Rust bindings build with the installed
nightly toolchain; no model or paid API is needed for the CPU contract tests.

```bash
PATH="$HOME/.cargo/bin:$PATH" RUSTUP_TOOLCHAIN=nightly uv sync --locked
uv run --no-sync basedpyright
uv run --no-sync ruff check
uv run --no-sync ruff format --check
uv run --no-sync pytest -q
```

`flake.nix` delegates Python formatting to Ruff. Nix is unavailable on the test
host, so the configured Python formatter was applied directly to all Python
files. Nix/Rust/dashboard-wide flake checks were not run and are not implied by
these Python gates. No Rust, Swift, dashboard or Nix implementation changed.

Pillow is now an explicit development dependency for the image contract tests and
strict typing. Relocking also moves `hf-xet` from unavailable `1.2.1rc0` to `1.7.0`;
attempting to retain `1.2.1rc0` produced an unsatisfiable resolution. Other existing
package versions are unchanged. `uv lock --check` and a locked sync succeed.

## Recorded local result

On macOS/Python 3.13.11 after locked sync: **0 strict type errors or warnings**,
**0 Ruff lint errors**, **131 files passing Ruff formatting**, and **224 tests
passed / 13 skipped** in 37.87 seconds. The skipped tests are described below.
There are 210 existing multiprocessing/fork deprecation warnings from the
node-ID persistence tests. These are software checks, not inference benchmarks.

## Qualification limits

The optional GPT-OSS tests require the actual indexed safetensor files plus
`config.json`, `tokenizer.json` and `tokenizer_config.json`. The local cache had
only configuration and an index. Their precondition now reports missing files
rather than starting inference with no weights; corrupt existing indexes still
fail. The tests remain present and run when the artifacts are complete.

The GGUF pipeline currently generates on the head and relays token text. Selecting
layer metadata and choosing GPU-layer offload counts do **not** implement partial
weight loading or transformer computation on worker nodes. The RPC main path is
not wired to workers and now fails explicitly if they are requested. All nodes
using the experimental TCP relay must run the new JSON protocol; old pickle
packets are intentionally incompatible and rejected.

The main `InfoGatherer` stream still does not publish NVIDIA measurements.
Physical NVIDIA collection, CUDA/Metal inference, real llama.cpp/mflux integration,
multi-node model inference, dashboard display and performance claims require
separate qualification. No GPU throughput, speedup, model-quality or production
readiness claim is made by these mocked/unit checks.
