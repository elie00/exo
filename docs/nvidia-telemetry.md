# NVIDIA telemetry in this fork

This is a contribution to the `elie00/exo` fork of `exo-explore/exo`, not a claim
to have created the upstream distributed inference engine. The collector lives
at `exo.utils.info_gatherer.nvidia_monitor`; the GGUF detector and tests use that
location after the upstream directory move.

## Measurement contract

- Install NVIDIA's optional `nvidia-ml-py` Python package on a host with a working
  NVIDIA driver. It provides the `pynvml` import. Mac development and mock tests
  do not require NVIDIA bindings or hardware.
- Device identity and VRAM are required for a usable device snapshot. Sensor
  failures produce `None` / JSON `null` for utilization, temperature or power.
  Zero means a measured zero, not an unsupported or unavailable sensor.
- Aggregate utilization and temperature are means; power is a sum, with NVML
  milliwatts converted to watts. An aggregate is unavailable if any contributing
  sensor or detected GPU is missing. It never averages missing values as zero.
- `gpu_count` and memory totals cover readable devices. `detected_gpu_count`
  records the enumeration count; `unavailable_gpu_indices` exposes partial
  collection. Device indices stay the physical NVML indices. Fields named
  `*_mb` retain the existing API names but use **MiB** (1024² bytes).
- The reported CUDA version is the driver's supported CUDA version, not proof
  of an installed toolkit, a CUDA-enabled inference backend, or a working kernel.
- Initialization, polling and shutdown are serialized with a reentrant lock.
  Failed initialization can be retried; shutdown clears local state even on a
  driver error. Async polling runs driver calls on a thread to avoid blocking
  the event loop. A driver call that hangs is not forcibly timed out.

## Local protocol checks

From the checkout root, a lightweight environment can test this module without
building the inference engine or downloading a model:

```bash
PYTHONPATH=src uv run --no-project --python 3.13 \
  --with pydantic --with loguru --with pytest --with pytest-asyncio --with pytest-env \
  pytest src/exo/utils/info_gatherer/tests/test_nvidia_monitor.py -q
```

Tests cover fully and partially readable devices, unsupported sensors, unit
conversion, version decoding, retry after initialization failure, shutdown,
concurrent polls and event-loop isolation. Mock results are not GPU benchmarks.

On an actual NVIDIA host, capture a snapshot with the optional bindings installed:

```bash
PYTHONPATH=src python - <<'PY'
from exo.utils.info_gatherer.nvidia_monitor import get_metrics, shutdown
try:
    print(get_metrics().model_dump_json(indent=2))
finally:
    shutdown()
PY
```

Record driver, bindings, OS and device versions alongside the snapshot. Compare
its memory, temperature and power with `nvidia-smi` at approximately the same
time, including a device with unsupported power telemetry when possible.
Device UUIDs identify physical hardware: redact them before sharing a report.

## Qualification boundary

This change validates the telemetry contract with mocked NVML calls on macOS.
It does not qualify physical NVIDIA devices, CUDA inference, placement,
multi-node execution or dashboard presentation. Following the upstream merge,
the collector is not wired into the main `InfoGatherer` event stream; the GGUF
detector is a separate consumer. Those integrations need independent checks.

### Validation on 2026-10-08

Base: `894a687e` from `elie00/exo` main. The original local checkout has separate
unpublished work, so this patch is isolated on `codex/nvidia-telemetry` in
`/Users/eybo/PycharmProjects/exo-ai-credibility`.

The collector and its typed test doubles pass Ruff lint/format, strict
BasedPyright (**0 errors**) and **21 tests** on macOS/Python 3.13.11. No NVIDIA
hardware was used. The Rust bindings can be built using the installed nightly
toolchain with the rustup shims first on PATH; Homebrew's stable Cargo initially
failed on PyO3's experimental async features.

The global merge-repair gates now pass. The unchanged base had previously been
checked in a separate worktree with the same tool versions; the current result
includes the broader [integration repair](fork-integration-repair.md).

| Gate | Untouched base | Current repair |
| --- | --- | --- |
| Strict type checking, entire project | 728 errors | **0 errors, 0 warnings** |
| Ruff lint, entire project | 77 errors | **0 errors** |
| Full pytest | 2 import errors during collection | **224 passed, 13 skipped** |
| Python formatting | Nix unavailable on this host | **131 files pass Ruff format**, as configured by `flake.nix` |
| Lock consistency | Not rechecked for the old lock | **`uv lock --check` and locked sync pass** |

The 13 skipped GPT-OSS tests need model weights and tokenizer artifacts missing
from the local cache; they are not counted as successful model inference. The
suite also emits 210 pre-existing multiprocessing/fork deprecation warnings.

`nix fmt` itself and the full Nix flake checks were not available on this host;
Python formatting used the equivalent configured Ruff formatter directly.
The patch is ready for a draft PR against `elie00/exo`. The separate original
checkout and its unpublished commits, and the upstream remote, are unchanged.
