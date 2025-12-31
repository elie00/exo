# EXO + Ollama Integration: Distributed GGUF Inference

This module enables **TRUE weight sharding** for Ollama/GGUF models across multiple nodes in an EXO cluster. Each node loads only its assigned layers, and activations are passed between nodes via TCP.

## 🌟 Features

- ✅ **Automatic Ollama Discovery**: Detects all locally installed Ollama models
- ✅ **GGUF Parsing**: Reads model architecture, layer count, and tensor info from GGUF files
- ✅ **Layer-based Sharding**: Distributes model layers across multiple nodes
- ✅ **Memory Estimation**: Calculates memory requirements per shard
- ✅ **Distributed Pipeline**: TCP-based activation passing between nodes
- ✅ **EXO Integration**: Seamlessly integrates with EXO's instance and runner systems
- ✅ **HTTP API**: Server mode with OpenAI-compatible API

## 📦 Supported Models

Any Ollama model works! The system automatically detects:
- Model architecture (llama, qwen, mistral, nemotron, etc.)
- Number of layers for sharding
- Quantization type (Q4_K_M, Q8_0, etc.)
- Hidden dimensions and context length

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /path/to/exo
uv sync --extra ollama
```

### 2. Discover Available Models

```bash
uv run python -m exo.worker.engines.gguf.cli discover
```

Example output:
```
📦 Found 2 Ollama model(s):

📌 qwen3-coder:30b
   Size: 17.3 GB
   Architecture: qwen3moe
   Layers: 48
   Context Length: 262,144
   Quantization: MOSTLY_Q4_K_M

📌 nemotron-3-nano:30b
   Size: 22.6 GB
   Architecture: nemotron_h_moe
   Layers: 52
   Context Length: 1,048,576
   Quantization: MOSTLY_Q4_K_M
```

### 3. View Sharding Plan

```bash
uv run python -m exo.worker.engines.gguf.cli shard qwen3-coder:30b -n 2
```

Example output:
```
📦 Sharding Plan: qwen3-coder:30b
   Distributing across 2 node(s)

🏗️  Model Info
   Total Layers: 48
   Total Size: 17.3 GB

📊 Node Assignments
   Node 0: (embeddings)
      Layers: 0 - 23
      Estimated Memory: ~9.5 GB

   Node 1: (output)
      Layers: 24 - 47
      Estimated Memory: ~9.2 GB
```

### 4. Single-Node Test

```bash
uv run python -m exo.worker.engines.gguf.launcher \
    --model qwen3-coder:30b \
    --rank 0 \
    --world-size 1 \
    --nodes "localhost:50100" \
    --interactive
```

## 🌐 Multi-Node Distributed Deployment

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Distributed Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │
│   │   Node 0     │     │   Node 1     │     │   Node 2     │           │
│   │   (HEAD)     │────▶│   (MIDDLE)   │────▶│   (TAIL)     │           │
│   │              │     │              │     │              │           │
│   │ Embeddings   │     │ Layers 16-31 │     │ Layers 32-47 │           │
│   │ Layers 0-15  │     │              │     │ Output Layer │           │
│   │              │     │              │     │ Generation   │           │
│   └──────────────┘     └──────────────┘     └──────────────┘           │
│         │                    │                    │                     │
│         └───── TCP/RDMA Activation Passing ──────┘                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Prerequisites

Each node must have:
1. **Ollama installed** with the same model pulled
2. **EXO cloned** from the same repository
3. **Network connectivity** between all nodes (same subnet recommended)
4. **Ports open** for activation passing (default: 50100+)

### Step-by-Step Setup

#### 1. Pull the model on all nodes

```bash
# On each machine
ollama pull qwen3-coder:30b
```

#### 2. Clone and setup EXO on all nodes

```bash
# On each machine
git clone https://github.com/exo-explore/exo
cd exo
uv sync --extra ollama
```

#### 3. Start the distributed inference

**On Node 0 (HEAD - receives prompts):**
```bash
uv run python -m exo.worker.engines.gguf.launcher \
    --model qwen3-coder:30b \
    --rank 0 \
    --world-size 2 \
    --nodes "192.168.1.10:50100,192.168.1.11:50100" \
    --interactive
```

**On Node 1 (TAIL - generates output):**
```bash
uv run python -m exo.worker.engines.gguf.launcher \
    --model qwen3-coder:30b \
    --rank 1 \
    --world-size 2 \
    --nodes "192.168.1.10:50100,192.168.1.11:50100"
```

### Server Mode (HTTP API)

Run with `--server` to enable HTTP API:

```bash
# On head node
uv run python -m exo.worker.engines.gguf.launcher \
    --model qwen3-coder:30b \
    --rank 0 \
    --world-size 2 \
    --nodes "192.168.1.10:50100,192.168.1.11:50100" \
    --server \
    --api-port 8000
```

API endpoints:
- `POST /v1/generate` - Generate text
- `GET /health` - Health check

Example request:
```bash
curl -X POST http://192.168.1.10:8000/v1/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'
```

## 📊 Memory Requirements

| Model | 1 Node | 2 Nodes | 3 Nodes | 4 Nodes |
|-------|--------|---------|---------|---------|
| qwen3-coder:30b (17 GB) | 17.3 GB | ~9.2 GB | ~6.3 GB | ~4.8 GB |
| nemotron-3-nano:30b (23 GB) | 22.6 GB | ~12.1 GB | ~8.3 GB | ~6.4 GB |
| llama-3:70b (40 GB) | 40 GB | ~21 GB | ~14.5 GB | ~11 GB |

## 🔧 Architecture Overview

```
src/exo/worker/engines/gguf/
├── __init__.py            # Module exports
├── cli.py                 # Command-line interface
├── launcher.py            # Distributed launcher
├── ollama_discovery.py    # Detect local Ollama models
├── gguf_loader.py         # Parse GGUF file format
├── gguf_sharding.py       # Layer distribution logic
├── distributed_pipeline.py # TCP-based activation passing
├── gguf_inference.py      # Inference with llama.cpp
├── gguf_runner.py         # EXO runner integration
├── exo_integration.py     # Bridge to EXO model system
└── README.md              # This file
```

## 🔌 Programmatic API

### Discover and List Models

```python
from exo.worker.engines.gguf import discover_ollama_models, load_gguf_metadata

# Discover all Ollama models
models = discover_ollama_models()

for model in models:
    print(f"{model.full_name}: {model.size_gb:.1f} GB")
    
    # Load detailed metadata
    info = load_gguf_metadata(model.model_path)
    print(f"  Architecture: {info.architecture}")
    print(f"  Layers: {info.n_layers}")
```

### Calculate Sharding Plan

```python
from exo.worker.engines.gguf import calculate_shard_assignments

plan = calculate_shard_assignments(model.model_path, world_size=3)

for assignment in plan.assignments:
    print(f"Node {assignment.node_rank}: layers {assignment.start_layer}-{assignment.end_layer-1}")
    print(f"  Memory: ~{assignment.estimated_memory}")
```

### Run Distributed Pipeline

```python
from exo.worker.engines.gguf import create_distributed_pipeline
import asyncio

async def main():
    node_addresses = [("192.168.1.10", 50100), ("192.168.1.11", 50100)]
    
    pipeline = await create_distributed_pipeline(
        model_path=model.model_path,
        node_addresses=node_addresses,
        this_node_index=0,  # This is node 0
    )
    
    async def on_token(token):
        print(token, end="", flush=True)
    
    pipeline.set_generation_callback(on_token)
    await pipeline.initialize()
    await pipeline.run_inference("Hello!", max_tokens=100)
    await pipeline.shutdown()

asyncio.run(main())
```

## 🛣️ Roadmap

- [x] Automatic Ollama model discovery
- [x] GGUF file parsing
- [x] Layer-based sharding calculation
- [x] Single-node inference
- [x] TCP activation transport
- [x] Multi-node pipeline coordination
- [x] HTTP API server mode
- [ ] RDMA transport for Thunderbolt connections
- [ ] Tensor parallelism (in addition to pipeline)
- [ ] Automatic model sync across nodes
- [ ] Dashboard integration
- [ ] Performance benchmarks

## ⚙️ Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Ollama model name (e.g., `qwen3-coder:30b`) |
| `--rank` | Required | Node rank (0-indexed) |
| `--world-size` | Required | Total number of nodes |
| `--nodes` | Required | Comma-separated `host:port` list |
| `--prompt` | None | Single prompt to run |
| `--max-tokens` | 256 | Maximum tokens to generate |
| `--interactive` | False | Interactive chat mode |
| `--server` | False | Run HTTP API server |
| `--api-port` | 8000 | HTTP API port (server mode) |

## 🐛 Troubleshooting

### Model not found
```
Error: Model 'xxx' not found
```
Make sure the model is pulled with Ollama:
```bash
ollama pull qwen3-coder:30b
```

### Connection refused
```
Failed to connect to next node
```
- Check that all nodes are running
- Verify network connectivity: `ping <node_ip>`
- Check firewall rules for ports 50100+

### Out of memory
```
CUDA out of memory / Metal out of memory
```
- Use more nodes to reduce per-node memory
- Try a smaller quantization (Q4 instead of Q8)

## 📝 License

This module is part of EXO and is licensed under Apache 2.0.
