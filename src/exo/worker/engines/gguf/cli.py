#!/usr/bin/env python3
"""CLI tool for Ollama/GGUF integration with EXO.

This script provides utilities for:
- Discovering Ollama models
- Showing sharding plans
- Testing GGUF inference
- Running distributed inference

Usage:
    uv run python -m exo.worker.engines.gguf.cli discover
    uv run python -m exo.worker.engines.gguf.cli info <model_name>
    uv run python -m exo.worker.engines.gguf.cli shard <model_name> --nodes <N>
    uv run python -m exo.worker.engines.gguf.cli test <model_name> --prompt "Hello"
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from exo.worker.engines.gguf.ollama_discovery import (
    OllamaModel,
    discover_ollama_models,
    get_ollama_model_path,
)
from exo.worker.engines.gguf.gguf_loader import (
    GGUFModelInfo,
    load_gguf_metadata,
)
from exo.worker.engines.gguf.gguf_sharding import (
    calculate_shard_assignments,
    GGUFShardingPlan,
)


def cmd_discover(args: argparse.Namespace) -> None:
    """Discover and list all available Ollama models."""
    print("\n🔍 Discovering Ollama models...\n")
    
    models = discover_ollama_models()
    
    if not models:
        print("❌ No Ollama models found.")
        print("   Make sure Ollama is installed and you have models pulled.")
        print("   Try: ollama pull llama3.2")
        return
    
    print(f"📦 Found {len(models)} Ollama model(s):\n")
    print("-" * 80)
    
    for model in models:
        print(f"📌 {model.full_name}")
        print(f"   Size: {model.size_gb:.1f} GB")
        print(f"   Path: {model.model_path}")
        
        try:
            info = load_gguf_metadata(model.model_path, load_tensors=False)
            print(f"   Architecture: {info.architecture}")
            print(f"   Layers: {info.n_layers}")
            print(f"   Hidden Size: {info.hidden_size}")
            print(f"   Context Length: {info.context_length:,}")
            print(f"   Quantization: {info.quantization}")
        except Exception as e:
            print(f"   ⚠️  Could not load GGUF metadata: {e}")
        
        print("-" * 80)


def cmd_info(args: argparse.Namespace) -> None:
    """Show detailed info about a specific model."""
    model_name = args.model
    
    # Parse model name
    if ":" in model_name:
        name, tag = model_name.split(":", 1)
    else:
        name, tag = model_name, "latest"
    
    # Find the model
    models = discover_ollama_models()
    model = next(
        (m for m in models if m.name == name and m.tag == tag),
        None
    )
    
    if not model:
        print(f"❌ Model '{model_name}' not found.")
        print("   Run 'discover' to see available models.")
        return
    
    print(f"\n📌 Model: {model.full_name}")
    print("=" * 60)
    print(f"Size: {model.size_gb:.1f} GB ({model.size_bytes:,} bytes)")
    print(f"Path: {model.model_path}")
    
    try:
        info = load_gguf_metadata(model.model_path, load_tensors=True)
        
        print(f"\n🏗️  Architecture")
        print("-" * 40)
        print(f"Type: {info.architecture}")
        print(f"Layers: {info.n_layers}")
        print(f"Hidden Size: {info.hidden_size}")
        print(f"Attention Heads: {info.n_heads}")
        print(f"KV Heads: {info.n_kv_heads}")
        print(f"Intermediate Size: {info.intermediate_size}")
        print(f"Vocab Size: {info.vocab_size:,}")
        print(f"Context Length: {info.context_length:,}")
        
        print(f"\n📊 Technical Details")
        print("-" * 40)
        print(f"GGUF Version: {info.version}")
        print(f"Quantization: {info.quantization}")
        print(f"Total Tensors: {info.tensor_count}")
        print(f"RoPE Freq Base: {info.rope_freq_base:,.0f}")
        
        # Show some key metadata
        print(f"\n📋 Key Metadata")
        print("-" * 40)
        for key in sorted(info.metadata.keys())[:20]:
            value = info.metadata[key]
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            elif isinstance(value, list) and len(value) > 3:
                value = f"{value[:3]}... ({len(value)} items)"
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"⚠️  Could not load GGUF metadata: {e}")


def cmd_shard(args: argparse.Namespace) -> None:
    """Show sharding plan for a model."""
    model_name = args.model
    num_nodes = args.nodes
    
    # Parse model name
    if ":" in model_name:
        name, tag = model_name.split(":", 1)
    else:
        name, tag = model_name, "latest"
    
    # Find the model
    models = discover_ollama_models()
    model = next(
        (m for m in models if m.name == name and m.tag == tag),
        None
    )
    
    if not model:
        print(f"❌ Model '{model_name}' not found.")
        return
    
    print(f"\n📦 Sharding Plan: {model.full_name}")
    print(f"   Distributing across {num_nodes} node(s)")
    print("=" * 60)
    
    try:
        info = load_gguf_metadata(model.model_path, load_tensors=True)
        plan = calculate_shard_assignments(model.model_path, num_nodes, info)
        
        print(f"\n🏗️  Model Info")
        print(f"   Total Layers: {info.n_layers}")
        print(f"   Total Size: {model.size_gb:.1f} GB")
        print(f"   Total Tensors: {info.tensor_count}")
        
        print(f"\n📊 Node Assignments")
        print("-" * 60)
        
        for assignment in plan.assignments:
            role = ""
            if assignment.is_first_shard:
                role = " (embeddings)"
            if assignment.is_last_shard:
                role += " (output)"
            
            mem_gb = assignment.estimated_memory.in_bytes / (1024**3)
            
            print(f"   Node {assignment.node_rank}:{role}")
            print(f"      Layers: {assignment.start_layer} - {assignment.end_layer - 1}")
            print(f"      Layer Count: {assignment.layer_count}")
            print(f"      Estimated Memory: ~{mem_gb:.1f} GB")
            if assignment.tensor_names:
                print(f"      Tensors: {len(assignment.tensor_names)}")
            print()
        
    except Exception as e:
        print(f"⚠️  Could not compute sharding plan: {e}")
        raise


def cmd_test(args: argparse.Namespace) -> None:
    """Test inference with a model."""
    model_name = args.model
    prompt = args.prompt
    max_tokens = args.max_tokens
    
    # Parse model name
    if ":" in model_name:
        name, tag = model_name.split(":", 1)
    else:
        name, tag = model_name, "latest"
    
    # Find the model
    models = discover_ollama_models()
    model = next(
        (m for m in models if m.name == name and m.tag == tag),
        None
    )
    
    if not model:
        print(f"❌ Model '{model_name}' not found.")
        return
    
    print(f"\n🧪 Testing: {model.full_name}")
    print(f"   Prompt: {prompt}")
    print("=" * 60)
    
    try:
        from exo.worker.engines.gguf.gguf_runner import GGUFRunner
        
        print("\n⏳ Loading model...")
        runner = GGUFRunner(
            model_path=model.model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
        )
        runner.load()
        
        print("\n💬 Response:")
        print("-" * 40)
        
        for response in runner.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
        ):
            print(response.text, end="", flush=True)
        
        print("\n")
        print("-" * 40)
        print("✅ Test complete!")
        
        runner.unload()
        
    except ImportError:
        print("❌ llama-cpp-python is not installed.")
        print("   Install with: uv sync --extra ollama")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Ollama/GGUF integration for EXO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s discover                     # List all Ollama models
  %(prog)s info qwen3-coder:30b         # Show model details
  %(prog)s shard qwen3-coder:30b -n 2   # Show 2-node sharding plan
  %(prog)s test qwen3-coder:30b         # Test inference
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # discover command
    discover_parser = subparsers.add_parser("discover", help="Discover Ollama models")
    discover_parser.set_defaults(func=cmd_discover)
    
    # info command
    info_parser = subparsers.add_parser("info", help="Show model info")
    info_parser.add_argument("model", help="Model name (e.g., qwen3-coder:30b)")
    info_parser.set_defaults(func=cmd_info)
    
    # shard command
    shard_parser = subparsers.add_parser("shard", help="Show sharding plan")
    shard_parser.add_argument("model", help="Model name (e.g., qwen3-coder:30b)")
    shard_parser.add_argument("-n", "--nodes", type=int, default=2, help="Number of nodes")
    shard_parser.set_defaults(func=cmd_shard)
    
    # test command
    test_parser = subparsers.add_parser("test", help="Test model inference")
    test_parser.add_argument("model", help="Model name (e.g., qwen3-coder:30b)")
    test_parser.add_argument("-p", "--prompt", default="Hello! How are you?", help="Prompt")
    test_parser.add_argument("-m", "--max-tokens", type=int, default=100, help="Max tokens")
    test_parser.set_defaults(func=cmd_test)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
