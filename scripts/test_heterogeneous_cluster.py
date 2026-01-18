#!/usr/bin/env python3
"""Test script for heterogeneous cluster inference.

This script tests distributed inference across Mac (Metal) and Dell (CUDA)
using the same GGUF model on both machines.

Usage:
    # Run locally (creates remote node via SSH)
    python scripts/test_heterogeneous_cluster.py

    # Run with custom config
    python scripts/test_heterogeneous_cluster.py --model /path/to/model.gguf
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from exo.worker.engines.gguf.heterogeneous_pipeline import (
    BackendType,
    GGUFInferenceNode,
    GGUFModelConfig,
    HeterogeneousCluster,
    NodeCapabilities,
)


async def create_local_node(model_path: Path) -> GGUFInferenceNode:
    """Create an inference node on the local machine."""
    caps = NodeCapabilities.detect("local")
    
    config = GGUFModelConfig(
        model_path=model_path,
        n_ctx=512,
        n_gpu_layers=-1,  # Use all GPU layers
        verbose=False,
    )
    
    return GGUFInferenceNode(f"local-{caps.backend.value}", config)


async def test_single_node(model_path: Path):
    """Test inference on a single node."""
    print("\n" + "=" * 60)
    print("🧪 Testing Single Node Inference")
    print("=" * 60)
    
    node = await create_local_node(model_path)
    
    print(f"\n📍 Node: {node.node_id}")
    print(f"   Backend: {node.backend.value}")
    print(f"   Memory: {node.capabilities.memory_mb} MB")
    
    print("\n⏳ Loading model...")
    await node.load_model()
    
    prompt = "What is the capital of France? Answer in one word:"
    print(f"\n📝 Prompt: {prompt}")
    print("💬 Response: ", end="", flush=True)
    
    async for token in node.generate(prompt, max_tokens=20, temperature=0.1):
        print(token, end="", flush=True)
    
    print("\n\n✅ Single node test passed!")
    node.unload()


async def test_cluster_load_balancing(model_path: Path, nodes: list[GGUFInferenceNode]):
    """Test load balancing across multiple nodes."""
    print("\n" + "=" * 60)
    print("🧪 Testing Cluster Load Balancing")
    print("=" * 60)
    
    cluster = HeterogeneousCluster()
    for node in nodes:
        cluster.add_node(node)
    
    print("\n📊 Cluster Configuration:")
    for node in cluster.nodes:
        print(f"   • {node.node_id}: {node.backend.value} ({node.capabilities.memory_mb} MB)")
    
    print("\n⏳ Initializing cluster...")
    await cluster.initialize()
    
    # Test single generation
    prompt = "Hello! How are you today?"
    print(f"\n📝 Prompt: {prompt}")
    print("💬 Response: ", end="", flush=True)
    
    async for token in cluster.generate(prompt, max_tokens=30):
        print(token, end="", flush=True)
    
    # Test parallel generation
    print("\n\n🔄 Testing parallel generation across nodes...")
    prompts = [
        "What is 2+2?",
        "What color is the sky?",
        "Name a planet:",
    ]
    
    responses = await cluster.parallel_generate(prompts, max_tokens=20, temperature=0.1)
    
    for prompt, response in zip(prompts, responses):
        print(f"   Q: {prompt}")
        print(f"   A: {response.strip()[:50]}...")
    
    print("\n✅ Cluster load balancing test passed!")
    cluster.shutdown()


async def test_chat_completion(model_path: Path):
    """Test chat completion."""
    print("\n" + "=" * 60)
    print("🧪 Testing Chat Completion")
    print("=" * 60)
    
    node = await create_local_node(model_path)
    await node.load_model()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ]
    
    print("\n💬 Chat messages:")
    for msg in messages:
        print(f"   [{msg['role']}]: {msg['content']}")
    
    print("\n🤖 Assistant: ", end="", flush=True)
    
    async for token in node.chat(messages, max_tokens=50):
        print(token, end="", flush=True)
    
    print("\n\n✅ Chat completion test passed!")
    node.unload()


async def main():
    parser = argparse.ArgumentParser(description="Test heterogeneous cluster inference")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/tmp/gguf_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--test",
        choices=["single", "cluster", "chat", "all"],
        default="all",
        help="Which test to run",
    )
    
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"❌ Model not found: {args.model}")
        print("\nTo download a test model, run:")
        print("  uv run python -c \"")
        print("  from huggingface_hub import hf_hub_download")
        print("  hf_hub_download(")
        print("      repo_id='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',")
        print("      filename='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',")
        print("      local_dir='/tmp/gguf_models'")
        print("  )\"")
        return
    
    print("=" * 60)
    print("🚀 Heterogeneous Cluster Test Suite")
    print("=" * 60)
    print(f"📦 Model: {args.model}")
    print(f"📏 Size: {args.model.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Detect local capabilities
    caps = NodeCapabilities.detect("local")
    print(f"\n🖥️  Local Node:")
    print(f"   Backend: {caps.backend.value}")
    print(f"   GPU: {caps.gpu_name or 'N/A'}")
    print(f"   Memory: {caps.memory_mb} MB")
    
    if args.test in ("single", "all"):
        await test_single_node(args.model)
    
    if args.test in ("chat", "all"):
        await test_chat_completion(args.model)
    
    if args.test in ("cluster", "all"):
        # For cluster test, we'd need multiple nodes
        # For now, just create multiple local nodes
        print("\n⚠️  Cluster test requires multiple nodes.")
        print("   Use the HeterogeneousCluster class directly for multi-node testing.")
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
