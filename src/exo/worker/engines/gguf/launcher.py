#!/usr/bin/env python3
"""Distributed GGUF inference launcher.

This script coordinates distributed inference across multiple nodes,
enabling true weight sharding for Ollama/GGUF models.

Usage:
    # On each node, run with appropriate configuration:
    
    # Head node (Node 0):
    uv run python -m exo.worker.engines.gguf.launcher \
        --model qwen3-coder:30b \
        --rank 0 \
        --world-size 2 \
        --nodes "192.168.1.10:50100,192.168.1.11:50100"
    
    # Worker node (Node 1):
    uv run python -m exo.worker.engines.gguf.launcher \
        --model qwen3-coder:30b \
        --rank 1 \
        --world-size 2 \
        --nodes "192.168.1.10:50100,192.168.1.11:50100"
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from exo.worker.engines.gguf.ollama_discovery import discover_ollama_models
from exo.worker.engines.gguf.gguf_loader import load_gguf_metadata
from exo.worker.engines.gguf.gguf_sharding import calculate_shard_assignments
from exo.worker.engines.gguf.distributed_pipeline import (
    PipelineNodeConfig,
    DistributedGGUFPipeline,
    NodeRole,
)


def parse_node_addresses(nodes_str: str) -> list[tuple[str, int]]:
    """Parse node addresses from comma-separated string."""
    addresses = []
    for node in nodes_str.split(","):
        parts = node.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid node address: {node}")
        host = parts[0]
        port = int(parts[1])
        addresses.append((host, port))
    return addresses


def get_model_path(model_name: str) -> Optional[Path]:
    """Get path to Ollama model by name."""
    if ":" in model_name:
        name, tag = model_name.split(":", 1)
    else:
        name, tag = model_name, "latest"
    
    models = discover_ollama_models()
    for model in models:
        if model.name == name and model.tag == tag:
            return model.model_path
    
    return None


async def run_distributed_node(
    model_path: Path,
    rank: int,
    world_size: int,
    node_addresses: list[tuple[str, int]],
    prompt: Optional[str] = None,
    max_tokens: int = 256,
    interactive: bool = False,
) -> None:
    """Run a distributed inference node."""
    
    # Get sharding plan
    info = load_gguf_metadata(model_path, load_tensors=False)
    plan = calculate_shard_assignments(model_path, world_size, info)
    assignment = plan.get_assignment_for_rank(rank)
    
    # Build config
    this_host, this_port = node_addresses[rank]
    
    prev_host, prev_port = None, None
    if rank > 0:
        prev_host, prev_port = node_addresses[rank - 1]
    
    next_host, next_port = None, None
    if rank < world_size - 1:
        next_host, next_port = node_addresses[rank + 1]
    
    config = PipelineNodeConfig(
        node_id=f"node-{rank}",
        rank=rank,
        world_size=world_size,
        model_path=model_path,
        start_layer=assignment.start_layer,
        end_layer=assignment.end_layer,
        total_layers=assignment.total_layers,
        listen_host="0.0.0.0",
        listen_port=this_port,
        prev_node_host=prev_host,
        prev_node_port=prev_port,
        next_node_host=next_host,
        next_node_port=next_port,
    )
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"🚀 Distributed GGUF Node {rank}/{world_size}")
    print(f"{'='*60}")
    print(f"Role: {config.role.value.upper()}")
    print(f"Layers: {config.start_layer} - {config.end_layer - 1}")
    print(f"Listening on: {this_host}:{this_port}")
    if prev_host:
        print(f"Previous node: {prev_host}:{prev_port}")
    if next_host:
        print(f"Next node: {next_host}:{next_port}")
    print(f"{'='*60}\n")
    
    # Create pipeline
    pipeline = DistributedGGUFPipeline(config)
    
    # Set up token callback for tail node
    async def on_token(token: str):
        print(token, end="", flush=True)
    
    if config.is_tail:
        pipeline.set_generation_callback(on_token)
    
    # Initialize
    print("⏳ Initializing pipeline...")
    await pipeline.initialize()
    print("✅ Pipeline initialized\n")
    
    try:
        if config.is_head:
            # Head node handles prompts
            if interactive:
                await run_interactive(pipeline, max_tokens)
            elif prompt:
                print(f"📝 Prompt: {prompt}\n")
                print("💬 Response: ", end="")
                await pipeline.run_inference(prompt, max_tokens)
                print("\n")
            else:
                # Default to interactive mode
                await run_interactive(pipeline, max_tokens)
        else:
            # Worker nodes wait for activations
            print("⏳ Waiting for activations from previous node...")
            await pipeline.run_inference("", max_tokens)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted")
    
    finally:
        await pipeline.shutdown()
        print("👋 Node shutdown complete")


async def run_interactive(pipeline: DistributedGGUFPipeline, max_tokens: int):
    """Run interactive chat mode."""
    print("🎯 Interactive mode - type 'quit' to exit\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt:
                continue
            
            print("Assistant: ", end="")
            await pipeline.run_inference(prompt, max_tokens)
            print("\n")
            
        except EOFError:
            break


async def run_server_mode(
    model_path: Path,
    rank: int,
    world_size: int,
    node_addresses: list[tuple[str, int]],
    api_port: int = 8000,
) -> None:
    """Run node in server mode with HTTP API."""
    from aiohttp import web
    
    # Get sharding plan
    info = load_gguf_metadata(model_path, load_tensors=False)
    plan = calculate_shard_assignments(model_path, world_size, info)
    assignment = plan.get_assignment_for_rank(rank)
    
    # Build config
    this_host, this_port = node_addresses[rank]
    
    prev_host, prev_port = None, None
    if rank > 0:
        prev_host, prev_port = node_addresses[rank - 1]
    
    next_host, next_port = None, None
    if rank < world_size - 1:
        next_host, next_port = node_addresses[rank + 1]
    
    config = PipelineNodeConfig(
        node_id=f"node-{rank}",
        rank=rank,
        world_size=world_size,
        model_path=model_path,
        start_layer=assignment.start_layer,
        end_layer=assignment.end_layer,
        total_layers=assignment.total_layers,
        listen_host="0.0.0.0",
        listen_port=this_port,
        prev_node_host=prev_host,
        prev_node_port=prev_port,
        next_node_host=next_host,
        next_node_port=next_port,
    )
    
    pipeline = DistributedGGUFPipeline(config)
    
    # Token buffer for streaming
    token_buffer: list[str] = []
    
    async def on_token(token: str):
        token_buffer.append(token)
    
    if config.is_tail:
        pipeline.set_generation_callback(on_token)
    
    await pipeline.initialize()
    
    # HTTP handlers
    async def handle_generate(request: web.Request) -> web.Response:
        if not config.is_head:
            return web.json_response(
                {"error": "Only head node accepts requests"},
                status=400
            )
        
        data = await request.json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 256)
        
        token_buffer.clear()
        await pipeline.run_inference(prompt, max_tokens)
        
        response_text = "".join(token_buffer)
        return web.json_response({
            "response": response_text,
            "tokens": len(token_buffer),
        })
    
    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "rank": rank,
            "role": config.role.value,
            "layers": f"{config.start_layer}-{config.end_layer-1}",
        })
    
    # Set up routes
    app = web.Application()
    app.router.add_post("/v1/generate", handle_generate)
    app.router.add_get("/health", handle_health)
    
    # Start worker task if not head
    worker_task = None
    if not config.is_head:
        worker_task = asyncio.create_task(pipeline.run_inference("", 0))
    
    # Run server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", api_port)
    
    print(f"\n🌐 HTTP API running on http://0.0.0.0:{api_port}")
    print(f"   POST /v1/generate - Generate text")
    print(f"   GET  /health      - Health check\n")
    
    await site.start()
    
    try:
        await asyncio.Event().wait()  # Run forever
    finally:
        if worker_task:
            worker_task.cancel()
        await runner.cleanup()
        await pipeline.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Distributed GGUF inference launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two-node setup:
  
  # On machine 1 (head):
  %(prog)s --model qwen3-coder:30b --rank 0 --world-size 2 \\
           --nodes "localhost:50100,192.168.1.11:50100" \\
           --prompt "Hello, how are you?"
  
  # On machine 2 (worker):
  %(prog)s --model qwen3-coder:30b --rank 1 --world-size 2 \\
           --nodes "192.168.1.10:50100,localhost:50100"
  
  # Server mode with HTTP API:
  %(prog)s --model qwen3-coder:30b --rank 0 --world-size 2 \\
           --nodes "localhost:50100,192.168.1.11:50100" --server
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Ollama model name (e.g., qwen3-coder:30b)"
    )
    parser.add_argument(
        "--rank", "-r",
        type=int,
        required=True,
        help="Rank of this node (0-indexed)"
    )
    parser.add_argument(
        "--world-size", "-w",
        type=int,
        required=True,
        help="Total number of nodes"
    )
    parser.add_argument(
        "--nodes", "-n",
        required=True,
        help="Comma-separated list of node addresses (host:port)"
    )
    parser.add_argument(
        "--prompt", "-p",
        help="Prompt for generation (head node only)"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive chat mode (head node only)"
    )
    parser.add_argument(
        "--server", "-s",
        action="store_true",
        help="Run in server mode with HTTP API"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="HTTP API port (server mode only)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.rank < 0 or args.rank >= args.world_size:
        print(f"Error: rank must be between 0 and {args.world_size - 1}")
        sys.exit(1)
    
    # Parse node addresses
    try:
        node_addresses = parse_node_addresses(args.nodes)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if len(node_addresses) != args.world_size:
        print(f"Error: number of nodes ({len(node_addresses)}) must match world-size ({args.world_size})")
        sys.exit(1)
    
    # Find model
    model_path = get_model_path(args.model)
    if model_path is None:
        print(f"Error: Model '{args.model}' not found")
        print("Available models:")
        for model in discover_ollama_models():
            print(f"  - {model.full_name}")
        sys.exit(1)
    
    print(f"📦 Model: {args.model}")
    print(f"📂 Path: {model_path}")
    
    # Run
    if args.server:
        asyncio.run(run_server_mode(
            model_path=model_path,
            rank=args.rank,
            world_size=args.world_size,
            node_addresses=node_addresses,
            api_port=args.api_port,
        ))
    else:
        asyncio.run(run_distributed_node(
            model_path=model_path,
            rank=args.rank,
            world_size=args.world_size,
            node_addresses=node_addresses,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            interactive=args.interactive,
        ))


if __name__ == "__main__":
    main()
