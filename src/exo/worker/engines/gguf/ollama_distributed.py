"""Ollama API-based distributed inference for EXO.

This module uses Ollama's HTTP API for inference instead of directly
loading GGUF files with llama-cpp-python. This provides broader model
support including newer architectures like nemotron_h_moe.

The distribution strategy changes to:
- Each node runs Ollama serving the same model
- EXO coordinates the inference by splitting prompts/context
- Results are aggregated back

This is a simpler approach that leverages Ollama's native support.
"""

import asyncio
import aiohttp
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional, Any

from loguru import logger


@dataclass
class OllamaNode:
    """Represents an Ollama node in the distributed cluster."""
    host: str
    port: int = 11434
    model_name: Optional[str] = None
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    async def is_available(self) -> bool:
        """Check if the Ollama server is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[str]:
        """List available models on this node."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [m["name"] for m in data.get("models", [])]
                return []
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Generate text using Ollama API."""
        model = model or self.model_name
        if not model:
            raise ValueError("Model name required")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Ollama API error: {error}")
                
                async for line in resp.content:
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):
                            break
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Chat completion using Ollama API."""
        model = model or self.model_name
        if not model:
            raise ValueError("Model name required")
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Ollama API error: {error}")
                
                async for line in resp.content:
                    if line:
                        import json
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                        if data.get("done"):
                            break


@dataclass
class DistributedOllamaCluster:
    """Manages a cluster of Ollama nodes for distributed inference.
    
    This implementation uses Ollama's native model support and coordinates
    inference across multiple nodes by:
    1. Load balancing requests across nodes
    2. Using model replication (each node has the full model)
    3. Optionally splitting long contexts across nodes
    """
    
    nodes: list[OllamaNode] = field(default_factory=list)
    model_name: str = ""
    current_node_idx: int = 0
    
    def add_node(self, host: str, port: int = 11434) -> None:
        """Add an Ollama node to the cluster."""
        self.nodes.append(OllamaNode(host=host, port=port, model_name=self.model_name))
    
    async def initialize(self) -> None:
        """Initialize the cluster and verify all nodes are available."""
        logger.info(f"Initializing Ollama cluster with {len(self.nodes)} nodes")
        
        for i, node in enumerate(self.nodes):
            node.model_name = self.model_name
            if await node.is_available():
                models = await node.list_models()
                logger.info(f"Node {i} ({node.host}:{node.port}): {len(models)} models available")
                if self.model_name in models or any(self.model_name in m for m in models):
                    logger.info(f"  ✓ Model '{self.model_name}' is available")
                else:
                    logger.warning(f"  ✗ Model '{self.model_name}' not found on node {i}")
            else:
                logger.warning(f"Node {i} ({node.host}:{node.port}): Not available")
    
    def _get_next_node(self) -> OllamaNode:
        """Get the next node in round-robin fashion."""
        node = self.nodes[self.current_node_idx]
        self.current_node_idx = (self.current_node_idx + 1) % len(self.nodes)
        return node
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate text using a node from the cluster."""
        node = self._get_next_node()
        logger.debug(f"Using node {node.host}:{node.port} for generation")
        
        async for token in node.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield token
    
    async def generate_parallel(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate responses for multiple prompts in parallel across nodes.
        
        Each prompt is sent to a different node for true parallel processing.
        """
        async def _generate_one(node: OllamaNode, prompt: str) -> str:
            tokens = []
            async for token in node.generate(prompt, max_tokens, temperature):
                tokens.append(token)
            return "".join(tokens)
        
        # Distribute prompts across nodes
        tasks = []
        for i, prompt in enumerate(prompts):
            node = self.nodes[i % len(self.nodes)]
            tasks.append(_generate_one(node, prompt))
        
        return await asyncio.gather(*tasks)


async def run_distributed_ollama(
    model_name: str,
    node_addresses: list[tuple[str, int]],
    prompt: str,
    max_tokens: int = 256,
    interactive: bool = False,
) -> None:
    """Run distributed Ollama inference.
    
    Args:
        model_name: Ollama model name (e.g., 'nemotron-3-nano:30b')
        node_addresses: List of (host, ollama_port) for each node
        prompt: Prompt for generation (if not interactive)
        max_tokens: Maximum tokens to generate
        interactive: If True, run interactive chat
    """
    # Create cluster
    cluster = DistributedOllamaCluster(model_name=model_name)
    
    for host, port in node_addresses:
        cluster.add_node(host, port)
    
    # Initialize
    await cluster.initialize()
    
    print(f"\n{'='*60}")
    print(f"🚀 Distributed Ollama Cluster")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Nodes: {len(cluster.nodes)}")
    for i, node in enumerate(cluster.nodes):
        print(f"  [{i}] {node.host}:{node.port}")
    print(f"{'='*60}\n")
    
    if interactive:
        print("🎯 Interactive mode - type 'quit' to exit\n")
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ('quit', 'exit', 'q'):
                    break
                if not user_input:
                    continue
                
                print("Assistant: ", end="", flush=True)
                async for token in cluster.generate(user_input, max_tokens):
                    print(token, end="", flush=True)
                print("\n")
                
            except EOFError:
                break
    else:
        print(f"📝 Prompt: {prompt}\n")
        print("💬 Response: ", end="", flush=True)
        async for token in cluster.generate(prompt, max_tokens):
            print(token, end="", flush=True)
        print("\n")
    
    print("👋 Done!")


# CLI entry point
def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Distributed Ollama inference")
    parser.add_argument("--model", "-m", required=True, help="Model name")
    parser.add_argument("--nodes", "-n", required=True, help="Comma-separated host:port list")
    parser.add_argument("--prompt", "-p", help="Prompt for generation")
    parser.add_argument("--max-tokens", "-t", type=int, default=256, help="Max tokens")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Parse nodes
    node_addresses = []
    for node in args.nodes.split(","):
        parts = node.strip().split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 11434
        node_addresses.append((host, port))
    
    asyncio.run(run_distributed_ollama(
        model_name=args.model,
        node_addresses=node_addresses,
        prompt=args.prompt or "Hello!",
        max_tokens=args.max_tokens,
        interactive=args.interactive,
    ))


if __name__ == "__main__":
    main()
