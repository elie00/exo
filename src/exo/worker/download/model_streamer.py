"""
Model Streamer - P2P model transfer between cluster nodes.

This module provides functionality to stream model files directly from
other nodes in the EXO cluster, avoiding redundant HuggingFace downloads
when a model is already available on another node.
"""

import os
import aiohttp
import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator
from loguru import logger

from exo.shared.types.models import ModelId
from exo.shared.types.common import NodeId


class ModelStreamer:
    """
    Stream models from other nodes in the EXO cluster.
    
    This allows P2P model transfer to avoid redundant downloads from
    HuggingFace when another node already has the model cached.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the model streamer.
        
        Args:
            cache_dir: Directory to cache downloaded models. Defaults to HF cache.
        """
        self.cache_dir = cache_dir or Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def find_node_with_model(
        self,
        model_id: ModelId,
        node_addresses: dict[NodeId, str],
        api_port: int = 52415,
    ) -> Optional[tuple[NodeId, str]]:
        """
        Find a node in the cluster that has the specified model.
        
        Args:
            model_id: The model ID to search for
            node_addresses: Dict mapping NodeId to IP address
            api_port: The API port to query nodes on
            
        Returns:
            Tuple of (NodeId, IP address) if found, None otherwise
        """
        for node_id, ip in node_addresses.items():
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"http://{ip}:{api_port}/models/has/{model_id}"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("has_model", False):
                                logger.info(f"Found model {model_id} on node {node_id} ({ip})")
                                return (node_id, ip)
            except Exception as e:
                logger.debug(f"Could not query node {node_id} ({ip}): {e}")
                continue
        
        return None
    
    async def stream_model_from_node(
        self,
        model_id: ModelId,
        source_ip: str,
        api_port: int = 52415,
        chunk_size: int = 1024 * 1024,  # 1MB chunks
    ) -> AsyncIterator[tuple[str, bytes, int, int]]:
        """
        Stream model files from another node.
        
        Args:
            model_id: The model to download
            source_ip: IP of the source node
            api_port: API port of the source node
            chunk_size: Size of chunks to receive
            
        Yields:
            Tuple of (filename, data_chunk, bytes_received, total_bytes)
        """
        url = f"http://{source_ip}:{api_port}/models/stream/{model_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=3600)  # 1 hour timeout
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to stream model: HTTP {resp.status}")
                        return
                    
                    total_bytes = int(resp.headers.get("Content-Length", 0))
                    current_file = resp.headers.get("X-Model-Filename", "model.bin")
                    bytes_received = 0
                    
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        bytes_received += len(chunk)
                        yield (current_file, chunk, bytes_received, total_bytes)
                        
        except asyncio.TimeoutError:
            logger.error(f"Timeout streaming model from {source_ip}")
        except Exception as e:
            logger.error(f"Error streaming model: {e}")
    
    async def download_model_from_cluster(
        self,
        model_id: ModelId,
        node_addresses: dict[NodeId, str],
        api_port: int = 52415,
    ) -> Optional[Path]:
        """
        Download a model from another node in the cluster if available.
        
        Args:
            model_id: The model to download
            node_addresses: Dict of available nodes
            api_port: API port to use
            
        Returns:
            Path to the downloaded model, or None if not found
        """
        source = await self.find_node_with_model(model_id, node_addresses, api_port)
        if not source:
            logger.info(f"Model {model_id} not found on any cluster node")
            return None
        
        node_id, source_ip = source
        logger.info(f"Downloading model {model_id} from node {node_id}")
        
        # Create model directory
        model_dir = self.cache_dir / str(model_id).replace("/", "--")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        current_file: Optional[Path] = None
        current_handle = None
        
        try:
            async for filename, chunk, received, total in self.stream_model_from_node(
                model_id, source_ip, api_port
            ):
                # Handle file changes (multi-file models)
                file_path = model_dir / filename
                if current_file != file_path:
                    if current_handle:
                        current_handle.close()
                    current_file = file_path
                    current_handle = open(file_path, "wb")
                    logger.info(f"Streaming file: {filename}")
                
                if current_handle:
                    current_handle.write(chunk)
                
                # Log progress periodically
                if total > 0:
                    progress = (received / total) * 100
                    if received % (10 * 1024 * 1024) < len(chunk):  # Every ~10MB
                        logger.info(f"Download progress: {progress:.1f}%")
            
            if current_handle:
                current_handle.close()
            
            logger.info(f"Model {model_id} downloaded successfully to {model_dir}")
            return model_dir
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            if current_handle:
                current_handle.close()
            return None


# Singleton instance
_model_streamer: Optional[ModelStreamer] = None


def get_model_streamer() -> ModelStreamer:
    """Get or create the global ModelStreamer instance."""
    global _model_streamer
    if _model_streamer is None:
        _model_streamer = ModelStreamer()
    return _model_streamer
