"""
Network Profiler - Measure latency and bandwidth between EXO cluster nodes.

This module provides functionality for network profiling, including:
- Latency measurement (ping-style)
- Bandwidth estimation
- Connection type detection (Thunderbolt, Ethernet, WiFi)
"""

import asyncio
import time
import platform
import subprocess
from typing import Optional
from dataclasses import dataclass
from enum import Enum

import aiohttp
from loguru import logger

from exo.shared.types.common import NodeId


class ConnectionType(str, Enum):
    """Type of network connection between nodes."""
    THUNDERBOLT = "thunderbolt"
    ETHERNET = "ethernet"
    WIFI = "wifi"
    TAILSCALE = "tailscale"
    UNKNOWN = "unknown"


@dataclass
class NetworkMetrics:
    """Network metrics between two nodes."""
    source_node: NodeId
    target_node: NodeId
    target_ip: str
    
    # Latency in milliseconds
    latency_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_stddev_ms: float
    
    # Bandwidth in MB/s (estimated)
    bandwidth_mbps: Optional[float] = None
    
    # Connection info
    connection_type: ConnectionType = ConnectionType.UNKNOWN
    interface_name: Optional[str] = None
    
    # Measurement timestamp
    measured_at: float = 0.0


class NetworkProfiler:
    """
    Network profiler for measuring inter-node connectivity.
    
    Provides latency and bandwidth measurements between cluster nodes.
    """
    
    def __init__(self, api_port: int = 52415):
        self.api_port = api_port
        self._cache: dict[tuple[NodeId, NodeId], NetworkMetrics] = {}
        self._cache_ttl = 60.0  # Cache results for 60 seconds
    
    async def measure_latency(
        self,
        source_node: NodeId,
        target_ip: str,
        target_node: NodeId,
        num_pings: int = 5,
    ) -> NetworkMetrics:
        """
        Measure network latency to a target node using HTTP ping.
        
        Args:
            source_node: ID of the source node
            target_ip: IP address of the target node
            target_node: ID of the target node
            num_pings: Number of ping requests to average
            
        Returns:
            NetworkMetrics with latency measurements
        """
        latencies: list[float] = []
        
        for _ in range(num_pings):
            start_time = time.perf_counter()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{target_ip}:{self.api_port}/node_id",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            end_time = time.perf_counter()
                            latency_ms = (end_time - start_time) * 1000
                            latencies.append(latency_ms)
            except Exception as e:
                logger.debug(f"Ping failed to {target_ip}: {e}")
                continue
            
            await asyncio.sleep(0.1)  # Small delay between pings
        
        if not latencies:
            return NetworkMetrics(
                source_node=source_node,
                target_node=target_node,
                target_ip=target_ip,
                latency_ms=float('inf'),
                latency_min_ms=float('inf'),
                latency_max_ms=float('inf'),
                latency_stddev_ms=0,
                measured_at=time.time(),
            )
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Standard deviation
        if len(latencies) > 1:
            variance = sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)
            stddev = variance ** 0.5
        else:
            stddev = 0.0
        
        # Detect connection type
        conn_type = self._detect_connection_type(target_ip, avg_latency)
        
        return NetworkMetrics(
            source_node=source_node,
            target_node=target_node,
            target_ip=target_ip,
            latency_ms=avg_latency,
            latency_min_ms=min_latency,
            latency_max_ms=max_latency,
            latency_stddev_ms=stddev,
            connection_type=conn_type,
            measured_at=time.time(),
        )
    
    async def estimate_bandwidth(
        self,
        target_ip: str,
        target_node: NodeId,
        source_node: NodeId,
        test_size_kb: int = 256,
    ) -> float:
        """
        Estimate bandwidth to a target node by transferring test data.
        
        Args:
            target_ip: IP address of the target
            target_node: ID of the target node
            source_node: ID of the source node
            test_size_kb: Size of test payload in KB
            
        Returns:
            Estimated bandwidth in MB/s
        """
        # Create test payload
        test_data = b"x" * (test_size_kb * 1024)
        
        try:
            start_time = time.perf_counter()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{target_ip}:{self.api_port}/network/bandwidth_test",
                    data=test_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        # Read response (echo back)
                        await resp.read()
                        end_time = time.perf_counter()
                        
                        # Calculate bandwidth (round trip)
                        total_bytes = test_size_kb * 1024 * 2  # Upload + download
                        duration_seconds = end_time - start_time
                        bandwidth_mbps = (total_bytes / duration_seconds) / (1024 * 1024)
                        
                        return bandwidth_mbps
        except Exception as e:
            logger.warning(f"Bandwidth test failed to {target_ip}: {e}")
        
        return 0.0
    
    def _detect_connection_type(self, target_ip: str, latency_ms: float) -> ConnectionType:
        """
        Detect the connection type based on IP and latency.
        
        Args:
            target_ip: Target IP address
            latency_ms: Measured latency in milliseconds
            
        Returns:
            Detected ConnectionType
        """
        # Tailscale IPs typically start with 100.
        if target_ip.startswith("100."):
            return ConnectionType.TAILSCALE
        
        # Thunderbolt typically has very low latency (<1ms)
        if latency_ms < 1.0:
            return ConnectionType.THUNDERBOLT
        
        # Local network (fast) - likely Ethernet
        if latency_ms < 5.0:
            return ConnectionType.ETHERNET
        
        # Higher latency suggests WiFi
        if latency_ms < 50.0:
            return ConnectionType.WIFI
        
        return ConnectionType.UNKNOWN
    
    def get_interface_for_ip(self, target_ip: str) -> Optional[str]:
        """
        Get the network interface name used to reach a target IP.
        
        Args:
            target_ip: The target IP address
            
        Returns:
            Interface name (e.g., 'en0', 'eth0') or None
        """
        try:
            if platform.system().lower() == "darwin":
                # macOS
                result = subprocess.run(
                    ["route", "-n", "get", target_ip],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                for line in result.stdout.split("\n"):
                    if "interface:" in line:
                        return line.split(":")[1].strip()
            elif platform.system().lower() == "linux":
                # Linux
                result = subprocess.run(
                    ["ip", "route", "get", target_ip],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                parts = result.stdout.split()
                if "dev" in parts:
                    idx = parts.index("dev")
                    if idx + 1 < len(parts):
                        return parts[idx + 1]
        except Exception as e:
            logger.debug(f"Could not determine interface for {target_ip}: {e}")
        
        return None
    
    async def profile_all_connections(
        self,
        source_node: NodeId,
        node_addresses: dict[NodeId, str],
    ) -> list[NetworkMetrics]:
        """
        Profile connections to all nodes in the cluster.
        
        Args:
            source_node: ID of the local node
            node_addresses: Dict of NodeId -> IP address
            
        Returns:
            List of NetworkMetrics for each connection
        """
        results: list[NetworkMetrics] = []
        
        tasks = []
        for target_node, target_ip in node_addresses.items():
            if target_node == source_node:
                continue
            
            # Check cache
            cache_key = (source_node, target_node)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                if time.time() - cached.measured_at < self._cache_ttl:
                    results.append(cached)
                    continue
            
            # Schedule measurement
            tasks.append(self.measure_latency(source_node, target_ip, target_node))
        
        # Run measurements concurrently
        if tasks:
            measured = await asyncio.gather(*tasks, return_exceptions=True)
            for result in measured:
                if isinstance(result, NetworkMetrics):
                    self._cache[(result.source_node, result.target_node)] = result
                    results.append(result)
        
        return results


# Singleton instance
_network_profiler: Optional[NetworkProfiler] = None


def get_network_profiler(api_port: int = 52415) -> NetworkProfiler:
    """Get or create the global NetworkProfiler instance."""
    global _network_profiler
    if _network_profiler is None:
        _network_profiler = NetworkProfiler(api_port)
    return _network_profiler
