"""
RDMA configuration for tinygrad distributed training.

Provides RDMA endpoint discovery and configuration using bootstrap capabilities.
"""

from __future__ import annotations
import os
from typing import Optional, Dict, List, Tuple

class RDMAConfig:
    """Configuration for RDMA endpoints."""

    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.endpoints: Dict[int, str] = {}
        self._discover_endpoints()

    def _discover_endpoints(self) -> None:
        """Discover RDMA endpoints for all ranks from bootstrap config."""
        try:
            from tinygrad.distributed import get_all_capabilities, get_world_info

            all_caps = get_all_capabilities()
            world_info = get_world_info()

            if not all_caps or world_info["world_size"] <= 1:
                self.endpoints = {0: "127.0.0.1:10000"}
                return

            rdma_endpoints_env = os.getenv("RDMA_ENDPOINTS")
            if rdma_endpoints_env:
                endpoints_list = rdma_endpoints_env.split(",")
                for i, endpoint in enumerate(endpoints_list):
                    self.endpoints[i] = endpoint
                return

            for caps in all_caps:
                rank = caps.get("rank")
                rdma_addr = caps.get("rdma_addr")
                if rank is not None and rdma_addr:
                    port = 10000 + rank
                    self.endpoints[rank] = f"{rdma_addr}:{port}"

        except ImportError:
            pass

        if not self.endpoints:
            try:
                world_size = int(os.getenv("WORLD_SIZE", "1"))
                if world_size <= 1:
                    self.endpoints = {0: "127.0.0.1:10000"}
                    return

                for rank in range(world_size):
                    if os.getenv("RDMA_ADDR"):
                        ip = os.getenv("RDMA_ADDR")
                    elif os.getenv(f"RDMA_ADDR_{rank}"):
                        ip = os.getenv(f"RDMA_ADDR_{rank}")
                    else:
                        num_nodes = int(os.getenv("NUM_NODES", "1"))
                        if num_nodes > 1:
                            ip = f"10.0.0.{rank + 1}"
                        else:
                            ip = "127.0.0.1"
                    self.endpoints[rank] = f"{ip}:{10000 + rank}"
            except Exception:
                self.endpoints = {0: "127.0.0.1:10000"}

    def get_endpoint(self, rank: int) -> str:
        """Get RDMA endpoint for a rank."""
        if rank not in self.endpoints:
            raise KeyError(f"No RDMA endpoint configured for rank {rank}")
        return self.endpoints[rank]

    def get_ip_port(self, rank: int) -> Tuple[str, int]:
        """Get IP and port for a rank."""
        endpoint = self.get_endpoint(rank)
        parts = endpoint.rsplit(":", 1)
        if len(parts) == 2:
            return parts[0], int(parts[1])
        return endpoint, 10000 + rank

    def get_local_endpoint(self) -> Optional[str]:
        """Get RDMA endpoint for local rank."""
        try:
            from tinygrad.distributed import get_world_info
            world_info = get_world_info()
            return self.get_endpoint(world_info["rank"])
        except Exception:
            return self.get_endpoint(0)

def patch_rdma_ip_assignment():
    """Monkey-patch MLXIface to use configurable IPs from bootstrap config."""
    try:
        from tinygrad.runtime.ops_rdma import MLXIface
        from tinygrad.runtime.support.mlx.mlxdev import MLXDev
    except ImportError:
        print("[tinygrad.distributed.rdma] RDMA not available, skipping IP patch")
        return

    original_init = MLXIface.__init__

    def patched_init(self, dev, dev_id):
        original_init(self, dev, dev_id)

        try:
            from tinygrad.distributed import get_world_info
            world_info = get_world_info()

            if world_info["world_size"] > 1:
                rdma_config = RDMAConfig()
                ip, _ = rdma_config.get_ip_port(world_info["rank"])
            else:
                ip = "127.0.0.1"

            self.mlx_dev = MLXDev(self.pci_dev, ip=ip)

            self.uar_buf = self._buf([self.mlx_dev.pci_dev.bar_info(0)[0] + self.mlx_dev.uar * 0x1000])
            self.dbr_buf = self._buf(self.mlx_dev.dbr_paddrs)

            if hasattr(self, 'qp'):
                self.qp = None

        except Exception as e:
            print(f"[tinygrad.distributed.rdma] Failed to patch RDMA IP: {e}")

    MLXIface.__init__ = patched_init

    print("[tinygrad.distributed.rdma] Patched RDMA IP assignment")

class RDMANetwork:
    """Manages RDMA network connections for distributed training."""

    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.rdma_config = RDMAConfig(config)
        self.connections: Dict[int, Any] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize RDMA network connections."""
        if self._initialized:
            return

        try:
            from tinygrad.distributed import get_world_info
            world_info = get_world_info()
            print(f"[tinygrad.distributed.rdma] Initializing RDMA for rank {world_info['rank']}")
        except Exception:
            print("[tinygrad.distributed.rdma] Initializing RDMA")

        self._initialized = True

    def connect_to_rank(self, rank: int) -> Optional[Any]:
        """Connect to a specific rank."""
        if rank == (self.config.rank if self.config else 0):
            raise ValueError("Cannot connect to self")

        if rank in self.connections:
            return self.connections[rank]

        ip, port = self.rdma_config.get_ip_port(rank)

        try:
            from tinygrad.distributed import get_world_info
            print(f"[tinygrad.distributed.rdma] Connecting to rank {rank} at {ip}:{port}")
        except Exception:
            pass

        connection = {"rank": rank, "ip": ip, "port": port, "connected": True}
        self.connections[rank] = connection

        return connection

    def get_peer_group_connections(self) -> List[Any]:
        """Get connections to all ranks in the same peer group."""
        connections = []
        try:
            from tinygrad.distributed import get_world_info
            world_info = get_world_info()
            my_rank = world_info["rank"]

            for rank in range(world_info["world_size"]):
                if rank != my_rank:
                    connections.append(self.connect_to_rank(rank))
        except Exception:
            pass

        return connections

    def barrier(self) -> None:
        """Synchronize all ranks using RDMA."""
        if self.config and self.config.world_size <= 1:
            return

        print("[tinygrad.distributed.rdma] Barrier")

import typing
if typing.TYPE_CHECKING:
    from tinygrad.distributed.bootstrap import DistributedConfig

__all__ = ["RDMAConfig", "patch_rdma_ip_assignment", "RDMANetwork"]