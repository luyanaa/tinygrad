"""
Distributed bootstrap module for tinygrad.

Provides rank discovery and process coordination for multi-node training.
Supports MPI, Slurm, and environment variable-based bootstrapping.
"""

from __future__ import annotations
import os
import socket
import subprocess
from typing import Dict, List, Optional, Any

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None

GLOBAL_CONFIG: Optional[DistributedConfig] = None

def reset_global_config():
    """Reset the global config (for testing)."""
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = None

class DistributedConfig:
    """Configuration for distributed training."""

    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.local_size = 1
        self.node_rank = 0
        self.num_nodes = 1
        self.master_addr = "127.0.0.1"
        self.master_port = 29500
        self.gpus_per_node = 1
        self.node_name = socket.gethostname()
        self.capabilities: Dict[str, Any] = {}
        self.backend = "auto"
        self.all_caps: List[Dict[str, Any]] = []

    def __repr__(self) -> str:
        return (f"DistributedConfig(rank={self.rank}/{self.world_size}, "
                f"local_rank={self.local_rank}/{self.local_size}, "
                f"node={self.node_rank}/{self.num_nodes}, "
                f"node_name={self.node_name})")

def _detect_gpus() -> List[str]:
    """Detect available GPUs on this node."""
    gpus = []

    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        for platform in platforms:
            if "AMD" in platform.name or "Advanced Micro Devices" in platform.name:
                devices = platform.get_devices(cl.device_type.GPU)
                gpus.extend([f"AMD:{i}" for i in range(len(devices))])
    except ImportError:
        pass

    try:
        import pycuda.driver as cuda
        cuda.init()
        gpus.extend([f"NV:{i}" for i in range(cuda.Device.count())])
    except ImportError:
        pass

    try:
        import metal
        gpus.append("METAL:0")
    except ImportError:
        pass

    if not gpus:
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("ROCR_VISIBLE_DEVICES")
        if visible_devices:
            device_count = len(visible_devices.split(","))
            gpus = [f"UNKNOWN:{i}" for i in range(device_count)]
        else:
            gpus = ["CPU:0"]

    return gpus

def _discover_rdma_addr() -> Optional[str]:
    """Discover RDMA-capable network interface address."""
    rdma_addr = os.getenv("RDMA_ADDR")
    if rdma_addr:
        return rdma_addr

    try:
        import netifaces
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    addr = addr_info.get('addr')
                    if addr and (addr.startswith("10.0.0.") or addr.startswith("192.168.") or addr.startswith("172.")):
                        return f"{addr}"
    except ImportError:
        pass

    return None

def _parse_slurm_nodelist(nodelist: str) -> List[str]:
    """Parse Slurm nodelist format (supports basic patterns like node[1-3], node1,node2,node3)."""
    if not nodelist:
        return [socket.gethostname()]

    nodes = []
    parts = nodelist.split(',')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if '[' in part and ']' in part:
            prefix = part[:part.index('[')]
            range_str = part[part.index('[')+1:part.index(']')]
            suffix = part[part.index(']')+1:]

            for r in range_str.split(','):
                r = r.strip()
                if '-' in r:
                    start, end = r.split('-', 1)
                    for i in range(int(start), int(end) + 1):
                        nodes.append(f"{prefix}{i}{suffix}")
                else:
                    nodes.append(f"{prefix}{r}{suffix}")
        else:
            nodes.append(part)

    return nodes if nodes else [socket.gethostname()]

def _get_slurm_node_names() -> List[str]:
    """Get list of node names from Slurm."""
    try:
        nodelist = os.getenv("SLURM_NODELIST", "")
        if nodelist:
            return _parse_slurm_nodelist(nodelist)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["scontrol", "show", "hostname"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            nodes = result.stdout.strip().split('\n')
            if nodes:
                return nodes
    except Exception:
        pass

    return [socket.gethostname()]

def _bootstrap_mpi() -> Optional[DistributedConfig]:
    """Bootstrap using MPI4Py."""
    if not HAS_MPI:
        return None

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    node_name = MPI.Get_processor_name()

    all_node_names = comm.allgather(node_name)
    unique_nodes = sorted(set(all_node_names))
    node_rank = unique_nodes.index(node_name)
    num_nodes = len(unique_nodes)

    local_ranks = [i for i, name in enumerate(all_node_names) if name == node_name]
    local_rank = local_ranks.index(rank)
    local_size = len(local_ranks)

    config = DistributedConfig()
    config.rank = rank
    config.world_size = world_size
    config.local_rank = local_rank
    config.local_size = local_size
    config.node_rank = node_rank
    config.num_nodes = num_nodes
    config.node_name = node_name
    config.backend = "mpi"

    if rank == 0:
        config.master_addr = socket.gethostbyname(node_name)
    config.master_addr = comm.bcast(config.master_addr, root=0)

    return config

def _bootstrap_slurm() -> Optional[DistributedConfig]:
    """Bootstrap using Slurm environment variables."""
    if not all(os.getenv(var) for var in ["SLURM_PROCID", "SLURM_NTASKS"]):
        return None

    rank = int(os.getenv("SLURM_PROCID", "0"))
    world_size = int(os.getenv("SLURM_NTASKS", "1"))
    node_name = socket.gethostname()

    node_names = _get_slurm_node_names()
    unique_nodes = sorted(set(node_names))
    num_nodes = len(unique_nodes)
    node_rank = unique_nodes.index(node_name) if node_name in unique_nodes else 0

    local_rank = int(os.getenv("SLURM_LOCALID", "0"))
    gpus_on_node_env = os.getenv("SLURM_GPUS_ON_NODE")
    local_size = int(gpus_on_node_env if gpus_on_node_env else os.getenv("SLURM_JOB_GPUS", "1").count(',') + 1 if os.getenv("SLURM_JOB_GPUS") else 1)

    config = DistributedConfig()
    config.rank = rank
    config.world_size = world_size
    config.local_rank = local_rank
    config.local_size = local_size
    config.node_rank = node_rank
    config.num_nodes = num_nodes
    config.node_name = node_name
    config.backend = "slurm"

    if rank == 0:
        config.master_addr = socket.gethostbyname(node_name)

    return config

def _bootstrap_env() -> Optional[DistributedConfig]:
    """Bootstrap using environment variables."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    local_size = int(os.getenv("LOCAL_SIZE", "1"))
    node_rank = int(os.getenv("NODE_RANK", "0"))
    num_nodes = int(os.getenv("NUM_NODES", "1"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29500"))

    if world_size <= 1 and num_nodes <= 1:
        return None

    config = DistributedConfig()
    config.rank = rank
    config.world_size = world_size
    config.local_rank = local_rank
    config.local_size = local_size
    config.node_rank = node_rank
    config.num_nodes = num_nodes
    config.node_name = socket.gethostname()
    config.master_addr = master_addr
    config.master_port = master_port
    config.backend = "env"

    return config

def _exchange_capabilities(config: DistributedConfig) -> List[Dict[str, Any]]:
    """Exchange capabilities across all ranks using the bootstrap backend."""
    gpus = _detect_gpus()
    config.gpus_per_node = len(gpus)

    rdma_addr = _discover_rdma_addr()
    tcp_addr = f"{socket.gethostbyname(config.node_name)}:{20000 + config.rank}"

    my_caps = {
        "rank": config.rank,
        "node": config.node_name,
        "node_rank": config.node_rank,
        "gpus": gpus,
        "rdma_addr": rdma_addr,
        "tcp_addr": tcp_addr,
        "backend": config.backend,
    }
    config.capabilities = my_caps

    if config.world_size <= 1:
        config.all_caps = [my_caps]
        return [my_caps]

    if config.backend == "mpi" and HAS_MPI:
        comm = MPI.COMM_WORLD
        config.all_caps = list(comm.allgather(my_caps))
    elif config.backend == "slurm":
        local_node_ranks = [i for i, caps in enumerate(config.all_caps) if caps["node"] == config.node_name] if config.all_caps else list(range(config.world_size))
        config.all_caps = [my_caps] * config.world_size
    else:
        config.all_caps = [my_caps]

    return config.all_caps

def init_distributed(backend: str = "auto") -> Optional[DistributedConfig]:
    """
    Initialize distributed training.

    Args:
        backend: Bootstrap backend to use ("auto", "mpi", "slurm", "env")

    Returns:
        DistributedConfig if distributed training is enabled, None otherwise.
    """
    global GLOBAL_CONFIG

    if GLOBAL_CONFIG is not None:
        return GLOBAL_CONFIG

    config = None

    backends_to_try = []
    if backend == "auto":
        backends_to_try = ["mpi", "slurm", "env"]
    else:
        backends_to_try = [backend]

    for backend_name in backends_to_try:
        if backend_name == "mpi":
            config = _bootstrap_mpi()
        elif backend_name == "slurm":
            config = _bootstrap_slurm()
        elif backend_name == "env":
            config = _bootstrap_env()

        if config is not None:
            break

    if config is None:
        return None

    _exchange_capabilities(config)

    os.environ["RANK"] = str(config.rank)
    os.environ["WORLD_SIZE"] = str(config.world_size)
    os.environ["LOCAL_RANK"] = str(config.local_rank)
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)

    print(f"[tinygrad.distributed] Initialized rank {config.rank}/{config.world_size} "
          f"(local {config.local_rank}/{config.local_size}) on node {config.node_name} "
          f"using {config.backend} backend")

    GLOBAL_CONFIG = config
    return config

def get_world_info() -> Dict[str, Any]:
    """
    Get world information for debugging.

    Returns:
        Dictionary with world information.
    """
    global GLOBAL_CONFIG

    if GLOBAL_CONFIG is not None:
        return {
            "rank": GLOBAL_CONFIG.rank,
            "world_size": GLOBAL_CONFIG.world_size,
            "local_rank": GLOBAL_CONFIG.local_rank,
            "local_size": GLOBAL_CONFIG.local_size,
            "node_rank": GLOBAL_CONFIG.node_rank,
            "num_nodes": GLOBAL_CONFIG.num_nodes,
            "node_name": GLOBAL_CONFIG.node_name,
            "master_addr": GLOBAL_CONFIG.master_addr,
            "master_port": GLOBAL_CONFIG.master_port,
            "backend": GLOBAL_CONFIG.backend,
            "gpus_per_node": GLOBAL_CONFIG.gpus_per_node,
            "gpus": GLOBAL_CONFIG.capabilities.get("gpus", []),
        }

    config = DistributedConfig()
    gpus = _detect_gpus()
    return {
        "rank": config.rank,
        "world_size": config.world_size,
        "local_rank": config.local_rank,
        "local_size": config.local_size,
        "node_rank": config.node_rank,
        "num_nodes": config.num_nodes,
        "node_name": config.node_name,
        "master_addr": config.master_addr,
        "master_port": config.master_port,
        "backend": config.backend,
        "gpus_per_node": len(gpus),
        "gpus": gpus,
    }

def get_all_capabilities() -> List[Dict[str, Any]]:
    """Get all capabilities from all ranks."""
    global GLOBAL_CONFIG
    if GLOBAL_CONFIG is not None and GLOBAL_CONFIG.all_caps:
        return GLOBAL_CONFIG.all_caps
    return [get_world_info()]

__all__ = ["init_distributed", "DistributedConfig", "get_world_info", "get_all_capabilities", "reset_global_config"]