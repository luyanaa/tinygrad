"""
TCP fallback transport for tinygrad distributed training.

Provides TCP-based data transfer for cross-node communication when RDMA is not available.
"""

from __future__ import annotations
import socket
import struct
import threading
import queue
import ctypes
from typing import Optional, Tuple, List, Dict, Any
from tinygrad.helpers import getenv, DEBUG, GlobalCounters
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocatorBase, HCQBuffer, HWQueue, HCQSignal
from tinygrad.runtime.support.system import PCIIfaceBase, RemoteCmd
from tinygrad.runtime.support.memory import VirtMapping, AddrSpace
from tinygrad.device import Buffer, BufferSpec

CHUNK_SIZE = getenv("TCP_CHUNK_SIZE", 1024 * 1024)  # 1MB chunks
MAX_INFLIGHT = getenv("TCP_MAX_INFLIGHT", 4)

class TCPServer:
    """TCP server for receiving remote transfers."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.pending_transfers: queue.Queue = queue.Queue()

    def start(self):
        """Start the TCP server thread."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(16)
        self.running = True
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()
        if DEBUG >= 1:
            print(f"[TCPServer] Listening on {self.host}:{self.port}")

    def _server_loop(self):
        """Main server loop to handle incoming connections."""
        while self.running:
            try:
                client_sock, addr = self.sock.accept()
                if DEBUG >= 2:
                    print(f"[TCPServer] Connection from {addr}")
                threading.Thread(target=self._handle_client, args=(client_sock, addr), daemon=True).start()
            except Exception as e:
                if self.running:
                    if DEBUG >= 1:
                        print(f"[TCPServer] Accept error: {e}")

    def _handle_client(self, client_sock: socket.socket, addr: tuple):
        """Handle a client connection for data transfer."""
        try:
            client_sock.settimeout(30.0)
            while True:
                header = self._recv_exact(client_sock, 24)
                if not header:
                    break

                cmd, va_addr, size, remote_offset = struct.unpack('=IIQQ', header)

                if cmd == RemoteCmd.SYSMEM_WRITE:
                    data = self._recv_exact(client_sock, size)
                    if len(data) != size:
                        break

                    ctypes.memmove(va_addr + remote_offset, ctypes.addressof(ctypes.c_char.from_buffer(bytearray(data))), size)

                    client_sock.sendall(struct.pack('=I', 0))

                    if DEBUG >= 3:
                        print(f"[TCPServer] Wrote {size} bytes to {va_addr:#x}+{remote_offset}")
                else:
                    break
        except Exception as e:
            if DEBUG >= 1:
                print(f"[TCPServer] Client handler error: {e}")
        finally:
            client_sock.close()

    def _recv_exact(self, sock: socket.socket, size: int) -> bytes:
        """Receive exact number of bytes."""
        data = b''
        while len(data) < size:
            chunk = sock.recv(min(size - len(data), 65536))
            if not chunk:
                break
            data += chunk
        return data

    def stop(self):
        """Stop the server."""
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=2.0)

class TCPConnection:
    """TCP connection to a remote node."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.lock = threading.Lock()

    def connect(self) -> socket.socket:
        """Establish TCP connection."""
        with self.lock:
            if self.sock is None:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.settimeout(getenv("TCP_TIMEOUT", 30))
                self.sock.connect((self.host, self.port))
                self.sock.settimeout(None)

                buf_size = getenv("TCP_BUFFER_SIZE", 16 * 1024 * 1024)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buf_size)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buf_size)

                if DEBUG >= 1:
                    print(f"[TCPConnection] Connected to {self.host}:{self.port}")

        return self.sock

    def send_buffer(self, va_addr: int, remote_va_addr: int, size: int, remote_port: int):
        """Send buffer contents to remote node via TCP."""
        sock = self.connect()

        header = struct.pack('=IIQQ', RemoteCmd.SYSMEM_WRITE, remote_va_addr, size, 0)

        with self.lock:
            sock.sendall(header)

            ctypes.memmove_out = ctypes.memmove
            data = bytearray(size)
            ctypes.memmove(ctypes.addressof(data), va_addr, size)

            sent = 0
            while sent < size:
                chunk = data[sent:sent + CHUNK_SIZE]
                sock.sendall(chunk)
                sent += len(chunk)

            response = sock.recv(4)
            if DEBUG >= 3:
                print(f"[TCPConnection] Sent {size} bytes to {self.host}:{remote_port}")

    def close(self):
        """Close the connection."""
        with self.lock:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None

class TCPCopyQueue(HWQueue):
    """TCP-based copy queue for cross-node transfers without RDMA."""

    def __init__(self, dev: 'TCPDevice'):
        self.dev = dev
        super().__init__()
        self.chunk_size = CHUNK_SIZE
        self.max_inflight = MAX_INFLIGHT

    def copy(self, dest: HCQBuffer, src: HCQBuffer, sz: int) -> 'TCPCopyQueue':
        """Enqueue a TCP copy operation."""
        remote_dev = dest.owner
        if not isinstance(remote_dev, HCQCompiled):
            raise RuntimeError("Destination buffer must be owned by an HCQCompiled device")

        self._q.append((dest, src, sz, remote_dev))
        return self

    def _submit(self, dev: 'TCPDevice'):
        """Submit all enqueued TCP copy operations."""
        for dest, src, sz, remote_dev in self._q:
            self._perform_tcp_copy(dest, src, sz, remote_dev)
        self._q.clear()

    def _perform_tcp_copy(self, dest: HCQBuffer, src: HCQBuffer, sz: int, remote_dev: HCQCompiled):
        """Perform actual TCP copy operation."""
        if not hasattr(remote_dev, 'iface') or not hasattr(remote_dev.iface, 'peer_group'):
            raise RuntimeError(f"Remote device {remote_dev.device} missing proper interface for TCP transfer")

        remote_iface = remote_dev.iface
        peer_group = remote_iface.peer_group

        conn = self.dev.get_connection(peer_group)
        if conn is None:
            raise RuntimeError(f"No TCP connection to peer group {peer_group}")

        if hasattr(src, 'va_addr') and hasattr(dest, 'va_addr'):
            remote_va = dest.va_addr
            local_va = src.va_addr

            try:
                tcp_addr = self.dev.get_remote_tcp_addr(peer_group)
                if tcp_addr:
                    host, port = tcp_addr
                    conn.send_buffer(local_va, remote_va, sz, port)
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[TCPCopyQueue] TCP copy failed: {e}")
                raise

class TCPIface(PCIIfaceBase):
    """TCP interface for remote device communication."""

    def __init__(self, dev: 'TCPDevice', host: str, port: int, peer_group: str):
        self.dev = dev
        self.host = host
        self.port = port
        self.peer_group = peer_group
        self.sock: Optional[socket.socket] = None

    def is_local(self) -> bool:
        return False

class TCPAllocator(HCQAllocatorBase):
    """TCP-based allocator for remote memory management."""

    def __init__(self, dev: 'TCPDevice'):
        super().__init__(dev, batch_cnt=0)

    def _transfer(self, dest: HCQBuffer, src: HCQBuffer, sz: int,
                 src_dev: HCQCompiled, dest_dev: HCQCompiled):
        """Transfer data between devices using TCP."""
        tcq = TCPCopyQueue(self.dev)
        tcq.copy(dest, src, sz).submit(self.dev)

class TCPDevice(HCQCompiled):
    """TCP device for fallback cross-node communication."""

    _instances: Dict[str, 'TCPDevice'] = {}
    _connections: Dict[str, TCPConnection] = {}
    _tcp_addrs: Dict[str, Tuple[str, int]] = {}
    _server: Optional[TCPServer] = None
    _lock = threading.Lock()

    def __new__(cls, device: str):
        if device in cls._instances:
            return cls._instances[device]

        instance = super().__new__(cls)
        cls._instances[device] = instance
        return instance

    def __init__(self, device: str):
        if hasattr(self, '_initialized'):
            return

        parts = device.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid TCP device string: {device}")

        if parts[0] != "TCP":
            raise ValueError(f"TCP device must start with 'TCP:', got {device}")

        host = "localhost"
        port = 7000
        self_peer_group = "default"

        try:
            from tinygrad.distributed import get_world_info, get_all_capabilities
            world_info = get_world_info()
            if world_info["world_size"] > 1:
                all_caps = get_all_capabilities()
                rank = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else world_info["rank"]

                if rank < len(all_caps):
                    caps = all_caps[rank]
                    tcp_addr = caps.get("tcp_addr", "")
                    if tcp_addr:
                        tcp_parts = tcp_addr.rsplit(":", 1)
                        if len(tcp_parts) == 2:
                            host, port = tcp_parts[0], int(tcp_parts[1])
                    self_peer_group = caps.get("node", f"node_{rank}")
                    port = 8000 + rank
        except Exception:
            if len(parts) == 3:
                host, port_str = parts[1], parts[2]
                port = int(port_str)
            self_peer_group = parts[1] if len(parts) > 1 else "default"

        self.peer_group = self_peer_group
        self.host = host
        self.port = port
        self.iface = TCPIface(self, host, port, self_peer_group)
        self.allocator = TCPAllocator(self)
        self._initialized = True

        super().__init__(device, allocator=self.allocator)

        TCPDevice._tcp_addrs[self_peer_group] = (host, port)

        with TCPDevice._lock:
            if TCPDevice._server is None:
                TCPDevice._server = TCPServer(host, port)
                TCPDevice._server.start()

    @classmethod
    def get_connection(cls, peer_group: str) -> Optional[TCPConnection]:
        """Get or create TCP connection to a peer group."""
        if peer_group not in cls._connections:
            tcp_addr = cls._tcp_addrs.get(peer_group)
            if tcp_addr:
                host, port = tcp_addr
                cls._connections[peer_group] = TCPConnection(host, port)
            else:
                return None
        return cls._connections[peer_group]

    @classmethod
    def get_remote_tcp_addr(cls, peer_group: str) -> Optional[Tuple[str, int]]:
        """Get TCP address for a peer group."""
        return cls._tcp_addrs.get(peer_group)

    def rdma_dev(self):
        """TCP device doesn't have RDMA."""
        raise RuntimeError("TCP device has no RDMA capability")

    def finalize(self):
        """Clean up TCP resources."""
        super().finalize()
        with TCPDevice._lock:
            if TCPDevice._server and len(TCPDevice._instances) == 1:
                TCPDevice._server.stop()
                TCPDevice._server = None

def register_tcp_device():
    """Register TCP device with tinygrad's device system."""
    pass

__all__ = ["TCPCopyQueue", "TCPIface", "TCPAllocator", "TCPDevice", "register_tcp_device", "TCPServer", "TCPConnection"]