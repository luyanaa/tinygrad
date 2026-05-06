"""
Test DeviceMesh functionality.
"""
import unittest
from tinygrad.distributed import DeviceMesh, create_data_parallel_mesh, create_tensor_parallel_mesh
from tinygrad.tensor import Tensor


class TestDeviceMesh(unittest.TestCase):
    def test_create_default_mesh(self):
        """Test creating a default DeviceMesh."""
        mesh = DeviceMesh()
        self.assertIsNotNone(mesh)
        self.assertIsInstance(mesh.devices, list)
        self.assertGreaterEqual(len(mesh.devices), 1)
    
    def test_get_device_coordinates(self):
        """Test converting devices to coordinates and back."""
        mesh = DeviceMesh(shape=(2, 2), devices=["d0", "d1", "d2", "d3"])
        self.assertEqual(mesh.get_device(0, 0), "d0")
        self.assertEqual(mesh.get_device(1, 1), "d3")
        self.assertEqual(mesh.get_coordinates("d2"), (1, 0))
    
    def test_submesh(self):
        """Test creating a submesh."""
        mesh = DeviceMesh(shape=(2, 2), devices=["d0", "d1", "d2", "d3"])
        submesh = mesh.submesh(dim=0, index=0)
        self.assertEqual(len(submesh.devices), 2)
        self.assertEqual(submesh.devices[0], "d0")
    
    def test_all_submeshes(self):
        """Test getting all submeshes along a dimension."""
        mesh = DeviceMesh(shape=(2, 2), devices=["d0", "d1", "d2", "d3"])
        submeshes = mesh.all_submeshes(dim=0)
        self.assertEqual(len(submeshes), 2)
        self.assertEqual(len(submeshes[0].devices), 2)
    
    def test_peer_group(self):
        """Test getting peer group for devices."""
        mesh = DeviceMesh()
        peer_group = mesh.peer_group("CPU:0")
        self.assertEqual(peer_group, "localhost")
    
    def test_same_peer_group(self):
        """Test checking if two devices are in the same peer group."""
        mesh = DeviceMesh()
        self.assertTrue(mesh.is_same_peer_group("CPU:0", "CPU:0"))
    
    def test_create_data_parallel_mesh(self):
        """Test create_data_parallel_mesh (stub)."""
        mesh = create_data_parallel_mesh()
        self.assertIsNotNone(mesh)
    
    def test_create_tensor_parallel_mesh(self):
        """Test create_tensor_parallel_mesh (stub)."""
        mesh = create_tensor_parallel_mesh()
        self.assertIsNotNone(mesh)


if __name__ == "__main__":
    unittest.main()
