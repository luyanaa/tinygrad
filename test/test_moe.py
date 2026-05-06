"""
Test MoE (Mixture of Experts) functionality.
"""
import unittest
from tinygrad import Tensor
from tinygrad.nn.moe import MoELayer, AllToAll


class TestMoE(unittest.TestCase):
    def test_moe_layer_creation(self):
        """Test creating MoELayer."""
        moe = MoELayer(num_experts=8, dim=64, hidden_dim=256, activated_experts=2)
        self.assertEqual(moe.num_experts, 8)
        self.assertEqual(moe.dim, 64)
        self.assertEqual(moe.hidden_dim, 256)
        self.assertEqual(moe.activated_experts, 2)
    
    def test_moe_layer_forward(self):
        """Test forward pass of MoELayer."""
        moe = MoELayer(num_experts=4, dim=32, hidden_dim=128, activated_experts=2)
        x = Tensor.rand(2, 16, 32)  # (batch, seq_len, dim)
        out = moe(x)
        self.assertEqual(out.shape, (2, 16, 32))
    
    def test_moe_layer_forward_single_expert(self):
        """Test forward pass with single active expert."""
        moe = MoELayer(num_experts=2, dim=16, hidden_dim=64, activated_experts=1)
        x = Tensor.rand(4, 8, 16)
        out = moe(x)
        self.assertEqual(out.shape, (4, 8, 16))
    
    def test_all_to_all_creation(self):
        """Test creating AllToAll primitive (stub)."""
        all2all = AllToAll(["d0", "d1"])
        self.assertEqual(len(all2all.devices), 2)


if __name__ == "__main__":
    unittest.main()
