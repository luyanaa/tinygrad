"""
Test tensor parallelism functionality.
"""
import unittest
import math
from tinygrad import Tensor
from tinygrad.nn.parallel import TensorParallelLinear, TensorParallelColumnLinear, TensorParallelRowLinear, TensorParallelMLP


class TestTensorParallel(unittest.TestCase):
    def test_tensor_parallel_linear_creation(self):
        """Test creating TensorParallelLinear."""
        tp = TensorParallelLinear(32, 64)
        self.assertEqual(tp.in_features, 32)
        self.assertEqual(tp.out_features, 64)
    
    def test_column_parallel_linear(self):
        """Test creating TensorParallelColumnLinear."""
        tp_col = TensorParallelColumnLinear(32, 64)
        self.assertEqual(tp_col.parallel_dim, 0)
    
    def test_row_parallel_linear(self):
        """Test creating TensorParallelRowLinear."""
        tp_row = TensorParallelRowLinear(32, 64)
        self.assertEqual(tp_row.parallel_dim, 1)
    
    def test_tensor_parallel_mlp_creation(self):
        """Test creating TensorParallelMLP."""
        mlp = TensorParallelMLP(64, 256)
        self.assertEqual(mlp.dim, 64)
        self.assertEqual(mlp.hidden_dim, 256)
    
    def test_linear_forward(self):
        """Test forward pass of TensorParallelLinear."""
        tp = TensorParallelLinear(32, 64)
        x = Tensor.rand(2, 32)
        out = tp(x)
        self.assertEqual(out.shape, (2, 64))
    
    def test_column_linear_forward(self):
        """Test forward pass of TensorParallelColumnLinear."""
        tp_col = TensorParallelColumnLinear(32, 64)
        x = Tensor.rand(2, 32)
        out = tp_col(x)
        self.assertEqual(out.shape, (2, 64))
    
    def test_row_linear_forward(self):
        """Test forward pass of TensorParallelRowLinear."""
        tp_row = TensorParallelRowLinear(32, 64)
        x = Tensor.rand(2, 32)
        out = tp_row(x)
        self.assertEqual(out.shape, (2, 64))


if __name__ == "__main__":
    unittest.main()
