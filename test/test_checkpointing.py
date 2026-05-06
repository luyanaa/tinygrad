"""
Test activation checkpointing functionality.
"""
import unittest
from tinygrad import Tensor
from tinygrad.checkpoint import checkpoint, checkpoint_sequential


class TestCheckpoint(unittest.TestCase):
    def test_checkpoint_basic(self):
        """Test basic checkpoint API."""
        x = Tensor.rand(4, 4)
        out = checkpoint(lambda a: a @ a, x)
        ref = x @ x
        # Checkpoint preserves output shape
        self.assertEqual(out.shape, ref.shape)
    
    def test_checkpoint_sequential(self):
        """Test checkpoint_sequential API."""
        w = Tensor.rand(4, 8)
        def matmul(x):
            return x @ w
        
        def identity(x):
            return x
        
        layers = [matmul, identity]
        
        x = Tensor.rand(4, 4)
        out = checkpoint_sequential(layers, x)
        self.assertEqual(out.shape, (4, 8))
    
    def test_checkpoint_gradient_flow(self):
        """Test that checkpoint preserves gradient flow."""
        Tensor.training = True
        x = Tensor.rand(4, 8, requires_grad=True)
        
        out = checkpoint(lambda a: a * 2.0 + 1.0, x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        Tensor.training = False
    
    def test_checkpoint_tuple_output(self):
        """Test checkpoint with function returning tuple."""
        x = Tensor.rand(4, 4)
        y = Tensor.rand(4, 4)
        
        def compute_two(a, b):
            return a @ b, a + b
        
        out_a, out_b = checkpoint(compute_two, x, y)
        self.assertEqual(out_a.shape, (4, 4))
        self.assertEqual(out_b.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
