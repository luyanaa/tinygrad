"""
Test ZeRO optimizer functionality.
"""
import unittest
import shutil
import numpy as np
from tinygrad import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import ZeroAdam, ZeroSGD, AdamW

_CLANG_AVAILABLE = shutil.which("clang") is not None

class TestZeRO(unittest.TestCase):
    def test_zero_adam_creation(self):
        """Test creating ZeroAdam."""
        model = Linear(32, 64)
        optimizer = ZeroAdam([model.weight, model.bias], lr=1e-3)
        self.assertIsNotNone(optimizer)

    def test_zero_sgd_creation(self):
        """Test creating ZeroSGD."""
        model = Linear(32, 64)
        optimizer = ZeroSGD([model.weight, model.bias], lr=1e-3)
        self.assertIsNotNone(optimizer)

    def test_zero_adam_zero_stage(self):
        """Test creating ZeroAdam with explicit zero_stage."""
        model = Linear(32, 64)
        optimizer = ZeroAdam([model.weight, model.bias], lr=1e-3, zero_stage=1)
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.zero_stage, 1)

    @unittest.skipUnless(_CLANG_AVAILABLE, "clang compiler not available")
    def test_zero_step(self):
        """Verify ZeroAdam reduces loss and updates parameters functionally."""
        Tensor.training = True

        model = Linear(2, 1)
        optimizer = ZeroAdam([model.weight, model.bias], lr=0.1, weight_decay=0)

        w_before = model.weight.numpy().copy()
        b_before = model.bias.numpy().copy()

        losses = []
        for _ in range(20):
            x = Tensor.rand(8, 2).realize()
            y = (x.sum(axis=1, keepdim=True) * 2).realize()

            out = model(x)
            loss = (out - y).square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.numpy().item())

        w_after = model.weight.numpy().copy()
        b_after = model.bias.numpy().copy()

        self.assertLess(losses[-1], losses[0] * 0.5, "loss should decrease by at least 50%")
        self.assertFalse(np.allclose(w_before, w_after), "weight should change after training")
        self.assertFalse(np.allclose(b_before, b_after), "bias should change after training")

        Tensor.training = False

    @unittest.skipUnless(_CLANG_AVAILABLE, "clang compiler not available")
    def test_zero_identical_to_adam(self):
        """ZeroAdam without zero_devices should be identical to Adam."""
        Tensor.training = True

        m1, m2 = Linear(2, 1), Linear(2, 1)
        m2.weight.assign(m1.weight.detach()).realize()
        m2.bias.assign(m1.bias.detach()).realize()

        opt_za = ZeroAdam([m1.weight, m1.bias], lr=0.1, weight_decay=0)
        opt_a = AdamW([m2.weight, m2.bias], lr=0.1, weight_decay=0)

        for _ in range(5):
            x = Tensor.rand(8, 2).realize()
            y = (x.sum(axis=1, keepdim=True) * 2).realize()

            for model, opt in [(m1, opt_za), (m2, opt_a)]:
                out = model(x)
                loss = (out - y).square().mean()
                loss.backward()
                opt.step()
                opt.zero_grad()

        self.assertTrue(np.allclose(m1.weight.numpy(), m2.weight.numpy(), atol=1e-5),
                        "ZeroAdam and Adam should produce identical weights")
        self.assertTrue(np.allclose(m1.bias.numpy(), m2.bias.numpy(), atol=1e-5),
                        "ZeroAdam and Adam should produce identical biases")

        Tensor.training = False


if __name__ == "__main__":
    unittest.main()