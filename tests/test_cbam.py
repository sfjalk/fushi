"""Unit tests for the CBAM attention module."""

import torch
import pytest

from modules.cbam import CBAM, ChannelAttention, SpatialAttention


class TestChannelAttention:
    def test_output_shape(self):
        ca = ChannelAttention(channels=64)
        x = torch.randn(2, 64, 16, 16)
        out = ca(x)
        # channel-attention output is a weight map broadcastable over spatial dims
        assert out.shape == (2, 64, 1, 1)

    def test_output_range(self):
        ca = ChannelAttention(channels=32)
        x = torch.randn(4, 32, 8, 8)
        out = ca(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_small_reduction(self):
        # reduction ratio larger than channels should not crash (clamped to 1)
        ca = ChannelAttention(channels=4, reduction=16)
        x = torch.randn(1, 4, 4, 4)
        assert ca(x).shape == (1, 4, 1, 1)


class TestSpatialAttention:
    @pytest.mark.parametrize("kernel_size", [3, 7])
    def test_output_shape(self, kernel_size):
        sa = SpatialAttention(kernel_size=kernel_size)
        x = torch.randn(2, 64, 16, 16)
        out = sa(x)
        assert out.shape == (2, 1, 16, 16)

    def test_output_range(self):
        sa = SpatialAttention()
        x = torch.randn(2, 128, 20, 20)
        out = sa(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_invalid_kernel_raises(self):
        with pytest.raises(AssertionError):
            SpatialAttention(kernel_size=5)


class TestCBAM:
    def test_output_shape_preserved(self):
        for c in [32, 64, 128, 256]:
            cbam = CBAM(channels=c)
            x = torch.randn(2, c, 14, 14)
            out = cbam(x)
            assert out.shape == x.shape, f"Shape mismatch for channels={c}"

    def test_gradients_flow(self):
        cbam = CBAM(channels=64)
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        loss = cbam(x).sum()
        loss.backward()
        assert x.grad is not None

    def test_batch_independence(self):
        """Each sample in a batch should produce the same result as when
        processed individually (no cross-batch leakage)."""
        cbam = CBAM(channels=16)
        cbam.eval()
        x = torch.randn(4, 16, 8, 8)
        batch_out = cbam(x)
        for i in range(4):
            single_out = cbam(x[i : i + 1])
            assert torch.allclose(batch_out[i : i + 1], single_out, atol=1e-6)
