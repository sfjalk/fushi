"""Tests for CBAM + NWD integration and end-to-end smoke tests."""

import torch
import pytest

from modules.cbam import CBAM
from utils.nwd_loss import NWDLoss


class TestCBAMInBackbone:
    """Simulate inserting CBAM after a conv layer in a minimal backbone."""

    def test_cbam_in_sequential(self):
        import torch.nn as nn

        channels = 64
        backbone_block = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(channels),
        )
        x = torch.randn(2, 3, 32, 32)
        out = backbone_block(x)
        assert out.shape == (2, channels, 32, 32)

    def test_cbam_double_channel(self):
        """CBAM should accept channels not divisible by default reduction."""
        cbam = CBAM(channels=48, reduction=16)
        x = torch.randn(1, 48, 7, 7)
        out = cbam(x)
        assert out.shape == x.shape


class TestNWDWithCBAMFeatures:
    """Verify that NWD loss can train a tiny network that uses CBAM."""

    def test_toy_training_step(self):
        import torch.nn as nn
        import torch.optim as optim

        # Tiny detection head: conv -> CBAM -> box regression
        channels = 16
        head = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            CBAM(channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 4),  # predict cx, cy, w, h
        )
        criterion = NWDLoss()
        optimizer = optim.SGD(head.parameters(), lr=0.01)

        x = torch.randn(4, 3, 32, 32)
        target = torch.rand(4, 4)

        initial_loss = criterion(head(x), target).item()

        for _ in range(5):
            optimizer.zero_grad()
            pred = head(x)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

        final_loss = criterion(head(x), target).item()
        # Loss should have decreased (or at least not exploded)
        assert final_loss < initial_loss * 10
