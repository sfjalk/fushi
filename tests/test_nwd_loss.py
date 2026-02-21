"""Unit tests for the NWD (Normalized Wasserstein Distance) loss."""

import math

import torch
import pytest

from utils.nwd_loss import NWDLoss, nwd_loss, wasserstein2_distance_sq


class TestWasserstein2DistanceSq:
    def test_identical_boxes_give_zero(self):
        boxes = torch.tensor([[10.0, 20.0, 4.0, 6.0]])
        dist = wasserstein2_distance_sq(boxes, boxes)
        assert torch.allclose(dist, torch.zeros(1))

    def test_known_value(self):
        # pred center=(0,0), size=(2,2); target center=(3,4), size=(4,4)
        # W2^2 = (0-3)^2 + (0-4)^2 + (1-2)^2 + (1-2)^2
        #       = 9 + 16 + 1 + 1 = 27
        pred   = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        target = torch.tensor([[3.0, 4.0, 4.0, 4.0]])
        dist = wasserstein2_distance_sq(pred, target)
        assert torch.allclose(dist, torch.tensor([27.0]))

    def test_symmetric(self):
        a = torch.rand(8, 4)
        b = torch.rand(8, 4)
        assert torch.allclose(
            wasserstein2_distance_sq(a, b),
            wasserstein2_distance_sq(b, a),
        )

    def test_non_negative(self):
        a = torch.rand(16, 4)
        b = torch.rand(16, 4)
        assert (wasserstein2_distance_sq(a, b) >= 0).all()


class TestNwdLoss:
    def test_identical_boxes_zero_loss(self):
        boxes = torch.rand(4, 4)
        loss = nwd_loss(boxes, boxes)
        assert torch.allclose(loss, torch.zeros(1), atol=1e-6)

    def test_loss_in_range(self):
        pred   = torch.rand(32, 4)
        target = torch.rand(32, 4)
        loss = nwd_loss(pred, target)
        assert 0.0 <= loss.item() <= 1.0

    def test_reduction_none_shape(self):
        pred   = torch.rand(10, 4)
        target = torch.rand(10, 4)
        loss = nwd_loss(pred, target, reduction="none")
        assert loss.shape == (10,)

    def test_reduction_sum(self):
        pred   = torch.rand(5, 4)
        target = torch.rand(5, 4)
        loss_none = nwd_loss(pred, target, reduction="none")
        loss_sum  = nwd_loss(pred, target, reduction="sum")
        assert torch.allclose(loss_none.sum(), loss_sum)

    def test_reduction_mean(self):
        pred   = torch.rand(6, 4)
        target = torch.rand(6, 4)
        loss_none = nwd_loss(pred, target, reduction="none")
        loss_mean = nwd_loss(pred, target, reduction="mean")
        assert torch.allclose(loss_none.mean(), loss_mean)

    def test_gradients_flow(self):
        pred   = torch.rand(4, 4, requires_grad=True)
        target = torch.rand(4, 4)
        loss = nwd_loss(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_constant_sensitivity(self):
        """Larger constant → smoother loss surface (lower absolute loss value)."""
        pred   = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        target = torch.tensor([[3.0, 4.0, 4.0, 4.0]])
        loss_small = nwd_loss(pred, target, constant=1.0)
        loss_large = nwd_loss(pred, target, constant=100.0)
        assert loss_small > loss_large


class TestNWDLossModule:
    def test_forward_matches_function(self):
        pred   = torch.rand(8, 4)
        target = torch.rand(8, 4)
        criterion = NWDLoss(constant=12.8)
        assert torch.allclose(criterion(pred, target), nwd_loss(pred, target, 12.8))

    def test_is_nn_module(self):
        import torch.nn as nn
        assert isinstance(NWDLoss(), nn.Module)
