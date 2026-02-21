"""
Normalized Wasserstein Distance (NWD) bounding-box loss for small objects.

Reference:
    Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object
    Detection", arXiv:2110.13389 (2021).

The NWD replaces or augments standard IoU-based localisation losses and
produces smoother gradients for boxes that do not overlap at all — a common
situation when ground-truth objects are very small (< 32 × 32 px).

Formula
-------
Given two axis-aligned boxes represented as Gaussians:

    μ  = (cx, cy),  Σ = diag(w/2, h/2) ** 2

The squared 2-Wasserstein distance is:

    W2² = ||μ1 - μ2||² + (√σ1w - √σ2w)² + (√σ1h - √σ2h)²
        = (cx1-cx2)² + (cy1-cy2)² + (w1/2 - w2/2)² + (h1/2 - h2/2)²

NWD normalises this by a constant C (default 1.0) so that the value stays in
[0, 1]:

    nwd = exp(-W2 / C)

Loss = 1 - nwd  (higher is worse, like 1-IoU)
"""

import torch
from torch import Tensor


def wasserstein2_distance_sq(
    pred: Tensor,
    target: Tensor,
) -> Tensor:
    """Squared 2-Wasserstein distance between two Gaussian distributions
    derived from axis-aligned bounding boxes.

    Args:
        pred:   Predicted boxes in ``(cx, cy, w, h)`` format, shape ``[N, 4]``.
        target: Ground-truth boxes in ``(cx, cy, w, h)`` format, shape ``[N, 4]``.

    Returns:
        Tensor of shape ``[N]`` with W2² values.
    """
    center_dist_sq = ((pred[:, :2] - target[:, :2]) ** 2).sum(dim=-1)
    size_dist_sq = ((pred[:, 2:] / 2 - target[:, 2:] / 2) ** 2).sum(dim=-1)
    return center_dist_sq + size_dist_sq


def nwd_loss(
    pred: Tensor,
    target: Tensor,
    constant: float = 12.8,
    reduction: str = "mean",
) -> Tensor:
    """NWD bounding-box regression loss.

    Args:
        pred:      Predicted boxes ``(cx, cy, w, h)``, shape ``[N, 4]``.
        target:    Ground-truth boxes ``(cx, cy, w, h)``, shape ``[N, 4]``.
        constant:  Normalisation constant ``C`` (tune to your image/box scale).
                   The paper suggests ``C ≈ 12.8`` for normalised coordinates.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.

    Returns:
        Scalar loss (or per-sample losses when ``reduction='none'``).
    """
    w2_sq = wasserstein2_distance_sq(pred, target)
    nwd = torch.exp(-torch.sqrt(w2_sq) / constant)
    loss = 1.0 - nwd
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class NWDLoss(torch.nn.Module):
    """Module wrapper around :func:`nwd_loss` for easy integration.

    Drop-in replacement for ``nn.SmoothL1Loss`` / ``CIoULoss`` in your
    bounding-box regression head.

    Example::

        criterion = NWDLoss(constant=12.8)
        loss = criterion(pred_boxes, target_boxes)
    """

    def __init__(self, constant: float = 12.8, reduction: str = "mean") -> None:
        super().__init__()
        self.constant = constant
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return nwd_loss(pred, target, self.constant, self.reduction)
