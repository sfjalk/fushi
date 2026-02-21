"""
Training script for the modified YOLOv8 traffic-sign detector.

Modifications applied on top of the stock YOLOv8:
  1. P2 detection head  – extra small-object output at stride 4 (yolov8-p2.yaml)
  2. CBAM attention     – injected into the backbone after specified layers
  3. NWD loss           – replaces CIoU for bounding-box regression

Usage
-----
    python train.py \\
        --data   data/tt100k.yaml \\
        --model  configs/yolov8-p2.yaml \\
        --epochs 100 \\
        --imgsz  640 \\
        --batch  16 \\
        --device 0

All other ultralytics keyword arguments (e.g. --lr0, --workers) are forwarded
directly to YOLO.train().
"""

from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Patch ultralytics to register CBAM and swap in NWD loss
# ---------------------------------------------------------------------------


def _patch_ultralytics() -> None:
    """Register custom modules so ultralytics can find them in .yaml configs."""
    try:
        import ultralytics.nn.modules as M  # noqa: PLC0415
        from modules.cbam import CBAM  # noqa: PLC0415

        if not hasattr(M, "CBAM"):
            M.CBAM = CBAM
            # Also expose at the package root so parse_model() can find it
            import ultralytics.nn.tasks as T  # noqa: PLC0415

            if not hasattr(T, "CBAM"):
                T.CBAM = CBAM
    except ImportError as exc:
        print(f"[warn] Could not register CBAM: {exc}", file=sys.stderr)


def _patch_nwd_loss() -> None:
    """Replace the bbox regression loss in ultralytics with NWD loss."""
    try:
        import ultralytics.utils.loss as L  # noqa: PLC0415
        from utils.nwd_loss import NWDLoss  # noqa: PLC0415

        # ultralytics v8 uses BboxLoss which internally calls bbox_iou.
        # We wrap it so that the box term uses NWD instead.
        _OrigBboxLoss = L.BboxLoss  # noqa: N806

        class _NWDBboxLoss(_OrigBboxLoss):  # noqa: N801
            def __init__(self, reg_max: int, use_dfl: bool = True) -> None:
                super().__init__(reg_max, use_dfl)
                from utils.nwd_loss import NWDLoss  # noqa: PLC0415

                self._nwd = NWDLoss(constant=12.8)

            def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                        target_scores, target_scores_sum, fg_mask):
                # Let the parent compute DFL loss; override only the box loss.
                loss_iou, loss_dfl = super().forward(
                    pred_dist, pred_bboxes, anchor_points,
                    target_bboxes, target_scores, target_scores_sum, fg_mask,
                )
                if fg_mask.sum():
                    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
                    # NWD expects cxcywh; pred_bboxes and target_bboxes are xyxy
                    def _xyxy2cxcywh(b):
                        return (b[..., :2] + b[..., 2:]) / 2, b[..., 2:] - b[..., :2]

                    pc, ps = _xyxy2cxcywh(pred_bboxes[fg_mask])
                    tc, ts = _xyxy2cxcywh(target_bboxes[fg_mask])
                    import torch  # noqa: PLC0415

                    pred_cxcywh = torch.cat([pc, ps], dim=-1)
                    tgt_cxcywh = torch.cat([tc, ts], dim=-1)
                    loss_iou = (
                        self._nwd(pred_cxcywh, tgt_cxcywh) * weight
                    ).sum() / target_scores_sum

                return loss_iou, loss_dfl

        L.BboxLoss = _NWDBboxLoss
    except (ImportError, AttributeError) as exc:
        print(f"[warn] Could not patch NWD loss: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 with P2 head + CBAM + NWD loss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",   default="data/tt100k.yaml",       help="dataset yaml")
    parser.add_argument("--model",  default="configs/yolov8-p2.yaml", help="model yaml or .pt")
    parser.add_argument("--epochs", type=int,   default=100)
    parser.add_argument("--imgsz",  type=int,   default=640)
    parser.add_argument("--batch",  type=int,   default=16)
    parser.add_argument("--device", default="0", help="cuda device, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int,  default=8)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name",    default="train")
    parser.add_argument(
        "--no-cbam",  dest="cbam",  action="store_false",
        help="disable CBAM patch (use stock C2f layers)"
    )
    parser.add_argument(
        "--no-nwd",   dest="nwd",   action="store_false",
        help="disable NWD loss patch (use stock CIoU)"
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.cbam:
        _patch_ultralytics()
    if args.nwd:
        _patch_nwd_loss()

    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        sys.exit(
            "ultralytics is not installed. "
            "Run: pip install ultralytics"
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
