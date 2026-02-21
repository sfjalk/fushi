"""
Traffic-sign small-object detection — Gradio demo.

Run:
    python demo.py

Then open the local URL printed in the console (default http://127.0.0.1:7860).
"""

import pathlib
import textwrap

import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------------
# Optional: resolve a trained weights path automatically.
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS = "yolov8n.pt"  # fallback to pretrained nano weights

_RUNS_DIR = pathlib.Path("runs/detect")
if _RUNS_DIR.exists():
    # pick the most recently modified best.pt
    candidates = sorted(
        _RUNS_DIR.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime
    )
    if candidates:
        _DEFAULT_WEIGHTS = str(candidates[-1])


def _load_model(weights: str):
    """Load an ultralytics YOLO model (lazy import so gradio starts quickly)."""
    from ultralytics import YOLO  # noqa: PLC0415

    return YOLO(weights)


# Cache the model at module level so it is only loaded once per session.
_model = None
_loaded_weights: str = ""


def detect(image: Image.Image, weights: str, conf: float, iou: float):
    """Run inference on *image* and return the annotated result.

    Args:
        image:   PIL image uploaded by the user.
        weights: Path to a YOLO ``.pt`` weights file.
        conf:    Confidence threshold (0–1).
        iou:     NMS IoU threshold (0–1).

    Returns:
        Tuple of (annotated PIL image, detection-summary text).
    """
    global _model, _loaded_weights  # noqa: PLW0603

    weights = weights.strip() or _DEFAULT_WEIGHTS

    if _model is None or weights != _loaded_weights:
        try:
            _model = _load_model(weights)
            _loaded_weights = weights
        except Exception as exc:  # noqa: BLE001
            return image, f"❌ Failed to load model: {exc}"

    try:
        results = _model.predict(
            source=image,
            conf=conf,
            iou=iou,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        return image, f"❌ Inference error: {exc}"

    result = results[0]
    annotated = Image.fromarray(result.plot())

    # Build a human-readable summary
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        summary = "No objects detected."
    else:
        names = result.names
        lines = [f"Detected **{len(boxes)}** object(s):\n"]
        for cls_id, conf_val in zip(
            boxes.cls.tolist(), boxes.conf.tolist()
        ):
            lines.append(f"- {names[int(cls_id)]}: {conf_val:.2%}")
        summary = "\n".join(lines)

    return annotated, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_DESCRIPTION = textwrap.dedent(
    """
    ## 🚦 Traffic Sign Small-Object Detector

    Upload a road-scene image and click **Detect**.
    The model will highlight every traffic sign it finds.

    * **Weights** – path to a local `.pt` file, or leave blank to use the
      default YOLOv8n pretrained weights.
    * **Confidence** – minimum score to keep a detection.
    * **IoU** – NMS overlap threshold.
    """
)

with gr.Blocks(title="Traffic Sign Detector") as demo:
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            inp_image = gr.Image(type="pil", label="Input image")
            inp_weights = gr.Textbox(
                value=_DEFAULT_WEIGHTS,
                label="Weights path",
                placeholder="e.g. runs/detect/train/weights/best.pt",
            )
            with gr.Row():
                inp_conf = gr.Slider(0.01, 1.0, value=0.25, step=0.01, label="Confidence")
                inp_iou = gr.Slider(0.01, 1.0, value=0.45, step=0.01, label="IoU (NMS)")
            btn = gr.Button("Detect", variant="primary")

        with gr.Column(scale=1):
            out_image = gr.Image(type="pil", label="Annotated result")
            out_text = gr.Markdown(label="Summary")

    btn.click(
        fn=detect,
        inputs=[inp_image, inp_weights, inp_conf, inp_iou],
        outputs=[out_image, out_text],
    )

    gr.Examples(
        examples=[],  # add local image paths here for quick demos
        inputs=inp_image,
    )

if __name__ == "__main__":
    demo.launch()
