# 交通标志小目标检测 — YOLOv8 魔改教程

基于 [ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)，针对**小目标交通标志检测**进行三项改进，并附带完整的训练脚本与 Gradio 可视化 Demo。

---

## 改进点概述

| 改进 | 文件 | 说明 |
|------|------|------|
| **P2 检测头** | `configs/yolov8-p2.yaml` | 在原有 P3/P4/P5 三个尺度的基础上增加 stride=4 的 P2 输出层，提升小目标召回率 |
| **CBAM 注意力** | `modules/cbam.py` | 通道注意力 + 空间注意力，插入主干特征层后以增强显著特征响应 |
| **NWD 损失** | `utils/nwd_loss.py` | 以归一化 Wasserstein 距离替换 CIoU，对不重叠的小目标框产生更平滑的梯度 |

---

## 项目结构

```
├── configs/
│   └── yolov8-p2.yaml      # P2 检测头配置（4 个输出尺度）
├── modules/
│   ├── __init__.py
│   └── cbam.py             # ChannelAttention / SpatialAttention / CBAM
├── utils/
│   ├── __init__.py
│   └── nwd_loss.py         # wasserstein2_distance_sq / nwd_loss / NWDLoss
├── tests/
│   ├── test_cbam.py
│   ├── test_nwd_loss.py
│   └── test_integration.py
├── train.py                # 一键训练（自动注入 CBAM + NWD）
├── demo.py                 # Gradio 可视化推理界面
└── README.md
```

---

## 环境搭建

```bash
# 推荐使用 AutoDL / 本地 GPU 服务器
conda create -n yolov8 python=3.9 -y
conda activate yolov8

# PyTorch（根据 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# YOLOv8 官方库 + Gradio
pip install ultralytics gradio
```

---

## 数据集准备

推荐使用 [Roboflow](https://roboflow.com/) 上的 **TT100K** 或 **CCTSDB** 数据集（已转为 YOLO 格式）。

解压后目录结构示例：

```
data/tt100k/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

---

## 训练

```bash
# 使用全部改进（P2 头 + CBAM + NWD）
python train.py \
    --data   data/tt100k/data.yaml \
    --model  configs/yolov8-p2.yaml \
    --epochs 100 \
    --imgsz  640 \
    --batch  16 \
    --device 0

# 消融实验：仅 P2 头，不使用 CBAM 和 NWD
python train.py --data data/tt100k/data.yaml --no-cbam --no-nwd
```

训练权重保存在 `runs/detect/train/weights/best.pt`。

---

## Gradio Demo

```bash
python demo.py
# 浏览器打开 http://127.0.0.1:7860
```

上传交通场景图片，点击 **Detect** 即可看到检测结果与置信度。

---

## 单元测试

```bash
pip install pytest
pytest tests/ -v
```

---

## 消融实验参考

| 模型配置 | P2 检测头 | CBAM | NWD | mAP@50 |
|---------|----------|------|-----|--------|
| Baseline | ✗ | ✗ | ✗ | — |
| 改进 A | ✔ | ✗ | ✗ | — |
| 改进 B | ✔ | ✔ | ✗ | — |
| Ours | ✔ | ✔ | ✔ | — |

> 请用你自己的训练结果填写 mAP 数值。

---

## 参考资料

- 官方代码库：<https://github.com/ultralytics/ultralytics>
- NWD 论文：Wang et al., *A Normalized Gaussian Wasserstein Distance for Tiny Object Detection*, arXiv:2110.13389
- CBAM 论文：Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018