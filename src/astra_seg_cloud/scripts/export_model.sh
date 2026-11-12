#!/usr/bin/env bash
# 导出 yolov8s-seg.onnx（只需跑一次）。
# 用独立 venv，避免把 torch/ultralytics 塞进系统 Python。
set -euo pipefail

OUT_DIR="${OUT_DIR:-$HOME/astra_models}"
VENV="${VENV:-/tmp/yolo_export}"
MODEL="${MODEL:-yolov8s-seg.pt}"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

if [ ! -f "$MODEL" ]; then
  echo "[1/4] 下载 $MODEL"
  curl -L --retry 3 -o "$MODEL" \
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/$MODEL"
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "[2/4] 创建 venv: $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -U pip
fi

echo "[3/4] 安装 torch(CPU) + ultralytics"
"$VENV/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install -q ultralytics onnxslim onnx onnxruntime

echo "[4/4] 导出 ONNX (imgsz=640, opset=12)"
cd /tmp
"$VENV/bin/yolo" export model="$OUT_DIR/$MODEL" format=onnx imgsz=640 opset=12 simplify=True

echo "完成: $OUT_DIR/${MODEL%.pt}.onnx"
