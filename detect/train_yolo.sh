#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from ultralytics import YOLO
model = YOLO("yolov8m.pt")  # downloads weights if missing
model.train(
    data="detect/yolo-dataspur.yaml",
    epochs=50,
    imgsz=960,
    project="runs/detect",
    name="yolov8m-dataspur",
    device=0
)
PY
