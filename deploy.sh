#!/bin/bash
# ============================================
# 一键部署：古文字OCR推理 (YOLO + EfficientNet)
# 上传本脚本到魔搭，执行: bash deploy.sh
# ============================================
set -e

CODE_DIR="/mnt/workspace/sais_ocr"
echo "===== 创建代码目录 ====="
mkdir -p $CODE_DIR/src $CODE_DIR/models

# ===== 1. run_inference.py =====
echo "创建 run_inference.py ..."
cat > $CODE_DIR/src/run_inference.py << 'PYEOF'
#!/usr/bin/env python3
"""
古文字OCR推理脚本
使用 YOLO 检测 + EfficientNet 识别
"""
import json
import os
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

INPUT_DIR = Path(os.getenv("INPUT_DIR", "/saisdata/13/eval/images"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))
MODEL_DIR = Path("/app/models")
YOLO_PATH = MODEL_DIR / "yolo_best.pt"
RECOGNITION_PATH = MODEL_DIR / "recognition_best.pth"
ID_TO_CHAR_PATH = MODEL_DIR / "id_to_char.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
IMG_SIZE = 128


def find_images():
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if INPUT_DIR.exists():
        return sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in suffixes])
    fallback = Path("/saisdata")
    if fallback.exists():
        return sorted([p for p in fallback.rglob("*") if p.suffix.lower() in suffixes])
    return []


def load_models():
    print(f"设备: {DEVICE}")
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_PATH))

    with open(ID_TO_CHAR_PATH, "r") as f:
        id_to_char = json.load(f)
    checkpoint = torch.load(RECOGNITION_PATH, map_location=DEVICE, weights_only=False)
    id_to_idx = checkpoint["id_to_idx"]
    idx_to_char = {v: id_to_char.get(k, k) for k, v in id_to_idx.items()}
    num_classes = len(id_to_idx)

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"模型加载完成! {num_classes} 类")
    return yolo, model, transform, idx_to_char


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    image_paths = find_images()
    print(f"图片数: {len(image_paths)}")

    if not image_paths:
        with open(OUTPUT_FILE, "w") as f:
            json.dump({}, f)
        return

    yolo, rec_model, transform, idx_to_char = load_models()
    results = {}

    for idx, img_path in enumerate(image_paths, 1):
        if idx == 1 or idx % 50 == 0:
            print(f"[{idx}/{len(image_paths)}] {img_path.name}")
        image_id = img_path.stem
        try:
            img = Image.open(img_path).convert("RGB")
            dets = yolo(str(img_path), conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            boxes = []
            if dets and dets[0].boxes is not None:
                for box in dets[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    if x2 - x1 > 5 and y2 - y1 > 5:
                        boxes.append([x1, y1, x2 - x1, y2 - y1])

            detections = []
            for box in boxes:
                x, y, w, h = box
                crop = img.crop((x, y, x + w, y + h))
                tensor = transform(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = rec_model(tensor)
                    pred = outputs.argmax(1).item()
                char = idx_to_char.get(pred, f"UNK_{pred}")
                detections.append({"bbox": [x, y, w, h], "text": char})

            detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
            results[image_id] = detections
        except Exception as e:
            print(f"跳过 {img_path.name}: {e}")
            results[image_id] = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"完成! -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
PYEOF

# ===== 2. run.sh =====
echo "创建 run.sh ..."
cat > $CODE_DIR/run.sh << 'SHEOF'
#!/bin/bash
set -euo pipefail
echo "Starting YOLO + EfficientNet OCR inference..."

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ] \
  && [ "${NVIDIA_VISIBLE_DEVICES}" != "all" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "void" ]; then
  export CUDA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"
fi

python3 -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')" || true

[ ! -d "/saisresult" ] && mkdir -p /saisresult
python3 /app/src/run_inference.py

PREDICTION_FILE="${OUTPUT_FILE:-/saisresult/prediction.json}"
[ ! -f "${PREDICTION_FILE}" ] && echo "Error: ${PREDICTION_FILE} not found" && exit 1
echo "Done! ${PREDICTION_FILE}"
SHEOF
chmod +x $CODE_DIR/run.sh

# ===== 3. requirements.txt =====
echo "创建 requirements.txt ..."
cat > $CODE_DIR/requirements.txt << 'REQ'
numpy==1.26.4
opencv-python-headless==4.8.1.78
pillow==10.4.0
ultralytics==8.3.0
torch>=2.0.0
torchvision>=0.15.0
timm==1.0.3
REQ

# ===== 4. Dockerfile =====
echo "创建 Dockerfile ..."
cat > $CODE_DIR/Dockerfile << 'DOCKER'
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

WORKDIR /app
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary -r /app/requirements.txt

COPY models/ /app/models/
COPY src/ /app/src/
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh
ENTRYPOINT ["bash", "/app/run.sh"]
DOCKER

# ===== 5. 复制模型文件 =====
echo "复制模型文件..."
YOLO_BEST="/mnt/workspace/runs/detect/train/weights/best.pt"
RECOG_BEST="/mnt/workspace/recognition_model/best_model.pth"

if [ -f "$YOLO_BEST" ]; then
    cp "$YOLO_BEST" "$CODE_DIR/models/yolo_best.pt"
    echo "  YOLO 模型 OK"
else
    echo "  ⚠️ YOLO 模型未找到: $YOLO_BEST"
fi

if [ -f "$RECOG_BEST" ]; then
    cp "$RECOG_BEST" "$CODE_DIR/models/recognition_best.pth"
    echo "  识别模型 OK"
else
    echo "  ⚠️ 识别模型未找到: $RECOG_BEST"
fi

# 生成 id_to_char.json
python3 -c "
import json, torch
ckpt = torch.load('$RECOG_BEST', map_location='cpu', weights_only=False)
import os
path = '/mnt/workspace/HUST-OBC/deciphered/ID_to_chinese.json'
if os.path.exists(path):
    with open(path) as f:
        id_to_char = json.load(f)
else:
    id_to_char = {k: k for k in ckpt['id_to_idx']}
with open('$CODE_DIR/models/id_to_char.json', 'w') as f:
    json.dump(id_to_char, f, ensure_ascii=False, indent=2)
print('  id_to_char.json OK')
" 2>/dev/null || echo "  ⚠️ id_to_char.json 生成失败"

# ===== 完成 =====
echo ""
echo "========================================"
echo "  ✅ 部署文件已就绪！"
echo "========================================"
echo ""
echo "目录: $CODE_DIR"
echo ""
echo "接下来2步:"
echo ""
echo "步骤1: 关联 Codeup 仓库（仅首次）"
echo "  cd $CODE_DIR"
echo "  git init"
echo "  git remote add origin <你的Codeup仓库地址>"
echo ""
echo "步骤2: 推送代码"
echo "  cd $CODE_DIR"
echo '  git add .'
echo '  git commit -m "feat: YOLO + EfficientNet OCR"'
echo "  git push -u origin main"
echo ""
echo "推送后阿里云会自动构建 Docker 镜像~"
echo "========================================"
