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
import numpy as np
import timm

INPUT_DIR = Path(os.getenv("INPUT_DIR", "/saisdata/13/eval/images"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))

# 模型路径（打包进镜像时放在 /app/models/ 下）
MODEL_DIR = Path("/app/models")
YOLO_PATH = MODEL_DIR / "yolo_best.pt"
RECOGNITION_PATH = MODEL_DIR / "recognition_best.pth"
ID_TO_CHAR_PATH = MODEL_DIR / "id_to_char.json"
ID_TO_IDX_PATH = MODEL_DIR / "id_to_idx.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
IMG_SIZE = 128


def find_images():
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if INPUT_DIR.exists():
        return sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in suffixes])
    fallback_root = Path("/saisdata")
    if fallback_root.exists():
        return sorted([p for p in fallback_root.rglob("*") if p.suffix.lower() in suffixes])
    return []


def load_models():
    """加载 YOLO 检测模型和 EfficientNet 识别模型"""
    print(f"使用设备: {DEVICE}")

    # 加载 YOLO
    from ultralytics import YOLO
    print(f"加载 YOLO 检测模型: {YOLO_PATH}")
    yolo = YOLO(str(YOLO_PATH))

    # 加载识别模型
    print(f"加载识别模型: {RECOGNITION_PATH}")
    checkpoint = torch.load(RECOGNITION_PATH, map_location=DEVICE, weights_only=False)
    id_to_idx = checkpoint["id_to_idx"]
    id_to_char = checkpoint.get("id_to_char", {})
    num_classes = len(id_to_idx)

    # 重建模型
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # 构建 idx → char 映射
    idx_to_char = {}
    for id_, idx in id_to_idx.items():
        idx_to_char[idx] = id_to_char.get(id_, id_)

    # 识别预处理
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"模型加载完成! 识别类别数: {num_classes}")
    return yolo, model, transform, idx_to_char


def detect_characters(yolo, image_path):
    """YOLO 检测，返回边界框列表 [x1, y1, x2, y2]"""
    results = yolo(str(image_path), conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            if w > 5 and h > 5:
                boxes.append([x1, y1, w, h])
    return boxes


def recognize_character(model, transform, image, box):
    """从图片中裁剪并识别单个字符"""
    x, y, w, h = box
    crop = image.crop((x, y, x + w, y + h))
    tensor = transform(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        pred_idx = outputs.argmax(1).item()
        confidence = torch.softmax(outputs, 1)[0, pred_idx].item()

    return pred_idx, confidence


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    image_paths = find_images()
    print(f"输入目录: {INPUT_DIR}")
    print(f"找到图片: {len(image_paths)}张")

    if not image_paths:
        print("未找到图片，输出空结果")
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return

    # 加载模型
    yolo, recognition_model, transform, idx_to_char = load_models()

    results = {}
    for idx, image_path in enumerate(image_paths, 1):
        if idx == 1 or idx % 50 == 0:
            print(f"[{idx}/{len(image_paths)}] {image_path.name}")

        image_id = image_path.stem
        try:
            img = Image.open(image_path).convert("RGB")
            boxes = detect_characters(yolo, image_path)
            detections = []
            for box in boxes:
                pred_idx, confidence = recognize_character(recognition_model, transform, img, box)
                char = idx_to_char.get(pred_idx, f"UNK_{pred_idx}")
                detections.append({
                    "bbox": [int(v) for v in box],
                    "text": char,
                })
            detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
            results[image_id] = detections
        except Exception as e:
            print(f"处理失败 {image_path.name}: {e}")
            traceback.print_exc()
            results[image_id] = []

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"保存结果: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
