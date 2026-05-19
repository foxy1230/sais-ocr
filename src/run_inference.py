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
