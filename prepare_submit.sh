#!/bin/bash
# 一键整理推理代码 + 模型文件，准备推送到 Codeup
set -e

CODE_DIR="/mnt/workspace/sais_ocr"
echo "创建代码目录: $CODE_DIR"
mkdir -p $CODE_DIR/src
mkdir -p $CODE_DIR/models

# 复制推理代码
echo "复制推理脚本..."
cp /mnt/workspace/run_inference.py $CODE_DIR/src/
cp /mnt/workspace/run.sh $CODE_DIR/
cp /mnt/workspace/requirements.txt $CODE_DIR/
cp /mnt/workspace/Dockerfile $CODE_DIR/

# 复制模型文件
echo "复制模型文件..."
cp /mnt/workspace/runs/detect/train/weights/best.pt $CODE_DIR/models/yolo_best.pt
cp /mnt/workspace/recognition_model/best_model.pth $CODE_DIR/models/recognition_best.pth

# 提取 id_to_char 和 id_to_idx（从训练好的 checkpoint 里）
python3 -c "
import torch
ckpt = torch.load('/mnt/workspace/recognition_model/best_model.pth', map_location='cpu', weights_only=False)
import json

with open('/mnt/workspace/HUST-OBC/deciphered/ID_to_chinese.json', 'r') as f:
    id_to_char = json.load(f)

id_to_idx = ckpt['id_to_idx']

with open('/mnt/workspace/sais_ocr/models/id_to_char.json', 'w') as f:
    json.dump(id_to_char, f, ensure_ascii=False, indent=2)
with open('/mnt/workspace/sais_ocr/models/id_to_idx.json', 'w') as f:
    json.dump(id_to_idx, f, ensure_ascii=False, indent=2)
print('id_to_char.json 和 id_to_idx.json 已生成')
"

echo ""
echo "✅ 文件整理完毕！"
echo "目录结构: $CODE_DIR"
ls -la $CODE_DIR/
echo "  ├── src/run_inference.py"
echo "  ├── run.sh"
echo "  ├── requirements.txt"
echo "  ├── Dockerfile"
echo "  └── models/"
echo "      ├── yolo_best.pt"
echo "      ├── recognition_best.pth"
echo "      ├── id_to_char.json"
echo "      └── id_to_idx.json"
echo ""
echo "接下来需要推送到 Codeup，请告诉我你的 Codeup 仓库地址~"
