#!/bin/bash
# ============================================
# 在魔搭终端运行：bash export_models.sh
# 打包模型文件到模型文件夹，方便下载
# ============================================
set -e

OUTPUT_DIR="/mnt/workspace/exported_models"
mkdir -p "$OUTPUT_DIR"

echo "===== 导出模型文件 ====="

# YOLO 模型
if [ -f /mnt/workspace/runs/detect/train/weights/best.pt ]; then
    cp /mnt/workspace/runs/detect/train/weights/best.pt "$OUTPUT_DIR/yolo_best.pt"
    echo "✅ YOLO 模型"
else
    echo "❌ YOLO 模型未找到"
fi

# 识别模型
if [ -f /mnt/workspace/recognition_model/best_model.pth ]; then
    cp /mnt/workspace/recognition_model/best_model.pth "$OUTPUT_DIR/recognition_best.pth"
    echo "✅ 识别模型"
else
    echo "❌ 识别模型未找到"
fi

# 字符映射
if [ -f /mnt/workspace/HUST-OBC/deciphered/ID_to_chinese.json ]; then
    cp /mnt/workspace/HUST-OBC/deciphered/ID_to_chinese.json "$OUTPUT_DIR/ID_to_chinese.json"
    echo "✅ 字符映射表"
else
    echo "❌ 字符映射表未找到"
fi

echo ""
echo "===== 打包中 ====="
cd /mnt/workspace
tar -czf models.tar.gz -C exported_models .
echo "✅ 打包完成: /mnt/workspace/models.tar.gz"
ls -lh /mnt/workspace/models.tar.gz

echo ""
echo "===== 使用方法 ====="
echo "1. 在魔搭文件管理器下载 models.tar.gz"
echo "2. 解压到本地 HUST-OBS/models/ 目录："
echo "   tar -xzf models.tar.gz -C /path/to/HUST-OBS/models/"
echo "3. 然后 git add models/ && git commit && git push"
echo "4. GitHub Actions 会自动构建 Docker 镜像"
