"""
古文字识别模型训练脚本
基于 EfficientNet-B0，使用 HUST-OBC 数据集
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# ========== 配置参数 ==========
DATA_ROOT = "/mnt/workspace/HUST-OBC/deciphered"
JSON_PATH = "/mnt/workspace/HUST-OBC/deciphered/ID_to_chinese.json"
OUTPUT_DIR = "/mnt/workspace/recognition_model"
BATCH_SIZE = 64
EPOCHS = 30
IMG_SIZE = 128
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1. 加载标签映射 ==========
with open(JSON_PATH, "r", encoding="utf-8") as f:
    id_to_char = json.load(f)

# 构建 ID 到索引的映射
all_ids = sorted(os.listdir(DATA_ROOT))
valid_ids = [d for d in all_ids if os.path.isdir(os.path.join(DATA_ROOT, d)) and d in id_to_char]
id_to_idx = {id_: i for i, id_ in enumerate(valid_ids)}
num_classes = len(id_to_idx)
print(f"类别数: {num_classes}")
print(f"有效类别数: {len(valid_ids)}")

# ========== 2. 收集所有图片路径和标签 ==========
all_images = []
for id_ in valid_ids:
    folder = os.path.join(DATA_ROOT, id_)
    imgs_in_folder = []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            imgs_in_folder.append((os.path.join(folder, fname), id_to_idx[id_]))
    all_images.extend(imgs_in_folder)

print(f"总图片数: {len(all_images)}")
print(f"总类别数: {len(valid_ids)}")

# 统计每个类别的样本数，过滤掉样本太少的
from collections import Counter
label_counts = Counter(label for _, label in all_images)
rare_labels = {label for label, count in label_counts.items() if count < 5}
filtered_images = [(p, l) for p, l in all_images if l not in rare_labels]
print(f"过滤掉 {len(rare_labels)} 个样本过少的类别, 剩余图片数: {len(filtered_images)}")

# 按标签分层划分训练/验证集
labels = [label for _, label in filtered_images]
train_imgs, val_imgs = train_test_split(
    filtered_images, test_size=0.15, random_state=42, stratify=labels
)
print(f"训练集: {len(train_imgs)}  验证集: {len(val_imgs)}")

# ========== 3. 数据集类 ==========
class OracleDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path, label = self.image_list[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# 训练数据增强（针对拓片特点）
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = OracleDataset(train_imgs, transform=train_transform)
val_dataset = OracleDataset(val_imgs, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ========== 4. 构建模型 ==========
try:
    print("正在下载 EfficientNet-B0 预训练权重（国内镜像）...")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    print("使用预训练权重！")
except Exception as e:
    print(f"下载失败 ({e})，改为无预训练从头训练...")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ========== 5. 训练循环 ==========
best_acc = 0.0
for epoch in range(EPOCHS):
    # ---- 训练 ----
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [训练]")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        train_loss += loss.item() * images.size(0)
        train_correct += preds.eq(labels).sum().item()
        train_total += labels.size(0)
        pbar.set_postfix({"loss": f"{train_loss/train_total:.4f}", "acc": f"{train_correct/train_total:.4f}"})

    train_acc = train_correct / train_total
    train_loss_avg = train_loss / train_total

    # ---- 验证 ----
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [验证]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = outputs.max(1)
            val_loss += loss.item() * images.size(0)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss_avg = val_loss / val_total

    scheduler.step()

    print(f"\nEpoch {epoch+1:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss_avg:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss_avg:.4f} Acc: {val_acc:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "id_to_idx": id_to_idx,
            "id_to_char": id_to_char,
        }, os.path.join(OUTPUT_DIR, "best_model.pth"))
        print(f"  → 保存最佳模型 (val_acc={val_acc:.4f})")

    # 保存最新模型（用于中断后继续）
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "id_to_idx": id_to_idx,
        "id_to_char": id_to_char,
    }, os.path.join(OUTPUT_DIR, "latest_checkpoint.pth"))

print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")
print(f"模型保存在: {OUTPUT_DIR}/best_model.pth")
