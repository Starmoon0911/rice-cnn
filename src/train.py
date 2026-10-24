from pathlib import Path
import argparse
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image

if __package__:
    from .device import training_device, cpu_state_dict
else:
    from device import training_device, cpu_state_dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# =========================================================
# 1. 基本設定
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description='Train RiceCNN on Intel XPU or CPU.')
parser.add_argument('--device', choices=('xpu', 'cpu'), default='xpu')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--model-dir', type=Path)
parser.add_argument('--result-dir', type=Path)
parser.add_argument('--figure-dir', type=Path)
args = parser.parse_args()
if args.epochs < 1:
    parser.error('--epochs must be >= 1')
try:
    DEVICE = training_device(args.device)
except RuntimeError as error:
    parser.error(str(error))

DATA_DIR = PROJECT_ROOT / "dataset" / "rice_leaf_diseases"
MODEL_DIR = args.model_dir or PROJECT_ROOT / 'models'
RESULT_DIR = args.result_dir or PROJECT_ROOT / ('results/xpu' if DEVICE.type == 'xpu' else 'results')
FIGURE_DIR = args.figure_dir or PROJECT_ROOT / ('figures/xpu' if DEVICE.type == 'xpu' else 'figures')

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = args.epochs
LEARNING_RATE = 0.001
PATIENCE = 50


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("邱裕勝－水稻葉片病害 CNN")
print("=" * 60)
print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {DEVICE}")
if DEVICE.type == 'xpu':
    print(f"GPU             : {torch.xpu.get_device_name(DEVICE)}")
print(f"CPU threads     : {torch.get_num_threads()}")
print(f"Dataset path    : {DATA_DIR}")
print("=" * 60)


# =========================================================
# 2. 讀取圖片路徑
# =========================================================

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"\n找不到資料夾：\n{DATA_DIR}\n"
        "請確認水稻資料集已解壓縮到正確位置。"
    )

class_names = sorted([
    folder.name
    for folder in DATA_DIR.iterdir()
    if folder.is_dir()
])

print("\n分類：")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")

image_paths = []
labels = []

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

for class_index, class_name in enumerate(class_names):

    class_folder = DATA_DIR / class_name

    for img_path in class_folder.rglob("*"):

        if img_path.suffix.lower() in extensions:
            image_paths.append(img_path)
            labels.append(class_index)

print(f"\n總圖片數：{len(image_paths)}")

if len(image_paths) == 0:
    raise RuntimeError("找不到任何圖片。")


# =========================================================
# 3. Train / Validation / Test 分割
# =========================================================

indices = np.arange(len(image_paths))

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels,
)

temp_labels = [labels[i] for i in temp_indices]

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels,
)

print("\n資料分割：")
print(f"Train      : {len(train_indices)}")
print(f"Validation : {len(val_indices)}")
print(f"Test       : {len(test_indices)}")


# =========================================================
# 4. Image Dataset
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class RiceDataset(Dataset):

    def __init__(self, indices, transform):
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        real_index = self.indices[index]

        image_path = image_paths[real_index]
        label = labels[real_index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = RiceDataset(
    train_indices,
    train_transform,
)

val_dataset = RiceDataset(
    val_indices,
    eval_transform,
)

test_dataset = RiceDataset(
    test_indices,
    eval_transform,
)


# Windows 建議先 num_workers=0
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# =========================================================
# 5. CNN 模型
# =========================================================

if __package__:
    from .model import RiceCNN
else:
    from model import RiceCNN


model = RiceCNN(
    num_classes=len(class_names)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
)


# =========================================================
# 6. 評估函數
# =========================================================

def evaluate(loader):

    model.eval()

    total_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                batch_labels,
            )

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(
                batch_labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions,
    )


# =========================================================
# 7. 訓練
# =========================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_macro_f1": [],
}

best_val_loss = float("inf")

patience_counter = 0

best_model_path = MODEL_DIR / f"best_rice_cnn_{DEVICE.type}.pth"

print(f"\n開始 {DEVICE.type.upper()} CNN 訓練\n")


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0

    for images, batch_labels in train_loader:

        images = images.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            batch_labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    (
        val_loss,
        val_accuracy,
        val_f1,
        _,
        _
    ) = evaluate(val_loader)

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)
    history["val_macro_f1"].append(val_f1)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss={train_loss:.4f} | "
        f"Val Loss={val_loss:.4f} | "
        f"Val Acc={val_accuracy:.4f} | "
        f"Val Macro-F1={val_f1:.4f}"
    )

    # Early stopping
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        patience_counter = 0

        torch.save(
            {
                "model_state_dict": cpu_state_dict(model),
                "device": DEVICE.type,
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
            },
            best_model_path,
        )

        print("  ★ 儲存目前最佳模型")

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly Stopping："
                f"連續 {PATIENCE} epochs 未改善"
            )

            break


# =========================================================
# 8. 載入最佳模型
# =========================================================

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# =========================================================
# 9. Test Set 最終評估
# =========================================================

(
    test_loss,
    test_accuracy,
    test_macro_f1,
    true_labels,
    predictions,
) = evaluate(test_loader)

print("\n" + "=" * 60)

print("TEST RESULT")

print("=" * 60)

print(f"Test Loss     : {test_loss:.4f}")

print(f"Test Accuracy : {test_accuracy:.4f}")

print(f"Test Macro-F1 : {test_macro_f1:.4f}")


# =========================================================
# 10. Classification Report
# =========================================================

report = classification_report(
    true_labels,
    predictions,
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print("\nClassification Report:\n")

print(report)

with open(
    RESULT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(report)


# =========================================================
# 11. Confusion Matrix
# =========================================================

cm = confusion_matrix(
    true_labels,
    predictions,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

display.plot(
    ax=ax,
    xticks_rotation=30,
)

plt.title(
    "Rice Leaf Disease - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "confusion_matrix.png",
    dpi=300,
)

plt.close()


# =========================================================
# 12. Loss 曲線
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["train_loss"],
    label="Train Loss",
)

plt.plot(
    history["val_loss"],
    label="Validation Loss",
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training History")

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "training_loss.png",
    dpi=300,
)

plt.close()


# =========================================================
# 13. 儲存結果
# =========================================================

metrics = {
    "device": DEVICE.type,

    "test_loss": float(test_loss),

    "test_accuracy": float(test_accuracy),

    "test_macro_f1": float(test_macro_f1),

    "classes": class_names,

    "train_samples": len(train_dataset),

    "validation_samples": len(val_dataset),

    "test_samples": len(test_dataset),
}

with open(
    RESULT_DIR / "metrics.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        ensure_ascii=False,
        indent=4,
    )


print("\n完成！")

print(f"最佳模型：{best_model_path}")

print(
    f"混淆矩陣："
    f"{FIGURE_DIR / 'confusion_matrix.png'}"
)

print(
    f"訓練曲線："
    f"{FIGURE_DIR / 'training_loss.png'}"
)

print(
    f"分類報告："
    f"{RESULT_DIR / 'classification_report.txt'}"
)

print(
    f"評估指標："
    f"{RESULT_DIR / 'metrics.json'}"
)
