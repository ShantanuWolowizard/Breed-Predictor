#!/usr/bin/env python3
"""
evaluate_test_with_mistakes.py

Saves misclassified test images into Test_mistakes/<true_class>/ with filenames
like: image123__pred_Murrah_actual_nili_ravi.jpg
"""

import os
import shutil
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# CONFIG: update this if your folder is elsewhere
# ---------------------
data_dir = "/Users/shantanuchaturvedi/Documents/Breed data/dataset"  # <- update only if needed
test_folder_name = "Test"  # your test folder name (as in your screenshot)
model_path = os.path.join(os.getcwd(), "best_model.pth")  # assumes best_model.pth in same folder as script
out_mistakes = os.path.join(os.getcwd(), "Test_mistakes")

# ---------------------
# device
# ---------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ---------------------
# transforms & loader
# ---------------------
input_size = 320
test_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dir = os.path.join(data_dir, test_folder_name)
if not os.path.isdir(test_dir):
    raise SystemExit(f"Test folder not found at {test_dir} - update data_dir/test_folder_name in script")

test_ds = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
class_names = test_ds.classes
num_classes = len(class_names)
print("Test classes:", class_names)
print("Number of test images:", len(test_ds))

# ---------------------
# build model architecture (must match the training script)
# ---------------------
model = models.efficientnet_b0(weights=None)  # create architecture
# replace classifier head to match num classes
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, num_classes)

# load weights
state = torch.load(model_path, map_location="cpu")
model.load_state_dict(state)
model = model.to(device)
model.eval()

# ---------------------
# prepare output folder for mistakes
# ---------------------
if os.path.exists(out_mistakes):
    print("Clearing existing Test_mistakes folder...")
    shutil.rmtree(out_mistakes)
os.makedirs(out_mistakes, exist_ok=True)
for cls in class_names:
    os.makedirs(os.path.join(out_mistakes, cls), exist_ok=True)

# ---------------------
# Evaluate & save mistakes
# ---------------------
total = 0
correct1 = 0
correct3 = 0

per_class_total = [0] * num_classes
per_class_correct = [0] * num_classes

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # top1 and top3
        _, pred1 = outputs.max(1)
        _, pred3 = outputs.topk(3, 1, True, True)

        # update totals
        total += labels.size(0)
        correct1 += (pred1 == labels).sum().item()
        # correct3: check membership
        correct3 += sum([labels[i].item() in pred3[i].cpu().tolist() for i in range(labels.size(0))])

        # per-class stats and save mistakes
        for i in range(labels.size(0)):
            true = labels[i].item()
            p1 = pred1[i].item()
            per_class_total[true] += 1
            if p1 == true:
                per_class_correct[true] += 1
            # Save misclassified examples
            if p1 != true:
                # find original file path for this image
                # ImageFolder stores paths in dataset.samples
                global_index = batch_idx * test_loader.batch_size + i
                img_path, _ = test_ds.samples[global_index]
                # Compose filename and save a copy in Test_mistakes/<true_class>/
                base = os.path.basename(img_path)
                save_name = f"{os.path.splitext(base)[0]}__pred_{class_names[p1]}__actual_{class_names[true]}{os.path.splitext(base)[1]}"
                save_path = os.path.join(out_mistakes, class_names[true], save_name)
                try:
                    # copy original file (not the transformed one)
                    shutil.copy(img_path, save_path)
                except Exception as e:
                    print("Failed to copy", img_path, "->", save_path, e)

        # accumulate labels/preds for confusion matrix
        all_preds.extend(pred1.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

top1 = correct1 / total if total else 0.0
top3 = correct3 / total if total else 0.0

print("\n===== Test Evaluation =====")
print(f"Total images: {total}")
print(f"Top-1 Accuracy: {top1:.4f}")
print(f"Top-3 Accuracy: {top3:.4f}\n")

print("Per-class accuracy:")
for idx, cls in enumerate(class_names):
    tot = per_class_total[idx]
    corr = per_class_correct[idx]
    acc = (corr / tot) if tot else 0.0
    print(f"  {cls}: {corr}/{tot} = {acc:.3f}")

# Confusion matrix
try:
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Could not plot confusion matrix:", e)

print(f"\nMisclassified images saved into: {out_mistakes}")
