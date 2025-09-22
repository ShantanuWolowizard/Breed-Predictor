import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# Device setup
# ---------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ---------------------
# Paths
# ---------------------
data_dir = "/Users/shantanuchaturvedi/Documents/Breed data/dataset"
test_dir = os.path.join(data_dir, "test")

# ---------------------
# Transforms (no heavy augments for test)
# ---------------------
input_size = 320
test_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------
# Dataset & Loader
# ---------------------
test_ds = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
class_names = test_ds.classes
print("Test classes:", class_names)

# ---------------------
# Load model
# ---------------------
from train_cnn_final import model  # import model definition
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ---------------------
# Evaluation
# ---------------------
correct1, correct3, total = 0, 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)

        # Top-1
        _, pred1 = outputs.max(1)

        # Top-3
        _, pred3 = outputs.topk(3, 1, True, True)

        total += y.size(0)
        correct1 += (pred1 == y).sum().item()
        correct3 += sum([y[i] in pred3[i] for i in range(y.size(0))])

        all_preds.extend(pred1.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

top1_acc = correct1 / total
top3_acc = correct3 / total

print(f"\n📊 Test Results:")
print(f"Top-1 Accuracy: {top1_acc:.3f}")
print(f"Top-3 Accuracy: {top3_acc:.3f}")

# ---------------------
# Confusion Matrix
# ---------------------
cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.show()
