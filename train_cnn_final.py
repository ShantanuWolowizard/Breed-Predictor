import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ---------------------
# Device setup (MPS for Mac M1/M2, else CUDA/CPU)
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
train_dir = os.path.join(data_dir, "train")
val_dir   = os.path.join(data_dir, "val")

# ---------------------
# Transforms (robust augmentations)
# ---------------------
input_size = 320
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(25),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3))
])

val_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------
# Datasets & Loaders
# ---------------------
batch_size = 16
train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
val_ds   = datasets.ImageFolder(val_dir, transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
class_names = train_ds.classes
num_classes = len(class_names)
print("Classes:", class_names)

# ---------------------
# Model
# ---------------------
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False  # freeze backbone

# Replace classifier head
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

# ---------------------
# Evaluation helper
# ---------------------
def evaluate(model, loader):
    model.eval()
    correct1, correct3, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, pred1 = outputs.max(1)
            _, pred3 = outputs.topk(3, 1, True, True)
            total += y.size(0)
            correct1 += (pred1 == y).sum().item()
            correct3 += sum([y[i] in pred3[i] for i in range(y.size(0))])
    return correct1/total, correct3/total

# ---------------------
# Training Loop
# ---------------------
if __name__ == "__main__":
    best_acc = 0
    epochs = 25
    for epoch in range(epochs):
        # Stage 2: unfreeze backbone after 5 epochs
        if epoch == 5:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            print("Unfroze backbone for fine-tuning.")

        model.train()
        running_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc1, val_acc3 = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} "
              f"- Val Top1: {val_acc1:.3f} - Val Top3: {val_acc3:.3f}")

        if val_acc1 > best_acc:
            best_acc = val_acc1
            torch.save(model.state_dict(), "best_model.pth")

    print("Training complete. Best Top1 Acc:", best_acc)

# ---------------------
# Grad-CAM for Explainability
# ---------------------
def grad_cam(model, img_path, target_layer="features.6"):
    model.eval()
    img = cv2.imread(img_path)[:, :, ::-1]
    img_resized = cv2.resize(img, (input_size, input_size))
    tensor = val_transforms(Image.fromarray(img_resized)).unsqueeze(0).to(device)

    gradients, activations = [], []
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    def forward_hook(module, input, output):
        activations.append(output.detach())

    layer = dict([*model.named_modules()])[target_layer]
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    output = model(tensor)
    class_id = output.argmax().item()
    score = output[0, class_id]
    model.zero_grad()
    score.backward()

    grad = gradients[0].mean(dim=[2, 3], keepdim=True)
    act = activations[0]
    cam = (grad * act).sum(dim=1).squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    plt.imshow(overlay)
    plt.title(f"Predicted: {class_names[class_id]}")
    plt.axis("off")
    plt.show()

# Example usage (after training completes):
# grad_cam(model, "/Users/shantanuchaturvedi/Documents/Breed data/test/Gir/example.jpg")
