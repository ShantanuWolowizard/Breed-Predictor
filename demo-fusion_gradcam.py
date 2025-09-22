#!/usr/bin/env python3
"""
demo_fusion_gradcam.py

End-to-end demo pipeline with visualization:
1. CNN prediction + Grad-CAM heatmap
2. Tabular model prediction (dummy simulated)
3. Fusion logic with crossbreed logging
4. Side-by-side visualization (original, heatmap, final decision)
"""

import os, argparse, csv, random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# ---------------- CONFIG ----------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CLASS_NAMES = ["Gir", "H_F", "Jersey", "Murrah", "nili_ravi"]
CNN_WEIGHTS = "/Users/shantanuchaturvedi/Documents/Breed data/best_cnn.pth"
LOG_FILE = "/Users/shantanuchaturvedi/Documents/Breed data/crossbreed_log.csv"
# ----------------------------------------

# ---------- CNN MODEL LOADING -----------
def load_model(weights_path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

# ---------- IMAGE TRANSFORM -------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- GRADCAM ---------------------
def gradcam(model, img_tensor, target_class):
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    layer = model.features[-1]
    h1 = layer.register_forward_hook(forward_hook)
    h2 = layer.register_backward_hook(backward_hook)

    out = model(img_tensor)
    model.zero_grad()
    class_loss = out[0, target_class]
    class_loss.backward()

    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    features = features.squeeze(0)
    for i in range(features.shape[0]):
        features[i, :, :] *= pooled_grads[i]

    heatmap = features.mean(dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    h1.remove(); h2.remove()
    return heatmap

def overlay_gradcam(orig_img, heatmap):
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

# ---------- TABULAR (DUMMY) -------------
def tabular_predict(features):
    return np.random.dirichlet(np.ones(len(CLASS_NAMES)))  # random distribution

# ---------- FUSION + LOGGING ------------
def fuse_and_decide(cnn_probs, tab_probs, img_path):
    cnn_conf = np.max(cnn_probs)
    cnn_pred = CLASS_NAMES[np.argmax(cnn_probs)]
    tab_pred = CLASS_NAMES[np.argmax(tab_probs)]

    if cnn_conf >= 0.8:
        final_pred = cnn_pred
        note = "CNN confident"
    else:
        fused_probs = 0.7*cnn_probs + 0.3*tab_probs
        final_pred = CLASS_NAMES[np.argmax(fused_probs)]
        note = "Fusion used"

    if cnn_pred != tab_pred and abs(cnn_conf - np.max(tab_probs)) < 0.2:
        note = "Possible crossbreed"
        log_crossbreed(img_path, cnn_pred, tab_pred, final_pred, cnn_probs, tab_probs)

    return final_pred, cnn_conf, note

def log_crossbreed(img_path, cnn_pred, tab_pred, final_pred, cnn_probs, tab_probs):
    header = ["image", "cnn_pred", "tab_pred", "final_pred", "cnn_probs", "tab_probs"]
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            img_path, cnn_pred, tab_pred, final_pred,
            cnn_probs.tolist(), tab_probs.tolist()
        ])
    print(f"⚠️ Logged crossbreed case for {img_path}")

# ---------- VISUALIZATION ---------------
def make_side_by_side(orig, heatmap_img, final_pred, conf, note, out_path):
    h = max(orig.shape[0], heatmap_img.shape[0])
    combined = np.hstack([
        cv2.resize(orig, (orig.shape[1], h)),
        cv2.resize(heatmap_img, (heatmap_img.shape[1], h))
    ])
    label = f"Final: {final_pred} ({conf*100:.1f}%) - {note}"
    cv2.putText(combined, label, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, combined)
    print(f"📸 Visualization saved -> {out_path}")

# ---------- MAIN ------------------------
def main(img_path):
    model = load_model(CNN_WEIGHTS)

    pil_img = Image.open(img_path).convert("RGB")
    img_tensor = transform(pil_img)
    orig_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0).to(DEVICE))
        cnn_probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    cnn_pred = CLASS_NAMES[np.argmax(cnn_probs)]
    print(f"🔹 CNN: {cnn_pred}, probs={cnn_probs}")

    heatmap = gradcam(model, img_tensor, np.argmax(cnn_probs))
    gradcam_img = overlay_gradcam(orig_cv, heatmap)

    tab_probs = tabular_predict({
        "milk_yield": random.randint(5, 20),
        "weight": random.randint(300, 600)
    })
    tab_pred = CLASS_NAMES[np.argmax(tab_probs)]
    print(f"📊 Tabular: {tab_pred}, probs={tab_probs}")

    final_pred, conf, note = fuse_and_decide(cnn_probs, tab_probs, img_path)
    print(f"✅ Final decision: {final_pred} ({note})")

    out_path = os.path.splitext(img_path)[0] + "_demo.jpg"
    make_side_by_side(orig_cv, gradcam_img, final_pred, conf, note, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to test image")
    args = parser.parse_args()
    main(args.img)
