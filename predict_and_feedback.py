#!/usr/bin/env python3
"""
predict_and_feedback.py  — patched with empty-crop safeguard
"""

import os, sys, argparse, json, csv, datetime, shutil, subprocess
from PIL import Image
import numpy as np
import torch, torch.nn as nn
from torchvision import models, transforms
import cv2
import math

# ---------- CONFIG ----------
BASE = os.getcwd()
TRAITS_CSV = os.path.join(BASE, "breed_traits.csv")
WEIGHTS = os.path.join(BASE, "best_model.pth")
SILH_DIR = os.path.join(BASE, "silhouettes")
OUT_DIR = os.path.join(BASE, "results")
CROSSLOG = os.path.join(BASE, "crossbreed_log.csv")
FEEDBACK = os.path.join(BASE, "feedback_corrections.csv")
CROSS_SAMPLES_DIR = os.path.join(BASE, "crossbreed_samples")
THRESHOLD_CNN = 0.7
CROSS_TOP_DIFF = 0.15
CROSS_TOP1_MIN = 0.50

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CROSS_SAMPLES_DIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ---------- helpers ----------
def load_trait_csv(path):
    rows = []
    with open(path, "r", encoding="utf8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows

def get_breeds_from_traits(rows):
    return [r['breed'] for r in rows]

def build_stats_map(rows):
    m = {}
    for r in rows:
        b = r['breed']
        m[b] = {
            'milk_mean': float(r.get('milk_mean') or 0.0),
            'milk_std' : float(r.get('milk_std') or 1.0),
            'lact_mean': float(r.get('lact_mean') or 0.0),
            'lact_std' : float(r.get('lact_std') or 1.0),
            'weight_mean': float(r.get('weight_mean') or 0.0),
            'weight_std' : float(r.get('weight_std') or 1.0),
            'disease_mean': float(r.get('disease_mean') or 0.0),
            'disease_std' : float(r.get('disease_std') or 1.0),
        }
    return m

def load_model(weights_path, num_classes, device):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    sd = torch.load(weights_path, map_location=device)
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.",""):v for k,v in sd.items()}
    model.load_state_dict(sd)
    model.to(device); model.eval()
    return model

transform_input = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def compute_gradcam(model, pil_img, target_class, device):
    model.zero_grad()
    x = transform_input(pil_img).unsqueeze(0).to(device)
    features = None; grads = None
    def f_hook(module, inp, out):
        nonlocal features; features = out
    def b_hook(module, grad_in, grad_out):
        nonlocal grads; grads = grad_out[0]
    h1 = model.features[-1].register_forward_hook(f_hook)
    h2 = model.features[-1].register_full_backward_hook(b_hook)
    out = model(x)
    loss = out[0, target_class]
    loss.backward()
    fmap = features.detach().cpu()[0].numpy()
    g = grads.detach().cpu()[0].numpy()
    weights = g.mean(axis=(1,2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i,w in enumerate(weights):
        cam += w * fmap[i]
    cam = np.maximum(cam, 0.0)
    if cam.max() > 0:
        cam = cam / (cam.max()+1e-8)
    h1.remove(); h2.remove()
    return cam

def overlay_heatmap_on_bgr(orig_bgr, heatmap, alpha=0.45):
    hm = cv2.resize(heatmap, (orig_bgr.shape[1], orig_bgr.shape[0]))
    hm_u8 = np.uint8(255 * hm)
    hm_col = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 1-alpha, hm_col, alpha, 0)
    return overlay

def bbox_from_heatmap(cam, thresh_rel=0.2):
    h,w = cam.shape
    thr = cam.max() * thresh_rel
    mask = (cam >= thr).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0,0,w,h)
    x,y,ww,hh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return (x,y,x+ww,y+hh)

def bbox_from_silhouette(sil_bgr):
    gray = cv2.cvtColor(sil_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0,0,sil_bgr.shape[1],sil_bgr.shape[0])
    x,y,ww,hh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return (x,y,x+ww,y+hh)

# --- gaussian tabular ---
EPS = 1e-9
def gaussian_pdf(x, mu, sigma):
    sigma = max(sigma, 1e-2)
    return (1.0 / (math.sqrt(2*math.pi)*sigma)) * math.exp(-0.5 * ((x-mu)/sigma)**2)

def compute_tabular_probs(farmer_input, stats_map, breed_order):
    logls = []
    for b in breed_order:
        s = stats_map[b]
        ll = 0.0
        ll += math.log(max(gaussian_pdf(farmer_input['milk'], s['milk_mean'], s['milk_std']), EPS))
        ll += math.log(max(gaussian_pdf(farmer_input['lact'], s['lact_mean'], s['lact_std']), EPS))
        ll += math.log(max(gaussian_pdf(farmer_input['weight'], s['weight_mean'], s['weight_std']), EPS))
        ll += math.log(max(gaussian_pdf(farmer_input['disease'], s['disease_mean'], s['disease_std']), EPS))
        logls.append(ll)
    arr = np.exp(np.array(logls) - np.max(logls))
    return arr / (arr.sum() + EPS)

def append_row_csv(path, rowdict):
    exist = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rowdict.keys()))
        if not exist:
            w.writeheader()
        w.writerow(rowdict)

# ---------- main ----------
def run_one_image(img_path, milk_raw, lact_raw, weight_raw, disease_raw, model, breeds, stats_map):
    pil = Image.open(img_path).convert("RGB")
    x = transform_input(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    top1 = int(np.argmax(probs))
    pred_breed = breeds[top1]
    print("\nTop-3 CNN predictions:")
    for i in np.argsort(-probs)[:3]:
        print(f"  {breeds[int(i)]}: {probs[int(i)]:.4f}")

    cam = compute_gradcam(model, pil, top1, DEVICE)
    orig_bgr = np.array(pil)[:,:,::-1].copy()
    demo_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(img_path))[0] + "_demo.jpg")
    cv2.imwrite(demo_path, overlay_heatmap_on_bgr(orig_bgr, cam))
    print("Saved original+heatmap:", demo_path)

    sil_overlay_path = None
    sil_file = os.path.join(SILH_DIR, pred_breed + ".jpg")
    if os.path.isfile(sil_file):
        sil_bgr = cv2.imread(sil_file)
        bx_cam = bbox_from_heatmap(cv2.resize(cam, (orig_bgr.shape[1], orig_bgr.shape[0])))
        x0,y0,x1,y1 = bx_cam
        cx0,cy0,cx1,cy1 = map(int, [x0/ orig_bgr.shape[1]*224, y0/orig_bgr.shape[0]*224, x1/orig_bgr.shape[1]*224, y1/orig_bgr.shape[0]*224])
        cam_crop = cam[cy0:cy1, cx0:cx1]
        if cam_crop is None or cam_crop.size == 0:
            cam_crop = cam.copy()
        bx_sil = bbox_from_silhouette(sil_bgr)
        sx0,sy0,sx1,sy1 = bx_sil
        target_w, target_h = max(1,sx1-sx0), max(1,sy1-sy0)
        resized_cam = cv2.resize(cam_crop, (target_w,target_h))
        heat_canvas = np.zeros(sil_bgr.shape[:2], dtype=np.float32)
        heat_canvas[sy0:sy1, sx0:sx1] = resized_cam
        sil_overlay = overlay_heatmap_on_bgr(sil_bgr, heat_canvas)
        sil_overlay_path = os.path.join(OUT_DIR, pred_breed + "_sil_overlay.jpg")
        cv2.imwrite(sil_overlay_path, sil_overlay)
        print("Saved silhouette+heatmap:", sil_overlay_path)

    # show images
    if sys.platform == "darwin":
        subprocess.run(["open", demo_path])
        if sil_overlay_path: subprocess.run(["open", sil_overlay_path])
    else:
        print("👉 Please open:", demo_path, sil_overlay_path or "")

    # Feedback step
    print("\n=== Farmer feedback ===")
    ans = input("Accept prediction? (y/n): ").strip().lower()
    if ans == "y":
        print("Prediction accepted.")
    else:
        print("Enter scores 0-10 for each breed:")
        scores = {b: float(input(f"{b}: ")) for b in breeds}
        print("Manual scores:", scores)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img", required=True)
    p.add_argument("--milk", type=float, default=0.0)
    p.add_argument("--lact", type=float, default=0.0)
    p.add_argument("--weight", type=float, default=0.0)
    p.add_argument("--disease", type=float, default=0.0)
    args = p.parse_args()

    rows = load_trait_csv(TRAITS_CSV)
    breeds = get_breeds_from_traits(rows)
    stats_map = build_stats_map(rows)
    model = load_model(WEIGHTS, len(breeds), DEVICE)

    run_one_image(args.img, args.milk, args.lact, args.weight, args.disease, model, breeds, stats_map)