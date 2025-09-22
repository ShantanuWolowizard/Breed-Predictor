#!/usr/bin/env python3
"""
Final app.py

- Safe model loading: supports weights saved as state_dict OR as full model object.
- /predict : POST image + milk,lact,weight,disease -> returns top3, final_pred, crossbreed_flag,
             and URLs to heatmap and silhouette overlay images (saved in static/results).
- /feedback: POST JSON {image, predicted_final, action, scores} -> saved to feedback_log.csv
- CORS enabled.
"""

import os, io, csv, json, datetime, traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# --------- CONFIG ----------
BASE = os.getcwd()
WEIGHTS = os.path.join(BASE, "best_model.pth")
TRAITS_CSV = os.path.join(BASE, "breed_traits.csv")
SILH_DIR = os.path.join(BASE, "silhouettes")
RESULTS_DIR = os.path.join(BASE, "static", "results")
CROSSLOG = os.path.join(BASE, "crossbreed_log.csv")
FEEDBACK_LOG = os.path.join(BASE, "feedback_log.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SILH_DIR, exist_ok=True)

# Replace with your breed list (order must match model training class order)
BREEDS = ["Gir", "H_F", "Murrah", "jersey", "nili_ravi"]

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("Using device:", DEVICE)

# -------- transforms ----------
transform_input = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --------- model load (handles state_dict or full object) ----------
def load_model(weights_path, num_classes, device):
    # build architecture expected (EfficientNet-B0 used during training earlier in conversation)
    model = models.efficientnet_b0(weights=None)
    # replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    data = torch.load(weights_path, map_location=device)
    if isinstance(data, dict) and not any(hasattr(v, '__call__') for v in data.values()):
        # likely a state_dict
        try:
            model.load_state_dict(data)
            print("Loaded weights as state_dict.")
        except Exception as e:
            # try to detect if the state_dict was saved with 'module.' prefixes
            new_sd = {}
            for k,v in data.items():
                new_sd[k.replace("module.","")] = v
            model.load_state_dict(new_sd)
            print("Loaded weights as state_dict (module. prefix removed).")
    else:
        # saved full model object
        try:
            model = data
            print("Loaded full model object from checkpoint.")
            # ensure classifier size matches; if not, adjust (best-effort)
            if isinstance(model, nn.Module):
                # if classifier mismatched, attempt to re-init
                try:
                    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
                        if getattr(model.classifier[1], "out_features", None) != num_classes:
                            in_f = model.classifier[1].in_features
                            model.classifier[1] = nn.Linear(in_f, num_classes)
                            print("Replaced classifier to match num_classes.")
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError("Unknown weights format; please provide state_dict or full model.") from e

    model.to(device)
    model.eval()
    return model

# load model
if not os.path.isfile(WEIGHTS):
    raise FileNotFoundError(f"Model weights not found at: {WEIGHTS}")
model = load_model(WEIGHTS, len(BREEDS), DEVICE)

# -------- Grad-CAM helper ----------
def compute_gradcam(model, pil_img, class_idx):
    # returns heatmap in 0..1 with shape roughly 7x7.. then we'll resize
    model.zero_grad()
    x = transform_input(pil_img).unsqueeze(0).to(DEVICE)
    features = None
    grads = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    # attach hooks to last conv block
    # efficientnet_b0: model.features[-1] is last block (tensor shape CxHxW)
    target_module = model.features[-1]
    fh = target_module.register_forward_hook(forward_hook)
    bh = target_module.register_full_backward_hook(backward_hook)

    out = model(x)  # forward
    if isinstance(out, tuple):
        out_tensor = out[0]
    else:
        out_tensor = out
    score = out_tensor[0, class_idx]
    score.backward(retain_graph=False)

    fmap = features.detach().cpu().numpy()[0]   # C,H,W
    g = grads.detach().cpu().numpy()            # C,H,W
    weights = g.mean(axis=(1,2))                # C
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)  # H,W
    for i,w in enumerate(weights):
        cam += w * fmap[i]
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)
    else:
        cam = np.zeros_like(cam)
    fh.remove(); bh.remove()
    return cam  # float array 0..1

def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.45):
    # pil_img: PIL RGB, heatmap: float HxW (0..1)
    orig = np.array(pil_img)[:,:,::-1].copy()  # BGR for cv2
    hm_u8 = np.uint8(255 * cv2.resize(heatmap, (orig.shape[1], orig.shape[0])))
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 1-alpha, hm_color, alpha, 0)
    return overlay  # BGR numpy

# --------- CSV helpers ----------
def append_csv(path, rowdict):
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rowdict.keys()))
        if not exists:
            w.writeheader()
        w.writerow(rowdict)

# ---------- Flask app ----------
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error":"no image uploaded"}), 400
        f = request.files['image']
        # parse numeric fields (default 0)
        try:
            milk = float(request.form.get('milk', 0.0))
            lact = float(request.form.get('lact', 0.0))
            weight = float(request.form.get('weight', 0.0))
            disease = float(request.form.get('disease', 0.0))
        except Exception:
            milk = lact = weight = disease = 0.0

        # read image into PIL
        img_bytes = f.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # forward pass
        x = transform_input(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]  # numpy array

        # top3
        idxs = probs.argsort()[::-1][:3]
        top3 = [(BREEDS[int(i)], float(probs[int(i)])) for i in idxs]

        top1_idx = int(idxs[0]); top1_prob = float(probs[top1_idx])
        pred_breed = BREEDS[top1_idx]

        # grad-cam
        cam = compute_gradcam(model, pil, top1_idx)  # HxW float
        heat_overlay = overlay_heatmap_on_image(pil, cam, alpha=0.45)

        # save original heatmap overlay
        stamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        base = f"{os.path.splitext(f.filename)[0]}_{stamp}"
        heat_fn = f"{base}_heat.jpg"
        heat_path = os.path.join(RESULTS_DIR, heat_fn)
        cv2.imwrite(heat_path, heat_overlay)

        # silhouette overlay: try to read silhouette image for predicted breed
        sil_fn = None
        sil_in_path = None
        # try various extensions/cases
        for ext in (".jpg", ".jpeg", ".png"):
            cand = os.path.join(SILH_DIR, f"{pred_breed}{ext}")
            cand_low = os.path.join(SILH_DIR, f"{pred_breed.lower()}{ext}")
            if os.path.isfile(cand):
                sil_in_path = cand; break
            if os.path.isfile(cand_low):
                sil_in_path = cand_low; break
        if sil_in_path is not None:
            sil_img = Image.open(sil_in_path).convert("RGB")
            # resize silhouette to same size as overlay
            sil_resized = sil_img.resize((heat_overlay.shape[1], heat_overlay.shape[0]))
            # compute a resized cam for silhouette
            cam_resized = cv2.resize(cam, (sil_resized.size[0], sil_resized.size[1]))
            sil_overlay = overlay_heatmap_on_image(sil_resized, cam_resized, alpha=0.45)
            sil_fn = f"{base}_sil_overlay.jpg"
            sil_path = os.path.join(RESULTS_DIR, sil_fn)
            cv2.imwrite(sil_path, sil_overlay)
        else:
            sil_fn = None

        # fusion / crossbreed logic - minimal: if top1 < threshold OR top1-top2 small -> possible crossbreed
        top2_prob = float(probs[idxs[1]])
        cross_flag = bool((top1_prob < 0.7) or ((top1_prob - top2_prob) < 0.15))

        # log initial decision
        append_csv(CROSSLOG, {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "image_filename": f.filename,
            "predicted_cnn": pred_breed,
            "pred_conf_cnn": top1_prob,
            "top3": json.dumps(top3),
            "milk": milk, "lact": lact, "weight": weight, "disease": disease,
            "cross_flag": bool(cross_flag)
        })

        response = {
            "top3": top3,
            "final_pred": pred_breed,
            "cnn_conf": top1_prob,
            "crossbreed_flag": bool(cross_flag),
            "heatmap_url": f"/static/results/{heat_fn}",
            "silhouette_url": (f"/static/results/{sil_fn}" if sil_fn else None)
        }
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json(force=True)
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "image": data.get("image"),
            "predicted_final": data.get("predicted_final"),
            "action": data.get("action"),
            "scores": json.dumps(data.get("scores")) if data.get("scores") else ""
        }
        append_csv(FEEDBACK_LOG, entry)
        return jsonify({"status":"saved"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/static/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)

if __name__ == "__main__":
    # quick checks
    print("App root:", BASE)
    print("Results dir:", RESULTS_DIR)
    print("Silhouettes dir:", SILH_DIR)
    app.run(host="0.0.0.0", port=5000, debug=True)